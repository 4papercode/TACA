from model.GNN_model import GNN, GNNLoRA
import torch
import torch.nn as nn
import os
from torch_geometric.transforms import SVDFeatureReduction
from util import get_dataset, act, SMMDLoss, mkdir, get_ppr_weight
from util import get_few_shot_mask, batched_smmd_loss, batched_gct_loss
from torch_geometric.utils import to_dense_adj, add_remaining_self_loops, degree
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader
import random
import networkx as nx
import gudhi as gd

from pd import pairs_from_gudhi, compute_pd
from torch_topological.nn import WassersteinDistance, PersistenceInformation
from torch_topological.nn import SlicedWassersteinDistance

import ot
from sklearn.metrics import recall_score, f1_score

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    print(f"Random seed set to {seed}")

class GlobalFeatureFusion(nn.Module):
    def __init__(self, pl_dim, embed_dim, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.pl_proj = nn.Linear(pl_dim, embed_dim)        
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)        
        self.layernorm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, node_features, pl_feature):
        N = node_features.shape[0]
        projected_pl = self.pl_proj(pl_feature)  # Shape: [embed_dim]
        expanded_pl = projected_pl.unsqueeze(0).expand(N, -1).unsqueeze(1) # [N, 1, embed_dim]
        unsqueezed_nodes = node_features.unsqueeze(1) # [N, 1, embed_dim]       
        sequences = torch.cat([unsqueezed_nodes, expanded_pl], dim=1) # [node_i, global_feature]  
        attn_output, _ = self.attention(sequences, sequences, sequences) # attn_output shape: [N, 2, embed_dim]
        fused_nodes_update = attn_output[:, 0, :] # Shape: [N, embed_dim]        
        h = self.layernorm(node_features + fused_nodes_update)
        h = h + self.ffn(h)    
        return h

class Projector(nn.Module):
    def __init__(self, input_size, output_size):
        super(Projector, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.initialize()

    def forward(self, x):
        return self.fc(x)

    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)


class LogReg(nn.Module):
    def __init__(self, hid_dim, out_dim):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(hid_dim, out_dim)
        self.initialize()

    def forward(self, x):
        return self.fc(x)
    
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)

def transfer(pretrain_data, downstream_data, pretrained_state, source_pl_features, target_pl_features, args, config, gpu_id, seed, is_reduction = False, year_tag="unknown"):
    device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')
    set_seed(seed)
    if is_reduction:
        feature_reduce = SVDFeatureReduction(out_channels=100)
        pretrain_data = feature_reduce(pretrain_data)
        downstream_data = feature_reduce(downstream_data)
    
    pretrain_data.edge_index = add_remaining_self_loops(pretrain_data.edge_index, num_nodes=pretrain_data.num_nodes)[0]
    downstream_data.edge_index = add_remaining_self_loops(downstream_data.edge_index, num_nodes=downstream_data.num_nodes)[0]
    pretrain_data = pretrain_data.to(device)
    downstream_data = downstream_data.to(device)

    epsilon = 1e-8
    # target_pl
    target_pl_features = torch.from_numpy(target_pl_features).float().to(device)
    mean_t = target_pl_features.mean(axis=0)
    std_t = target_pl_features.std(axis=0)
    target_pl_features_scaled = (target_pl_features - mean_t) / (std_t + epsilon)
    # source_pl
    source_pl_features = torch.from_numpy(source_pl_features).float().to(device)
    mean_s = source_pl_features.mean(axis=0)
    std_s = source_pl_features.std(axis=0)
    source_pl_features_scaled = (source_pl_features - mean_s) / (std_s + epsilon)    

    gnn = GNN(pretrain_data.x.shape[1], config['output_dim'], act(config['activation']), config['gnn_type'], config['num_layers'])
    gnn.load_state_dict(pretrained_state['gnn'])
    gnn.to(device)
    gnn.eval()
    for param in gnn.conv.parameters():
        param.requires_grad = False

    pl_dim = source_pl_features_scaled.shape[0]
    pretrain_fusion_module = GlobalFeatureFusion(pl_dim=pl_dim, embed_dim=pretrain_data.x.shape[1], num_heads=4)
    pretrain_fusion_module.load_state_dict(pretrained_state['fusion_module'])
    pretrain_fusion_module.to(device)
    pretrain_fusion_module.eval()
    with torch.no_grad():
        source_pl_projection = pretrain_fusion_module.pl_proj(source_pl_features_scaled).detach()

    fusion_module = GlobalFeatureFusion(pl_dim=pl_dim, embed_dim=pretrain_data.x.shape[1], num_heads=4)
    fusion_module.load_state_dict(pretrained_state['fusion_module'])
    fusion_module.to(device)
    # fusion_module.eval()
    # for param in fusion_module.parameters():
    #     param.requires_grad = False
    fusion_module.train() 

    gnn2 = GNNLoRA(pretrain_data.x.shape[1], config['output_dim'], act(config['activation']), gnn, config['gnn_type'], config['num_layers'], r=args.r)
    gnn2.to(device)
    gnn2.train()

    SMMD = SMMDLoss().to(device)

    projector = Projector(downstream_data.x.shape[1], pretrain_data.x.shape[1])
    projector = projector.to(device)
    projector.train()

    num_classes = 13
    logreg = LogReg(config['output_dim'], num_classes)
    logreg = logreg.to(device)
    loss_fn = nn.CrossEntropyLoss()

    filtration_hidden = 256
    num_filtrations = 1

    filtration_mlp_out = nn.Sequential(
        nn.Linear(pretrain_data.x.shape[1], filtration_hidden),
        nn.ReLU(),
        nn.Linear(filtration_hidden, num_filtrations)
    ).to(device)

    wdist = SlicedWassersteinDistance().to('cpu')

    index = np.arange(downstream_data.x.shape[0])
    np.random.shuffle(index)
    train_mask = torch.zeros(downstream_data.x.shape[0]).bool().to(device)
    test_mask = torch.zeros(downstream_data.x.shape[0]).bool().to(device)
    train_mask[index[:int(len(index) * 0.7)]] = True
    test_mask[index[int(len(index) * 0.7):]] = True

    mask = torch.zeros((downstream_data.x.shape[0], downstream_data.x.shape[0])).to(device)
    idx_a = torch.tensor([]).to(device)
    idx_b = torch.tensor([]).to(device)
    for i in range(num_classes):
        train_idx = torch.nonzero(train_mask, as_tuple=False).squeeze()
        train_label = downstream_data.y[train_idx]
        idx_a = torch.concat((idx_a, train_idx[train_label == i].repeat_interleave(len(train_idx[train_label == i]))))
        idx_b = torch.concat((idx_b, train_idx[train_label == i].repeat(len(train_idx[train_label == i]))))
    mask = torch.sparse_coo_tensor(indices=torch.stack((idx_a, idx_b)), values=torch.ones(len(idx_a)).to(device), size=[downstream_data.x.shape[0], downstream_data.x.shape[0]]).to_dense()
    mask = args.sup_weight * (mask - torch.diag_embed(torch.diag(mask))) + torch.eye(downstream_data.x.shape[0]).to(device)
    
    optimizer = torch.optim.Adam([
        {"params": projector.parameters(), 'lr': args.lr1, 'weight_decay': args.wd1},
        {"params": logreg.parameters(), 'lr': args.lr2, 'weight_decay': args.wd2},
        {"params": gnn2.parameters(), 'lr': args.lr3, 'weight_decay': args.wd3},
        {"params": fusion_module.parameters(), 'lr': args.lr4, 'weight_decay': args.wd4},
        {"params": filtration_mlp_out.parameters(), 'lr': args.lr5, 'weight_decay': args.wd5}
    ])

    downstream_data.train_mask = train_mask
    downstream_data.test_mask = test_mask

    train_labels = downstream_data.y[train_mask]
    test_labels = downstream_data.y[test_mask]

    pretrain_graph_loader = DataLoader(pretrain_data.x, batch_size=128, shuffle=True)

    best_epoch = 0
    best_loss = 0
    best_train_acc = 0
    best_train_recall = 0
    best_train_f1 = 0
    best_test_acc = 0
    best_test_recall = 0
    best_test_f1 = 0

    num_nodes = downstream_data.x.shape[0]
    target_adj = to_dense_adj(downstream_data.edge_index, max_num_nodes=num_nodes)[0]
    ppr_weight = get_ppr_weight(downstream_data)

    row, col = downstream_data.edge_index
    deg = degree(col, downstream_data.x.size(0), dtype=torch.float).to(device)
    deg_mean = deg.mean()
    deg_std = deg.std()
    f_v_in = (deg - deg_mean) / (deg_std + 1e-6)
    pd0_in = compute_pd(downstream_data.edge_index, f_v_in)
    pd0_in_cpu = pd0_in.detach().cpu()

    for epoch in range(0, args.num_epochs):
        logreg.train()
        projector.train()
        gnn2.train()
        fusion_module.train()
        filtration_mlp_out.train()
  
        pos_weight = float(target_adj.shape[0] * target_adj.shape[0] - target_adj.sum()) / target_adj.sum()
        weight_mask = target_adj.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0)).to(device)
        weight_tensor[weight_mask] = pos_weight

        feature_map = projector(downstream_data.x)

        f_v_out = filtration_mlp_out(feature_map).squeeze()
        pd0_out = compute_pd(downstream_data.edge_index, f_v_out)
        
        # with torch.no_grad():
        feature_map_w_pl = fusion_module(feature_map, target_pl_features_scaled)
        emb, emb1, emb2 = gnn2(feature_map_w_pl, downstream_data.edge_index)
        optimizer.zero_grad()

        if pd0_in.numel() == 0 and pd0_out.numel() == 0:
            loss_topo = torch.tensor(0.0, device=device)
        else:
            pd0_out_cpu = pd0_out.cpu()
            loss_topo_cpu = wdist(PersistenceInformation(pairing=None, diagram=pd0_in_cpu, dimension=0), PersistenceInformation(pairing=None, diagram=pd0_out_cpu, dimension=0))
            loss_topo = loss_topo_cpu.to(device)

        smmd_loss_f = batched_smmd_loss(feature_map, pretrain_graph_loader, SMMD, ppr_weight, 128)
        ct_loss = 0.5 * (batched_gct_loss(emb1, emb2, 1000, mask, args.tau) + batched_gct_loss(emb2, emb1, 1000, mask, args.tau)).mean()

        logits = logreg(emb)
        rec_adj = torch.sigmoid(torch.matmul(torch.softmax(logits, dim=1), torch.softmax(logits, dim=1).T))
        loss_rec = F.binary_cross_entropy(rec_adj.view(-1), target_adj.view(-1), weight=weight_tensor)

        train_logits = logits[train_mask]
        train_preds = torch.argmax(train_logits, dim=1)
        cls_loss = loss_fn(train_logits, train_labels)

        current_target_pl_projection = fusion_module.pl_proj(target_pl_features_scaled)
        loss_pl_align = F.mse_loss(current_target_pl_projection, source_pl_projection)

        loss = args.l1 * cls_loss + args.l2 * smmd_loss_f +  args.l3 * ct_loss + args.l4 * loss_rec + args.l5 * loss_topo + args.l6 * loss_pl_align
        loss.backward()
        optimizer.step()

        train_acc = torch.sum(train_preds == train_labels).float() / train_labels.shape[0]
        train_preds_np = train_preds.cpu().numpy()
        train_labels_np = train_labels.cpu().numpy()
        train_recall = recall_score(train_labels_np, train_preds_np, average='macro', zero_division=0)
        train_f1 = f1_score(train_labels_np, train_preds_np, average='macro', zero_division=0)

        logreg.eval()
        projector.eval()
        filtration_mlp_out.eval()
        fusion_module.eval()

        with torch.no_grad():
            test_logits = logits[test_mask]
            test_preds = torch.argmax(test_logits, dim=1)
            test_acc = torch.sum(test_preds == test_labels).float() / test_labels.shape[0]

            test_labels_np = test_labels.cpu().numpy()
            test_preds_np = test_preds.cpu().numpy()
            test_recall = recall_score(test_labels_np, test_preds_np, average='macro', zero_division=0)
            test_f1 = f1_score(test_labels_np, test_preds_np, average='macro', zero_division=0)

            print(
                'Epoch: {}, loss: {:.4f}, cls: {:.4f}, smmd: {:.4f}, ct: {:.4f}, rec: {:.4f}, topo: {:.4f}, pl_align: {:.4f}, '
                'train_acc: {:.4f}, '
                'test_acc: {:4f}'.format(
                    epoch,
                    loss,
                    cls_loss,
                    smmd_loss_f,
                    ct_loss,
                    loss_rec,
                    loss_topo,
                    loss_pl_align,
                    train_acc,
                    test_acc,
                )
            )

            if best_test_acc <= test_acc:
                best_test_acc = test_acc
                best_test_recall = test_recall
                best_test_f1 = test_f1
                best_epoch = epoch + 1
                best_loss = loss
                best_train_acc = train_acc
                best_train_recall = train_recall
                best_train_f1 = train_f1

                print("✓ New best test_acc.")

                preds_to_save = test_preds.cpu().numpy()
                labels_to_save = test_labels.cpu().numpy()
                output_data = np.vstack((preds_to_save, labels_to_save)).T
    print(
        '{}: epoch: {}, train_acc: {:4f}, test_acc: {:4f}'.format(
            year_tag,
            best_epoch,
            best_train_acc,
            best_test_acc,
        )
    )

    result_path = './result'
    mkdir(result_path)

    with open(result_path + '/result.txt', 'a') as f:
        f.write(
            'seed: %d, r: %d, lrs: (%g, %g, %g, %g, %g), '
            'wds: (%g, %g, %g, %g, %g), '
            'ls: (%g, %g, %g, %g, %g, %g), '
            '%s: epoch: %d, train_loss: %f, train_acc: %f, '
            'test_acc: %f\n'
            %
            (
                seed,
                args.r,
                args.lr1,
                args.lr2,
                args.lr3,
                args.lr4,
                args.lr5,
                args.wd1,
                args.wd2,
                args.wd3,
                args.wd4,
                args.wd5,
                args.l1,
                args.l2,
                args.l3,
                args.l4,
                args.l5,
                args.l6,
                year_tag,
                best_epoch,
                best_loss,
                best_train_acc,
                best_test_acc,
            )
        )