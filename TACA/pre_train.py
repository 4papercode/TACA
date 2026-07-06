from model.GNN_model import GNN
from model.GRACE_model import GRACE
from time import time
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from util import get_dataset, act, mkdir
from torch_geometric.transforms import SVDFeatureReduction

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
        # attn_output shape: [N, 2, embed_dim]
        attn_output, _ = self.attention(sequences, sequences, sequences)
        fused_nodes_update = attn_output[:, 0, :] # Shape: [N, embed_dim]        

        h = self.layernorm(node_features + fused_nodes_update)
        h = h + self.ffn(h)    
        return h

def pretrain(data, pl_features, pretext, config, gpu, is_reduction=False):
    if is_reduction:
        feature_reduce = SVDFeatureReduction(out_channels=100)
        data = feature_reduce(data)
    device = torch.device('cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    pl_features = torch.from_numpy(pl_features).float().to(device)
    mean = pl_features.mean(axis=0)
    std = pl_features.std(axis=0)
    epsilon = 1e-8
    pl_features_scaled = (pl_features - mean) / (std + epsilon)

    pre_trained_model_path = './pre_trained_gnn/'
    mkdir(pre_trained_model_path)
    print("create PreTrain instance...")
    input_dim = data.x.shape[1]
    output_dim = config['output_dim']
    num_proj_dim = config['num_proj_dim']
    activation = act(config['activation'])
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    num_epochs = config['num_epochs']
    tau = config['tau']
    gnn_type = config['gnn_type']
    num_layers = config['num_layers']
    drop_edge_rate = config['drop_edge_rate']
    drop_feature_rate = config['drop_feature_rate']
    gnn = GNN(input_dim, output_dim, activation, gnn_type, num_layers)
    if pretext == 'GRACE':
        pretrain_model = GRACE(gnn, output_dim, num_proj_dim, drop_edge_rate, drop_feature_rate, tau)
    else:
        pretrain_model = GRACE(gnn, output_dim, num_proj_dim, drop_edge_rate, drop_feature_rate, tau)
    pretrain_model.to(device)
    print("pre-training...")
    
    pl_dim = pl_features_scaled.shape[0]
    fusion_module = GlobalFeatureFusion(pl_dim=pl_dim, embed_dim=input_dim, num_heads=4)
    fusion_module.to(device)
    optimizer = torch.optim.Adam([
        {"params": pretrain_model.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay},
        {"params": fusion_module.parameters(), 'lr': learning_rate, 'weight_decay': weight_decay}
        ])

    start = time()
    prev = start
    pretrain_model.train()
    fusion_module.train()
    min_loss = 100000
    model_path = pre_trained_model_path + "{}.pth".format('pre')
    best_gnn_state = None
    best_fusion_module_state = None

    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        feature_w_pl = fusion_module(data.x, pl_features_scaled)
        loss = pretrain_model.compute_loss(feature_w_pl, data.edge_index)
        loss.backward()
        optimizer.step()
        now = time()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        if min_loss > loss:
            min_loss = loss
            best_gnn_state = pretrain_model.gnn.state_dict()
            best_fusion_module_state = fusion_module.state_dict()  
            torch.save({
                'gnn': best_gnn_state,
                'fusion_module': best_fusion_module_state,
            }, model_path)
            print("+++model saved ! {}.pth".format('pre'))
    print("=== Final ===")

    return {
        'gnn': best_gnn_state,
        'fusion_module': best_fusion_module_state
    }