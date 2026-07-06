import yaml
from yaml import SafeLoader
import argparse
import torch
from pre_train import pretrain
from model.TACA import transfer
import numpy as np
import random
import os
import re

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

def extract_year(file_path):
    basename = os.path.basename(file_path)
    match = re.search(r"\d{4}", basename)
    if match is None:
        raise ValueError(f"Cannot extract year from file path: {file_path}")
    return match.group(0)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrain_graph_file_path", type=str, required=True)
    parser.add_argument("--downstream_graph_file_path", type=str, required=True)
    parser.add_argument("--source_pl_file_path", type=str, required=True)
    parser.add_argument("--target_pl_file_path", type=str, required=True)

    parser.add_argument("--lr1", type=float, required=True)
    parser.add_argument("--lr2", type=float, required=True)
    parser.add_argument("--lr3", type=float, required=True)
    parser.add_argument("--lr4", type=float, required=True)
    parser.add_argument("--lr5", type=float, required=True)

    return parser.parse_args()

def build_args(args_config, cli_args):
    args = argparse.Namespace()

    args.r = int(args_config["r"])
    args.tau = float(args_config["tau"])
    args.sup_weight = float(args_config["sup_weight"])

    args.lr1 = cli_args.lr1
    args.lr2 = cli_args.lr2
    args.lr3 = cli_args.lr3
    args.lr4 = cli_args.lr4
    args.lr5 = cli_args.lr5

    args.wd1 = float(args_config["wd1"])
    args.wd2 = float(args_config["wd2"])
    args.wd3 = float(args_config["wd3"])
    args.wd4 = float(args_config["wd4"])
    args.wd5 = float(args_config["wd5"])

    args.l1 = float(args_config["l1"])
    args.l2 = float(args_config["l2"])
    args.l3 = float(args_config["l3"])
    args.l4 = float(args_config["l4"])
    args.l5 = float(args_config["l5"])
    args.l6 = float(args_config["l6"])

    args.num_epochs = int(args_config["num_epochs"])

    return args


def main():
    cli_args = parse_args()
    set_seed(42)

    source_year = extract_year(cli_args.pretrain_graph_file_path)
    target_year = extract_year(cli_args.downstream_graph_file_path)
    year_tag = f"{source_year} to {target_year}"
    print("Transfer year:", year_tag)
    
    pretrain_graph = torch.load(cli_args.pretrain_graph_file_path)
    downstream_graph = torch.load(cli_args.downstream_graph_file_path)
    source_pl_features = torch.load(cli_args.source_pl_file_path)
    target_pl_features = torch.load(cli_args.target_pl_file_path)

    print("\n" + "="*30 + "\nPRE-TRAINING\n" + "="*30)
    config_pretrain = yaml.load(open('config.yaml'), Loader=SafeLoader)['H1N1']    
    gpu_id = 0
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    pretrained_state = pretrain(pretrain_graph, source_pl_features, "GRACE", config_pretrain, gpu_id, is_reduction=False)

    if pretrained_state:
        print("\n" + "="*30 + "\nLoRA FINE-TUNING\n" + "="*30)
        args_config = yaml.load(open('config2.yaml'), Loader=SafeLoader)['public']['H1N1']
        args = build_args(args_config, cli_args)      
        config_finetune = config_pretrain       
        for seed in range(5):
            transfer(
                pretrain_graph,
                downstream_graph,
                pretrained_state,
                source_pl_features,
                target_pl_features,
                args,
                config_finetune,
                gpu_id,
                seed=seed,
                is_reduction=False,
                year_tag=year_tag,
            )

if __name__ == '__main__':
    main()
