#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# file paths
PRETRAIN_GRAPH="./dataset/H1N1_graph_2010.pt"
DOWNSTREAM_GRAPH="./dataset/H1N1_graph_2011.pt"
SOURCE_PL="./dataset/H1N1_pl_2010.pt"
TARGET_PL="./dataset/H1N1_pl_2011.pt"

# learning rates
LR1=0.000695223
LR2=2.24687e-05
LR3=0.00255742
LR4=3.08742e-05
LR5=0.00623945

python main.py \
  --pretrain_graph_file_path ${PRETRAIN_GRAPH} \
  --downstream_graph_file_path ${DOWNSTREAM_GRAPH} \
  --source_pl_file_path ${SOURCE_PL} \
  --target_pl_file_path ${TARGET_PL} \
  --lr1 ${LR1} \
  --lr2 ${LR2} \
  --lr3 ${LR3} \
  --lr4 ${LR4} \
  --lr5 ${LR5}