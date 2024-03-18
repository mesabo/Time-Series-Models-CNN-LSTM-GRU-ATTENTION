#!/bin/bash
#SBATCH --gres=gpu:1
source ~/.bashrc
hostname
echo USED GPUs=$CUDA_VISIBLE_DEVICES
source activate hoge
cd /home/23r9802_chen/messou/multivariate_time_series/
python ./src/sample.py