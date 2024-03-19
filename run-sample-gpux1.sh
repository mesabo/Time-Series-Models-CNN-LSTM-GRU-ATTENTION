#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodelist=ai-gpgpu14
source ~/.bashrc
hostname
echo USED GPUs=$CUDA_VISIBLE_DEVICES
source  activate tf-gpu
pwd
cd /home/23r9802_chen/messou/multivariate_time_series/

python ./src/sample.py
