# Export install packages in the current environment
> pip list --format=freeze > requirements.txt

# Submit a code on gpu
> sbatch /home/23r9802_chen/messou/multivariate_time_series/run-sample-gpux1.sh 

# Install tensorflow gpu
> pip install tensorflow[and-cuda]

# List running process
> watch -n 5 nvidia-smi

