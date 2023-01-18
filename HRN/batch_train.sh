#!/bin/bash
##SBATCH -A mobility_arfs
##SBATCH -n 3
#SBATCH --partition=ihub
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=6
##SBATCH --nodelist=gnode110
#SBATCH --mem-per-cpu=12000
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END
##SBATCH --reservation ndq



source /home2/praffulkumar/miniconda3/etc/profile.d/conda.sh
conda activate conda_env_1

cd /home2/praffulkumar/HRN/

echo ---Starting Training---

CUDA_VISIBLE_DEVICES=0,1,2,3 python hrn-fgvd.py 
###CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node 4 hrn-fgvd.py
echo ----Training Complete----

