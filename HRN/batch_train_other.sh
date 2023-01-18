#!/bin/bash
##SBATCH -A research
##SBATCH -n 3
#SBATCH --partition=ihub
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=6
#SBATCH --nodelist=gnode112
#SBATCH --mem-per-cpu=12000
#SBATCH --time=4-00:00:00
##SBATCH --mail-type=END
##SBATCH --reservation ndq



source /home2/praffulkumar/miniconda3/etc/profile.d/conda.sh
conda activate conda_env_1

cd /home2/praffulkumar/HRN/

echo ---Starting Training---

CUDA_VISIBLE_DEVICES=0,1 python hrn-fgvd(1).py 
echo ----Training Complete----

