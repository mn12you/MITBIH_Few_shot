#!/bin/bash
#SBATCH -A MST112277       # Account name/project number
#SBATCH -J train_siamense_relation     # Job name
#SBATCH --partition=gp1d
#SBATCH --nodes=1                ## 索取 2 節點
#SBATCH --ntasks-per-node=2      ## 每個節點運行 8 srun tasks
#SBATCH --cpus-per-task=4        ## 每個 srun task 索取 4 CPUs
#SBATCH --gres=gpu:2             ## 每個節點索取 8 GPUs
#SBATCH -o TWCC_log/%j.out           # Path to the standard output file
#SBATCH -e TWCC_log/%j.err           # Path to the standard error ouput file
#SBATCH --mail-user=s1212mn@gmail.com   # email
#SBATCH --mail-type=BEGIN,END




module purge
ml miniconda3
conda activate ECG_SHAP_39 #進入 conda 環境
# python -u tain_triplet.py --dataset=baseline_ml_diag --model_name=resnet1d1
# python -u "train_relation.py" --model_name=Siamese_CNN
python -u "train_CNN.py" --model_name=CNN 
# python -u "train_CNN_cwt.py" --model_name=CNN --test_set=spe

