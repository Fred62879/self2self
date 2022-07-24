#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=pdr3_s2s_512_10_inpaint_179_479_12
#SBATCH --output=./output/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/env/astro_env/bin/activate
cd ../

#python main.py --config configs/inpaint.ini --sample_ratio_cho $SLURM_ARRAY_TASK_ID
python main.py --config configs/inpaint.ini --sample_ratio_cho 1 --recon_restore --mask_bands 179
python main.py --config configs/inpaint.ini --sample_ratio_cho 2 --recon_restore --mask_bands 179

python main.py --config configs/inpaint.ini --sample_ratio_cho 1 --recon_restore --mask_bands 479
python main.py --config configs/inpaint.ini --sample_ratio_cho 2 --recon_restore --mask_bands 479
