#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --account=def-kyi-ab
#SBATCH --job-name=pdr3_s2s_512_10_inpaint_179
#SBATCH --output=./output/%x-%j.out
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=40000

source ~/env/astro_env/bin/activate
cd ../

#python main.py --config configs/inpaint.ini --sample_ratio_cho $SLURM_ARRAY_TASK_ID
python main.py --config configs/inpaint.ini --sample_ratio_cho 1 --recon_restore
python main.py --config configs/inpaint.ini --sample_ratio_cho 2 --recon_restore
python main.py --config configs/inpaint.ini --sample_ratio_cho 3 --recon_restore
python main.py --config configs/inpaint.ini --sample_ratio_cho 4 --recon_restore
python main.py --config configs/inpaint.ini --sample_ratio_cho 5 --recon_restore
python main.py --config configs/inpaint.ini --sample_ratio_cho 6 --recon_restore
python main.py --config configs/inpaint.ini --sample_ratio_cho 7 --recon_restore
#python main.py --config configs/inpaint.ini --sample_ratio_cho 8 --recon_restore
#python main.py --config configs/inpaint.ini --sample_ratio_cho 9 --recon_restore
