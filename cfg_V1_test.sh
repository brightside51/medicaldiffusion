#!/bin/bash
#
#SBATCH --partition=gpu_min80gb                                   # Partition (check with "$sinfo")
#SBATCH --output=../MetaBreast/logs/medical_diffusion/V1/output_sample.out           # Filename with STDOUT. You can use special flags, such as %N and %j.
#SBATCH --error=../MetaBreast/logs/medical_diffusion/V1/error_sample.err             # (Optional) Filename with STDERR. If ommited, use STDOUT.
#SBATCH --job-name=meddif                                        # (Optional) Job name
#SBATCH --time=14-00:00                                             # (Optional) Time limit (D: days, HH: hours, MM: minutes)
#SBATCH --qos=gpu_min80GB                                           # (Optional) 01.ctm-deep-05

#Commands / scripts to run (e.g., python3 train.py)
#$ enroot import docker://nvcr.io#nvidia/pytorch:21.04-py3
#$ enroot create --name container_name nvidia+pytorch+21.04-py3.sqsh

#conda activate medicaldiffusion
#PL_TORCH_DISTRIBUTED_BACKEND=gloo
#python train/train_vqgan.py dataset=default dataset.root_dir=../../../../nas-ctm01/datasets/private/METABREST/T1W_Breast/video_data model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='own_dataset' model.precision=16 model.embedding_dim=64 model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=8 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=5 model.n_codes=16384 model.accumulate_grad_batches=1 
#python train/train_ddpm.py model=ddpm dataset=default model.results_folder_postfix='own_dataset' model.vqgan_ckpt=../../../../nas-ctm01/homes/pfsousa/MedDiff/DEFAULT/own_dataset/lightning_logs/version_5610/checkpoints/latest_checkpoint.ckpt model.diffusion_img_size=64 model.diffusion_depth_size=8 model.diffusion_num_channels=64 model.dim_mults=[1,2,4,8] model.batch_size=5 model.gpus=1
#python train/train_vqgan.py dataset=default dataset.root_dir=../../../../nas-ctm01/datasets/private/METABREST/T1W_Breast/video_data model=vq_gan_3d model.gpus=1 model.default_root_dir_postfix='own_dataset' model.precision=16 model.embedding_dim=8 model.n_hiddens=16 model.downsample=[2,2,2] model.num_workers=4 model.gradient_clip_val=1.0 model.lr=3e-4 model.discriminator_iter_start=10000 model.perceptual_weight=4 model.image_gan_weight=1 model.video_gan_weight=1 model.gan_feat_weight=4 model.batch_size=1 model.n_codes=16384 model.accumulate_grad_batches=1 
#python train/train_ddpm.py model=ddpm dataset=default model.results_folder_postfix='own_dataset' model.vqgan_ckpt=../../../../nas-ctm01/homes/pfsousa/MedDiff/DEFAULT/own_dataset/lightning_logs/version_6008/checkpoints/latest_checkpoint.ckpt model.diffusion_img_size=64 model.diffusion_depth_size=15 model.diffusion_num_channels=8 model.dim_mults=[1,2,4,8] model.batch_size=2 model.gpus=1
python train/sample_V1.py