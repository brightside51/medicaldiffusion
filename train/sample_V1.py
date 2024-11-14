GPU_NUM = 0

import torch
import os
import sys
from re import I
os.environ["CUDA_VISIBLE_DEVICES"]=f"{GPU_NUM}"
#%env CUDA_VISIBLE_DEVICES=1
#torch.cuda.set_device(GPU_NUM)
device = f'cuda'
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)
sys.path.append('vq_gan_3d/model')
from vqgan import VQGAN
sys.path.append('evaluation/pytorch_ssim')
from ssim_script import msssim_3d
#sys.path.append('../')
#from vq_gan_3d.model.vqgan import VQGAN
sys.path.append('dataset')
from brats import BRATSDataset
from mrnet import MRNetDataset
from adni import ADNIDataset
from duke import DUKEDataset
from lidc import LIDCDataset
from default import DEFAULTDataset
#from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, DEFAULTDataset
sys.path.append('ddpm')
from diffusion import Unet3D, GaussianDiffusion, Trainer
#from train.get_dataset import get_dataset
sys.path.append('train')
from get_dataset_V1 import get_dataset

import matplotlib.pyplot as plt
import SimpleITK as sitk
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from omegaconf import OmegaConf
#import pytorch_ssim
import SimpleITK as sitk

# ============================================================================================

USE_DATASET='DEFAULT'

DDPM_CHECKPOINT = "/nas-ctm01/homes/pfsousa/MedDiff/ddpm/V1/own_dataset/model-42.pt"
#VQGAN_CHECKPOINT = '/nas-ctm01/homes/pfsousa/MedDiff/V1/own_dataset/lightning_logs/version_6008/checkpoints/latest_checkpoint.ckpt'
VQGAN_CHECKPOINT = 'V1/own_dataset/lightning_logs/version_6008/checkpoints/latest_checkpoint.ckpt'

if USE_DATASET=='DEFAULT':
	
	with initialize(config_path="../config/"):
			cfg=compose(config_name="base_cfg.yaml", overrides=[
				"model=ddpm",
				"dataset=default",
				f"model.vqgan_ckpt={VQGAN_CHECKPOINT}",
				"model.diffusion_img_size=32",
				"model.diffusion_depth_size=15",
				"model.diffusion_num_channels=8",
				"model.dim_mults=[1,2,4,8]",
				"model.batch_size=5 ",
				"model.gpus=0 ",
				])

"""
if USE_DATASET=='LIDC':
	#DDPM_CHECKPOINT = '/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/ddpm/LIDC/model-78.pt'
	#VQGAN_CHECKPOINT = '/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/LIDC/lightning_logs/version_0/checkpoints/epoch\=100-step\=102000-train/recon_loss\=0.33.ckpt'

	with initialize(config_path="../config/"):
			cfg=compose(config_name="base_cfg.yaml", overrides=[
				"model=ddpm",
				"dataset=lidc",
				f"model.vqgan_ckpt={VQGAN_CHECKPOINT}",
				"model.diffusion_img_size=16",
				"model.diffusion_depth_size=16",
				"model.diffusion_num_channels=8",
				"model.dim_mults=[1,2,4,8]",
				"model.batch_size=40 ",
				"model.gpus=0 ",
				])
elif USE_DATASET=='ADNI':
	DDPM_CHECKPOINT = '/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/ddpm/ADNI/roi/model-34.pt'
	VQGAN_CHECKPOINT = '/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/ADNI/roi/lightning_logs/version_1/checkpoints/epoch\=99-step\=99000-train/recon_loss\=0.05.ckpt'

	with initialize(config_path="../config/"):
			cfg=compose(config_name="base_cfg.yaml", overrides=[
				"model=ddpm",
			 	"dataset=adni",
			 	f"model.vqgan_ckpt={VQGAN_CHECKPOINT}",
			 	"model.diffusion_img_size=32",
			 	"model.diffusion_depth_size=32",
			 	"model.diffusion_num_channels=8",
			 	"model.dim_mults=[1,2,4,8]",
			 	"model.batch_size=40 ",
			 	"model.gpus=0 ",
				])
elif USE_DATASET=='DUKE':
	DDPM_CHECKPOINT = '/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/ddpm/DUKE/model-83.pt'
	VQGAN_CHECKPOINT = '/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/DUKE/lightning_logs/version_0/checkpoints/epoch\=58-step\=108000-train/recon_loss\=0.17.ckpt'

	with initialize(config_path="../config/"):
			cfg=compose(config_name="base_cfg.yaml", overrides=[
				"model=ddpm",
			 	"dataset=duke",
			 	f"model.vqgan_ckpt={VQGAN_CHECKPOINT}",
			 	"model.diffusion_img_size=32",
			 	"model.diffusion_depth_size=4",
			 	"model.diffusion_num_channels=8",
			 	"model.dim_mults=[1,2,4,8]",
			 	"model.batch_size=40 ",
			 	"model.gpus=0 ",
				])
elif USE_DATASET=='MRNet':
	DDPM_CHECKPOINT = '/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/ddpm/MRNet/model-77.pt'
	VQGAN_CHECKPOINT = '/data/home/firas/Desktop/work/other_groups/medicaldiffusion/checkpoints/vq_gan/MRNet/lightning_logs/version_0/checkpoints/epoch\=126-step\=114000-train/recon_loss\=0.47.ckpt'

	with initialize(config_path="../config/"):
			cfg=compose(config_name="base_cfg.yaml", overrides=[
				"model=ddpm",
			 	"dataset=mrnet",
			 	f"model.vqgan_ckpt={VQGAN_CHECKPOINT}",
			 	"model.diffusion_img_size=32",
			 	"model.diffusion_depth_size=4",
			 	"model.diffusion_num_channels=8",
			 	"model.dim_mults=[1,2,4,8]",
			 	"model.batch_size=40 ",
			 	"model.gpus=0 ",
				])
"""

train_dataset, _, _ = get_dataset(cfg)
print(cfg.model.diffusion_img_size)

# ============================================================================================

model = Unet3D(
    dim=cfg.model.diffusion_img_size,
    dim_mults=cfg.model.dim_mults,
    channels=cfg.model.diffusion_num_channels,
).cuda()

diffusion = GaussianDiffusion(
    model,
    vqgan_ckpt=cfg.model.vqgan_ckpt,
    image_size=cfg.model.diffusion_img_size,
    num_frames=cfg.model.diffusion_depth_size,
    channels=cfg.model.diffusion_num_channels,
    timesteps=cfg.model.timesteps,
    # sampling_timesteps=cfg.model.sampling_timesteps,
    loss_type=cfg.model.loss_type,
    # objective=cfg.objective
).cuda()

trainer = Trainer(
    diffusion,
    cfg=cfg,
    dataset=train_dataset,
    train_batch_size=cfg.model.batch_size,
    save_and_sample_every=cfg.model.save_and_sample_every,
    train_lr=cfg.model.train_lr,
    train_num_steps=cfg.model.train_num_steps,
    gradient_accumulate_every=cfg.model.gradient_accumulate_every,
    ema_decay=cfg.model.ema_decay,
    amp=cfg.model.amp,
    num_sample_rows=cfg.model.num_sample_rows,
    results_folder=cfg.model.results_folder,
    num_workers=cfg.model.num_workers,
    # logger=cfg.model.logger
)

trainer.load(DDPM_CHECKPOINT, map_location='cuda:0')
#for i in range(0, 25):
#    sample = trainer.ema_model.sample(batch_size=1)
    #print(sample.shape)
    #sample = (sample + 1.0) * 127.5
#sitk.WriteImage(sitk.GetImageFromArray(sample[0][0].cpu()), '../debug/test.nii')

sum_ssim = 0
for i in range(0, 2):
    trainer.ema_model.eval()
    with torch.no_grad():
        sample = trainer.ema_model.sample(batch_size = 2)

    #msssim = msssim_3d(sample[0].cpu(), sample[1].cpu())
    #sum_ssim = sum_ssim + msssim
    #print(sum_ssim / (i + 1.0))
    torch.save(sample, f"evaluation/V1/sampling_1/sample_{i}.pt")

print(f"Final: {sum_ssim / 1000}")

# ============================================================================================

