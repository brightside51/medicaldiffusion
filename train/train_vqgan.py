"Adapted from https://github.com/SongweiGe/TATS"

import os
import torch
print(torch.__version__)
from torch import Tensor
import pytorch_lightning as pl
import sys
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
print(pl.__version__)

sys.path.append('ddpm')
from diffusion import default
sys.path.append('vq_gan_3d/model')
from vqgan import VQGAN
sys.path.append('train')
from callbacks import ImageLogger, VideoLogger
from get_dataset import get_dataset


#from ddpm.diffusion import default
#from vq_gan_3d.model import VQGAN
#from train.callbacks import ImageLogger, VideoLogger
#from train.get_dataset import get_dataset
import hydra
from omegaconf import DictConfig, open_dict


@hydra.main(config_path='../config', config_name='base_cfg', version_base=None)
def run(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)

    train_dataset, val_dataset, sampler = get_dataset(cfg)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=cfg.model.batch_size,
                                  num_workers=cfg.model.num_workers, sampler=sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.model.batch_size,
                                shuffle=False, num_workers=cfg.model.num_workers)

    # automatically adjust learning rate
    bs, base_lr, ngpu, accumulate = cfg.model.batch_size, cfg.model.lr, cfg.model.gpus, cfg.model.accumulate_grad_batches

    with open_dict(cfg):
        cfg.model.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr
        cfg.model.default_root_dir = os.path.join(
            cfg.model.default_root_dir, cfg.dataset.name, cfg.model.default_root_dir_postfix)
    print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
        cfg.model.lr, accumulate, ngpu/8, bs/4, base_lr))

    model = VQGAN(cfg)

    callbacks = []
    callbacks.append(ModelCheckpoint(monitor='val/Recon Loss',
                     save_top_k=1, mode='min', filename='latest_checkpoint_recon'))
    callbacks.append(ModelCheckpoint(monitor='train/SSIM Index',
                     save_top_k=1, mode='max', filename='latest_checkpoint_ssim'))
    callbacks.append(ModelCheckpoint(every_n_train_steps=3000,
                     save_top_k=-1, filename='{epoch}-{step}-{train/Recon Loss:.2f}'))
    #callbacks.append(ModelCheckpoint(every_n_train_steps=10000, save_top_k=1,
    #                 filename='{epoch}-{step}-10000-{train/Recon Loss:.2f}-{train/SSIM Index:.2f}'))
    callbacks.append(ImageLogger(
        batch_frequency=10000, max_images=4, clamp=True))
    callbacks.append(VideoLogger(
        batch_frequency=10000, max_videos=4, clamp=True))

    # load the most recent checkpoint file
    base_dir = os.path.join(cfg.model.default_root_dir, 'lightning_logs')
    if os.path.exists(base_dir):
        log_folder = ckpt_file = ''
        version_id_used = step_used = 0
        for folder in os.listdir(base_dir):
            version_id = int(folder.split('_')[1])
            if version_id > version_id_used:
                version_id_used = version_id
                log_folder = folder
        if len(log_folder) > 0:
            ckpt_folder = os.path.join(base_dir, log_folder, 'checkpoints')
            for fn in os.listdir(ckpt_folder):
                if fn == 'latest_checkpoint.ckpt':
                    ckpt_file = 'latest_checkpoint_prev.ckpt'
                    os.rename(os.path.join(ckpt_folder, fn),
                              os.path.join(ckpt_folder, ckpt_file))
            if len(ckpt_file) > 0:
                cfg.model.resume_from_checkpoint = os.path.join(
                    ckpt_folder, ckpt_file)
                print('will start from the recent ckpt %s' %
                      cfg.model.resume_from_checkpoint)

    accelerator = 'cuda'
    #if cfg.model.gpus > 1:
    #    accelerator = 'ddp'

    if cfg.model.resume_from_checkpoint is not None:
        vqgan = VQGAN.load_from_checkpoint(cfg.model.resume_from_checkpoint).cuda()

    trainer = pl.Trainer(
        #gpus=cfg.model.gpus,
        num_sanity_val_steps = 0,
        accumulate_grad_batches=cfg.model.accumulate_grad_batches,
        default_root_dir=cfg.model.default_root_dir,
        #resume_from_checkpoint=cfg.model.resume_from_checkpoint,
        callbacks=callbacks,
        max_steps=cfg.model.max_steps,
        max_epochs=cfg.model.max_epochs,
        precision=cfg.model.precision,
        #gradient_clip_val=cfg.model.gradient_clip_val,
        accelerator=accelerator,
    )

    trainer.fit(model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    run()
