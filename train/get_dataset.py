import argparse
import sys
import torch
sys.path.append('dataset')
sys.path.append('../dataset')
from brats import BRATSDataset
from mrnet import MRNetDataset
from adni import ADNIDataset
from duke import DUKEDataset
from lidc import LIDCDataset
from default import DEFAULTDataset
#from default import NCDataset

#from dataset import MRNetDataset, BRATSDataset, ADNIDataset, DUKEDataset, LIDCDataset, DEFAULTDataset, NCDataset
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Subset, ConcatDataset

# ============================================================================================

# Non-Conditional 3D Diffusion Model Parser Initialization
if True:
    ncdiff_parser = argparse.ArgumentParser(
        description = "Non-Conditional 3D Diffusion Model")
    ncdiff_parser.add_argument('--model_type', type = str,            # Chosen Model / Diffusion
                                choices =  {'video_diffusion',
                                            'blackout_diffusion',
                                            'mcvd', 'ramvid', 'meddiff'},
                                default = 'meddiff')
    ncdiff_parser.add_argument('--model_version', type = int,         # Model Version Index
                                default = 2)
    ncdiff_parser.add_argument('--data_version', type = int,          # Dataset Version Index
                                default = 4)
    settings = ncdiff_parser.parse_args("")

    # ============================================================================================

    # Directories and Path Arguments
    ncdiff_parser.add_argument('--reader_folderpath', type = str,         # Path for Dataset Reader Directory
                                default = '../../pfsousa/MedDiff/dataset')
                                #default = "X:/nas-ctm01/homes/pfsousa/MedDiff/dataset")
    ncdiff_parser.add_argument('--public_data_folderpath', type = str,    # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
                                default = "../../../datasets/public/MEDICAL/Duke-Breast-Cancer-T1")
    ncdiff_parser.add_argument('--private_data_folderpath', type = str,   # Path for Private Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/METABREST/T1W_Breast")
                                default = '../../../datasets/private/METABREST/T1W_Breast')
    ncdiff_parser.add_argument( '--lung_data_folderpath', type = str,     # Path for LUCAS Dataset Directory
                                #default = "X:/nas-ctm01/datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")
                                default = "../../../datasets/private/LUCAS/lidc/TCIA_LIDC-IDRI_20200921/LIDC-IDRI")

    # Directory | Model-Related Path Arguments
    ncdiff_parser.add_argument('--model_folderpath', type = str,          # Path for Model Architecture Directory
                                default = f'../../models/{settings.model_type}')
    ncdiff_parser.add_argument('--script_folderpath', type = str,         # Path for Model Training & Testing Scripts Directory
                                default = f'../../scripts/{settings.model_type}')
    ncdiff_parser.add_argument('--logs_folderpath', type = str,           # Path for Model Saving Directory
                                default = f'../../logs/{settings.model_type}')
    ncdiff_parser.add_argument('--verbose', type = bool,                  # Verbose Control Variable
                                default = False)
        
    # ============================================================================================

    # Dataset | Dataset General Arguments
    ncdiff_parser.add_argument('--data_format', type = str,           # Chosen Dataset Format for Reading
                                choices =  {'mp4', 'dicom'},
                                default = 'mp4')
    ncdiff_parser.add_argument('--img_size', type = int,              # Generated Image Resolution
                                default = 128)
    ncdiff_parser.add_argument('--num_slice', type = int,             # Number of 2D Slices in MRI
                                default = 30)
    ncdiff_parser.add_argument('--slice_spacing', type = bool,        # Usage of Linspace for Slice Spacing
                                default = False)
    ncdiff_parser.add_argument('--slice_bottom_margin', type = int,   # Number of 2D Slices to be Discarded in Bottom Margin
                                default = 5)
    ncdiff_parser.add_argument('--slice_top_margin', type = int,      # Number of 2D Slices to be Discarded in Top Margin
                                default = 15)
    ncdiff_parser.add_argument('--data_prep', type = bool,            # Usage of Dataset Pre-Processing Control Value
                                default = True)
    ncdiff_parser.add_argument('--h_flip', type = int,                # Percentage of Horizontally Flipped Subjects
                                default = 50)

    # Dataset | Dataset Splitting Arguments
    ncdiff_parser.add_argument('--train_subj', type = int,            # Number of Random Subjects in Training Set
                                default = 0)                          # PS: Input 0 for all Subjects in the Dataset
    ncdiff_parser.add_argument('--val_subj', type = int,              # Number of Random Subjects in Validation Set
                                default = 0)
    ncdiff_parser.add_argument('--test_subj', type = int,             # Number of Random Subjects in Test Set
                                default = 10)

    # Dataset | DataLoader Arguments
    ncdiff_parser.add_argument('--batch_size', type = int,            # DataLoader Batch Size Value
                                default = 1)
    ncdiff_parser.add_argument('--num_fps', type = int,               # Number of Video Frames per Second
                                default = 4)
    ncdiff_parser.add_argument('--shuffle', type = bool,              # DataLoader Subject Shuffling Control Value
                                default = True)
    ncdiff_parser.add_argument('--num_workers', type = int,           # Number of DataLoader Workers
                                default = 12)
    ncdiff_parser.add_argument('--prefetch_factor', type = int,       # Number of Prefetched DataLoader Batches per Worker
                                default = 1)

    # ============================================================================================

    # Model | Architecture-Defining Arguments
    ncdiff_parser.add_argument('--seed', type = int,                  # Randomised Generational Seed
                                default = 0)
    ncdiff_parser.add_argument('--dim', type = int,                   # Input Dimensionality (Not Necessary)
                                default = 64)
    ncdiff_parser.add_argument('--num_channel', type = int,           # Number of Input Channels for Dataset
                                default = 1)
    ncdiff_parser.add_argument('--mult_dim', type = tuple,            # Dimensionality for all Conditional Layers
                                default = (1, 2, 4, 8))

    # Model | Training & Diffusion Arguments
    ncdiff_parser.add_argument('--noise_type', type = str,            # Diffusion Noise Distribution
                                default = 'gaussian')
    #ncdiff_parser.add_argument('--num_epochs', type = int,           # Number of Training Epochs
    #                            default = 30)
    ncdiff_parser.add_argument('--num_ts', type = int,                # Number of Scheduler Timesteps
                                default = 500)
    ncdiff_parser.add_argument('--num_steps', type = int,             # Number of Diffusion Training Steps
                                default = 150000)
    ncdiff_parser.add_argument('--lr_base', type = float,             # Base Learning Rate Value
                                default = 1e-4)
    ncdiff_parser.add_argument('--lr_decay', type = float,            # Learning Rate Decay Value
                                default = 0.999)
    ncdiff_parser.add_argument('--lr_step', type = float,             # Number of Steps inbetween Learning Rate Decay
                                default = 250)
    ncdiff_parser.add_argument('--lr_min', type = float,              # Minimum Decayed Learning Rate Value
                                default = 1e-6)
    
    # Model | Result Logging Arguments 
    ncdiff_parser.add_argument('--save_interval', type = int,         # Number of Training Step Interval inbetween Image Saving
                                default = 500)
    #ncdiff_parser.add_argument('--log_interval', type = int,          # Number of Training Step Interval inbetween Result Logging (not a joke i swear...)
    #                           default = 1)
    ncdiff_parser.add_argument('--save_img', type = int,              # Square Root of Number of Images Saved for Manual Evaluation
                                default = 2)
    ncdiff_parser.add_argument('--log_method', type = str,            # Metric Logging Methodology
                                choices = {'wandb', 'tensorboard', None},
                                default = 'tensorboard')
    
    settings = ncdiff_parser.parse_args("")
    settings.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

# ============================================================================================

def get_dataset(cfg):
    if cfg.dataset.name == 'MRNet':
        train_dataset = MRNetDataset(
            root_dir=cfg.dataset.root_dir, task=cfg.dataset.task, plane=cfg.dataset.plane, split='train')
        val_dataset = MRNetDataset(root_dir=cfg.dataset.root_dir,
                                   task=cfg.dataset.task, plane=cfg.dataset.plane, split='valid')
        sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weight, num_samples=len(train_dataset.sample_weight))
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'BRATS':
        train_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
        val_dataset = BRATSDataset(
            root_dir=cfg.dataset.root_dir, imgtype=cfg.dataset.imgtype, train=True, severity=cfg.dataset.severity, resize=cfg.dataset.resize)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'ADNI':
        train_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = ADNIDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DUKE':
        train_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DUKEDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'LIDC':
        train_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        val_dataset = LIDCDataset(
            root_dir=cfg.dataset.root_dir, augmentation=True)
        sampler = None
        return train_dataset, val_dataset, sampler
    if cfg.dataset.name == 'DEFAULT':
        private_train_dataset = DEFAULTDataset( settings,
                                                mode = 'train',
                                                dataset = 'private')
        public_train_dataset = DEFAULTDataset(  settings,
                                                mode = 'train',
                                                dataset = 'public')
        train_dataset = ConcatDataset([private_train_dataset, public_train_dataset])
        private_test_dataset = DEFAULTDataset(  settings,
                                                mode = 'test',
                                                dataset = 'private')
        public_test_dataset = DEFAULTDataset(   settings,
                                                mode = 'test',
                                                dataset = 'public')
        val_dataset = ConcatDataset([private_test_dataset, public_test_dataset])
        sampler = None
    else: raise ValueError(f'{cfg.dataset.name} Dataset is not available')
    return train_dataset, val_dataset, sampler

    """    
    if cfg.dataset.name == 'DEFAULT':
        train_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        val_dataset = DEFAULTDataset(
            root_dir=cfg.dataset.root_dir)
        sampler = None
    """
    
