import argparse
import json
import os
import warnings
import cv2

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

from models import MultiClsModels, MultiSegModels
from utils import predict_batch
from utils.utils import mask2rle, post_process, load_model
from utils.functions import resize_batch_images
from utils.config import load_config
from datasets import make_loader, INV_CLASSES
from transforms import get_transforms

KAGGLE_WORK_DIR = '/kaggle/working'
SUB_HEIGHT, SUB_WIDTH = 350, 525


def ensemble():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # parmeters and configs
    # ------------------------------------------------------------------------------------------------------------
    config_paths320 = [
                        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold0.yml',
                        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold1.yml',
                        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold2.yml',
                        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold3.yml',
                        'config/seg/017_efnet_b3_Unet_img320_cutout5_aug_fold4.yml',
                        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold0.yml',
                        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold1.yml',
                        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold2.yml',
                        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold3.yml',
                        'config/seg/030_efnet_b0_Unet_bs16_half_cosine_fold4.yml',
                        ]
    config_paths384 = [
                        'config/seg/032_efnet_b3_Unet_img384_RandomSizedCrop_half_cosine_fold0.yml',
                        'config/seg/032_efnet_b3_Unet_img384_RandomSizedCrop_half_cosine_fold1.yml',
                        'config/seg/032_efnet_b3_Unet_img384_RandomSizedCrop_half_cosine_fold2.yml',
                        'config/seg/032_efnet_b3_Unet_img384_RandomSizedCrop_half_cosine_fold3.yml',
                        'config/seg/032_efnet_b3_Unet_img384_RandomSizedCrop_half_cosine_fold4.yml',  
                        'config/seg/048_resnet34_FPN_img384_mixup_fold0.yml',
                        'config/seg/048_resnet34_FPN_img384_mixup_fold1.yml',
                        'config/seg/048_resnet34_FPN_img384_mixup_fold2.yml',
                        'config/seg/048_resnet34_FPN_img384_mixup_fold3.yml',
                        'config/seg/048_resnet34_FPN_img384_mixup_fold4.yml',                                                                     
                        ]
    LABEL_THRESHOLDS = [0.68, 0.69, 0.69, 0.67]
    MASK_THRESHOLDS = [0.31, 0.36, 0.31, 0.34]
    MIN_SIZES = [7500, 10000, 7500, 7500]    
    WEIGHTS = [0.5, 0.5]
    # ------------------------------------------------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------------------------------------------------
    config = load_config('config/base_config.yml')
    
    def get_model_and_loader(config_paths):
        config = load_config(config_paths[0])

        models = []
        for c in config_paths:
            models.append(load_model(c))

        model = MultiSegModels(models)

        testloader = make_loader(
            data_folder=config.data.test_dir,
            df_path=config.data.sample_submission_path,
            phase='test',
            img_size=(config.data.height, config.data.width),
            batch_size=config.test.batch_size,
            num_workers=config.num_workers,
            transforms=get_transforms(config.transforms.test)
        )
        return model, testloader

    model320, loader320 = get_model_and_loader(config_paths320)
    model384, loader384 = get_model_and_loader(config_paths384)

    predictions = []
    with torch.no_grad():
        for (batch_fnames320, batch_images320), (batch_fnames384, batch_images384) in tqdm(zip(loader320, loader384)):    
            batch_images320 = batch_images320.to(config.device)
            batch_images384 = batch_images384.to(config.device)
            
            batch_preds320 = predict_batch(model320, batch_images320, tta=config.test.tta)
            batch_preds384 = predict_batch(model384, batch_images384, tta=config.test.tta)

            batch_preds320 = resize_batch_images(batch_preds320, SUB_HEIGHT, SUB_WIDTH)
            batch_preds384 = resize_batch_images(batch_preds384, SUB_HEIGHT, SUB_WIDTH)            
            batch_preds = batch_preds320 * WEIGHTS[0] + batch_preds384 * WEIGHTS[1]

            batch_labels320 = torch.nn.functional.adaptive_max_pool2d(torch.sigmoid(torch.Tensor(batch_preds320)),1).view(batch_preds320.shape[0], -1)
            batch_labels384 = torch.nn.functional.adaptive_max_pool2d(torch.sigmoid(torch.Tensor(batch_preds384)),1).view(batch_preds384.shape[0], -1)
            batch_labels = batch_labels320 * WEIGHTS[0] + batch_labels384 * WEIGHTS[1]

            for fname, preds, labels in zip(batch_fnames320, batch_preds, batch_labels):
                for cls in range(4):
                    if labels[cls] <= LABEL_THRESHOLDS[cls]:
                        pred = np.zeros(preds[cls, :, :].shape)
                    else:
                        pred, _ = post_process(preds[cls, :, :], MASK_THRESHOLDS[cls], MIN_SIZES[cls], height=SUB_HEIGHT, width=SUB_WIDTH)
                    rle = mask2rle(pred)
                    cls_name = INV_CLASSES[cls]
                    name = fname + f"_{cls_name}"
                    predictions.append([name, rle])

    # ------------------------------------------------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------------------------------------------------
    sub_df = pd.DataFrame(predictions, columns=['Image_Label', 'EncodedPixels'])

    sample_submission = pd.read_csv(config.data.sample_submission_path)
    df_merged = pd.merge(sample_submission, sub_df, on='Image_Label', how='left')
    df_merged.fillna('', inplace=True)
    df_merged['EncodedPixels'] = df_merged['EncodedPixels_y']
    df_merged = df_merged[['Image_Label', 'EncodedPixels']]

    df_merged.to_csv("submission.csv", index=False)

    if 'COLAB_GPU' in os.environ:
        config.work_dir = '/content/drive/My Drive/kaggle_cloud/'
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        config.work_dir = '/kaggle/working/'
    else:
        config.work_dir = '.'
    df_merged.to_csv(config.work_dir + '/submission.csv', index=False)


if __name__ == '__main__':
    ensemble()
