import cv2
from sklearn.metrics import accuracy_score, f1_score
from transforms import get_transforms
from datasets import make_loader
from utils.metrics import dice_score
from utils.functions import resize_batch_images
from utils.utils import post_process, dict_to_json
from utils.config import load_config
from utils import predict_batch, load_model
from models import CustomNet
import segmentation_models_pytorch as smp
from catalyst.dl.utils import load_checkpoint
import argparse
import os
import warnings

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")


SUB_HEIGHT, SUB_WIDTH = 350, 525


def validation(config_file_seg):

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    config = load_config(config_file_seg)
    if 'COLAB_GPU' in os.environ:
        config.work_dir = '/content/drive/My Drive/kaggle_cloud/' + config.work_dir
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        config.work_dir = '/kaggle/working/' + config.work_dir

    validloader = make_loader(
        data_folder=config.data.train_dir,
        df_path=config.data.train_df_path,
        phase='valid',
        img_size=(config.data.height, config.data.width),
        batch_size=config.test.batch_size,
        num_workers=config.num_workers,
        idx_fold=config.data.params.idx_fold,
        transforms=get_transforms(config.transforms.test),
        num_classes=config.data.num_classes,
    )

    model = load_model(config_file_seg)

    min_sizes = np.arange(0, 20000, 5000)
    label_thresholds = [0.6, 0.7, 0.8]
    mask_thresholds = [0.2, 0.3, 0.4]
    all_dice = np.zeros(
        (4, len(label_thresholds), len(mask_thresholds), len(min_sizes)))
    count = 0

    with torch.no_grad():
        for i, (batch_images, batch_masks) in enumerate(tqdm(validloader)):
            batch_images = batch_images.to(config.device)
            batch_preds = predict_batch(
                model, batch_images, tta=config.test.tta)

            batch_labels = torch.nn.functional.adaptive_max_pool2d(
                torch.sigmoid(torch.Tensor(batch_preds)), 1).view(batch_preds.shape[0], -1)

            batch_masks = batch_masks.cpu().numpy()
            batch_labels = batch_labels.cpu().numpy()

            batch_masks = resize_batch_images(
                batch_masks, SUB_HEIGHT, SUB_WIDTH)
            batch_preds = resize_batch_images(
                batch_preds, SUB_HEIGHT, SUB_WIDTH)

            for labels, masks, preds in zip(batch_labels, batch_masks, batch_preds):
                for cls in range(config.data.num_classes):
                    for i, label_th in enumerate(label_thresholds):
                        for j, mask_th in enumerate(mask_thresholds):
                            for k, min_size in enumerate(min_sizes):
                                if labels[cls] <= label_th:
                                    pred = np.zeros(preds[cls, :, :].shape)
                                else:
                                    pred, _ = post_process(
                                        preds[cls, :, :],
                                        mask_th,
                                        min_size,
                                        height=SUB_HEIGHT,
                                        width=SUB_WIDTH
                                    )
                                mask = masks[cls, :, :]

                                dice = dice_score(pred, mask)
                                all_dice[cls, i, j, k] += dice
                count += 1

    all_dice = all_dice / (count)
    np.save('all_dice', all_dice)

    parameters = {}
    parameters['label_thresholds'] = []
    parameters['mask_thresholds'] = []
    parameters['min_sizes'] = []
    parameters['dice'] = []
    cv_score = 0

    for cls in range(4):
        i, j, k = np.where((all_dice[cls] == all_dice[cls].max()))
        parameters['label_thresholds'].append(float(label_thresholds[i[0]]))
        parameters['mask_thresholds'].append(float(mask_thresholds[j[0]]))
        parameters['min_sizes'].append(int(min_sizes[k[0]]))
        parameters['dice'].append(float(all_dice[cls].max()))
        cv_score += all_dice[cls].max() / 4

    print('cv_score:', cv_score)
    dict_to_json(parameters, config.work_dir + '/parameters.json')
    print(pd.DataFrame(parameters))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    print('segmentation validation.')
    validation(args.config_file)


if __name__ == '__main__':
    main()
