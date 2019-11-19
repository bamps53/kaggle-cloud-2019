import argparse
import json
import os
import warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import cv2

warnings.filterwarnings("ignore")

from utils import predict_batch, load_model
from utils.utils import mask2rle, post_process
from utils.config import load_config
from datasets import make_loader, INV_CLASSES
from transforms import get_transforms

SUB_HEIGHT, SUB_WIDTH = 350, 525


def run_seg(config_file_seg):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # ------------------------------------------------------------------------------------------------------------
    # 2. segmentation inference
    # ------------------------------------------------------------------------------------------------------------
    config = load_config(config_file_seg)
    if 'COLAB_GPU' in os.environ:
        config.work_dir = '/content/drive/My Drive/kaggle_cloud/' + config.work_dir
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        config.work_dir = '/kaggle/working/' + config.work_dir


    if os.path.exists('cls_preds.csv'):
        testloader = make_loader(
            data_folder=config.data.test_dir,
            df_path='cls_preds.csv',
            phase='filtered_test',
            img_size=(config.data.height, config.data.width),
            batch_size=config.test.batch_size,
            num_workers=config.num_workers,
            transforms=get_transforms(config.transforms.test)
        )
    else:
        testloader = make_loader(
            data_folder=config.data.test_dir,
            df_path=config.data.sample_submission_path,
            phase='test',
            img_size=(config.data.height, config.data.width),
            batch_size=config.test.batch_size,
            num_workers=config.num_workers,
            transforms=get_transforms(config.transforms.test)
        )

    model = load_model(config_file_seg)

    if os.path.exists(config.work_dir + '/threshold_search.json'):
        with open(config.work_dir + '/threshold_search.json') as json_file:
            data = json.load(json_file)
        df = pd.DataFrame(data)
        min_sizes = list(df.T.idxmax().values.astype(int))
        print('load best threshold from validation:', min_sizes)
    else:
        min_sizes = config.test.min_size
        print('load default threshold:', min_sizes)

    predictions = []
    with torch.no_grad():
        for i, (batch_fnames, batch_images) in enumerate(tqdm(testloader)):
            batch_images = batch_images.to(config.device)
            batch_preds = predict_batch(model, batch_images, tta=config.test.tta)

            for fname, preds in zip(batch_fnames, batch_preds):
                for cls in range(preds.shape[0]):
                    pred, _ = post_process(
                        preds[cls, :, :],
                        config.test.best_threshold,
                        min_sizes[cls],
                        height=config.transforms.test.Resize.height,
                        width=config.transforms.test.Resize.width
                    )
                    pred = cv2.resize(pred, (SUB_WIDTH, SUB_HEIGHT))
                    pred = (pred > 0.5).astype(int)
                    rle = mask2rle(pred)
                    cls_name = INV_CLASSES[cls]
                    name = fname + f"_{cls_name}"
                    predictions.append([name, rle])

    # ------------------------------------------------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------------------------------------------------
    df = pd.DataFrame(predictions, columns=['Image_Label', 'EncodedPixels'])
    df.to_csv(config.work_dir + "/submission.csv", index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Severstal')
    parser.add_argument('--cls_config', dest='config_file_cls',
                        help='configuration file path',
                        default=None, type=str)
    parser.add_argument('--seg_config', dest='config_file_seg',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    print('segmentation inference.')
    run_seg(args.config_file_seg)


if __name__ == '__main__':
    main()
