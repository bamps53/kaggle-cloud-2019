import json
import os

import cv2
import numpy as np
import segmentation_models_pytorch as smp
from catalyst.dl.utils import load_checkpoint
from catalyst.utils import set_global_seed, prepare_cudnn

from models import CustomNet
from .config import load_config


def dict_to_json(dict_obj, file_name):
    with open(file_name, 'w') as fp:
        json.dump(dict_obj, fp)


def seed_all(SEED):
    set_global_seed(SEED)
    prepare_cudnn(deterministic=True)


def prepare_train_directories(config):
    out_dir = config.train.dir
    os.makedirs(os.path.join(out_dir, 'checkpoint'), exist_ok=True)


def mask2rle(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formatted
    '''
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def make_mask(row_id, df, height=1400, width=2100):
    fname = df.iloc[row_id].name
    labels = df.iloc[row_id][:4]
    masks = np.zeros((height, width, 4), dtype=np.float32)

    for idx, label in enumerate(labels.values):
        if label is not np.nan:
            label = label.split(" ")
            positions = map(int, label[0::2])
            length = map(int, label[1::2])
            mask = np.zeros(height * width, dtype=np.uint8)
            for pos, le in zip(positions, length):
                mask[pos:(pos + le)] = 1
            masks[:, :, idx] = mask.reshape(height, width, order='F')
    return fname, masks


def resize_rle(rle, before=(1400, 2100), after=(350, 525)):
    if rle is not np.nan:
        mask = rle2mask(rle, before[0], before[1])
        mask = cv2.resize(mask, (after[1], after[0]))
        return mask2rle(mask)
    else:
        return rle


def post_process(probability, threshold, min_size, height=1400, width=2100):
    '''Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored'''
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((height, width), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def load_model(config_path):
    config = load_config(config_path)
    if 'COLAB_GPU' in os.environ:
        config.work_dir = '/content/drive/My Drive/kaggle_cloud/' + config.work_dir
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        config.work_dir = '/kaggle/working/' + config.work_dir

    if config.checkpoint_path == None:
        config.checkpoint_path = config.work_dir + '/checkpoints/best.pth'
    print(config.checkpoint_path)

    if config.model.arch == 'Classification':
        model = CustomNet(config.model.encoder,
                          config.data.num_classes, pretrained=False)
    else:
        # create segmentation model with pre-trained encoder
        model = getattr(smp, config.model.arch)(
            encoder_name=config.model.encoder,
            encoder_weights=None,
            classes=config.data.num_classes,
            activation=None,
        )

    model.to(config.device)
    model.eval()

    checkpoint = load_checkpoint(config.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
