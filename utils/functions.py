import torch
import numpy as np
import cv2

def predict_batch(model, batch_images, tta=False, task='seg'):
    batch_preds = torch.sigmoid(model(batch_images))

    if tta:
        # h_flip
        h_images = torch.flip(batch_images, dims=[3])
        h_batch_preds = torch.sigmoid(model(h_images))
        if task == 'seg':
            batch_preds += torch.flip(h_batch_preds, dims=[3])
        else:
            batch_preds += h_batch_preds

        # v_flip
        v_images = torch.flip(batch_images, dims=[2])
        v_batch_preds = torch.sigmoid(model(v_images))
        if task == 'seg':
            batch_preds += torch.flip(v_batch_preds, dims=[2])
        else:
            batch_preds += v_batch_preds

        # hv_flip
        hv_images = torch.flip(torch.flip(batch_images, dims=[3]), dims=[2])
        hv_batch_preds = torch.sigmoid(model(hv_images))
        if task == 'seg':
            batch_preds += torch.flip(torch.flip(hv_batch_preds, dims=[3]), dims=[2])
        else:
            batch_preds += hv_batch_preds

        batch_preds /= 4

    return batch_preds.detach().cpu().numpy()


def predict_batch_with_softmax(model, batch_images, tta=False, task='seg'):
    batch_preds = torch.softmax(model(batch_images), 1)

    if tta:
        # h_flip
        h_images = torch.flip(batch_images, dims=[3])
        h_batch_preds = torch.softmax(model(h_images), 1)
        if task == 'seg':
            batch_preds += torch.flip(h_batch_preds, dims=[3])
        else:
            batch_preds += h_batch_preds

        # v_flip
        v_images = torch.flip(batch_images, dims=[2])
        v_batch_preds = torch.softmax(model(v_images), 1)
        if task == 'seg':
            batch_preds += torch.flip(v_batch_preds, dims=[2])
        else:
            batch_preds += v_batch_preds

        # hv_flip
        hv_images = torch.flip(torch.flip(batch_images, dims=[3]), dims=[2])
        hv_batch_preds = torch.softmax(model(hv_images), 1)
        if task == 'seg':
            batch_preds += torch.flip(torch.flip(hv_batch_preds, dims=[3]), dims=[2])
        else:
            batch_preds += hv_batch_preds

        batch_preds /= 4

    return batch_preds.detach().cpu().numpy()


def resize_batch_images(batch_images, h, w):
    final_output = None
    for img in batch_images:
        img = np.transpose(img,[1,2,0])
        img = cv2.resize(img,(w, h))
        img = np.transpose(img,[2,0,1])
        img = img.reshape(1,img.shape[0],img.shape[1],img.shape[2])
        if final_output is None:
            final_output = img
        else:
            final_output = np.concatenate((final_output, img), 0)
    return final_output
