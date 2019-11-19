# Understanding Clouds from Satellite Images
This is Catalyst + pytorch segmentation model baseline for https://www.kaggle.com/c/understanding_cloud_organization/

My solution was below:
1. Unet/efficientnet-b3/image size 320x480/5fold
2. Unet/efficientnet-b0/image size 320x480/cosineanealing/5fold
3. Unet/efficientnet-b3/image size 384x576/cosineanealing/5fold
4. FPN/resnet34/image size 384x576/mixup/5fold
5. Ensemble above 20 models
6. Triplet thresholding(label threshold/mask threshold/min componet)

## Usage

```
$ python split_fold.py --config config/base_config.yml
$ python train.py --config config/seg/* #run all config in the folder
$ python ensemble.py  #I was lazy, parameter of ensembing is hard coded in ensemble.py
```
