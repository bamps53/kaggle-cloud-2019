from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomCrop,
    Resize,
    Cutout,
    Normalize,
    Compose,
    GaussNoise,
    IAAAdditiveGaussianNoise,
    RandomContrast,
    RandomGamma,
    RandomRotate90,
    RandomSizedCrop,
    RandomBrightness,
    ShiftScaleRotate,
    MotionBlur,
    MedianBlur,
    Blur,
    OpticalDistortion,
    GridDistortion,
    IAAPiecewiseAffine,
    OneOf)
from albumentations.pytorch import ToTensor

HEIGHT, WIDTH = 1400, 2100


def get_transforms(phase_config):
    list_transforms = []
    if phase_config.Resize.p > 0:
        list_transforms.append(
            Resize(
                phase_config.Resize.height,
                phase_config.Resize.width,
                p=1)
        )
    if phase_config.HorizontalFlip:
        list_transforms.append(HorizontalFlip())
    if phase_config.VerticalFlip:
        list_transforms.append(VerticalFlip())
    if phase_config.RandomCropScale:
        if phase_config.Resize.p > 0:
            height = phase_config.Resize.height
            width = phase_config.Resize.width
        else:
            height = HEIGHT
            width = WIDTH
        list_transforms.append(
            RandomSizedCrop(
                min_max_height=(int(height * 0.90), height),
                height=height,
                width=width,
                w2h_ratio=width/height)
            )
    if phase_config.ShiftScaleRotate:
        list_transforms.append(ShiftScaleRotate(p=1))

    if phase_config.RandomCrop.p > 0:
        list_transforms.append(
            RandomCrop(phase_config.RandomCrop.height, phase_config.RandomCrop.width, p=1)
        )
    if phase_config.Noise:
        list_transforms.append(
            OneOf([
                GaussNoise(),
                IAAAdditiveGaussianNoise(),
            ], p=0.5),
        )
    if phase_config.Contrast:
        list_transforms.append(
            OneOf([
                RandomContrast(0.5),
                RandomGamma(),
                RandomBrightness(),
            ], p=0.5),
        )
    if phase_config.Blur:
        list_transforms.append(
            OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.5)
        )
    if phase_config.Distort:
        list_transforms.append(
            OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.5)
        )

    if phase_config.Cutout.num_holes > 0:
        num_holes = phase_config.Cutout.num_holes
        hole_size = phase_config.Cutout.hole_size
        list_transforms.append(Cutout(num_holes, hole_size))

    list_transforms.extend(
        [
            Normalize(mean=phase_config.mean, std=phase_config.std, p=1),
            ToTensor(),
        ]
    )

    return Compose(list_transforms)

