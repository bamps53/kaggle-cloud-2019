from transforms import get_transforms
from schedulers import get_scheduler
from losses import get_loss
from optimizers import get_optimizer
from datasets import make_loader
from utils.callbacks import CutMixCallback
from utils.config import load_config, save_config
import segmentation_models_pytorch as smp
from catalyst.utils import get_device
from catalyst.dl.callbacks import DiceCallback, IouCallback, CheckpointCallback, MixupCallback, EarlyStoppingCallback, OptimizerCallback, CriterionCallback
from catalyst.dl import SupervisedRunner
import argparse
import os
import warnings

warnings.filterwarnings("ignore")


def run(config_file):
    config = load_config(config_file)
    if 'COLAB_GPU' in os.environ:
        config.work_dir = '/content/drive/My Drive/kaggle_cloud/' + config.work_dir
    elif 'KAGGLE_WORKING_DIR' in os.environ:
        config.work_dir = '/kaggle/working/' + config.work_dir
    print('working directory:', config.work_dir)

    if not os.path.exists(config.work_dir):
        os.makedirs(config.work_dir, exist_ok=True)
    save_config(config, config.work_dir + '/config.yml')

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    all_transforms = {}
    all_transforms['train'] = get_transforms(config.transforms.train)
    all_transforms['valid'] = get_transforms(config.transforms.test)

    dataloaders = {
        phase: make_loader(
            data_folder=config.data.train_dir,
            df_path=config.data.train_df_path,
            phase=phase,
            img_size=(config.data.height, config.data.width),
            batch_size=config.train.batch_size,
            num_workers=config.num_workers,
            idx_fold=config.data.params.idx_fold,
            transforms=all_transforms[phase],
            num_classes=config.data.num_classes,
            pseudo_label_path=config.train.pseudo_label_path,
            debug=config.debug
        )
        for phase in ['train', 'valid']
    }

    # create segmentation model with pre trained encoder
    model = getattr(smp, config.model.arch)(
        encoder_name=config.model.encoder,
        encoder_weights=config.model.pretrained,
        classes=config.data.num_classes,
        activation=None,
    )

    # train setting
    criterion = get_loss(config)
    params = [
        {'params': model.decoder.parameters(), 'lr': config.optimizer.params.decoder_lr},
        {'params': model.encoder.parameters(), 'lr': config.optimizer.params.encoder_lr},
    ]
    optimizer = get_optimizer(params, config)
    scheduler = get_scheduler(optimizer, config)

    # model runner
    runner = SupervisedRunner(model=model, device=get_device())

    callbacks = [DiceCallback(), IouCallback()]

    if config.train.early_stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(
            patience=config.train.early_stop_patience))

    if config.train.accumulation_size > 0:
        accumulation_steps = config.train.accumulation_size // config.train.batch_size
        callbacks.extend(
            [CriterionCallback(),
             OptimizerCallback(accumulation_steps=accumulation_steps)]
        )

    # to resume from check points if exists
    if os.path.exists(config.work_dir + '/checkpoints/best.pth'):
        callbacks.append(CheckpointCallback(
            resume=config.work_dir + '/checkpoints/last_full.pth'))

    if config.train.mixup:
        callbacks.append(MixupCallback())

    if config.train.cutmix:
        callbacks.append(CutMixCallback())

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=dataloaders,
        logdir=config.work_dir,
        num_epochs=config.train.num_epochs,
        main_metric=config.train.main_metric,
        minimize_metric=config.train.minimize_metric,
        callbacks=callbacks,
        verbose=True,
        fp16=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_file',
                        help='configuration file path',
                        default=None, type=str)
    return parser.parse_args()


def main():
    print('train Segmentation model.')
    args = parse_args()
    if args.config_file is None:
        raise Exception('no configuration file')
    print('load config from {}'.format(args.config_file))
    run(args.config_file)


if __name__ == '__main__':
    main()
