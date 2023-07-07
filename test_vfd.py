from FaceModel.model import AuditoryTransformer, VisualTransformer
import torch.nn
from datasets import FakeAVCeleb_Datasets, DFDC_Datasets, TIMIT_Datasets
from torch.utils.data import DataLoader
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
from utils.Augmentation import *
import pprint
from utils.saving import *
import os
from test import validate_on_fake_dataset


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(996)

def main():
    global args, best_prec1
    global global_step
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', default='')
    parser.add_argument('--log_time', default='')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'],
                               args.log_time)
    wandb.init(project=config['network']['type'],
               name='{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch']))
    print('-' * 80)
    print(' ' * 20, "working dir: {}".format(working_dir))
    print('-' * 80)

    print('-' * 80)
    print(' ' * 30, "Config")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)
    print('-' * 80)

    config = DotMap(config)

    Path(working_dir).mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, working_dir)
    shutil.copy('test.py', working_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.

    if config['network']['drop_out'] > 0.:
        dpr = [x.item() for x in torch.linspace(0, config['network']['drop_out'], config['network']['layers'],)]
    else:
        dpr = None

    voice_model = AuditoryTransformer(
        input_resolution=config['network']['voice']['input_resolution'],
        patch_size=config['network']['voice']['patch_size'],
        width=config['network']['voice']['width'],
        layers=config['network']['voice']['layers'],
        heads=config['network']['voice']['heads'],
        output_dim=config['network']['voice']['output_dim'],
        joint=config['network']['voice']['joint'],
        dropout=dpr,
        emb_dropout=config['network']['emb_dropout']
    )

    face_model = VisualTransformer(
        input_resolution=config['network']['face']['input_resolution'],
        patch_size=config['network']['face']['patch_size'],
        width=config['network']['face']['width'],
        layers=config['network']['face']['layers'],
        heads=config['network']['face']['heads'],
        output_dim=config['network']['face']['output_dim'],
        joint=config['network']['face']['joint'],
        dropout=dpr,
        emb_dropout=config['network']['emb_dropout']
    )


    transform_val = get_augmentation(False, config)

    voice_model = torch.nn.DataParallel(voice_model).cuda()
    face_model = torch.nn.DataParallel(face_model).cuda()
    wandb.watch(voice_model)
    wandb.watch(face_model)


    if config.data.dataset == 'FakeAVCeleb':
        val_data = FakeAVCeleb_Datasets(config.data.val_face_list, config.data.label_list,
                                        random_seed=config.seed, transformer=transform_val, neg_sample=0)
    elif config.data.dataset == 'DFDC':
        val_data = DFDC_Datasets(config.data.val_face_list, config.data.label_list,
                                        random_seed=config.seed, transformer=transform_val, neg_sample=0)
    else:
        val_data = TIMIT_Datasets(config.data.val_face_list, config.data.label_list,
                                        random_seed=config.seed, transformer=transform_val, neg_sample=0)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                            shuffle=False, pin_memory=False, drop_last=True)

    start_epoch = config.solver.start_epoch


    if config.pretrain.split:
        if os.path.isfile(config.pretrain.pretrain_face) and os.path.isfile(config.pretrain.pretrain_voice):
            print(("=> loading checkpoint_face '{}'".format(config.pretrain.pretrain_face)))
            print(("=> loading checkpoint_voice '{}'".format(config.pretrain.pretrain_voice)))
            checkpoint_face = torch.load(config.pretrain.pretrain_face)
            checkpoint_voice = torch.load(config.pretrain.pretrain_voice)
            face_model.load_state_dict(checkpoint_face['face_model_state_dict'])
            voice_model.load_state_dict(checkpoint_voice['voice_model_state_dict'])
            del checkpoint_face, checkpoint_voice
        else:
            print(("=> no checkpoint found at '{}' or '{}'".format(config.pretrain_face, config.checkpoint_voice)))
    else:
        if os.path.isfile(config.pretrain.pretrain):
            print(("=> loading checkpoint '{}'".format(config.pretrain.pretrain)))
            checkpoint = torch.load(config.pretrain.pretrain)
            face_model.load_state_dict(checkpoint['face_model_state_dict'])
            voice_model.load_state_dict(checkpoint['voice_model_state_dict'])
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}''".format(config.pretrain)))


    best_prec1 = 0.0
    if config.data.dataset == 'FakeAVCeleb':
        prec1 = validate_on_fake_dataset(start_epoch, val_loader, device, voice_model, face_model, config)
    elif config.data.dataset == 'DFDC':
        prec1 = validate_on_fake_dataset(start_epoch, val_loader, device, voice_model, face_model, config)
    elif config.data.dataset == 'TIMIT':
        prec1 = validate_on_fake_dataset(start_epoch, val_loader, device, voice_model, face_model, config)
    else:
        print('Not Support.')


if __name__ == '__main__':
    main()
