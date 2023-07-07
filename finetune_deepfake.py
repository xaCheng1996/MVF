from FaceModel.model import AuditoryTransformer, VisualTransformer
import torch.nn
from datasets import FakeAVCeleb_Datasets, DFDC_Datasets, TIMIT_Datasets
from utils.CustomInfoNCE import CustomInfoNCE
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import argparse
import shutil
from pathlib import Path
import yaml
from dotmap import DotMap
from utils.Augmentation import *
import pprint
from test import validate_on_fake_dataset
from utils.solver import _optimizer_face_voice_match, _lr_scheduler
from utils.saving import *
import os

print(os.environ['PATH'])
os.environ["PATH"] += ':/opt/anaconda3/envs/torch_cv/bin/'
envdir_list = [os.curdir] + os.environ["PATH"].split(os.pathsep)
print(envdir_list)


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
        config = yaml.safe_load(f)
    working_dir = os.path.join('./exp', config['network']['type'], config['network']['arch'],
                               config['data']['dataset']['train_dataset'],
                               config['data']['exp_name'], args.log_time)
    wandb.init(project=config['network']['type'],
               name='{}_{}_{}_{}_{}'.format(args.log_time, config['network']['type'], config['network']['arch'],
                                            config['data']['dataset']['train_dataset'], config['data']['exp_name']))
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
    shutil.copy('finetune_deepfake.py', working_dir)

    # If using GPU then use mixed precision training.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform_train = get_augmentation(True, config)
    transform_val = get_augmentation(False, config)

    if config.data.randaug.N > 0:
        transform_train = randAugment(transform_train, config)

    if config['network']['drop_out'] > 0.:
        dpr = [x.item() for x in torch.linspace(0, config['network']['drop_out'],
                                                config['network']['layers'])]  # stochastic depth decay rule
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


    voice_model = torch.nn.DataParallel(voice_model).cuda()
    face_model = torch.nn.DataParallel(face_model).cuda()
    wandb.watch(voice_model)
    wandb.watch(face_model)


    if config.data.dataset.train_dataset == 'FakeAVCeleb':
        train_data = FakeAVCeleb_Datasets(config.data.train_face_list, config.data.label_list,
                                          neg_sample=config.data.neg_sample,
                                          random_seed=config.data.random_seed,
                                          transformer=transform_train)
    elif config.data.dataset.train_dataset == 'DFDC':
        train_data = DFDC_Datasets(config.data.train_face_list, config.data.label_list,
                                   neg_sample=config.data.neg_sample,
                                   random_seed=config.data.random_seed,
                                   transformer=transform_train)
    elif config.data.dataset.train_dataset == 'TIMIT':
        train_data = TIMIT_Datasets(config.data.train_face_list, config.data.label_list,
                                    neg_sample=config.data.neg_sample,
                                    random_seed=config.data.random_seed,
                                    transformer=transform_train)
    else:
        print("Not supporting this training dataset %s!" % config.data.dataset.train_dataset)
        return

    if config.data.dataset.val_dataset == 'FakeAVCeleb':
        val_data = FakeAVCeleb_Datasets(config.data.val_face_list, config.data.label_list,
                                        random_seed=config.data.random_seed,
                                        neg_sample=0,
                                        transformer=transform_val)
    elif config.data.dataset.val_dataset == 'DFDC':
        val_data = DFDC_Datasets(config.data.val_face_list, config.data.label_list,
                                 random_seed=config.data.random_seed,
                                 neg_sample=0,
                                 transformer=transform_val)
    elif config.data.dataset.train_dataset == 'TIMIT':
        val_data = TIMIT_Datasets(config.data.val_face_list, config.data.label_list,
                                  random_seed=config.data.random_seed,
                                  neg_sample=0,
                                  transformer=transform_val)
    else:
        print("Not supporting this testing dataset %s!" % config.data.dataset.val_dataset)
        return
    train_loader = DataLoader(train_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                              shuffle=True, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=config.data.batch_size, num_workers=config.data.workers,
                            shuffle=False, pin_memory=False, drop_last=True)

    start_epoch = config.solver.start_epoch

    loss_fake_nce = CustomInfoNCE(temperature=0.1)


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

    if config.resume:
        if os.path.isfile(config.resume):
            print(("=> loading checkpoint '{}'".format(config.resume)))
            checkpoint = torch.load(config.resume)
            face_model.load_state_dict(checkpoint['face_model_state_dict'])
            voice_model.load_state_dict(checkpoint['voice_model_state_dict'])
            start_epoch = checkpoint['epoch']
            print(("=> loaded checkpoint '{}' (epoch {})"
                   .format(config.evaluate, start_epoch)))
            del checkpoint
        else:
            print(("=> no checkpoint found at '{}'".format(config.resume)))

    optimizer = _optimizer_face_voice_match(config, voice_model=voice_model, face_model=face_model)
    lr_scheduler = _lr_scheduler(config, optimizer)

    best_prec1 = 0.0
    if config.solver.evaluate:
        prec1 = validate_on_fake_dataset(start_epoch, val_loader, device, voice_model, face_model, config)
        return

    for epoch in range(start_epoch, config.solver.epochs):
        voice_model.train()
        face_model.train()

        for kkk, real_fake_data in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            real_img = real_fake_data['real']['img']
            real_mel = real_fake_data['real']['mel']
            if config.data.dataset.train_dataset == 'DFDC':
                fake_img = real_fake_data['fake']['img']
                fake_mel = real_fake_data['fake']['mel']
            elif config.data.dataset.train_dataset == 'FakeAVCeleb':
                fake_img = real_fake_data['fake_video']['img']
                fake_mel = real_fake_data['fake_audio']['mel']
            elif config.data.dataset.train_dataset == 'TIMIT':
                fake_img = real_fake_data['fake']['img']
                fake_mel = real_fake_data['fake']['mel']
            else:
                print('Not support this dataset!')
                raise

            # with torch.no_grad():
            real_mel = real_mel.view((-1, 1) + real_mel.size()[-2:]).to(device)
            voice_embedding = voice_model(real_mel)

            mis_mel = fake_mel.view((-1, 1) + fake_mel.size()[-2:])
            mis_match_voice_embedding = voice_model(mis_mel)
            mis_match_voice_embedding = mis_match_voice_embedding.view(config['data']['batch_size'],
                                                                       config['data']['neg_sample'],
                                                                       config['network']['voice']['output_dim'])

            images = real_img.view((-1, 3) + real_img.size()[-2:]).to(device)
            face_embedding = face_model(images)

            mis_match_images = fake_img.view((-1, 3) + fake_img.size()[-2:]).to(device)
            mis_match_face_embedding = face_model(mis_match_images)
            mis_match_face_embedding = mis_match_face_embedding.view(config['data']['batch_size'],
                                                                     config['data']['neg_sample'],
                                                                     config['network']['face']['output_dim'])

            loss_voice_anchor_fake_pair = loss_fake_nce(voice_embedding, face_embedding, mis_match_voice_embedding,
                                                        mis_match_face_embedding)
            total_loss = loss_voice_anchor_fake_pair

            total_loss.backward()
            wandb.log({"loss_voice_anchor_fake_pair": loss_voice_anchor_fake_pair.detach().item()})
            wandb.log({"lr": optimizer.param_groups[0]['lr']})

            if device == "cpu":
                optimizer.step()
            else:
                optimizer.step()

            if config.solver.type != 'monitor':
                if (kkk + 1) == 1 or (kkk + 1) % 10 == 0:
                    lr_scheduler.step(epoch + kkk / len(train_loader))

        if epoch % config.logging.eval_freq == 0:
            prec1 = validate_on_fake_dataset(epoch, val_loader, device, voice_model, face_model, config)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        print('Testing: {}/{}'.format(prec1, best_prec1))
        filename = "{}/last_model.pt".format(working_dir)
        print('Saving:{}'.format(filename))
        wandb.log({"epoch": epoch})
        epoch_saving_voice_face(epoch, voice_model, face_model, optimizer, filename)
        if is_best:
            best_saving_voice_face(working_dir, epoch, voice_model, face_model, optimizer)


if __name__ == '__main__':
    main()
