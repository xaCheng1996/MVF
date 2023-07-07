import torch.nn as nn
from tqdm import tqdm
import wandb
import numpy as np
import torch
import sklearn.metrics


def validate_voice_face(epoch, val_loader, device, config, voice_model, face_model):
    voice_model.eval()
    face_model.eval()
    match = []
    target = []
    plt_pos = []

    with torch.no_grad():
        for iii, (match_pair, mis_match_pair) in enumerate(tqdm(val_loader)):
            # if iii > 100: break
            image = match_pair['img']
            mel = match_pair['mel']
            list_id = match_pair['label']

            image = image.view((-1, 3) + image.size()[-2:])
            b, c, h, w = image.size()
            image_input = image.to(device).view(-1, c, h, w)
            image_features = face_model(image_input)

            voice = mel.view((-1, 1) + mel.size()[-2:])
            b, c, h, w = voice.size()
            voice_input = voice.to(device).view(-1, c, h, w)
            voice_features = voice_model(voice_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            voice_features /= voice_features.norm(dim=-1, keepdim=True)
            tmp = torch.sum(image_features * voice_features, dim=1, keepdim=True).cpu()
            match.append(tmp)
            plt_pos.append(tmp)
            for num in range(voice_features.shape[0]):
                target.append(1)

            mis_match_img = mis_match_pair['img']
            mis_match_img = mis_match_img.view((-1, 3) + mis_match_img.size()[-2:])
            b, c, h, w = mis_match_img.size()
            mis_match_image_input = mis_match_img.to(device).view(-1, c, h, w)
            mis_match_image_features = face_model(mis_match_image_input)
            mis_match_image_features = mis_match_image_features.view(config['data']['neg_sample'],
                                                                     config['data']['batch_size'],
                                                                     config['network']['face']['output_dim'])
            mis_match_image_features /= mis_match_image_features.norm(dim=-1, keepdim=True)

            for mid in range(config['data']['neg_sample']):
                # match.append(pdist(image_features, voice_features).cpu())
                tmp = torch.sum(mis_match_image_features[mid] * voice_features, dim=1, keepdim=True).cpu()
                match.append(tmp)
                for num in range(voice_features.shape[0]):
                    target.append(0)


    match = np.stack(match).flatten()
    # print(match.shape)
    target = np.array(target)
    auc = sklearn.metrics.roc_auc_score(target, match)
    wandb.log({"auc": auc})
    print('Epoch: [{}/{}]: auc: {}'.format(epoch, config.solver.epochs, auc))
    return auc


def validate_on_fake_dataset(epoch, val_loader, device, voice_model, face_model, config):
    voice_model.eval()
    face_model.eval()
    match = []
    target = []
    match_pred = []
    with torch.no_grad():
        for iii, match_pair in enumerate(tqdm(val_loader)):
            image = match_pair['img']
            mel = match_pair['mel']
            list_id = match_pair['label']

            image = image.view((-1, 3) + image.size()[-2:])
            b, c, h, w = image.size()
            image_input = image.to(device).view(-1, c, h, w)
            image_features = face_model(image_input)

            voice = mel.view((-1, 1) + mel.size()[-2:])
            b, c, h, w = voice.size()
            voice_input = voice.to(device).view(-1, c, h, w)
            voice_features = voice_model(voice_input)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            voice_features /= voice_features.norm(dim=-1, keepdim=True)

            tmp = torch.sum(voice_features * image_features, dim=1, keepdim=True)
            match.append(tmp.cpu())

            for label in list_id:
                if int(label.cpu()) == 0:
                    target.append(1)
                else:
                    target.append(0)

    match = np.stack(match).flatten()
    # match_pred = np.stack(match_pred).flatten()
    for att in match:
        if att > 0.0:
            match_pred.append(1)
        else:
            match_pred.append(0)

    target = np.array(target)
    match_pred = np.array(match_pred)
    auc = sklearn.metrics.roc_auc_score(target, match)
    acc = sklearn.metrics.accuracy_score(target, match_pred)
    print('Epoch: [{}/{}]: auc: {}, acc: {}'.format(epoch, config.solver.epochs, auc, acc))
    return auc
