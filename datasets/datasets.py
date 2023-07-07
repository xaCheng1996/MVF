import einops
import torch.utils.data as data
import os
import os.path
import numpy as np
from numpy.random import randint, choice
import matplotlib
import matplotlib.pyplot as plt
import pdb
import io
import time
import pandas as pd
import torchvision
from pydub import AudioSegment
import random
import torchaudio
from PIL import Image, ImageOps, ImageFile
import cv2
import numbers
from tqdm import tqdm
import math
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True

class GroupTransform(object):
    def __init__(self, transform):
        self.worker = transform

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()


class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                print(len(img_group))
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                print(len(img_group))
                rst = np.concatenate(img_group, axis=2)
                return rst


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class Action_DATASETS(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file, sep=';')
        return classes_all.values.tolist()

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(
                    self.total_length) + randint(record.num_frames // 2),
                              record.num_frames) + self.index_bias
            offsets = np.concatenate((
                np.arange(record.num_frames),
                randint(record.num_frames,
                        size=self.total_length - record.num_frames)))
            return np.sort(offsets) + self.index_bias
        offsets = list()
        ticks = [i * record.num_frames // self.num_segments
                 for i in range(self.num_segments + 1)]

        for i in range(self.num_segments):
            tick_len = ticks[i + 1] - ticks[i]
            tick = ticks[i]
            if tick_len >= self.seg_length:
                tick += randint(tick_len - self.seg_length + 1)
            offsets.extend([j for j in range(tick, tick + self.seg_length)])
        return np.array(offsets) + self.index_bias

    def _get_val_indices(self, record):
        if self.num_segments == 1:
            return np.array([record.num_frames // 2], dtype=np.int) + self.index_bias

        if record.num_frames <= self.total_length:
            if self.loop:
                return np.mod(np.arange(self.total_length), record.num_frames) + self.index_bias
            return np.array([i * record.num_frames // self.total_length
                             for i in range(self.total_length)], dtype=np.int) + self.index_bias
        offset = (record.num_frames / self.num_segments - self.seg_length) / 2.0
        return np.array([i * record.num_frames / self.num_segments + offset + j
                         for i in range(self.num_segments)
                         for j in range(self.seg_length)], dtype=np.int) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        # print(segment_indices)
        return self.get(record, segment_indices)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def get(self, record, indices):
        images = list()
        for i, seg_ind in enumerate(indices):
            p = int(seg_ind)
            # print(p)
            try:
                seg_imgs = self._load_image(record.path, p)
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(seg_imgs)
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)


class FrameRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def label(self):
        return int(self._data[1])


class VoxDatasets(data.Dataset):
    def __init__(self, list_file, labels_file,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1):

        self.list_file = list_file
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False

    def _load_image(self, idx):
        return [Image.open(idx).convert('RGB')]


    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file, sep=';')
        return classes_all.values.tolist()

    def _parse_list(self):
        self.video_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.video_list[index].path
        img = self._load_image(record)
        label = self.video_list[index].label
        return self.transform(img), label

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def __len__(self):
        return len(self.video_list)


class VoxAudioDatasets(data.Dataset):
    def __init__(self, list_file, labels_file,
                 num_segments=1, new_length=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False, index_bias=1):

        self.list_file = list_file
        self.num_segments = num_segments
        self.seg_length = new_length
        self.image_tmpl = image_tmpl
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.loop = False
        self.index_bias = index_bias
        self.labels_file = labels_file

        if self.index_bias is None:
            if self.image_tmpl == "frame{:d}.jpg":
                self.index_bias = 0
            else:
                self.index_bias = 1
        self._parse_list()
        self.initialized = False
        self.mel_spc = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160,
                                                            win_length=480)

    def _load_wav(self, idx):
        # print(idx)
        # wav = AudioSegment.from_file(idx)
        # print(wav.raw_data)
        # sr = wav.frame_rate
        # print(sr)
        # if wav.duration_seconds < 4:
        #     wav = wav * 2
        # wav = wav[1000:4000]

        # handler = wav.export(format='wav')
        aud, sr = torchaudio.load(idx)
        aud = aud[0][:40960]
        mel_out = self.mel_spc(aud.to(torch.float))
        # print(mel_out)
        return mel_out

    @property
    def total_length(self):
        return self.num_segments * self.seg_length

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file, sep=';')
        return classes_all.values.tolist()

    def _parse_list(self):
        self.audio_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def pre(self):
        for pat in tqdm(self.audio_list):
            idx = pat.path
            idx = idx.split('/')
            fa_path = ''
            for ele in idx[:-1]:
                if ele == 'aac':
                    ele = 'wav'
                fa_path = os.path.join('/', fa_path, ele)
            # print(fa_path)
            if not os.path.exists(fa_path):
                os.makedirs(fa_path)
            wav = AudioSegment.from_file(pat.path)
            # print(wav.raw_data)
            sr = wav.frame_rate
            # print(sr)
            if wav.duration_seconds < 4:
                wav = wav * 2
            wav = wav[1000:4000]
            res_path = os.path.join(fa_path, idx[-1].replace('m4a', 'wav'))
            handler = wav.export(out_f=res_path, format='wav')

    def __getitem__(self, index):
        record = self.audio_list[index].path
        mel = self._load_wav(record)
        label = self.audio_list[index].label
        return mel, label

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def __len__(self):
        return len(self.audio_list)


class Vox_MVA_Datasets(data.Dataset):
    def __init__(self, list_video_file, list_wav_file, labels_file, neg_sample, random_seed=None, transformer=None):

        self.list_video_file = list_video_file
        self.list_wav_file = list_wav_file
        self.labels_file = labels_file
        self.transformer = transformer
        self.neg_sample = neg_sample

        if random_seed:
            self.random_seed = random_seed
        torch.manual_seed(self.random_seed)

        self._parse_list()
        self.initialized = False
        self.mel_spc = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160,
                                                            win_length=480)

    def _load_wav(self, idx):
        aud, sr = torchaudio.load(idx)
        aud = aud[0][:40960]
        mel_out = self.mel_spc(aud.to(torch.float))
        return mel_out

    def _load_image(self, idx):
        return [Image.open(idx).convert('RGB')]

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file, sep=',')
        return classes_all.values.tolist()

    def _parse_list(self):
        self.audio_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_wav_file)]
        self.video_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_video_file)]
        self.dict_wav_img = {}
        self.dict_cnt = {}
        classes_all = pd.read_csv(self.labels_file, delimiter=',')
        classes = classes_all.values.tolist()
        id_list = []
        for id_p in classes:
            id_list.append(int(id_p[0]))
        cnt = 0
        for video in self.video_list:
            if not self.dict_wav_img.get(video.label):
                self.dict_wav_img[video.label] = {
                    'img': [],
                    'wav': []
                }
            self.dict_wav_img[video.label]['img'].append(video.path)

        for audio in self.audio_list:
            if not self.dict_wav_img.get(audio.label): continue
            self.dict_wav_img[audio.label]['wav'].append(audio.path)

        for video_label in self.dict_wav_img.keys():
            img_list = self.dict_wav_img[video_label]['img']
            wav_list = self.dict_wav_img[video_label]['wav']
            len_id = min(len(img_list), len(wav_list))
            for ind in range(len_id):
                self.dict_cnt[cnt] = {
                    'id': video_label,
                    'img': img_list[ind],
                    'wav': wav_list[ind],
                }
                cnt += 1


    def __getitem__(self, index):
        record = self.dict_cnt[index]
        record_wav = record['wav']
        mel = self._load_wav(record_wav)
        # rand_i = random.randint(0, len(self.dict_wav_img[record['id']]['img']) - 1)
        # record_img = self.dict_wav_img[record['id']]['img'][rand_i]
        record_img = record['img']
        img = self._load_image(record_img)

        # label = self.audio_list[index].label
        label = record['id']

        match_pair = {
            'img': self.transformer(img),
            'mel': mel,
            'label': label
        }
        # print(self.dict_wav_img.keys())

        mis_img_list = []
        mis_id_list = []
        for sa in range(self.neg_sample):
            mis_id = 0
            for i in range(10):
                mis_id = choice(list(self.dict_wav_img.keys()))
                if int(mis_id) != int(label):
                    break
            rand_i = random.randint(0, len(self.dict_wav_img[mis_id]['img']) - 1)
            mis_img = self.dict_wav_img[mis_id]['img'][rand_i]
            mis_img = self._load_image(mis_img)
            mis_img_list.append(self.transformer(mis_img))
            mis_id_list.append(mis_id)

        mis_img = torch.stack(mis_img_list, dim=0)
        mismatch_pair = {
            'img': mis_img,
            'mel': mel,
            'truth_label': label,
            'sample_label': mis_id_list,
        }

        return match_pair, mismatch_pair

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def __len__(self):
        return len(self.dict_cnt)


class FakeAVCeleb_Datasets(data.Dataset):
    def __init__(self, list_face_file, labels_file, neg_sample=0, random_seed=1024, transformer=None):
        self.list_face_file = list_face_file
        self.labels_file = labels_file
        self.transformer = transformer
        self.neg_sample = neg_sample

        if random_seed:
            self.random_seed = random_seed
            print('use seed %d' % self.random_seed)
            random.seed(self.random_seed)

        self._parse_list()
        self.initialized = False
        self.mel_spc = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160,
                                                            win_length=480)
    def _load_wav(self, idx):
        aud, sr = torchaudio.load(idx)
        start_point = random.randrange(1600, 16000, 160)
        num_channels, num_frames = aud.shape
        if num_frames < start_point + 40960:
            # self.plot_waveform(aud, sr, title=idx)
            # self.plot_specgram(aud, sr, title=idx)
            # aud = aud.numpy()
            repeat_time = (start_point + 40960) // num_frames
            aud = einops.repeat(aud, 'n f -> n (c f)', c=repeat_time + 1)
        aud = torch.mean(aud, dim=0)
        aud = aud[start_point:start_point + 40960]
        mel_out = self.mel_spc(aud.to(torch.float))
        return mel_out

    def _load_image(self, idx):
        return [Image.open(idx).convert('RGB')]

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file, sep=';')
        return classes_all.values.tolist()

    def _parse_list(self):
        self.video_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_face_file)]
        self.img_wav_list = []
        self.img_to_id = {}
        self.id_to_img = {}
        self.id_sample_list = []
        self.img_wav_list_real = []
        for video in self.video_list:
            # if video.label == 1: continue
            # if video.label == 2: continue
            video_path = video.path
            audio_path = video_path.replace('/face/', '/voice/').replace('.jpg', '.wav')
            self.img_wav_list.append([video_path, audio_path, video.label])
            if video.label == 0:
                self.img_wav_list_real.append([video_path, audio_path, video.label])
            id_info = video_path.split('/')[-2]
            self.img_to_id[video_path] = id_info
            if not self.id_to_img.get(id_info):
                self.id_to_img[id_info] = {
                    'real': [],
                    'fake_audio': [],
                    'fake_video': []
                }
            if video.label == 0:
                self.id_to_img[id_info]['real'].append([video_path, audio_path])
            elif video.label == 1:
                self.id_to_img[id_info]['fake_video'].append([video_path, audio_path])
            elif video.label == 2:
                self.id_to_img[id_info]['fake_audio'].append([video_path, audio_path])
            elif video.label == 3:
                # continue
                self.id_to_img[id_info]['fake_video'].append([video_path, audio_path])
                self.id_to_img[id_info]['fake_audio'].append([video_path, audio_path])
            else:
                print('Wrong video class %s %s, check the data list !' % (id_info, video_path))
        for key in self.id_to_img.keys():
            # print(len(self.id_to_img[key]['real']), len(self.id_to_img[key]['fake_video']), len(self.id_to_img[key]['fake_audio']))
            if len(self.id_to_img[key]['real']) > 0 and len(self.id_to_img[key]['fake_video']) > 5 \
                    and len(self.id_to_img[key]['fake_audio']) > 5:
                self.id_sample_list.append(key)
        # self.img_wav_list_real = self.img_wav_list_real * 20
        return self.img_wav_list

    def __getitem__(self, index):
        if self.neg_sample > 0:
            record = self.img_wav_list_real[index]
            record_img = record[0]
            record_wav = record[1]
            record_label = int(record[2])
            id_info = self.img_to_id[record_img]
            id_sample_list = self.id_sample_list
            fake_img_list = []
            fake_mel_list = []
            for sa in range(self.neg_sample):
                rand_fake_video = random.randint(0, len(self.id_to_img[id_info]['fake_video']) - 1) \
                    if len(self.id_to_img[id_info]['fake_video']) > 1 else 0
                rand_fake_audio = random.randint(0, len(self.id_to_img[id_info]['fake_audio']) - 1) \
                    if len(self.id_to_img[id_info]['fake_audio']) > 1 else 0
                if len(self.id_to_img[id_info]['fake_video']) == 0:
                    id_sample = choice(id_sample_list)
                    rand_fake_video = random.randint(0, len(self.id_to_img[id_sample]['fake_video']) - 1)
                    fake_img = self.id_to_img[id_sample]['fake_video'][rand_fake_video][0]
                else:
                    fake_img = self.id_to_img[id_info]['fake_video'][rand_fake_video][0]

                if len(self.id_to_img[id_info]['fake_audio']) == 0:
                    id_sample = choice(id_sample_list)
                    rand_fake_audio = random.randint(0, len(self.id_to_img[id_sample]['fake_audio']) - 1)
                    fake_mel = self.id_to_img[id_sample]['fake_audio'][rand_fake_audio][1]
                else:
                    fake_mel = self.id_to_img[id_info]['fake_audio'][rand_fake_audio][1]
                fake_img = self._load_image(fake_img)
                fake_img_list.append(self.transformer(fake_img))
                fake_mel_list.append(self._load_wav(fake_mel))
            fake_img_list = torch.stack(fake_img_list, dim=0)
            fake_mel_list = torch.stack(fake_mel_list, dim=0)

            if record_label == 0:
                real_fake_data = {
                    'real': {
                        'img': self.transformer(self._load_image(record_img)),
                        'mel': self._load_wav(record_wav)
                    },
                    'fake_video': {
                        'img': fake_img_list
                    },
                    'fake_audio': {
                        'mel': fake_mel_list
                    }
                }
            else:
                if len(self.id_to_img[id_info]['real']) == 0:
                    id_sample = choice(id_sample_list)
                    rand_real = random.randint(0, len(self.id_to_img[id_sample]['real']) - 1)
                    real_img, real_mel = self.id_to_img[id_sample]['real'][rand_real]
                else:
                    rand_real = random.randint(0, len(self.id_to_img[id_info]['real']) - 1)
                    real_img, real_mel = self.id_to_img[id_info]['real'][rand_real]
                real_fake_data = {
                    'real': {
                        'img': self.transformer(self._load_image(real_img)),
                        'mel': self._load_wav(real_mel)
                    },
                    'fake_video': {
                        'img': fake_img_list
                    },
                    'fake_audio': {
                        'mel': fake_mel_list
                    }
                }
        else:
            record = self.img_wav_list[index]
            record_img = record[0]
            record_wav = record[1]
            record_label = int(record[2])
            real_fake_data = {
                'img': self.transformer(self._load_image(record_img)),
                'mel': self._load_wav(record_wav),
                'label': record_label
            }
        return real_fake_data

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def __len__(self):
        if self.neg_sample > 0:
            return len(self.img_wav_list_real)
        else:
            return len(self.img_wav_list)

    def plot_waveform(self, waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c])
            axes[c].grid(False)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)
        plt.show(block=False)

    def plot_specgram(self, waveform, sample_rate, title="Spectrogram", xlim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show(block=False)


class DFDC_Datasets(data.Dataset):
    def __init__(self, list_face_file, labels_file, neg_sample=0, random_seed=1024, transformer=None):
        self.list_face_file = list_face_file
        self.labels_file = labels_file
        self.transformer = transformer
        self.neg_sample = neg_sample

        # self.voice_path_list = '/share_data/DFDC/dev/voice'

        if random_seed:
            self.random_seed = random_seed
            print('use seed %d' % self.random_seed)
            random.seed(self.random_seed)

        self._parse_list()
        self.initialized = False
        self.mel_spc = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160,
                                                            win_length=480)
    def _load_wav(self, idx):
        aud, sr = torchaudio.load(idx)
        start_point = random.randrange(1600, 16000, 160)
        num_channels, num_frames = aud.shape
        if num_frames < start_point + 40960:
            # self.plot_waveform(aud, sr, title=idx)
            # self.plot_specgram(aud, sr, title=idx)
            # aud = aud.numpy()
            repeat_time = (start_point + 40960) // num_frames
            aud = einops.repeat(aud, 'n f -> n (c f)', c=repeat_time + 1)
        aud = torch.mean(aud, dim=0)
        aud = aud[start_point:start_point + 40960]
        mel_out = self.mel_spc(aud.to(torch.float))
        return mel_out

    def _load_image(self, idx):
        return [Image.open(idx).convert('RGB')]

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file, sep=';')
        return classes_all.values.tolist()

    def _parse_list(self):
        self.video_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_face_file)]
        self.img_wav_list = []
        self.img_wav_list_fake = []
        self.img_to_id = {}
        self.id_to_img = {}
        self.id_sample_list = []
        cnt = 0
        for video in self.video_list:
            video_name = video.path
            if self.neg_sample > 0:
                video_path = video.path
                audio_path = video_path.replace('/face/', '/voice/').replace('.jpg', '.wav').replace('train_sec', 'train')
                if video.label == 0:
                    self.img_wav_list.append([video_path, audio_path, video.label])
                else:
                    self.img_wav_list_fake.append([video_path, audio_path, video.label])

            else:
                # if int(video.label) == 1 and cnt > 4000: continue
                # else:
                video_path = video.path
                audio_path = video_path.replace('/face/', '/voice/').replace('.jpg', '.wav')
                # if not os.path.exists(audio_path) or self.neg_sample is None:
                #     video_name = video_name.split('/')[-1].split('.jpg')[0][:-2]
                #     for i in os.listdir(os.path.join(audio_path)):
                #         audio_path = os.path.join(self.voice_path_list, video_name, i)
                self.img_wav_list.append([video_path, audio_path, video.label])
                    # if int(video.label) == 1:
                    #     cnt += 1
        # return self.img_wav_list

    def __getitem__(self, index):
        record = self.img_wav_list[index]
        record_img = record[0]
        record_wav = record[1]
        record_label = int(record[2])

        if self.neg_sample > 0:
            fake_img_list = []
            fake_mel_list = []
            for ind in range(self.neg_sample):
                rand_fake_video = random.randint(0, len(self.img_wav_list_fake) - 1)

                fake_img = self.img_wav_list_fake[rand_fake_video][0]
                fake_img = self._load_image(fake_img)
                fake_img_list.append(self.transformer(fake_img))

                fake_mel = self.img_wav_list_fake[rand_fake_video][1]
                fake_mel_list.append(self._load_wav(fake_mel))
            fake_img_list = torch.stack(fake_img_list, dim=0)
            fake_mel_list = torch.stack(fake_mel_list, dim=0)

            real_img = self.transformer(self._load_image(record_img))
            real_mel = self._load_wav(record_wav)
            real_fake_data = {
                'real': {
                    'img': real_img,
                    'mel': real_mel,
                },
                'fake': {
                    'img': fake_img_list,
                    'mel': fake_mel_list
                }
            }
        else:
            real_fake_data = {
                'img': self.transformer(self._load_image(record_img)),
                'mel': self._load_wav(record_wav),
                'label': record_label,
                'path': record_img
            }
        return real_fake_data

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def __len__(self):
        if self.neg_sample:
            # return int(5000)
            return len(self.img_wav_list)
        else:
            # return int(6000)
            return len(self.img_wav_list)

    def plot_waveform(self, waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c])
            axes[c].grid(False)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)
        plt.show()

    def plot_specgram(self, waveform, sample_rate, title="Spectrogram", xlim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show(block=False)


class TIMIT_Datasets(data.Dataset):
    def __init__(self, list_face_file, labels_file, neg_sample=0, random_seed=None, transformer=None):
        self.list_face_file = list_face_file
        self.labels_file = labels_file
        self.transformer = transformer
        self.neg_sample = neg_sample

        if random_seed:
            self.random_seed = random_seed
            print('use seed %d' % self.random_seed)
            random.seed(self.random_seed)

        self._parse_list()
        self.initialized = False
        self.mel_spc = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, hop_length=160,
                                                            win_length=480)

    def _load_wav(self, idx):
        aud, sr = torchaudio.load(idx)
        start_point = random.randrange(1600, 16000, 160)
        num_channels, num_frames = aud.shape
        if num_frames < start_point + 40960:
            # self.plot_waveform(aud, sr, title=idx)
            # self.plot_specgram(aud, sr, title=idx)
            # aud = aud.numpy()
            repeat_time = (start_point + 40960) // num_frames
            aud = einops.repeat(aud, 'n f -> n (c f)', c=repeat_time + 1)
        aud = torch.mean(aud, dim=0)
        aud = aud[start_point:start_point + 40960]
        mel_out = self.mel_spc(aud.to(torch.float))
        return mel_out

    def _load_image(self, idx):
        return [Image.open(idx).convert('RGB')]

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file, sep=';')
        return classes_all.values.tolist()

    def _parse_list(self):
        self.video_list = [FrameRecord(x.strip().split(' ')) for x in open(self.list_face_file)]
        self.img_wav_list = []
        self.img_wav_list_fake = []
        self.img_to_id = {}
        self.id_to_img = {}
        self.id_sample_list = []
        for video in self.video_list:
            video_name = video.path
            if self.neg_sample > 0:
                video_path = video.path
                audio_path = video_path.replace('/face/', '/voice/').replace('.jpg', '.wav')
                if video.label == 0:
                    self.img_wav_list.append([video_path, audio_path, video.label])
                else:
                    self.img_wav_list_fake.append([video_path, audio_path, video.label])

            else:
                video_path = video.path
                audio_path = video_path.replace('/face/', '/voice/').replace('.jpg', '.wav')
                self.img_wav_list.append([video_path, audio_path, video.label])
        return self.img_wav_list

    def __getitem__(self, index):
        record = self.img_wav_list[index]
        record_img = record[0]
        record_wav = record[1]
        record_label = int(record[2])

        if self.neg_sample > 0:
            fake_img_list = []
            fake_mel_list = []
            for ind in range(self.neg_sample):
                rand_fake_video = random.randint(0, len(self.img_wav_list_fake) - 1)

                fake_img = self.img_wav_list_fake[rand_fake_video][0]
                fake_img = self._load_image(fake_img)
                fake_img_list.append(self.transformer(fake_img))

                fake_mel = self.img_wav_list_fake[rand_fake_video][1]
                fake_mel_list.append(self._load_wav(fake_mel))
            fake_img_list = torch.stack(fake_img_list, dim=0)
            fake_mel_list = torch.stack(fake_mel_list, dim=0)

            real_img = self.transformer(self._load_image(record_img))
            real_mel = self._load_wav(record_wav)
            real_fake_data = {
                'real': {
                    'img': real_img,
                    'mel': real_mel
                },
                'fake': {
                    'img': fake_img_list,
                    'mel': fake_mel_list
                }
            }
        else:
            real_fake_data = {
                'img': self.transformer(self._load_image(record_img)),
                'mel': self._load_wav(record_wav),
                'label': record_label,
                'path': record_img
            }
        return real_fake_data

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]

    def __len__(self):
        if self.neg_sample:
            return int(1000)
        #     # return len(self.img_wav_list)
        else:
            return len(self.img_wav_list)

    def plot_waveform(self, waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, waveform[c])
            axes[c].grid(False)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
            if ylim:
                axes[c].set_ylim(ylim)
        figure.suptitle(title)
        plt.show()

    def plot_specgram(self, waveform, sample_rate, title="Spectrogram", xlim=None):
        waveform = waveform.numpy()

        num_channels, num_frames = waveform.shape
        time_axis = torch.arange(0, num_frames) / sample_rate

        figure, axes = plt.subplots(num_channels, 1)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].specgram(waveform[c], Fs=sample_rate)
            if num_channels > 1:
                axes[c].set_ylabel(f'Channel {c + 1}')
            if xlim:
                axes[c].set_xlim(xlim)
        figure.suptitle(title)
        plt.show(block=False)
# os.environ["PATH"] += ':/opt/anaconda3/envs/torch_cv/bin/'
# envdir_list = [os.curdir] + os.environ["PATH"].split(os.pathsep)
# Vox = VoxAudioDatasets(list_file='../lists/Voxceleb2_voice/train_frame_ori.txt', labels_file='../lists/voxceleb2.csv')
# Vox.pre()
