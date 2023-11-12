# Voice-Face Homogeneity Tells Deepfake

Code for 'Voice-Face Homogeneity Tells Deepfake' [[Arxiv]][https://arxiv.org/abs/2203.02195], which is designed to detect deepfake images via the matching view of voices and faces.



### Data Preparation

1. Download the DFDC , DF-TIMIT, or FakeAVCeleb Datasets.

2. Extract the frames and audio from the videos, and store them in the format as described in ./lists/[Dataset]/train_frame.txt. For instance, the frames and corresponding audios can be stored as:

   `/data/FakeAVCeleb/test/face/RealVideo-RealAudio/African/women/id04245/00001.jpg 0`

   and 

   `/data/FakeAVCeleb/test/voice/RealVideo-RealAudio/African/women/id04245/00001.wav 0`

   The first item is the **path** of image/audio, the second item is the label (real for 0, and fake for 1/2/3)

   The other datasets, e.g., DFDC, can also be formatted.

### Quick Start

1. Download the pre-trained model from:

   DFDC: [link](https://drive.google.com/drive/folders/11YZ91OG1qFxDKRS8D1ihGL9wDsl7ybk7?usp=drive_link)

   FakeAVCeleb: [link](https://drive.google.com/drive/folders/1DhnCb0nS3EM3Ym3tPIOcJGElLgv3YsJ8?usp=drive_link)

   and put them into ./exp/[Dataset]

2. Run:

   `python test_vfd.py --config ./configs/DFDC/test.yaml`

   `python test_vfd.py --config ./configs/FakeAVCeleb/test.yaml`

###  Citation

Kindly cite us if you find this paper helps :)

````
@article{VFD,
author = {Cheng, Harry and Guo, Yangyang and Wang, Tianyi and Li, Qi and Chang, Xiaojun and Nie, Liqiang},
title = {Voice-Face Homogeneity Tells Deepfake},
year = {2023},
publisher = {ACM},
volume = {20},
number = {3},
doi = {10.1145/3625231},
journal = {ACM Trans. Multimedia Comput. Commun. Appl.},
}
````

