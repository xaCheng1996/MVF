# Voice-Face Homogeneity Tells Deepfake

> Detects deepfakes by exploiting the natural identity-level homogeneity between voice and face — a cross-modal consistency that deepfake generation breaks.

## Authors

**Harry Cheng**<sup>1</sup>, **Yangyang Guo**<sup>2</sup>\*, **Tianyi Wang**<sup>3</sup>, **Qi Li**<sup>1</sup>, **Xiaojun Chang**<sup>4</sup>, **Liqiang Nie**<sup>5</sup>\*

<sup>1</sup> School of Computer Science and Technology, Shandong University
<sup>2</sup> School of Computing, National University of Singapore
<sup>3</sup> Department of Computer Science, The University of Hong Kong
<sup>4</sup> Faculty of Engineering and Information Technology, University of Technology Sydney
<sup>5</sup> Department of Computer Science and Technology, Harbin Institute of Technology (Shenzhen)
\* Corresponding authors

## Links

- **Paper**: [Voice-Face Homogeneity Tells Deepfake](https://dl.acm.org/doi/10.1145/3625231)
- **arXiv**: [2203.02195](https://arxiv.org/abs/2203.02195)
- **Checkpoint (DFDC)**: [Google Drive](https://drive.google.com/file/d/1D1M6cg4KShfsqrvpqhuNI3TNn-U8vk4h/view?usp=sharing)
- **Checkpoint (FakeAVCeleb)**: [Google Drive](https://drive.google.com/file/d/1kPneI1lrx9T6m-fT-Z79uJSlMeMClQsG/view?usp=drive_link)
- **Code Repository**: [GitHub](https://github.com/iLearn-Lab/ToMM23-VFD)

---

## Table of Contents

- [Updates](#updates)
- [Introduction](#introduction)
- [Highlights](#highlights)
- [Method / Framework](#method--framework)
- [Project Structure](#project-structure)
- [Checkpoints / Models](#checkpoints--models)
- [Dataset / Benchmark](#dataset--benchmark)
- [Usage](#usage)
- [TODO](#todo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [License](#license)

---

## Updates

- [04/2026] Transfer repos to iLearn-Lab
- [2023] Paper published in ACM Transactions on Multimedia Computing, Communications and Applications (ToMM), Vol. 20, Issue 3

---

## Introduction

This repository is the official implementation of **Voice-Face Homogeneity Tells Deepfake**, published in **ACM ToMM 2023**.

Real videos exhibit a natural identity-level homogeneity between a person's voice and face — their vocal and visual characteristics are correlated through shared identity. Deepfake generation typically manipulates only one modality, breaking this natural cross-modal consistency.

**VFD** (Voice-Face Deepfake detection) detects deepfakes by measuring the matching degree between the voice and face in a video clip. A mismatch signals a potential forgery.

---

## Highlights

- Exploits **voice-face identity homogeneity** as a natural, annotation-free detection signal
- Detects audio-visual deepfakes across DFDC, DF-TIMIT, and FakeAVCeleb
- Provides pretrained checkpoints for DFDC and FakeAVCeleb

---

## Method / Framework

VFD trains a cross-modal matching model to determine whether the voice and face in a video clip belong to the same identity. Real videos produce high consistency scores; deepfakes that manipulate one modality produce a detectable mismatch.

---

## Project Structure

```text
.
├── FaceModel/              # Face feature extraction model
├── configs/                # Dataset-specific configuration files
│   ├── DFDC/
│   └── FakeAVCeleb/
├── datasets/               # Dataset class definitions
├── lists/                  # Annotation list files (train/test splits)
├── utils/                  # Utility functions
├── finetune_deepfake.py    # Fine-tuning script
├── pretrain_general.py     # General pretraining script
├── test.py                 # Testing script
├── test_vfd.py             # Main evaluation script
└── README.md
```

---

## Checkpoints / Models

Download checkpoints and place into `./exp/[Dataset]/`:

- **DFDC**: [Google Drive](https://drive.google.com/file/d/1D1M6cg4KShfsqrvpqhuNI3TNn-U8vk4h/view?usp=sharing)
- **FakeAVCeleb**: [Google Drive](https://drive.google.com/file/d/1kPneI1lrx9T6m-fT-Z79uJSlMeMClQsG/view?usp=drive_link)

---

## Dataset / Benchmark

Supports DFDC, DF-TIMIT, and FakeAVCeleb. Steps:

### 1. Download datasets

Download the original datasets from their official sources.

### 2. Extract frames and audio

Extract frames and audio, and organize annotation files under `./lists/[Dataset]/`:

```
/data/FakeAVCeleb/test/face/RealVideo-RealAudio/African/women/id04245/00001.jpg 0
/data/FakeAVCeleb/test/voice/RealVideo-RealAudio/African/women/id04245/00001.wav 0
```

Format: `<file_path> <label>`, where label is `0` (real) or `1/2/3` (fake).

---

## Usage

### Testing

```bash
python test_vfd.py --config ./configs/DFDC/test.yaml
python test_vfd.py --config ./configs/FakeAVCeleb/test.yaml
```

---

## TODO

- [ ] Add training script documentation
- [ ] Release DF-TIMIT configuration and checkpoint

---

## Citation

If you find our paper useful, please cite:

```bibtex
@article{cheng2023voice,
  title={Voice-face homogeneity tells deepfake},
  author={Cheng, Harry and Guo, Yangyang and Wang, Tianyi and Li, Qi and Chang, Xiaojun and Nie, Liqiang},
  journal={ACM Transactions on Multimedia Computing, Communications and Applications},
  volume={20},
  number={3},
  pages={1--22},
  year={2023},
}
```

---

## Acknowledgement

- Thanks to the creators of DFDC, DF-TIMIT, and FakeAVCeleb for making their datasets available.
- Thanks to our supervisors and collaborators for their support.

---

## License

This project is released under the Apache License 2.0.
