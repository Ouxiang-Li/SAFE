# SAFE:  Simple Preserved and Augmented FEatures

This is the official Pytorch implementation of our paper:

> [Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspectives](https://arxiv.org/abs/2408.06741)
>
> Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, Fuli Feng

## News

- `2025/04` :new: Include evaluation on GPT-4o generations, achieving 98.92% (GenEval) and 96.32% (ReasoningEdit) accuracies — see [Getting the data](#getting-the-data).
- `2024/11` :fire: We collect a new testset [`DiTFake`](https://rec.ustc.edu.cn/share/bb75c2e0-aa6c-11ef-add8-4fbd6e5ad235), comprising three SOTA DiT-based generators (i.e., Flux, PixArt, and SD3). We hope this dataset could facilitate more comprehensive evaluations for SID.
- `2024/11` :tada: Our paper is accepted by KDD2025 ADS Track.

## Requirements

Install the environment as follows:

```python
# create conda environment
conda create -n SAFE -y python=3.9
conda activate SAFE
# install pytorch 
pip install torch==2.2.1 torchvision==0.17.1
# install other dependencies
pip install -r requirements.txt
```

We are using torch 2.2.1 in our production environment, but other versions should be fine as well.

## Getting the data

|             |                            paper                             |                             Url                              |
| :---------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  Train Set  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)            | [Link](https://cmu.app.box.com/s/4syr4womrggfin0tsfhxohaec5dh6n48) |
|  Val   Set  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)            | [Link](https://cmu.app.box.com/s/4syr4womrggfin0tsfhxohaec5dh6n48/folder/129187348352) |
|  Test Set1  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)            | [Link](https://cmu.app.box.com/s/4syr4womrggfin0tsfhxohaec5dh6n48/folder/129187348352) |
|  Test Set2  | [FreqNet AAAI2024](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) | [Link](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing) |
|  Test Set3  | [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect) | [Link](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=sharing) |
|  Test Set4  | [GenImage NeurIPS2023](https://github.com/GenImage-Dataset/GenImage)             | [Link](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS) |
|  Test Set5  | [DiTFake Ours](https://github.com/Ouxiang-Li/SAFE)                               | [Link](https://rec.ustc.edu.cn/share/bb75c2e0-aa6c-11ef-add8-4fbd6e5ad235) |
|  Test Set6  | [GPT-ImgEval](https://github.com/PicoTrex/GPT-ImgEval)                           | [Link](https://huggingface.co/datasets/Yejy53/GPT-ImgEval) |

The generation script for our dataset is provided in `data/generation.py`, we hope more synthetic images from up-to-date generative models coud be promptly evaluated and made publicly available. Details of our `DiTFake` testset and comparative results can be found in the latest [ArXiv](https://arxiv.org/abs/2408.06741) paper.

`2025/04` : Due to the impressive performance of **GPT-4o** in image generation tasks, it also poses new challenges for synthetic image detection. Here, we evaluate the generalization performance of our **SAFE** on this front, using two subsets collected by [GPT-ImgEval](https://github.com/PicoTrex/GPT-ImgEval): **GenEval** (555 fake images) and **ReasoningEdit** (190 fake images). Our method achieved **98.92%** and **96.32%** ACC on these two test sets, respectively.

## Directory structure

<details>
<summary> You should organize the above data as follows: </summary>

```
data/datasets
|-- train_ForenSynths
|   |-- train
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   |-- horse
|   |-- val
|   |   |-- car
|   |   |-- cat
|   |   |-- chair
|   |   |-- horse
|-- test1_ForenSynths/test
|   |-- biggan
|   |-- cyclegan
|   |-- deepfake
|   |-- gaugan
|   |-- progan
|   |-- stargan
|   |-- stylegan
|   |-- stylegan2
|-- test2_Self-Synthesis/test
|   |-- AttGAN
|   |-- BEGAN
|   |-- CramerGAN
|   |-- InfoMaxGAN
|   |-- MMDGAN
|   |-- RelGAN
|   |-- S3GAN
|   |-- SNGAN
|   |-- STGAN
|-- test3_Ojha/test
|   |-- dalle
|   |-- glide_100_10
|   |-- glide_100_27
|   |-- glide_50_27
|   |-- guided          # Also known as ADM.
|   |-- ldm_100
|   |-- ldm_200
|   |-- ldm_200_cfg
|-- test4_GenImage/test
|   |-- ADM
|   |-- BigGAN
|   |-- Glide
|   |-- Midjourney
|   |-- stable_diffusion_v_1_4
|   |-- stable_diffusion_v_1_5
|   |-- VQDM
|   |-- wukong
|-- test5_DiTFake/test
|   |-- FLUX.1-schnell
|   |-- PixArt-Sigma-XL-2-1024-MS
|   |-- stable-diffusion-3-medium-diffusers
```
</details>

## Training

```
bash scripts/train.sh
```

This script enables training with 4 GPUs, you can specify the number of GPUs by setting `GPU_NUM`.

## Inference

```
bash scripts/eval.sh
```

We provide the pretrained checkpoint in `./checkpoint/checkpoint-best.pth`, you can directly run the script to reproduce our results. 

## Citing
If you find this repository useful for your work, please consider citing it as follows:
```
@article{li2024improving,
  title={Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspective},
  author={Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and Jiang, Xiaolong and Hu, Yao and Feng, Fuli},
  journal={arXiv preprint arXiv:2408.06741},
  year={2024}
}
```