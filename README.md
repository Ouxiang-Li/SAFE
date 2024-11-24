# SAFE:  Simple Preserved and Augmented FEatures

This is the official Pytorch implementation of our paper:

> [Improving Synthetic Image Detection Towards Generalization: An Image Transformation Perspectives](https://arxiv.org/abs/2408.06741)
>
> Ouxiang Li, Jiayin Cai, Yanbin Hao, Xiaolong Jiang, Yao Hu, Fuli Feng

## News

- [2024/11] :fire: Suggested by reviewers, we collect a new testset [`DiTFake`](https://rec.ustc.edu.cn/share/bb75c2e0-aa6c-11ef-add8-4fbd6e5ad235), comprising three SOTA DiT-based models, including Flux, PixArt, and SD3. We hope this dataset could facilitate community for more comprehensive evaluation.
- [2024/11] :tada: Our paper is accepted by KDD25 ADS Track.

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
|  Train Set  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)            | [link](https://drive.google.com/file/d/1iVNBV0glknyTYGA9bCxT_d0CVTOgGcKh/view) |
|  Val   Set  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)            | [link](https://drive.google.com/file/d/1FU7xF8Wl_F8b0tgL0529qg2nZ_RpdVNL/view) |
|  Test Set1  | [CNNDetection CVPR2020](https://github.com/PeterWang512/CNNDetection)            | [link](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view) |
|  Test Set2  | [FreqNet AAAI2024](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) | [link](https://drive.google.com/drive/folders/11E0Knf9J1qlv2UuTnJSOFUjIIi90czSj?usp=sharing) |
|  Test Set3  | [UniversalFakeDetect CVPR2023](https://github.com/Yuheng-Li/UniversalFakeDetect) | [link](https://drive.google.com/drive/folders/1nkCXClC7kFM01_fqmLrVNtnOYEFPtWO-?usp=sharing) |
|  Test Set4  | [GenImage NeurIPS2023](https://github.com/GenImage-Dataset/GenImage)             | [link](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS) |
|  Test Set5  | DiTFake           | [link](https://rec.ustc.edu.cn/share/bb75c2e0-aa6c-11ef-add8-4fbd6e5ad235) |

Details of our `DiTFake` testset and comparative results will be provided in the forthcoming camera-ready version soon.

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