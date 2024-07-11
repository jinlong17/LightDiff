<!--
 * @Descripttion: 
 * @version: 
 * @Author: Jinlong Li CSU PhD
 * @Date: 2024-07-10 20:59:10
 * @LastEditors: Jinlong Li CSU PhD
 * @LastEditTime: 2024-07-10 21:47:55
-->



# [LightDiff](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Light_the_Night_A_Multi-Condition_Diffusion_Framework_for_Unpaired_Low-Light_CVPR_2024_paper.pdf): Light the Night: A Multi-Condition Diffusion Framework for Unpaired Low-Light Enhancement in Autonomous Driving (CVPR 2024)


[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2404.04804)
[![supplement](https://img.shields.io/badge/Supplementary-Material-red)]()
<!-- [![video](https://img.shields.io/badge/Video-Presentation-F9D371)]() -->




This is the official implementation of CVPR2024 paper Light the Night: A Multi-Condition Diffusion Framework for Unpaired Low-Light Enhancement in Autonomous Driving".

[Jinlong Li](https://jinlong17.github.io/)<sup>1*</sup>, [Baolu Li]()<sup>1*</sup>,[Zhengzhong Tu](https://github.com/vztu)<sup>2</sup>, [Xinyu Liu]()<sup>1</sup>, [Qing Guo]()<sup>3</sup>, [Felix Juefei-Xu]()<sup>4</sup>, [Runsheng Xu](https://derrickxunu.github.io/)<sup>5</sup>, [Hongkai Yu]()<sup>1</sup>


<sup>1</sup>Cleveland State University, <sup>2</sup>University of Texas at Austin,  <sup>3</sup>A*STAR, <sup>4</sup>New York University, <sup>5</sup>UCLA

Computer Vision and Pattern Recognition (CVPR), 2024


## [Project Page](https://genforce.github.io/freecontrol/) <br>


![teaser](./images/lightdiff.png)


## Getting Started

**Environment Setup**
- We proovide a [conda env file](environment.yml) for environment setup. 
```bash
conda env create -f environment.yml
conda activate lightdiff
```

- Following the installation of [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth) step by step. 


**Model Training**


**Model testing**


## DATA Preparation


- Download nuScenes official dataset.


**Training set**


We select all 616 daytime scenes of the nuScenes training set containing total **24,745 camera front images** as our training set. 


**Testing set**


## Multi-modality Data Generation



**Instruction prompt**


**Depth map**




## Citation
 If you are using our wokr for your research, please cite the following paper:
 ```bibtex
@inproceedings{li2024light,
  title={Light the Night: A Multi-Condition Diffusion Framework for Unpaired Low-Light Enhancement in Autonomous Driving},
  author={Li, Jinlong and Li, Baolu and Tu, Zhengzhong and Liu, Xinyu and Guo, Qing and Juefei-Xu, Felix and Xu, Runsheng and Yu, Hongkai},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={15205--15215},
  year={2024}
}
```



## Acknowledgment

This code is modified based on the code [ControlNet-v1-1-nightly](https://github.com/lllyasviel/ControlNet-v1-1-nightly) and [BEVDepth](https://github.com/Megvii-BaseDetection/BEVDepth). Thanks.