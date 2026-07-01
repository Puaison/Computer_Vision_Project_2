# Project 2: Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios
## Abstract:
AI-Generated Image detectors are usually trained in idealized conditions. But in the real world, images goes through a lot of post-processing transformations, which can introduce a degradation signature in the image-pattern and lead the detectors in error. In this context it is inserted the work of [Li et al. (2025)](#li2025): They created a new Dataset (called RRDataset) containing images in three different format categories: in their original format, in the modified format due to the transmition through the internet, and in the modified format due to the redigitalization process. They took most state-of-the-art detectors, fine-tuned them on their Dataset contaning these eterogeneous images and analized how the detection real/fake changed across the transformations (in respect to the origianl scenario).

In this project, we are going further:
- We tried not only to classify Real/AI Images across different transformations, but also to identify the category format to which the images belong; we analyzed if one task can benefit from the features and model-weights learned from the opposite task or if they compete, resulting in the degradation of both tasks' performance during joint-training. 
- We adapted the state-of-the-art [DRCTConvNext_Base](#chen2024drct) in order to take in input not only the RGB-values but also the amplitude of the Discrete Fourier Transform, following the hypothesis that some post-processing transformations can leave a mark in the spectrum (like low-pass or high-pass filters).

## Dataset

## Other stuffs
We introduce a second head which has the role of detecting which type of transformation the image was subjected, and then 
I modified the the best network found in the Benchamrk of [[1]](#li2025) (that is [DRCTConvB](#chen2024drct)) in order to add a Parallel STEM on DFT, and then sum the feature maps of this stem with the feature maps of the RGB stem. The hypothesis is that, for understanding what type of transformation the image is undergone, some of these information can be in the DFT magnitude. And in order to indicate to the Network if the magnitude in the recptive field is a high-frequency or low-fre, we overlapped with a Radial Map.
We are going to see in details:
Creation of the subset
Replication of the experiment in this subset and then seeing what are the results of the DFT

**N.B.** I took inspiration and adapted the Train and Test Loop skeletons from one of my previous projects in which I worked and collaborated in: https://github.com/cybernetic-m/DAgger4Robotics

However I did significant changes in order to adapt those skeletons to this work


# References

<a id="li2025"></a>

[1] C. Li, X. Wang, M. Li, B. Miao, P. Sun, Y. Zhang, X. Ji, and Y. Zhu,
“Bridging the Gap Between Ideal and Real-world Evaluation: Benchmarking AI-Generated Image Detection in Challenging Scenarios,”
arXiv preprint arXiv:2509.09172, 2025.
[Paper on arXiv](https://arxiv.org/abs/2509.09172)
    
<a id="chen2024drct"></a>

[2] B. Chen, J. Zeng, J. Yang, and R. Yang,
“DRCT: Diffusion Reconstruction Contrastive Training towards Universal Detection of Diffusion Generated Images,”
in Proceedings of the 41st International Conference on Machine Learning (ICML),
vol. 235, pp. 7621–7639, 2024.
[Paper Page](https://proceedings.mlr.press/v235/chen24ay.html), [PDF](https://raw.githubusercontent.com/mlresearch/v235/main/assets/chen24ay/chen24ay.pdf)
