# Project 2: Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios
## Abstract:
AI-Generated Image stae-of-the-art detectors are usally trained in idealized conditions. But in the real world, images goes through a lot of post-processing transformations, which can introduce a degradation signature in the image-pattern and lead the detectors in error. In this context it is inserted the work of [Li et al. (2025)](#li2025), which took most of the state-of-the-art detectors, fine-tuned them on their Dataset (representing the same images but first in the ideal scenario, then after transmitting them through the internet and after some redigitalization process) and analizing how the detection real/fake changed across the transformation (in respect to the ideal scenario).
In this project, we are going further:

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
