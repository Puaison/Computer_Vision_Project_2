# Project 2: Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios
## Abstract:
AI-Generated Image detectors are usually trained in idealized conditions. But in the real world, images goes through a lot of post-processing transformations, which can introduce a degradation signature in the image-pattern and lead the detectors in error. In this context it is inserted the work of [Li et al. (2025)](#li2025): They created a new Dataset (called RRDataset) containing images in three different format categories: in their original format, in the modified format due to the transmition through the internet, and in the modified format due to the redigitalization process. They took most state-of-the-art detectors, fine-tuned them on their Dataset contaning these eterogeneous images and analized how the detection real/fake changed across the transformations (in respect to the origianl scenario).

In this project, we are going further:
- We tried not only to classify Real/AI Images across different transformations, but also to identify the category format to which the images belong; we analyzed if one task can benefit from the features and model-weights learned from the opposite task or if they compete, resulting in the degradation of both tasks' performance during joint-training. 
- We adapted the state-of-the-art [DRCTConvNext_Base](#chen2024drct) (which is the best in the [[1]](#li2025) work) in order to take in input not only the RGB-values but also the amplitude of the Discrete Fourier Transform, following the hypothesis that some post-processing transformations can leave a mark in the spectrum (like low-pass or high-pass filters).


## Dataset
We trained the models in the Google Colab Environment (in particular on T4 and A100 GPUs) as we did not have access to powerful local machines. The drowback is that in every session we need to re-load the Dataset in the /content/, and this is impractical if we are using the complete RRDataset which size is around 20 GB. Also loading the Dataset in Google Drive is impractical as, during training, most of the time would be lost in I/O passages between Colab and Drive.
Thus, the more convinient solution was:
1) The Creation of a subset of 9000 Images (3.4 GB)
2) The Compression of this subset (this does not reduce the size, but the next steps will be faster).
3) The Saving of this archive in the Local Drive
4) The Download of this archive in the /content/ of Google Colab
5) The Extraction of this archive
6) The Creation of train,validation and test sets

While passage 1-2-3 can be done only once, we need to repeat passage 4-5-6 in every colab session. But downloading and extracting the archive is critically faster than downloading and using the subset folder as is. 

It is important to spend some words about step 1), in particular on the criteria of the balanced data split; I divided the Images in six big families, and took 1500 images for each macro-category:
- original/ai
- original/real
- redigital/ai
- redigital/real
- transfer/ai
- transfer/real

Then, during training, each macro-category of 1500 photos is divided in three sets:
-  train set 70% (in total 6300=6*1050)
-  validation set 15% (in total 1350=6*225)
-  test set 15% (in total 1350=6*225)

This ensures split balance both in real/fake classes and transformation categories.

Steps 1-2-3 are done separately in this [notebook](https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Subset_creator.ipynb)

Another important features is the possibility to choose between **Lazy Loading** and **Eager Loading**:
- With **Lazy Loading** the DataLoader will re-load the image and re-compute the DFT Amplitude every time the image is requested. With A100 GPU, every epoch it loses about 40/50 seconds in this process.
- With **Eager Loading** the DataLoader will load and compute the DFT Amplitude of the image at the beginning and then it will store them in the GPU memory. With A100 GPU the Eager loading lasts 4 minutes, but then we recover 40/50 seconds in each epoch of training. The drowaback is that we need around 17/18 GB of VRAM of the GPU. 

The subset archive can be downloaded [here](https://drive.google.com/file/d/1Y9WJSk2nGYXYGO9T6PcgerI0cK-aGbHn/view?usp=sharing), while the folder of the original RRDataset can be seen [here](https://drive.google.com/drive/folders/1fTFIHXxDNseudhx9EJI-QoA0ZtzvGv8O?usp=sharing)

## Network
As mentioned in the abstract, in this project We propose a new model based on the [DRCTConvNext_Base](#chen2024drct) detector that takes in input not only RGB features, but also the DFT Amplitude features. The name is DRCTConvNext_Base_DFT (abbreviated DRCTConvB_DFT). We compared both models on the two tasks separately and jointly. We found out that this modification improved slightly the classification Real/Ai task, but when combined with the transformation classification, it outperformed the DRCTConvNext_Base. This confirms the hypothesis that the study of the spectrum really helped the network in understanding the transformation category. We will show the results in the **Results** chapter; In this chapter we are going to describe in details the Architecture and the modifications.

The first thing that we need to analyze is the base model detector that the paper [DRCT](#chen2024drct) modified, that is [ConvNext_Base](#convnext). It is a variant of the ConvNeXt family, CNN-ResNet based models that try to achieve the same results of Vision Transformers by adapting some ideas that became popular and common in the ViT. The result is a CNN Model that can compete with Transformers in vision tasks.
<a id="ConvNeXt_architecture"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Base_Model_centered.jpg" alt="BaseConvNeXt" width="200">
  <br>
  <sub>
    <b>Figure 1:</b> ConvNeXt Base architecture.
  </sub>
</p>

Then the author of [DRCT](#chen2024drct) transformed the [ConvNext_Base](#convnext) head in a features extractor and glued a new head that has 2 as output-dimension (the beige block in the image below). Finally they trained it to became a detector of AI Images.

<a id="DRCT_architecture"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/DRCT_Detector_centered.jpg" alt="Model architecture" width="200">
  <br>
  <sub>
    <b>Figure 2:</b> DRCTConvB model detector.
  </sub>
</p>

In the work of [RRDataset](#li2025), the authors found out that this is the best model to be fine-tuned and used in real world detection of AI Images that can be degraded.


And now we come to **DRCTConvB_DFT**: as in this project we want to predict also the post-processing transformation, I added also a STEM that is the copy in size and layers of the STEM in the referred model. But in input we cannot pass only a DFT, as Convolutional Layer are not capable of distinguish in which portion of the image the patch was taken, and then could be impossible to distinguish between low frequencies and high frequencies. For this motivation, we concatenate in the channels a **Normalized Radial Map**, a special images that in each pixel there is the distance from the center, and then the STEM branch can associate the DFT Amplitude portion to a particular frequency. At the end, the feature maps of the DFT branch are summed pixel-wise with the feature maps of the RGB original STEM branch and then they continue together the flow in the network. The new added head takes in input the same embeddings of the Real/Ai Head, but it return 3 logits.

Below there is the complete structure of the proposed model with the two different heads. The block marked with green are the blocks that I added.
<a id="prposed_architecture"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/DFT_model_centered.jpg" alt="Model architecture" width="600">
  <br>
  <sub>
    <b>Figure 3:</b> Proposed multi-head architecture based on DRCTConvB, combining RGB and DFT-based features.
  </sub>
</p>

## Loss function
The loss used for both the two task is the CrossEntropy, i.e.:

$$
CE(y, \hat{y}) = - \log(\hat{y}_{y})
$$
where $ \hat{y}_{y} $ is the probability predicted of the correct class assigned to y by the model 
Despite when trained singularly there is no necessity to normalize to the same scale the loss, while comparing a binary with a ternary loss is necessary a normalization.
In fact, when the model assigns uniformely the sample to all the K classes of the problem, it will produce. 

$$
CE(y, \hat{y}) = - \log(\frac{1}{K}) = \log(K)
$$


Then we obtain that:

$$
CE_{\mathrm{norm}} =\frac{CE(y, \hat{y})}{\log(K)} 
$$

When the model assign the probability uniformily, then $CE_{\mathrm{norm}}=1$

The combined weighted loss for the joint tasks detection is given by:

$$
\mathcal{L} =
\frac{
w_{\mathrm{AI/R}} \cdot CE_{\mathrm{class,norm}}
+
w_{\mathrm{cat}} \cdot CE_{\mathrm{cat,norm}}
}
{
w_{\mathrm{AI/R}} + w_{\mathrm{cat}}
}
$$



## Experiments

**N.B.** I took inspiration and adapted the Train and Test Loop skeletons from one of my previous projects in which I worked and collaborated in: https://github.com/cybernetic-m/DAgger4Robotics

However I did significant changes in order to adapt those skeletons to this work

## Results







## Other stuffs
We introduce a second head which has the role of detecting which type of transformation the image was subjected, and then 
I modified the the best network found in the Benchamrk of [[1]](#li2025) (that is [DRCTConvB](#chen2024drct)) in order to add a Parallel STEM on DFT, and then sum the feature maps of this stem with the feature maps of the RGB stem. The hypothesis is that, for understanding what type of transformation the image is undergone, some of these information can be in the DFT magnitude. And in order to indicate to the Network if the magnitude in the recptive field is a high-frequency or low-fre, we overlapped with a Radial Map.
We are going to see in details:
Creation of the subset
Replication of the experiment in this subset and then seeing what are the results of the DFT


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

<a id="convnext"></a>
[3] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie,
“A ConvNet for the 2020s,” arXiv preprint arXiv:2201.03545, 2022.
[Paper on arXiv](https://arxiv.org/abs/2201.03545)
