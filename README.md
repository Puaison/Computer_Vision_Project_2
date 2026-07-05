# Project 2: Joint Detection of AI-Generated Images and Post-Processing Alterations in Real-World Scenarios
## Abstract:
AI-Generated Image detectors are usually trained and evaluated in idealized conditions. But in real world scenario, images undergo post-processing transformations, which can introduce a degradation signature and artifacts in the image-pattern and lead the detectors in error. In this context it is inserted the work of [Li et al. (2025)](#li2025). The authors introduced a new Benchmark Dataset (called **RRDataset**) containing images under three different format categories: original images, images modified by the transmition through the internet, and images modified by a redigitalization process. In their work, several state-of-the-art AI-detectors were fine-tuned on their Dataset contaning these eterogeneous images, and then they analyzed how the real/fake detection changed across the transformations (w.r.t. the origianl scenario).

In this project, we are going further and extended that setting in two main directions:
- We do not only classify Real/AI Images, but we also identify to which **post-processing  category transformaton** images belong; This is done in a **joint multi-task learning fashion** and this allows us to understand if one task can benefit from the features and model-weights learned from the other task or if they compete, resulting in the degradation of both tasks' performances during joint-training. 
- We adapted the [DRCTConvNext_Base detector](#chen2024drct) by adding a parallel Stem branch which processes the amplitude of the **Discrete Fourier Transform (DFT)**, The motivation relies on the hypothesis that some post-processing transformations can leave detectable traces in the frequency spectrum (and acting like a low-pass or high-pass filter).

The proposed model, called DRCTConvB-DFT, combines both RGB information and DFT informations as a shared backbone. It uses two heads to generate the output: one that takes into account Real/AI detection and the other for category post-processing classification.


## Dataset
All experiments were perfomred on a subset (both balanced in Real/Ai classes and in category transformations) of RRDataset. We trained the models in the Google Colab Environment (in particular on T4 and A100 GPUs) as we did not have access to powerful local machines. The drowback is that at the beginning of every Colab session we need to re-load the Dataset in /content/, and this is impractical if we are using the complete RRDataset which size is around 20 GB. Also using the Dataset from Google Drive is impractical as, during training, most of the time would be lost in I/O passages between Colab and Drive.

For this reason we used the following procedure:
1) The Creation of a subset of **9,000 Images** (**3.4 GB**)
2) The Compression of this subset (this does not reduce the size, but it makes copying faster).
3) The Storing of this archive in the Google Drive
4) The Download of this archive in /content/ of Google Colab
5) The Locally Extraction of this archive in /content/
6) The Creation of the train,validation and test splits

While passage 1-2-3 can be done only once, we need to repeat passage 4-5-6 at the beginning of every Colab session. This is faster than repeatedly downloading and using the subset raw folder as is. 

It is important to spend some words about step 1), in particular on the criteria of the balanced data split; I divided the Images in six big families, and took **1,500 images** for each group:
- original/ai
- original/real
- redigital/ai
- redigital/real
- transfer/ai
- transfer/real

Then each macro-category of 1,500 photos is divided into three splits:
-  train set 70% (in total 6,300=6*1,050)
-  validation set 15% (in total 1,350=6*225)
-  test set 15% (in total 1,350=6*225)

This ensures split balance both in real/fake classes and transformation categories.

The subset creation is implemented in this [notebook](https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Subset_creator.ipynb)

Another important features is the possibility to choose between **Lazy Loading** and **Eager Loading**:
- With **Lazy Loading** the DataLoader will re-load the image and re-compute the DFT Amplitude every time the image is requested. With A100 GPU, in every epoch it loses about 40/50 seconds in this process.
- With **Eager Loading** the DataLoader will load and compute the DFT Amplitude of the image at the beginning of the Colab Session and then it will store them in the GPU memory. With A100 GPU the Eager loading takes about 4 minutes, but then we recover 40/50 seconds in each epoch of training. The drowaback is that it requires around 17-18 GB of GPU memory. 

The subset archive can be downloaded [here](https://drive.google.com/file/d/1Y9WJSk2nGYXYGO9T6PcgerI0cK-aGbHn/view?usp=sharing), while the folder of the original RRDataset can be seen [here](https://drive.google.com/drive/folders/1fTFIHXxDNseudhx9EJI-QoA0ZtzvGv8O?usp=sharing)

## Network architecture

As mentioned in the abstract, in this project we propose a new multi-head model based on the [DRCTConvB](#chen2024drct) detector that takes in input not only RGB features, but also the DFT Amplitude features. The name is **DRCTConvB-DFT**. 

[DRCTConvB](#chen2024drct) (the baseline of this project) is a modification of the [ConvNeXt_Base](#convnext), a Network from the ConvNeXt family. ConvNeXt architectures modernize ResNet-style CNNs by adapting and incorporating some ideas and design choiches that became popular and common in the Vision Transformers, resulting in CNN Models that can compete with Transformers in vision tasks.
<a id="ConvNeXt_architecture"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Base_Model_centered.jpg" alt="BaseConvNeXt" width="200">
  <br>
  <sub>
    <b>Figure 1:</b> ConvNeXt Base architecture.
  </sub>
</p>

The author of [DRCT](#chen2024drct) transformed the [ConvNeXt_Base](#convnext) original head in a features extractor and glued a the end of the network a new binary head (the pink block in the image below) that outputs two logits: one for real images and one for AI-generated images.

<a id="DRCT_architecture"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/DRCT_Detector_centered.jpg" alt="Model architecture" width="200">
  <br>
  <sub>
    <b>Figure 2:</b> DRCTConvB model detector.
  </sub>
</p>

In [RRDataset](#li2025) benchamrk, DRCTConvB was reported as the strongest detector among the considered models when fine-tuned and used in real world detection of AI degraded Images.


Our proposed model extends DRCTConvB by adding a parallel frequency-domain branch.
The main idea is to give to the network access not only to the RGB Image processed by the original ConvNeXt Stem, but also to the DFT amplitude of the Image as it may contain informations about the undergone post-processing transformation.

The DFT Stem has the same structure as the original RGB Stem. But a convolutional layer, by itself, cannot distinguish if a local patch comes from the region of high frequencies or low frequencies. To solve this problem, we concatenate a **Normalized Radial Map** to the DFT amplitude along channel dimension. Each pixel of the Radial Map contains the normalized distance from the center. In this way, the network can associate the DFT Amplitude portions to their corresponding frequency ranges. A the end, the feature maps of the DFT branch are summed pixel-wise with the feature maps of the RGB original Stem branch.

To summaryze the structure and the flow of the new proposed model:
1. The RGB image is analyzed by the original RGB Stem.
2. The DFT amplitude and normalized radial map are analyzed by the DFT Stem.
3. The feature maps outputted by the two Stems are summed elemet-wise.
4. The combined feature maps are passed through the shared DRCTConvB backbone.
5. The final embedding is sent to two different heads:
   - a **real/AI class head** which returns 2 logits;
   - a **category transformation head** which returns 3 logits. 

Below there is the image of the structure of our new proposed model with the two different heads and the parallel DFT Stem. The block marked in green are the blocks that we added.
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

where $\hat{y}_{y}$ is the probability predicted of the correct class assigned to y by the model.
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



## Experiments Protocol

In the sections regarding we are going to report not only performances, but also answers to a bunch of questions about our proposed model with DFT Stem and the interaction between the two tasks.

</br>

**N.B.** I took inspiration and adapted the Train and Test Loop skeletons from one of my previous projects in which I worked and collaborated in: https://github.com/cybernetic-m/DAgger4Robotics

However the code was significantly modified to fit the models and tasks of this work.

### Tasks

We consider two supervised tasks:
| Task name | Description | Number of classes |
|---|---|---:|
| **Class task** | Real vs AI-generated image classification | 2 |
| **Category task** | Post-processing transformation classification | 3 |

### Experimental questions

1. **Does adding the parallel DFT Stem branch improve real/AI detection?**
2. **Does the DFT branch help when the model needs to recognize post-processing transformation categories?**
3. **Does category classification benefit from representations and embeddings learned for real/Ai detection task?**
4. **Does joint multi-head training improve both tasks compared with training them separately?**
5. **How do different combinations of loss weights affect the trade-off between the two tasks?**

### Starting checkpoint used in the experiments
| Checkpoint | Description |
|---|---|
| [DRCT Base Checkpoint](https://drive.google.com/file/d/1LXLXAlsomU5o3AjauINmOlokSvJIGE0q/view?usp=drive_link) | Original DRCTConvB checkpoint before fine-tuning on our Dataset. |
| [RGB-Only Class Task Checkpoint](https://drive.google.com/file/d/18BRyXCF1kSpfi2IGsXEI7j5fRCinoO-s/view?usp=sharing) | DRCTConvB fine-tuned on our Dataset in the class task only. |
| [DFT Class Task Checkpoint](https://drive.google.com/file/d/1kK0usJh56bbYRQF_q6rKHMVO0IYqv4uT/view?usp=drive_link) | DRCTConvB-DFT fine-tuned on our Dataset in the class task only. |

### Experiments roadmap

| ID | Experiment | Model(s) | Initialization | Metrics considered | Purpose |
|---|---|---|---|---|---|
| E1 | Single-head class task | DRCTConvB, DRCTConvB-DFT | DRCT Base Checkpoint | Class accuracy only | Test if DFT helps the real/AI task and if fine-tuning is necessary.|
| E2 | Multi-head fine-tuning from class checkpoints | DRCTConvB, DRCTConvB-DFT | RGB-Only Class Task Checkpoint, DFT Class Task Checkpoint | Category accuracy-F1 only |  Through a Category-head warm-up and then a joint loss, test if DFT features help category detection when the model was previously trained on class task. |
| E3 | Single-head category task | DRCTConvB-DFT | DRCT Base Checkpoint or DFT Class Task Checkpoint | Category accuracy-F1 only | Test if category task benefits from class task pretraining, and if catastrophic unlearning occurs. |
| E4 | Joint training from a common initialization | DRCTConvB-DFT | DRCT Base Checkpoint | Class and Category accuracy-F1 | Compare joint training against the corresponding single-task trainings from the same starting point. |
| E5 | Loss-weight ablation | DRCTConvB-DFT | DRCT Base Checkpoint | Class and Category accuracy-F1 | Using a Joint loss with different $w_{\mathrm{cat}}$ , keeping $w_{\mathrm{AI/R}} = 1$ , we study the trade-off between real/AI detection and category transformation classification. |

### Loss-weight ablation
Before running the final loss-weight ablation, we have done some experiments with differnt learning rates while keeping fixed $ w_{\mathrm{AI/R}} = 1 $ and $ w_{cat} = 1 $ to identify which is the best learning rate for the MultiHead configuration. We have found that the best is **3e-4**, which is then used for the loss-weight ablation.

The table below there reports the weights tested and their nominal relative contribution to the total combined loss (maintaining $ w_{\mathrm{AI/R}} = 1 $ fixed)

<a id="loss_weights"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/loss_weights.png" alt="loss weights" width="600">
</p>

## Results and discussion
This section follows the same order presented in the experimental roadmap. The goal is to give an answer to the raised questions that motivated the experiments.

### E1 -- Single Head Real/Ai Images Classification task
The first experiments evaluates how models detect real/AI images under different post-processing transformations. We compare:
- DRCTConvB with the original DRCT Base Checkpoint without additional fine-tuning;
- DRCTConvB fine-tuned on our Dataset;
- DRCTConvB-DFT (the proposed model) fine-tuned on our Dataset.

The validation and test metrics are reported in [table 2](#table_2). At first glance we can notice that DRCTConvB performs poorly without fine-tuning. This suggets that fine-tuning is necessary for this task.
<a id="table_2"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_2.png" alt="loss weights" width="800">
</p>

In fact, by looking at [table 3](#table_3) the DRCT Base Checkpoint predicts most of the images as "Real", regardless of the post-processing transformation.
<a id="table_3"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_3.png" alt="loss weights" width="800">
</p>

After fine-tuning, both DRCTConvB and DRCTConvB-DFT reach much stronger performances. Moreover we can assert that out proposed model slightly improves Real/AI detection accuracy despite it is less confident due to higher loss.  


By breaking down real/fake detection accuracy separately for each transformation category, the results for RGB-only checkpoint in [table 4](#table_4) and DFT checkpoint in [table 5](#table_5) suggest that both models behave differently w.r.t. post-processing condition, and in particular, they perform better in redigital category (w.r.t the original category) while perform less on the transfer category (in particular our proposed model)

<a id="table_4"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_4.png" alt="loss weights" width="800">
</p>

<a id="table_5"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_5.png" alt="loss weights" width="800">
</p>

To verify if our proposed model (DRCTConvB-DFT) is really using the DFT Stem branch, we performed an ablation at test time: after training, we set the parameters of the parallel DFT branch to zero and we compared how the predictions in the test set changed. Results are reported in [table 6](#table_6) and [table 7](#table_7). It is possible to note that there is a slight improvement in the fake detection across all categories, but it strongly reduce accuracy score in the detection of real images across categories. This suggests that our proposed model is effectively using and learning the DFT Stem.
<a id="table_6"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_6.png" alt="loss weights" width="800">
</p>

<a id="table_7"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_7.png" alt="loss weights" width="800">
</p>

**Multihead fine-tuning of both models starting from our checkpoints of the previous unimodal class training task ([RGB ONLY CLASS TASK CHECKPOINT](https://drive.google.com/file/d/18BRyXCF1kSpfi2IGsXEI7j5fRCinoO-s/view?usp=sharing) and [DFT CLASS TASK CHECKPOINT](https://drive.google.com/file/d/1kK0usJh56bbYRQF_q6rKHMVO0IYqv4uT/view?usp=drive_link)**):
We glued the second head and divided the process of fine tuning in two steps:
1) freezed all the model (for 4 epochs) apart from the new head and used only the category loss in order to train the new head to adapt to the features learned for the classification REal/AI task
2) Unlocked all the parts of the model and trained combined the two heads with $ w_{\mathrm{AI/R}} = 0.1 $ (as being already trained to detect fake images) and $ w_{cat} = 1 $. Moreover we set different learning rates fot the parts of the model. For example for the **class task** head we give a very low learning rate as we don't want to disrupt the learned features.
3) Then compared both DRCT_ConvB e DRCTConvB_DFT with the same experiment parameters and configuration

Brief result: Confirmed that our proposed model has sliglty better perfomances in AI/Real detection, while outperformed in category transformation detection. 

### **Multihead fine-tuning of both models starting from our checkpoints of the previous unimodal class training task ([RGB ONLY CLASS TASK CHECKPOINT](https://drive.google.com/file/d/18BRyXCF1kSpfi2IGsXEI7j5fRCinoO-s/view?usp=sharing) and [DFT CLASS TASK CHECKPOINT](https://drive.google.com/file/d/1kK0usJh56bbYRQF_q6rKHMVO0IYqv4uT/view?usp=drive_link)**)

By looking at the [table 8](#table_8) we can assert that our proposed model is more performing in respect to the DRCTConvB base model (despite it has more uncertainty). In particular it is important to note that thanks to the new DFT Stem branch, our model is hugely particularly more accurated and balanced in category recognition (+18,37 in accuracy and +27,56 in F1-macro) and this confirm our initial hypothesis that it is more simple to detect post-processing transformation by looking also to the spectrum.

<a id="table_8"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_8.png" alt="loss weights" width="800">
</p>

In [table 9](#table_9) are also reported the metrics for Test set, that confirms what we have seen in the validation set.
<a id="table_9"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_9.png" alt="loss weights" width="800">
</p>

From these first two experiments we noted that our proposed model is better than the base DRCT detector, so from now on all the experiment will be done with our proposed model.

**SingleHead Category transformation detection task** (from now we will refer to as **category task**):
- Starting from the [DRCT Base Checkpoint](https://drive.google.com/file/d/1LXLXAlsomU5o3AjauINmOlokSvJIGE0q/view?usp=drive_link) we fine tuned the proposed model on the Subset created from the RRDataset and used only the category detection loss.
- Starting from our personal [DFT CLASS TASK CHECKPOINT](https://drive.google.com/file/d/1kK0usJh56bbYRQF_q6rKHMVO0IYqv4uT/view?usp=drive_link), we fine tuned the proposed model on the Subset created from the RRDataset and used only the category detection loss.
- Compared the results

Brief result: If the model has already learned features for the **class task**, then **category task** benefits from them and reach better result while disrupting the class accuracy. 

**Multihead training of the proposed model starting from [DRCT Base Checkpoint](https://drive.google.com/file/d/1LXLXAlsomU5o3AjauINmOlokSvJIGE0q/view?usp=drive_link)** on the Subset created from the RRDataset and used the combined weighted loss $ w_{\mathrm{AI/R}} = 1 $ and $ w_{cat} = 1 $ with the same configuration and learning rates of the single head tasks. Then compared this joint learning with the unimodal learnings from the same starting checkpoint

Brief results: the joint training improve both the task with respect to two unimodal tasks that starts from the same inital point [DRCTConvB Base Checkpoint](https://drive.google.com/file/d/1LXLXAlsomU5o3AjauINmOlokSvJIGE0q/view?usp=drive_link)

**Multihead training of the proposed model starting from [DRCT Base Checkpoint](https://drive.google.com/file/d/1LXLXAlsomU5o3AjauINmOlokSvJIGE0q/view?usp=drive_link) with different loss weight combinations** and analyzed how the performances changed.

Brief result: We confirmed that the category task benefits and highly depends from the class task and it follows the class task's training velocity despite the weight assigned to it is bigger. And bigger the weight, more at the end bring better result in that category instead of the other task.




## Results
In this section we are going to show the metrics and performances of the five experiments conducted. Our proposed model will be higlited in blue.


### **SingleHead Category transformation detection task**
In [table 10](#table_10) and [table 11](#table_11) are reported the metrics for Validation set and Test set respectively. The first thing we can note is that starting from a checkpoint trained on Real/Fake unimodal task, it can reach better accuracies and security on the category task, althought the process unlearned the class task. This suggests that Category task can effectively benefit from the embeddings learned during the class task
<a id="table_10"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_10.png" alt="loss weights" width="800">
</p>
<a id="table_11"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_11.png" alt="loss weights" width="800">
</p>

### **Multihead training of the proposed model starting from [DRCT Base Checkpoint](https://drive.google.com/file/d/1LXLXAlsomU5o3AjauINmOlokSvJIGE0q/view?usp=drive_link)**
In [table 12](#table_12) and [table 13](#table_13) are reported the metrics for the joint training and the unimodal tranings both for class and category respectively for validation and test set. We can notice very quickly that the multimodal training brings better perfomances in both task, suggesting that both tasks can benefit from each other.
<a id="table_12"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_12.png" alt="loss weights" width="800">
</p>
<a id="table_13"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_13.png" alt="loss weights" width="800">
</p>

By looking to [table 14](#table_14) it seems that the Single-head model is better in finding AI Images, but losses perfomances as seen before in the transfer category. While Multi-head is more accurate on real Images.
<a id="table_14"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_14.png" alt="loss weights" width="800">
</p>

### **Multihead training of the proposed model starting from [DRCT Base Checkpoint](https://drive.google.com/file/d/1LXLXAlsomU5o3AjauINmOlokSvJIGE0q/view?usp=drive_link) with different loss weight combinations**
In [table 15](#table_15) and [table 16](#table_16) are reported the metrics for different weights combination on validation and test sets respectively. There is a very little differences, but we can assert that if the principal task is to detect Ai/Real, use $ w_{\mathrm{AI/R}} = 1 $ and $ w_{cat} = 1 $. If it is necessary something balanced then use $ w_{\mathrm{AI/R}} = 1 $ and $ w_{cat} = 2.7665 $. If the principal task is category detection, then use $ w_{\mathrm{AI/R}} = 1 $ and $ w_{cat} = 5 $. 

<a id="table_15"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_15.png" alt="loss weights" width="800">
</p>
<a id="table_16"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_16.png" alt="loss weights" width="800">
</p>

But what is important to note is in the training process:
By looking at the [training graphs](#graphs), regardless of the weight assigned to the Category term, it follows the velocity learning of the Class term. In fact, when the weight contribution regarding the class is very low, the category loss reached the amount of 0.20 only in the 6-th epoch (altough it had a very big weight), while in the other weightening configuration reached that value on the epochs before (even if the weight was really low). This confirms that the category task is "leeching" from the class task.

<a id="graphs"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Graphs.png" alt="loss weights" width="1600">
</p>



In [table 17](#table_17), we can notice that there is a general trend between the different configuration of weightings that in redigital and transfer category it is more powerful in detecting AI images but less powerful in detecting Real Images. 
<a id="table_17"></a>
<p align="center">
  <img src="https://github.com/Puaison/Computer_Vision_Project_2/blob/main/Images_GitHub/Tables/table_17.png" alt="loss weights" width="800">
</p>

## Conclusion
In this project we have seen that:
- Our proposed model that is DRCTConvB with a parallel DFT Stem is generally better than the base DRCTConvB in the classification of AI/Real Images.
- Jointly combining the two tasks can achieve even better results in detection and post-processing classification, but the most important thing is that category transformation classification benefits from the other task (it is like a leech)
In fact our proposed model overperformed the overall accuracy reported in the work of [RRDatatset](#li2025) even using 1/6 of the original dataset.

## Future works
It would interesting to replicate the fifth experiment with the whole [RRDatatset](#li2025) and using the same training configuration used by the authors of the Dataset to see what performances the proposed model can reach.


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
