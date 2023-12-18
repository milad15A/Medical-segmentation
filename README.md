<div align="center">
  <a href="https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation">
    <img src="cover.png" alt="Logo" width="1000" height="300">
  </a>
  <h1 align="center">Medical Image Segmentation</h1>
</div>

This repository has been created to delve into the realm of AI within the field of medicine. Our goal is to devise and execute a straightforward but precise initiative in this domain. I express my gratitude to Howsam Academy and Dr. Seyed Sajad Ashrafi for their assistance during challenging situations.

## 1. Problem Statement

This segment focuses on a prevalent form of cancer in the population and an effective deep-learning approach to aid in its treatment. However, before delving into that, we aim to provide a concise definition of semantic segmentation and its related concepts.
  <br/>1-Image segmentation: Image segmentation involves dividing an image into meaningful segments or regions based on certain characteristics, such as color, intensity, texture, or boundaries.
  <br/>2- Semantic segmentation: the algorithm attempts to label every single pixel.
  <br/> In medicine, segmenting can out the image exactly which pixels correspond to certain parts of the patient's anatomy

In summary, medical image segmentation is a critical component of modern healthcare, offering improvements in diagnosis, treatment planning, research, and education. 

  ### Main objective:

To perform segmentation on the stomach and intestines, aiming to support oncologists in the precise administration of X-ray treatments. the ultimate goal is to reduce patient's pain levels and enhance the overall effectiveness of medical interventions

## 2. Related Works
this section presents both common and innovative methods and architectures, aiming to provide readers with a comprehensive understanding of the subject. 

| Number | Architecture | Short Description | Link |
|--------|--------------|-------------------|------|
| 1   |   OneFormer    | new panoptic architectures used the same architecture to achieve top performance across diﬀerent tasks. a unique multi-task universal architecture with a task-conditioned joint training strategy that sets new state-of-the-art across semantic | [Link](https://paperswithcode.com/paper/oneformer-one-transformer-to-rule-universal) |
| 2      |   UNet++    | A deeply-supervised encoder-decoder network where the encoder and decoder sub-networks are connected through a series of nested, dense skip pathways. The re-designed skip pathways aim at reducing the semantic gap between the feature maps of the encoder and decoder sub-networks  |[Link](https://paperswithcode.com/paper/unet-a-nested-u-net-architecture-for-medical) |
| 3     |  Deeplabv3     |designed modules that employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates  |[Link](https://paperswithcode.com/method/deeplabv3) |
|4       |    BiSeNet          |designed a Spatial Path with a small stride to preserve the spatial information and generate high-resolution features. Meanwhile, a Context Path with a fast downsampling strategy is employed to obtain a sufficient receptive field. | [Link](https://paperswithcode.com/paper/bisenet-bilateral-segmentation-network-for)  |
| 5      |  U-Net      |The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization  |[Link](https://paperswithcode.com/paper/u-net-convolutional-networks-for-biomedical) |

 

## 3. The Proposed Method
Here, the proposed approach for solving the problem is detailed. It covers the algorithms, techniques, or deep learning models to be applied, explaining how they address the problem and why they were chosen.

## 4. Implementation
This section delves into the practical aspects of the project's implementation.

### 4.1. Dataset
Under this subsection, you'll find information about the dataset used for the medical image segmentation task. It includes details about the dataset source, size, composition, preprocessing, and loading applied to it.
[Dataset](https://drive.google.com/file/d/1-2ggesSU3agSBKpH-9siKyyCYfbo3Ixm/view?usp=sharing)

### 4.2. Model
In this subsection, the architecture and specifics of the deep learning model employed for the segmentation task are presented. It describes the model's layers, components, libraries, and any modifications made to it.

### 4.3. Configurations
This part outlines the configuration settings used for training and evaluation. It includes information on hyperparameters, optimization algorithms, loss function, metric, and any other settings that are crucial to the model's performance.

### 4.4. Train
Here, you'll find instructions and code related to the training of the segmentation model. This section covers the process of training the model on the provided dataset.
