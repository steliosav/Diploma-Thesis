# Diploma-Thesis
### Semantic Segmentation
The task of **semantic segmentation** refers to the objective of classifying each pixel in a given image, so that it corresponds to a single object class.
The pixels belonging to the same class can then be grouped together, usually by colour coding them, and create a segmentation mask.
For this reason, the task of semantic segmentation is often also refered to as **per-pixel classification**.

Below is an example

#### Real Image
![real_img](/assets/img.png)

#### Semantic Segmentation mask
![ground_truth_seg](/assets/gt_img.png)

A popular model for the task of semantic segmentation and the backbone for this work is the **Pyramid Scene Parsing Network** (PSPNet).

### ClusterNet
An augmented version of the Pyramid Scene Parsing Network (PSPNet, Zhao et. al) featuring a spectral clustering layer.

 Spectral clustering is used on the feature maps, right after the ResNet forward pass, to reduce feature space redundancy.
 The spectral clustering algorithm used is as described in Ng, Jordan and Weiss' work "On Spectral Clustering: Analysis and an algorithm" for NIPS 2001.

 Credit also goes to Torchcluster (https://pypi.org/project/torchcluster/) and Zhang Zhi for providing the framework for PyTorch supported clustering algorithms.
 Use **pip install torchcluster** (**conda install -c tczhangzhi torchcluster** for anaconda users) before running the code in this repository.

 The network can be retrained, including the added clustering layer, on the Cityscapes dataset.
 Go to https://www.cityscapes-dataset.com/, download **leftImg8bit_trainvaltest.zip** and **gtFine_trainvaltest.zip**, unzip them and
 place the **leftImg8bit** and **gtFine** directories inside the **cityscapes_dataset** directory such that the resulting path complies with the data lists in **train_set.txt** and **val_set.txt**
 
### Model
The modified training architecture, including the clustering layer, is depected in the image bellow. 
Credit for the initial image goes to Zhao et. al, "Pyramid Scene Parsing Network" for CVPR 2017 

![clusternet_arch](/assets/clusternet_arch.png)

A sample ground truth segmentation coupled with the model predictions after training

![sample_result](/assets/pretrained_clustering_concat_feat_40clusters.png)

### Notes
#### Credits
This implementation includes code segments provided within the scope of the Computer Vision & Graphics class of CEID University of Patras,
which are used for preprocessing and loading image datasets.

IMPORTANT: Code functionality has been tested for versions:

| Module | Version |
| ------------- | ------------- |
| Python  | 3.9.12  |
| PyTorch  | 1.11.0  |
| NumPy  | 1.22.3  |
| OpenCV  | 4.5.2 |
| SciPy  | 1.7.3  |