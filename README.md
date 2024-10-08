### Diploma-Thesis
# ClusterNet
An augmented version of the Pyramid Scene Parsing Network (PSPNet, Zhao et. al 2017) featuring a spectral clustering layer.

 Spectral clustering is used on the feature maps, right after the ResNet forward pass, to reduce feature space redundancy.
 The spectral clustering algorithm used is as described in Ng, Jordan and Weiss' work "On Spectral Clustering: Analysis and an algorithm" for NIPS 2001.

 Credit also goes to Torchcluster (https://pypi.org/project/torchcluster/) and Zhang Zhi for providing the framework for PyTorch supported clustering algorithms.
 Use **pip install torchcluster** (**conda install -c tczhangzhi torchcluster** for anaconda users) before running the code in this repository.

 The network can be retrained, including the added clustering layer, on the Cityscapes dataset.
 Go to https://www.cityscapes-dataset.com/ and download **leftImg8bit_trainvaltest.zip** and **gtFine_trainvaltest.zip** and
 place the images inside the cityscapes_dataset directory such that the resulting path complies with the data lists in **train_set.txt** and **val_set.txt**

 IMPORANT: Code functionality has been tested for versions: Python = 3.9.12
                                                            PyTorch = 1.11.0
                                                            NumPy = 1.22.3
                                                            OpenCV = 4.5.2
 
The training architecture is depected in the modified image bellow. Credit for the initial image goes to Zhao et. al, "Pyramid Scene Parsing Network" for CVPR 2017 

**Image Placeholder**

Some results from training coupled with corresponding ground truth segmentations

**Images Placeholder**