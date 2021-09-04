## Discriminative Feature Generation for Classification of Imbalanced Data in PyTorch
<img width="400" alt="Figure1" src="https://user-images.githubusercontent.com/27656658/84164881-b73aba00-aa7b-11ea-9905-bc182d670cde.png">

A Pytorch implementation of Two-Stage Generative Adversarial Networks for Document Image Binarization described in the paper:
* Sungho Suh, Paul Lukowicz, and Yong Oh Lee, "Discriminative feature generation for classification of imbalanced data", Pattern Recognition, 2021. [[Pattern Recognition](https://doi.org/10.1016/j.patcog.2021.108302)] [[arXiv](https://arxiv.org/abs/2010.12888)]

Abstract

The data imbalance problem is frequently bottleneck of the neural network performance in classification. In this paper, we propose a novel supervised discriminative feature generation method (DFG) for minority class dataset. DFG is based on the modified structure of Generative Adversarial Network consisting of four independent networks: generator, discriminator, feature extractor, and classifier. To augment the selected discriminative features of minority class data by adopting attention mechanism, the generator for class-imbalanced target task is trained while feature extractor and classifier are regularized with the pre-trained ones from large source data. The experimental results show that the generator of DFG enhances the augmentation of label-preserved and diverse features, and classification results are significantly improved on the target task.

## Models

The performance of each model

<table>
  <tr align="center">
    <td colspan="2"></td>
    <td colspan="2">LeNet-5 (EMNIST)</td>
    <td colspan="2">VGG-16 (CIFAR10)</td>
    <td colspan="2">ResNet-50 (ImageNet)</td>
  </tr>
  <tr align="center">
    <td colspan="2">Data Set</td>
    <td>SVHN</td>
    <td>F-MNIST</td>
    <td>STL-10</td>
    <td>CINIC-10</td>
    <td>CALTECH-256</td>
    <td>FOOD-101</td>
  </tr>
  <tr align="center">
    <td colspan="2">Imbalance ratio (IR)</td>
    <td>10:1</td>
    <td>40:1</td>
    <td>1:1</td>
    <td>10:1</td>
    <td>1:1</td>
    <td>5:1</td>
  </tr>
  <tr align="center">
    <td colspan="2">ORIGINAL</td>
    <td>76.57 ± 1.65</td>
    <td>77.13 ± 3.24</td>
    <td>66.73 ± 0.67</td>
    <td>58.14 ± 3.42</td>
    <td>43.30 ± 0.34</td>
    <td>30.17 ± 1.72</td>
  </tr>
  <tr align="center">
    <td colspan="2">FINE-TUNING</td>
    <td>76.66 ± 0.65</td>
    <td>75.39 ± 3.54</td>
    <td>72.89 ± 0.38</td>
    <td>64.42 ± 0.81</td>
    <td>81.99 ± 0.12</td>
    <td>68.99 ± 0.42</td>
  </tr>
  <tr align="center">
    <td colspan="2">DELTA</td>
    <td>78.47 ± 0.57</td>
    <td>78.91 ± 2.27</td>
    <td>79.41 ± 0.27</td>
    <td>68.89 ± 0.18</td>
    <td>85.33 ± 0.15</td>
    <td>72.03 ± 0.35</td>
  </tr>
  <tr align="center">
    <td colspan="2">DIFA + CMWGAN</td>
    <td>76.64 ± 1.32</td>
    <td>78.27 ± 2.32</td>
    <td>80.08 ± 0.19</td>
    <td>71.82 ± 0.35</td>
    <td>82.23 ± 0.32</td>
    <td>71.71 ± 0.08</td>
  </tr>
  <tr align="center" style="bold">
    <td colspan="2">OURS(DFG)</td>
    <td>80.81 ± 0.25</td>
    <td>82.65 ± 0.28</td>
    <td>81.09 ± 0.17</td>
    <td>72.09 ± 0.20</td>
    <td>86.29 ± 0.19</td>
    <td>76.00 ± 0.36</td>
  </tr>
</table>

## Prerequisites
- Linux (Ubuntu)
- Python >= 3.6
- NVIDIA GPU + CUDA CuDNN

## Installation


- Clone this repo:
```bash
git clone https://github.com/opensuh/DFG/
```


- Install [PyTorch](http://pytorch.org)
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, you can create a new Conda environment using `conda env create -f environment.yml`.

## DFG train/eval
- Prepare datasets 
  - [SVHN](http://ufldl.stanford.edu/housenumbers)
  - [F-MNIST](https://github.com/zalandoresearch/fashion-mnist)
  - [STL-10](https://cs.stanford.edu/~acoates/stl10)
  - [CINIC-10](https://github.com/BayesWatch/cinic-10)
  - [CALTECH-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/)
  - [FOOD-101](https://www.kaggle.com/dansbecker/food-101/home)


- Train a model:
```bash
./scripts/lenet_svhn_train.sh
./scripts/lenet_fmnist_train.sh
./scripts/vgg_stl_train.sh
./scripts/vgg_cinic_train.sh
./scripts/resnet_caltech_train.sh
./scripts/resnet_food_train.sh
```

- Evaluate the model (our pre-trained models are in ./pretrained_model)
- We plan to upload the pre-trained models on our Github page.
```bash
./scripts/lenet_svhn_eval.sh
./scripts/lenet_fmnist_eval.sh
./scripts/vgg_stl_eval.sh
./scripts/vgg_cinic_eval.sh
./scripts/resnet_caltech_eval.sh
./scripts/resnet_food_eval.sh
```
