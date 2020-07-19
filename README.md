# Residual Attention Network for Image Classification
###### Author: Jie (Jay) Li, Xiaofan (Frances) Zhang, Zhaoyang Wang

##  Introduction
This repo contains our re-implementation of Residual Attention Network based on the paper [Residual Attention Network for Image Classification](https://arxiv.org/pdf/1704.06904.pdf).

Convolutional Neural Network (CNN) is the most popular neural netowrk model being used for image classification problem which can help break images and extracts the high-level features. Rather than compress an entire image into a static representation, the Attention Module allows for salient features to dynamically come to the forefront as needed.

The Residual Attention Network, a convolutional neural network adopts mixed attention mechanism into very deep structure for image classification task. The Residual Attention Network can incorporate with state-of-art feed forward network architecture, and is built by stacking Attention Modules, which generate attention-aware features from low resolution and mapping back to orginal feature maps.

In our task, we propose a new architecture of Residual Attention Network different from original paper for image classification. First, we follow the original paper to re-construct and train Residual Attention Network on CIFAR-10 dataset. Then, we experiment different architectures and optimizers to improve our network. Lastly, we visualize the model performance in training, validation error and training time by Tensorbaord to compare our model with the original model from paper. In the end, we conduct all the detail and results in the papers: [Residual Attention Neural Network](https://github.com/Jay4869/Deep-Learning/blob/master/Residual%20Attention%20Neural%20Network.pdf).

## Dataset
In our task, We are using **CIFAR-10** and **CIFAR-100** which consist of 50,000 training set and 10,000 test set with 32 x 32 RGB images, representing 10/100 different image labels. We apply the data augmentation technique that generate image rotation, shifting, and horizontal flip, with the per-pixel RGB mean value subtracted.

## Results
We show the Residual Attention Network performance drops by 3.75% on CIFAR-10 dataset using Naive Residual Learning (**NAL**), compared with Attention Residual Learning (**ARL**) as table at below:

| Network     | ARL (error) | NAL (error)   |
| ----------- | ----------- | ------------- |
| Attention-56 | 5.28% | 9.03% |
| Attention-92 | 36.2% | 40.05% |

Also, we evaluate the Attention-56 and Attention-92 Networks on CIFAR-10 by validation error, test error and training time. For the small images (32 x 32), Attention-56 works very well and achieves 3.27% train error and 5.28% validation error spending 337 minutes on training. Attention-92, stacking more Attention Modules only ends up at 66% accuracy.

| Network     | Train Error |  Val Error  | Train Time  |
| ----------- | ----------- | ------------- | ------------- |
| Attention-56 | 3.27% | 5.28% | 337 min |
| Attention-92 | 36.1% | 33.19% | 403 min |

## Network Structure
| Layer       | Output Size | Detail        |
| ----------- | ----------- | ------------- |
| Conv2D | 32x32x32 | 5x5, stride=1 |
| Max pooling | 16x16x32 | 2x2, stride=2 |
| Residual Unit | 16x16x128    |  |
| Attention | 16x16x128 |  |
| Residual Unit | 8x8x256  |  |
| Attention | 8x8x256 | x1 |
| Residual Unit | 4x4x1024 | x3 |
| AvgPooling2D | 1x1x1024 | pool 4, stride=1 |
| Flatten | 1x1x1024 |  |
| Dropout | 1x1x1024 |  |
| Dense | 10 | L2 Norm, softmax |

## Dependencies
* Python 3.6+
* Tensorflow-gpu 2.0
* Tensorboard 2.0.2
* Google Cloud (NVIDIA Tesla P100)
* Reference: https://ecbm4040.bitbucket.io/2019_fall/EnvSetup/gcp.html

## Usage
### Quick Training Attention Model

Execute `training_cifar10.py`. It's training ResNet56 with Attention by default, and provides the model performance such as validation/testing accuracy, validation/testing loss, and runing time. Also, the model log will be stored into `Logs/` folder, named by excuation time. 

### Customize Training Attention Model

`Module/`: contains the core Modules scripts: Residual Unit and Attention Block

`Model/`: contains the Residual Attention Network structure: 56/92 layers

Jupyter notebook version is provided to you that contain the details of our development, and help you reporoduce the results above. `training_56.ipynb` and `training_92.ipynb` generate all work for Attention-56 and Attention-92 respectively.

### Model Performance Dashboard

Tensorboard is a powerful tool provided by TensorFlow. It allows to check their graph and trend of parameters, as well as the model performance. To start your Tensorboard, you need model logs file that generate by `Tensorflow callbacks`

Example: `$ tensorboard --logdir= 'Logs'`

You will see as following

`TensorBoard 2.0.2 at http://localhost:6006/ (Press CTRL+C to quit)`

Make sure Tensorboard is running, you can visit [http://localhost:6006](http://localhost:6006)







