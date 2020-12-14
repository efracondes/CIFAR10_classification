# CIFAR10 Classifier

This project has the purpose of showing how to train a CNN using PyTorch, saving a training routine that can be extended for future applications and providing a method to check if PyTorch is making use of the GPUs correctly. With this goal in mind, we try to solve the image classification task in the CIFAR10 dataset, which consists of 60,000 32x32 color images classified within ten classes. The classes represent objects such as airplanes, automobiles, birds, cats, etc.

We present two experiments:

* Custom Model: We build and train a model from scratch, defining each layer of the architecture and training all of its weights.
* Pretrained Model: We use transfer learning to implement a ResNet18 based model, using its weights pre-trained on ImageNet and training only the classification layer.
