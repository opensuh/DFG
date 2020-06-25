import numpy as np
import sys
import os
import utils
import time
import csv
import argparse
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, datasets
import torch.autograd as autograd
from torch.autograd import Variable

from model import *
from custom_data_loader import CinicDataset, Food101Dataset, SvhnDataset, FashionMnistDataset


class EvalOps(object):

    def __init__(self, gpu_num, base_model_name, batch_size, data_set_name):
        self.device = torch.device("cuda:%d" % gpu_num)
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.data_set_name = data_set_name

        if base_model_name == 'vgg':
            self.feature_dimension = 64
            self.resize_size = 32
            self.crop_size = 32
        elif base_model_name == 'resnet':
            self.feature_dimension = 256
            self.resize_size = 256
            self.crop_size = 224

        tmp = str(time.time())
        self.curtime = tmp[:9]
        self.code_start_time = time.time()

    def load_data_set(self, data_set_name, batch_size):
        if data_set_name == 'caltech': # resnet
            self.num_labels = 257
            transform_train = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),            
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    
            ])
            transform_test = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            image_datasets = datasets.ImageFolder('/data/open_dataset/caltech256/caltech_256_train_60/', transform_train)
            test_image_datasets = datasets.ImageFolder('/data/open_dataset/caltech256/caltech_256_test_20/', transform_test)
            return torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True), \
                    torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=True)

        elif data_set_name == 'stl': # vgg
            self.num_labels = 10
            transform_train = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.RandomCrop(self.crop_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),            
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    
            ])
            transform_test = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            image_datasets = datasets.ImageFolder('/data/open_dataset/stl10/stl_train', transform=transform_train)
            test_image_datasets = datasets.ImageFolder('/data/open_dataset/stl10/stl_test', transform=transform_test)
            return torch.utils.data.DataLoader(image_datasets, batch_size=batch_size, shuffle=True), \
                    torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=True)

        elif data_set_name == 'cinic': # vgg
            self.num_labels = 10
            transform_train = transforms.Compose([
                transforms.RandomCrop(self.crop_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            trainset = CinicDataset(root='/data/open_dataset/cinic10/train', transform=transform_train, minor_class_num=0, ratio=1.0)
            testset = CinicDataset(root='/data/open_dataset/cinic10/test', transform=transform_test, minor_class_num=0, ratio=1.0)
            return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), \
                    torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

        elif data_set_name == 'food': # resnet
            self.num_labels = 101
            transform_train = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.RandomCrop(self.crop_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),            
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),    
            ])
            transform_test = transforms.Compose([
                transforms.Resize((self.resize_size, self.resize_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            self.trainset = Food101Dataset(root='/data/open_dataset/food-101/resized_train/', transform=transform_train, minor_class_num=0, ratio=1.0)
            test_image_datasets = Food101Dataset(root='/data/open_dataset/food-101/resized_test/', transform=transform_test, minor_class_num=0, ratio=1.0)
            return torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True), \
                    torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=True)

        elif data_set_name == 'svhn': # lenet
            self.num_labels = 10
            transform_train = transforms.Compose([
                transforms.ToTensor(),            
                transforms.Normalize((0.5), (0.5)),    
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ])
            trainset = SvhnDataset(root='/data/open_dataset/svhn/', transform=transform_train, split='train',
                                        major_len=5000, minor_class_num=0, ratio=1.0)
            test_image_datasets = SvhnDataset(root='/data/open_dataset/svhn/', transform=transform_test, split='test',
                                        minor_class_num=0, ratio=1.0)
            return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), \
                    torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=False)

        elif data_set_name == 'fashion_mnist': # lenet
            self.num_labels = 10
            transform_train = transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),            
                transforms.Normalize((0.5), (0.5)),    
            ])
            transform_test = transforms.Compose([
                transforms.Pad(2),
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5)),
            ])
            trainset = FashionMnistDataset(root='/data/open_dataset/fashion_mnist/', transform=transform_train, train=True,
                                                minor_class_num=0, ratio=1.0)
            test_image_datasets = FashionMnistDataset(root='/data/open_dataset/fashion_mnist/', transform=transform_test, train=False,
                                                minor_class_num=0, ratio=1.0)
            return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), \
                    torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=True)
        else :
            raise ValueError('Unknown data set')

    def evaluate(self, extractor_weight_path, classifier_weight_path):
        _, self.data_loader_test = self.load_data_set(self.data_set_name, batch_size=self.batch_size)
        print('data set name: %s' % (self.data_set_name))
        print('test data loader len: %d' % ( len(self.data_loader_test) ) )

        self.feature_extractor = Feature_Extractor(base_model_name=self.base_model_name, pretrained_weight=None, num_classes=self.num_labels)
        self.feature_extractor.load_state_dict(torch.load(extractor_weight_path))
        self.feature_extractor.eval()
        self.feature_extractor.to(self.device)

        self.feature_classifier = Feature_Classifier(base_model_name=self.base_model_name, pretrained_weight=None, num_classes=self.num_labels)
        self.feature_classifier.load_state_dict(torch.load(classifier_weight_path))
        self.feature_classifier.eval()
        self.feature_classifier.to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        loss = 0.0
        step_acc = 0.0
        total_inputs_len = 0
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(self.data_loader_test):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                extracted_feature = self.feature_extractor(inputs)
                logits_real = self.feature_classifier(extracted_feature)
                _, preds = torch.max(logits_real, 1)
                loss += self.criterion(logits_real, labels).item() * inputs.size(0)

                corr_sum = torch.sum(preds == labels.data)
                step_acc += corr_sum.double()
                total_inputs_len += inputs.size(0)
                if idx % 20 == 0:
                    print('%d / %d' % (idx, len(self.data_loader_test)))

        loss /= total_inputs_len
        step_acc /= total_inputs_len

        print ('Test loss: [%.6f] accuracy: [%.4f]' % (loss, step_acc))
        print("time: %.3f" % (time.time() - self.code_start_time))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help=" 0, 1, 2 or 3 ")

    parser.add_argument("--source_dataset", type=str, default='imagenet',
                    help=" 'emnist', 'imagenet', 'cifar' ")

    parser.add_argument("--target_dataset", type=str, default='caltech',
                    help=" 'svhn', 'fashion mnist', 'caltech', 'cinic', 'stl', 'food' ")

    parser.add_argument("--base_model_name", type=str, default='resnet', help=" 'lenet or vgg or resnet' ")

    parser.add_argument("--batch_size", type=int, default=64, help='batch_size')

    # weight path
    parser.add_argument("--extractor_weight_path", type=str, default='./model/feature_extractor_target_resnet_caltech.pth', help='extractor weight path')
    parser.add_argument("--classifier_weight_path", type=str, default='./model/feature_classifier_target_resnet_caltech.pth', help='classifier weight path')

    parser.add_argument("--pretrained_base_model", type=str, default='', help='pretrained_base_model')

    # './checkpoint/vgg16_cifar10_source.pth'
    args = parser.parse_args()

    test = EvalOps(gpu_num=args.gpu, base_model_name=args.base_model_name, batch_size=args.batch_size, data_set_name=args.target_dataset)

    test.evaluate(args.extractor_weight_path, args.classifier_weight_path)