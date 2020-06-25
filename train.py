import numpy as np
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
import torch.optim as optim
from torchvision import transforms, datasets
import torch.autograd as autograd
from torch.autograd import Variable

from model import *
from custom_data_loader import CinicDataset, Food101Dataset, SvhnDataset, FashionMnistDataset

class TrainOps(object):

    def __init__(self, gpu_num, base_model_name, batch_size, data_set_name, extractor_learning_rate, classifier_learning_rate, discriminator_learning_rate,
                    generator_learning_rate, minor_class_num, minor_class_ratio, with_regularization, model_save_path):
        self.device = torch.device("cuda:%d" % gpu_num)
        self.base_model_name = base_model_name
        self.batch_size = batch_size
        self.data_set_name = data_set_name

        #Learning rate
        self.extractor_learning_rate = extractor_learning_rate
        self.classifier_learning_rate = classifier_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_learning_rate = generator_learning_rate

        if base_model_name == 'vgg':
            self.feature_dimension = 64
            self.resize_size = 32
            self.crop_size = 32
            self.noise_dim = 100
        elif base_model_name == 'resnet':
            self.feature_dimension = 256
            self.resize_size = 256
            self.crop_size = 224
            self.noise_dim = 4900
        elif base_model_name == 'lenet':
            self.feature_dimension = 6
            self.resize_size = 32
            self.crop_size = 32
            self.noise_dim = 100
        else:
            raise ValueError('Unknown base model')

        # for hook layer output
        self.layer_outputs_source = []
        self.layer_outputs_target = []

        self.minor_class_num = minor_class_num
        self.minor_class_ratio = minor_class_ratio

        self.with_regularization = with_regularization

        tmp = str(time.time())
        self.curtime = tmp[:9]

        self.model_save_path = model_save_path + self.base_model_name + '_' + self.data_set_name
        if not os.path.isdir(self.model_save_path):
            os.makedirs(self.model_save_path)

        self.csv_save_path = self.model_save_path + '/csv/'
        if not os.path.isdir(self.csv_save_path):
            os.makedirs(self.csv_save_path)

        self.generator_attention_class = self.csv_save_path + 'generator_attention_class_' + self.base_model_name + '_' + self.data_set_name + '_' + self.curtime + '.csv'
        self.wj_extractor_file = self.csv_save_path + 'wj_extractor_' + self.base_model_name + '_' + self.data_set_name + '_' + self.curtime + '.csv'
        self.generator_attention_file = self.csv_save_path + 'generator_attention_' + self.base_model_name + '_' + self.data_set_name + '_' + self.curtime + '.csv'

        self.cal_weight_path = self.model_save_path + '/config/'
        if not os.path.isdir(self.cal_weight_path):
            os.makedirs(self.cal_weight_path)

        self.transpose_wj_extractor_npy = self.cal_weight_path + 'transpose_wj_extractor_%s_%s.npy' % (self.base_model_name, self.data_set_name)
        self.generator_attention_npy = self.cal_weight_path + 'generator_attention_%s_%s.npy' % (self.base_model_name, self.data_set_name)
        self.channel_weight_json = self.cal_weight_path + 'channel_weight_%s_%s.json' % (self.base_model_name, self.data_set_name)

        self.optimal_attention_npy = self.model_save_path + '/optimal_attention_%s_%s_%s' % (self.base_model_name, self.data_set_name, self.curtime)
        self.feature_extractor_target_pth = self.model_save_path + '/feature_extractor_target_%s_%s_%s.pth' % (self.base_model_name, self.data_set_name, self.curtime)
        self.feature_classifier_target_pth = self.model_save_path + '/feature_classifier_target_%s_%s_%s.pth' % (self.base_model_name, self.data_set_name, self.curtime)
        self.feature_generator_pth = self.model_save_path + '/feature_generator_%s_%s_%s.pth' % (self.base_model_name, self.data_set_name, self.curtime)
        self.feature_discriminator_pth = self.model_save_path + '/feature_discriminator_%s_%s_%s.pth' % (self.base_model_name, self.data_set_name, self.curtime)
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
                    torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=False)

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
                    torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=False)

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
            trainset = CinicDataset(root='/data/open_dataset/cinic10/train', transform=transform_train, minor_class_num=self.minor_class_num, ratio=self.minor_class_ratio)
            testset = CinicDataset(root='/data/open_dataset/cinic10/test', transform=transform_test, minor_class_num=0, ratio=1.)
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
            trainset = Food101Dataset(root='/data/open_dataset/food-101/resized_train/', transform=transform_train, minor_class_num=self.minor_class_num, ratio=self.minor_class_ratio)
            test_image_datasets = Food101Dataset(root='/data/open_dataset/food-101/resized_test/', transform=transform_test, minor_class_num=0, ratio=1.0)
            return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), \
                    torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=False)

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
                                        major_len=5000, minor_class_num=self.minor_class_num, ratio=self.minor_class_ratio)
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
                                                minor_class_num=self.minor_class_num, ratio=self.minor_class_ratio)
            test_image_datasets = FashionMnistDataset(root='/data/open_dataset/fashion_mnist/', transform=transform_test, train=False,
                                                minor_class_num=0, ratio=1.0)
            return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True), \
                    torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size, shuffle=False)
        else :
            raise ValueError('Unknown data set')

    def calculate_weighting_feature_maps_extractor(self, extractor_model, classifier_model, layer_name, label_min=10):
        print('weight data loader len : %d' % len(self.data_loader))
        criterion = nn.CrossEntropyLoss(reduction='none')

        channel = extractor_model.state_dict()[layer_name + '.weight'].shape[0]
        print('channel number : %d' % channel)

        labels_cnt = [0 for i in range(self.num_labels)]
        labels_min = label_min

        # calculate base, jth loss
        total_start_time = time.time()
        base_loss = []
        jthfilter_loss_list = [[] for i in range(channel)]
        class_label_list = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.data_loader):
                step_start_time = time.time()
                class_inputs = []
                class_labels = []
                for batch_idx in range(inputs.size(0)):
                    if labels_cnt[labels[batch_idx].item()] >= labels_min:
                        continue
                    
                    labels_cnt[labels[batch_idx].item()] += 1
                    class_inputs.append(inputs[batch_idx])
                    class_labels.append(labels[batch_idx])

                if len(class_inputs) == 0:
                    if min(labels_cnt) < labels_min:
                        continue
                    else:
                        break
                
                class_label_list.extend(class_labels)
                class_inputs = torch.stack(class_inputs)
                class_labels = torch.stack(class_labels)
                
                class_inputs = class_inputs.to(self.device)
                class_labels = class_labels.to(self.device)

                feature_outputs = extractor_model(class_inputs)
                classifier_outputs = classifier_model(feature_outputs)
                base_loss.extend(criterion(classifier_outputs, class_labels).cpu())
                self.layer_outputs_source.clear()
                self.layer_outputs_target.clear()

                for j in range(channel):
                    j_tmp_weight = extractor_model.state_dict()[layer_name + '.weight'][j,:,:,:].clone()
                    extractor_model.state_dict()[layer_name + '.weight'][j,:,:,:] = 0

                    feature_outputs = extractor_model(class_inputs)
                    classifier_outputs = classifier_model(feature_outputs)
                    jthfilter_loss_list[j].extend(criterion(classifier_outputs, class_labels).cpu())

                    extractor_model.state_dict()[layer_name + '.weight'][j,:,:,:] = j_tmp_weight
                    self.layer_outputs_source.clear()
                    self.layer_outputs_target.clear()
                
                print('%d step loss len : %d, time : %.5f' % (i, len(base_loss), time.time() - step_start_time))

        print('total loss len : %d, class label len : %d, total time : %.5f' % (len(base_loss), len(class_label_list), time.time() - total_start_time))
        # memory clear
        self.layer_outputs_source.clear()
        self.layer_outputs_target.clear()
        return base_loss, jthfilter_loss_list, class_label_list

    def calculate_weighting_feature_maps_classifier(self, extractor_model, classifier_model, layer_name):
        print('weight data loader len : %d' % len(self.data_loader))

        filter_weight = []
        for i in range(len(layer_name)):
            channel = classifier_model.state_dict()[layer_name[i] + '.weight'].shape[0]
            layer_filter_weight = [0] * channel
            filter_weight.append(layer_filter_weight)         
            
        criterion = nn.CrossEntropyLoss()

        since = time.time()
        for i, (inputs, labels) in enumerate(self.data_loader):
            if i >= 4:
                break
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = extractor_model(inputs)
            outputs = classifier_model(outputs)
            loss = criterion(outputs, labels)

            for name, module in classifier_model.named_modules():
                if not name in layer_name:
                    continue
                layer_id = layer_name.index(name)
                channel = classifier_model.state_dict()[name + '.weight'].shape[0]
                for j in range(channel):
                    tmp = classifier_model.state_dict()[name + '.weight'][j,:,:,:].clone()
                    classifier_model.state_dict()[name + '.weight'][j,:,:,:] = 0
                    outputs = extractor_model(inputs)
                    outputs = classifier_model(outputs)
                    loss1 = criterion(outputs, labels)
                    diff = loss1 - loss
                    diff = diff.detach().cpu().numpy().item()
                    hist = filter_weight[layer_id][j]
                    filter_weight[layer_id][j] = 1.0 * (i * hist + diff) / (i + 1)
                    print('%s:%d %.4f %.4f' % (name, j, diff, filter_weight[layer_id][j]))
                    classifier_model.state_dict()[name + '.weight'][j,:,:,:] = tmp
                    
            print('step %d finished' % i)
            time_elapsed = time.time() - since
            print('step Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
        
        json.dump(filter_weight, open(self.channel_weight_json, 'w'))

    def for_hook_source(self, module, input, output):
        self.layer_outputs_source.append(output)

    def for_hook_target(self, module, input, output):
        self.layer_outputs_target.append(output)

    def register_hook(self, model, func, layer_name):
        for name, layer in model.named_modules():
            if name in layer_name:
                layer.register_forward_hook(func)

    def train_fc(self, feature_extractor, model):
        
        for name, param in model.named_parameters():
            if not name.startswith('classifier.'):
                param.requires_grad = False
            else:
                print(name)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                            lr = 0.01, momentum=0.9, weight_decay=1e-4)
        num_epochs = 10
        decay_epochs = 6 
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = math.exp(math.log(0.1) / decay_epochs))
        since = time.time()
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            model.train()
            running_loss = 0.0
            running_corrects = 0.0
            total = 0.0
            nstep = len(self.data_loader)
            for i, (inputs, labels) in enumerate(self.data_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                features = feature_extractor(inputs)
                outputs = model(features)

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer.step()

                if i % 10 == 0:
                    corr_sum = torch.sum(preds == labels.data)
                    step_acc = corr_sum.double() / len(labels)
                    print('step: %d/%d, loss = %.4f, top1 = %.4f' %(i, nstep, loss, step_acc))
                    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total += inputs.size(0)
                        
            scheduler.step()
            epoch_loss = running_loss / total
            epoch_acc = running_corrects.double() / total

            print('epoch: {:d} Loss: {:.4f} Acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc))

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        
        return model

    def channel_evaluation(self, pretrained_base_model=None):
        print ('get weighting_feature_map')
        
        print('loading %s data set' % (self.data_set_name))
        self.data_loader, self.data_loader_test = self.load_data_set(self.data_set_name, batch_size=self.batch_size)
        print('data loader len: %d' % ( len(self.data_loader) ) )

        # set feature extractor
        print(pretrained_base_model)
        self.feature_extractor = Feature_Extractor(base_model_name=self.base_model_name, pretrained_weight=pretrained_base_model, num_classes=self.num_labels)
        self.feature_extractor.to(self.device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        
        # set extractor hook for feature map
        if self.base_model_name == 'vgg':
            self.layer_name_extractor = 'model.2' # (64 in, 64 out)
        elif self.base_model_name == 'resnet':
            self.layer_name_extractor = 'model.4.2.conv3' # (64 in, 256 out)
        elif self.base_model_name == 'lenet':
            self.layer_name_extractor = 'model.0' # (6 in, 16 out)

        # set feature classifier
        self.feature_classifier = Feature_Classifier(base_model_name=self.base_model_name, pretrained_weight=pretrained_base_model, num_classes=self.num_labels)
        self.feature_classifier.to(self.device)

        if self.base_model_name == 'resnet' or (self.base_model_name == 'vgg' and pretrained_base_model is None):
            self.feature_classifer = self.train_fc(self.feature_extractor, self.feature_classifier)

        for param in self.feature_classifier.parameters():
            param.requires_grad = False
        self.feature_classifier.eval()
        
        # calculate weighting source extractor feature maps
        base_loss_extractor, jthfilter_loss_extractor, class_label_list = self.calculate_weighting_feature_maps_extractor(self.feature_extractor, 
                                                                            self.feature_classifier, layer_name=self.layer_name_extractor, label_min=10)
        extractor_number_filter = len(jthfilter_loss_extractor)

        wj_extractor = np.zeros((extractor_number_filter, len(base_loss_extractor)))
        base_loss_extractor = np.array(base_loss_extractor)
        print(wj_extractor.shape, base_loss_extractor.shape)

        for fidx in range(extractor_number_filter):
            wj_extractor[fidx] = base_loss_extractor - np.array(jthfilter_loss_extractor[fidx])
        
        transpose_wj_extractor = np.transpose(wj_extractor)
        print(transpose_wj_extractor.shape)
        np.save(self.transpose_wj_extractor_npy, transpose_wj_extractor.mean(axis=0))
        
        with open(self.wj_extractor_file, 'w') as wj_list:
            writer = csv.writer(wj_list, delimiter=',', quoting=csv.QUOTE_ALL)
            for t in range(transpose_wj_extractor.shape[0]):
                def softmax_loss(loss) : 
                    max_loss = np.max(loss) 
                    exp_loss = np.exp(loss-max_loss) 
                    sum_exp_loss = np.sum(exp_loss)
                    result = exp_loss / sum_exp_loss
                    return result

                transpose_wj_extractor[t] = softmax_loss(transpose_wj_extractor[t])
                writer.writerow(list(transpose_wj_extractor[t]))

        # initialize attention for feature generator      
        self.generator_attention = np.zeros((self.num_labels, self.feature_dimension))
        labels_count  = np.zeros(self.num_labels)
        iCnt=0
        for class_num in class_label_list:
            class_num = class_num.item()
            self.generator_attention[class_num, :] = 1.0 * (self.generator_attention[class_num, :] * labels_count[class_num] + transpose_wj_extractor[iCnt, :]) / (labels_count[class_num]+1)
            labels_count[class_num] += 1
            iCnt+=1

        np.save(self.generator_attention_npy, self.generator_attention)

        # save_generator_attention
        for filter_idx in range(self.generator_attention.shape[1]):
            filename = self.csv_save_path  + "/"+ "generator_attention_" + str(filter_idx) + "_" + self.curtime + ".csv"
            with open(filename, 'a') as generator_attention_save:
                writer = csv.writer(generator_attention_save, delimiter=',', quoting=csv.QUOTE_ALL)
                writer.writerow(list(self.generator_attention[:,filter_idx]))

        # set classifier hook for feature map
        if self.base_model_name == 'vgg':
            self.layer_name_classifier = ['feature.23'] # (512 in, 512 out)
        elif self.base_model_name == 'resnet':
            self.layer_name_classifier = ['feature.0.3.conv3', 'feature.1.5.conv3', 'feature.2.2.conv3'] # (512 in, 2048 out)
        elif self.base_model_name == 'lenet':
            self.layer_name_classifier = ['feature.0'] # (6 in, 16 out)

        # calculate weighting source classifier feature maps
        self.calculate_weighting_feature_maps_classifier(self.feature_extractor, 
                                                            self.feature_classifier, layer_name=self.layer_name_classifier)

    def flatten_outputs(self, fea):
        return torch.reshape(fea, (fea.shape[0], fea.shape[1], fea.shape[2] * fea.shape[3]))
    
    def extractor_att_fea_map(self, fm_src, fm_tgt):
        fea_loss = torch.tensor(0.).to(self.device)
        
        b, c, h, w = fm_src.shape
        fm_src = self.flatten_outputs(fm_src)
        fm_tgt = self.flatten_outputs(fm_tgt)
        div_norm = h * w
        distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
        distance = c * torch.mul(self.extractor_channel_weights, distance ** 2) / (h * w)
        fea_loss += 0.5 * torch.sum(distance)
        return fea_loss

    def reg_att_fea_map(self):
        fea_loss = torch.tensor(0.).to(self.device)

        for i, (fm_src, fm_tgt) in enumerate(zip(self.layer_outputs_source, self.layer_outputs_target)):
            b, c, h, w = fm_src.shape
            fm_src = self.flatten_outputs(fm_src)
            fm_tgt = self.flatten_outputs(fm_tgt)
            div_norm = h * w
            distance = torch.norm(fm_tgt - fm_src.detach(), 2, 2)
            distance = c * torch.mul(self.channel_weights[i], distance ** 2) / (h * w)
            fea_loss += 0.5 * torch.sum(distance)
        return fea_loss

    def reg_classifier(self):
        l2_cls = torch.tensor(0.).to(self.device)
        for name, param in self.feature_classifier_target.named_parameters():
            if name.startswith(self.fc):
                l2_cls += 0.5 * torch.norm(param) ** 2
        return l2_cls


    def compute_gradient_penalty(self, D, real_samples, fake_samples):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = Variable(torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
        # Get gradient w.r.t. interpolates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0] 
        gradients = gradients.view(gradients.size(0), -1) + 1e-16
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_discriminator(self):
        self.feature_discriminator.requires_grad_(True)
        self.feature_extractor_target.requires_grad_(False)
        self.feature_classifier_target.requires_grad_(False)
        self.feature_generator.requires_grad_(False)

        for inputs_tmp in self.inputs_gpu:
            bz_rand, bz_cat, _  = utils.z_sampler(self.batch_size, self.noise_dim, self.num_labels) # (batch, noise_dim), (batch, num_labels)
            bz_input = torch.FloatTensor(np.concatenate((bz_rand, bz_cat), axis=1))         # (batch, noise_dim + num_labels)
            bz_cat = torch.FloatTensor(bz_cat)
            einsum = torch.matmul(bz_cat, self.generator_attention)         # (batch, feature_dimension)

            bz_input = bz_input.to(self.device)
            einsum = einsum.to(self.device)
            generated_feature = self.feature_generator(bz_input, einsum)

            extracted_feature = self.feature_extractor_target(inputs_tmp)

            logits_fake = self.feature_discriminator(generated_feature)
            logits_real = self.feature_discriminator(extracted_feature)
            
            self.discriminator_optimizer.zero_grad()
            # wgan gp
            gradient_penalty = self.compute_gradient_penalty(self.feature_discriminator, extracted_feature, generated_feature)
            discriminator_loss = -torch.mean(logits_real) + torch.mean(logits_fake) + self.lambda_gp * gradient_penalty
            discriminator_loss.backward()

            self.discriminator_optimizer.step()
        return discriminator_loss.item()
        
    def train_generator(self):
        self.feature_discriminator.requires_grad_(False)
        self.feature_generator.requires_grad_(True)

        bz_rand, bz_cat, _  = utils.z_sampler(self.batch_size, self.noise_dim, self.num_labels) # (batch, noise_dim), (batch, num_labels)
        bz_input = torch.FloatTensor(np.concatenate((bz_rand, bz_cat), axis=1))  # (batch, noise_dim + num_labels)
        bz_cat = torch.FloatTensor(bz_cat)
        einsum = torch.matmul(bz_cat, self.generator_attention) # (batch, feature_dimension)

        bz_input = bz_input.to(self.device)
        einsum = einsum.to(self.device)
        generated_feature = self.feature_generator(bz_input, einsum)

        logits_fake = self.feature_discriminator(generated_feature)
        
        self.generator_optimizer.zero_grad()
        # wgan gp
        generator_loss = -torch.mean(logits_fake)
        generator_loss.backward()

        self.generator_optimizer.step()
        return generator_loss.item()
    
    def train_generator_with_fake_feature(self):
        bz_rand, bz_cat, fake_labels  = utils.z_sampler(self.batch_size, self.noise_dim, self.num_labels) # (batch, noise_dim), (batch, num_labels)
        bz_input = torch.FloatTensor(np.concatenate((bz_rand, bz_cat), axis=1))  # (batch, noise_dim + num_labels)
        bz_cat = torch.FloatTensor(bz_cat)
        einsum = torch.matmul(bz_cat, self.generator_attention) # (batch, feature_dimension)

        bz_input = bz_input.to(self.device)
        einsum = einsum.to(self.device)
        generated_feature = self.feature_generator(bz_input, einsum)
        fake_labels = torch.LongTensor(fake_labels).to(self.device)

        extracted_feature = self.feature_extractor_target(self.inputs_gpu[0])

        # merge gen, real feature
        merged_feature = torch.cat((generated_feature, extracted_feature), dim=0) # batch size * 2
        merged_labels = torch.cat((fake_labels, self.labels_gpu[0]), dim=0)

        self.generator_optimizer.zero_grad()

        # predict fake feature
        c_logits_merged = self.feature_classifier_target(merged_feature)
        c_loss_merged = self.criterion(c_logits_merged, merged_labels)
        c_loss_merged.backward()

        self.generator_optimizer.step()

    def train_extractor_classifier(self):
        self.feature_extractor_target.requires_grad_(True)
        self.feature_classifier_target.requires_grad_(True)
        self.feature_generator.requires_grad_(False)

        if self.with_regularization==True:
            self.layer_outputs_target.clear()
            self.layer_outputs_source.clear()

        # Feature Extractor by using Regularization
        extracted_feature_target = self.feature_extractor_target(self.inputs_gpu[0])
        e_c_loss = self.feature_classifier_target(extracted_feature_target)
        e_c_loss = self.criterion(e_c_loss, self.labels_gpu[0])

        extracted_feature_source = self.feature_extractor_source(self.inputs_gpu[0])
        
        # omega1 for extractor
        if self.with_regularization==True:
            loss_extractor_feature = self.extractor_att_fea_map(extracted_feature_source, extracted_feature_target)
            e_loss = e_c_loss + self.alpha_extractor * loss_extractor_feature
        else:
            e_loss = e_c_loss
            loss_extractor_feature = torch.zeros(1)

        # Feature Classifier by using Regularization
        # for hook layer
        if self.with_regularization==True:
            self.feature_classifier_source(extracted_feature_source)

        # omega1, omega2 for classifier
        if self.with_regularization==True:
            loss_feature = self.reg_att_fea_map()
            loss_classifier = self.reg_classifier()
            c_loss_real = e_c_loss + self.alpha_classifier * loss_feature + self.beta_classifier * loss_classifier
        else:
            c_loss_real = e_c_loss

        if self.with_regularization==True:
            loss = e_loss + c_loss_real
        else:
            loss = e_c_loss

        self.classifier_optimizer.zero_grad()
        self.extractor_optimizer.zero_grad()

        loss.backward()

        self.extractor_optimizer.step()
        self.classifier_optimizer.step()
        return e_loss.item(), c_loss_real.item(), loss_extractor_feature.item()

    def train_classifier_with_fake(self):
        if self.with_regularization==True:
            self.layer_outputs_target.clear()
            self.layer_outputs_source.clear()

        # Feature Extractor
        extracted_feature_target = self.feature_extractor_target(self.inputs_gpu[0])
        e_c_logit_real = self.feature_classifier_target(extracted_feature_target)
        e_c_loss_real = self.criterion(e_c_logit_real, self.labels_gpu[0])

        # Feature Classifier by using Regularization
        bz_rand, bz_cat, fake_labels  = utils.z_sampler(self.batch_size, self.noise_dim, self.num_labels) # (batch, noise_dim), (batch, num_labels)
        bz_input = torch.FloatTensor(np.concatenate((bz_rand, bz_cat), axis=1))  # (batch, noise_dim + num_labels)
        bz_cat = torch.FloatTensor(bz_cat)
        einsum = torch.matmul(bz_cat, self.generator_attention) # (batch, feature_dimension)

        bz_input = bz_input.to(self.device)
        einsum = einsum.to(self.device)
        generated_feature = self.feature_generator(bz_input, einsum)

        fake_labels = torch.LongTensor(fake_labels).to(self.device)
        e_c_logits_fake = self.feature_classifier_target(generated_feature)
        e_c_loss_fake = self.criterion(e_c_logits_fake, fake_labels)

        merged_feature = torch.cat((generated_feature, extracted_feature_target), dim=0) # batch size * 2
        merged_labels = torch.cat((fake_labels, self.labels_gpu[0]), dim=0)

        c_logits_merged = self.feature_classifier_target(merged_feature)
        c_loss_merged = self.criterion(c_logits_merged, merged_labels)

        # for hook layer
        if self.with_regularization==True:
            extracted_feature_source = self.feature_extractor_source(self.inputs_gpu[0])
            self.feature_classifier_source(extracted_feature_source)

            # omega1, omega2 for classifier
            loss_feature = self.reg_att_fea_map()
            loss_classifier = self.reg_classifier()
            e_c_loss_real = e_c_loss_real + self.alpha_classifier * loss_feature + self.beta_classifier * loss_classifier
        else:    
            loss_feature = torch.zeros(1)
            loss_classifier = torch.zeros(1)
        
        c_loss = 1.0 / 3.0 * e_c_loss_real + 1.0 / 3.0 * e_c_loss_fake + 1.0 / 3.0 * c_loss_merged
        
        self.classifier_optimizer.zero_grad()

        c_loss.backward()

        self.classifier_optimizer.step()

        return c_loss.item(), e_c_loss_real.item(), e_c_loss_fake.item(), loss_feature.item(), loss_classifier.item(), c_loss_merged.item()

    def update_generator_attention(self, feature_extractor_target, feature_classifier_target, layer_name_extractor):
        # Calculate weighting feature maps for extractor
        base_loss_extractor, jthfilter_loss_extractor, class_label_list = self.calculate_weighting_feature_maps_extractor(feature_extractor_target, feature_classifier_target, 
                                                                        layer_name=layer_name_extractor, label_min=10)

        extractor_number_filter = len(jthfilter_loss_extractor)
        wj_extractor = np.zeros((extractor_number_filter, len(base_loss_extractor)))
        print(wj_extractor.shape)
        base_loss_extractor = np.array(base_loss_extractor)

        for fidx in range(extractor_number_filter):
            wj_extractor[fidx] = base_loss_extractor - np.array(jthfilter_loss_extractor[fidx])
        
        transpose_wj_extractor = np.transpose(wj_extractor)
        for t in range(transpose_wj_extractor.shape[0]):
            def softmax_loss(loss) : 
                max_loss = np.max(loss) 
                exp_loss = np.exp(loss-max_loss) 
                sum_exp_loss = np.sum(exp_loss)
                result = exp_loss / sum_exp_loss
                return result
            transpose_wj_extractor[t] = softmax_loss(transpose_wj_extractor[t])
        
        #Initialize Attention for feature generator
        target_generator_attention = np.zeros((self.num_labels, self.feature_dimension))
        labels_count  = np.zeros(self.num_labels)
        iCnt=0
        for class_num in class_label_list:
            class_num = class_num.item()
            target_generator_attention[class_num, :] = 1.0 * (target_generator_attention[class_num, :] * labels_count[class_num] + transpose_wj_extractor[iCnt, :]) / (labels_count[class_num]+1)
            labels_count[class_num] += 1
            iCnt+=1
        
        generator_attention_np = self.rho * self.generator_attention.numpy() + (1 - self.rho) * target_generator_attention

        #save_generator_attention
        for filter_idx in range(generator_attention_np.shape[1]):
            filename = self.csv_save_path  + "/"+ "generator_attention_" + str(filter_idx) + "_" + self.curtime + ".csv"
            with open(filename, 'a') as generator_attention_save:
                writer = csv.writer(generator_attention_save, delimiter=',', quoting=csv.QUOTE_ALL)
                writer.writerow(list(generator_attention_np[:,filter_idx]))

        generator_attention_np = target_generator_attention * self.feature_dimension

        for class_label in range(self.num_labels):
            generator_attention_np[class_label, :] = np.where(generator_attention_np[class_label, :] >= 0.95, generator_attention_np[class_label, :]/self.feature_dimension, 0.0)
        
        # write number of zero generator attention per class
        with open(self.generator_attention_class, 'a') as wj_list:
            writer = csv.writer(wj_list, delimiter=',', quoting=csv.QUOTE_ALL)
            zero_generator_attention_list = []
            for t in range(generator_attention_np.shape[0]):
                zero_generator_attention_list.append(len(generator_attention_np[t][generator_attention_np[t] == 0]))
            writer.writerow(zero_generator_attention_list)

        print(len(generator_attention_np[generator_attention_np == 0]))
        self.generator_attention = torch.FloatTensor(generator_attention_np)

    def evaluate(self, step, generator_step):
        self.feature_generator.eval()
        self.feature_extractor_target.eval()
        self.feature_classifier_target.eval()
        self.feature_discriminator.eval()

        loss = 0.0
        step_acc = 0.0
        total_inputs_len = 0
        with torch.no_grad():
            for inputs, labels in self.data_loader_test:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                extracted_feature = self.feature_extractor_target(inputs)
                logits_real = self.feature_classifier_target(extracted_feature)
                _, preds = torch.max(logits_real, 1)
                loss += self.criterion(logits_real, labels).item() * inputs.size(0)

                corr_sum = torch.sum(preds == labels.data)
                step_acc += corr_sum.double()
                total_inputs_len += inputs.size(0)
                # gpu memory
                if self.with_regularization==True:
                    self.layer_outputs_target.clear()
                    self.layer_outputs_source.clear()

        loss /= total_inputs_len
        step_acc /= total_inputs_len

        print ('Step: [%d/%d] validation loss: [%.8f] validation accuracy: [%.4f]' % (step, generator_step, loss, step_acc))

        self.feature_generator.train()
        self.feature_extractor_target.train()
        self.feature_classifier_target.train()
        self.feature_discriminator.train()
        return loss, step_acc

    def train_gan(self, generator_step, rho, lambda_gp, alpha_extractor,
                    alpha_classifier, beta_classifier, num_d_iters, e_loss_for_6, 
                    step_for_3_4, step_for_5, step_for_6, pretrained_base_model):

        # Loss weight for gradient penalty
        self.rho = rho
        self.lambda_gp = lambda_gp
        self.alpha_extractor = alpha_extractor
        self.alpha_classifier = alpha_classifier
        self.beta_classifier = beta_classifier

        # start gan training
        self.data_loader, self.data_loader_test = self.load_data_set(self.data_set_name, batch_size=self.batch_size)
        print('data set name: %s' % self.data_set_name)
        print('train data loader len: %d' % ( len(self.data_loader) ) )
        print('test data loader len: %d' % ( len(self.data_loader_test) ) )
        
        self.feature_extractor_source = Feature_Extractor(base_model_name=self.base_model_name, pretrained_weight=pretrained_base_model, num_classes=self.num_labels)
        self.feature_extractor_source.requires_grad_(False)
        self.feature_extractor_source.to(self.device)
        self.feature_extractor_source.eval()
        
        self.feature_classifier_source = Feature_Classifier(base_model_name=self.base_model_name, pretrained_weight=pretrained_base_model, num_classes=self.num_labels)
        self.feature_classifier_source.requires_grad_(False)
        self.feature_classifier_source.to(self.device)
        self.feature_classifier_source.eval()
        
        self.feature_extractor_target = Feature_Extractor(base_model_name=self.base_model_name, pretrained_weight=pretrained_base_model, num_classes=self.num_labels)
        self.feature_extractor_target.to(self.device)
        self.feature_classifier_target = Feature_Classifier(base_model_name=self.base_model_name, pretrained_weight=pretrained_base_model, num_classes=self.num_labels)
        self.feature_classifier_target.to(self.device)
        
        # set extractor layer name for feature map
        if self.base_model_name == 'vgg':
            self.layer_name_extractor = 'model.2' # (64 in, 64 out)
        elif self.base_model_name == 'resnet':
            self.layer_name_extractor = 'model.4.2.conv3' # (64 in, 256 out)
        elif self.base_model_name == 'lenet':
            self.layer_name_extractor = 'model.0' # ( 6 in,  16 out)
        
        if self.with_regularization == True:
            # set classifier hook for feature map
            if self.base_model_name == 'vgg':
                self.layer_name_classifier = ['feature.23'] # (512 in, 512 out)
                self.register_hook(self.feature_classifier_source, self.for_hook_source, self.layer_name_classifier)
                self.register_hook(self.feature_classifier_target, self.for_hook_target, self.layer_name_classifier)
            elif self.base_model_name == 'resnet':
                self.layer_name_classifier = ['feature.0.3.conv3', 'feature.1.5.conv3', 'feature.2.2.conv3'] # (512 in, 2048 out)
                self.register_hook(self.feature_classifier_source, self.for_hook_source, self.layer_name_classifier)
                self.register_hook(self.feature_classifier_target, self.for_hook_target, self.layer_name_classifier)
            elif self.base_model_name == 'lenet':
                self.layer_name_classifier = ['feature.0'] # (6 in, 16 out)
                self.register_hook(self.feature_classifier_source, self.for_hook_source, self.layer_name_classifier)
                self.register_hook(self.feature_classifier_target, self.for_hook_target, self.layer_name_classifier)

            # set fc name
            if self.base_model_name == 'vgg':
                self.fc = 'classifier.6'
            elif self.base_model_name == 'resnet':
                self.fc = 'classifier'
            elif self.base_model_name == 'lenet':
                self.fc = 'classifier.4'
        
        self.feature_discriminator = Feature_Discriminator(in_channels=self.feature_dimension, base_model_name=self.base_model_name)
        self.feature_discriminator.to(self.device)

        self.feature_generator = Feature_Generator(base_model_name=self.base_model_name, noise_shape=self.noise_dim + self.num_labels)
        self.feature_generator.to(self.device)

        self.feature_generator.train()
        self.feature_extractor_target.train()
        self.feature_classifier_target.train()
        self.feature_discriminator.train()

        self.discriminator_optimizer = optim.Adam(self.feature_discriminator.parameters(), lr=self.discriminator_learning_rate, betas=(0.5, 0.9))
        self.generator_optimizer = optim.Adam(self.feature_generator.parameters(), lr=self.generator_learning_rate, betas=(0.5, 0.9))

        if self.base_model_name=='resnet':
            self.extractor_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.feature_extractor_target.parameters()),
                                lr=self.extractor_learning_rate, momentum=0.9)

            self.classifier_optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.feature_classifier_target.parameters()),
                                lr=self.classifier_learning_rate, momentum=0.9)
            
            decay_epochs = 0.5*int(generator_step) + 1
            lr_decay_extractor = optim.lr_scheduler.StepLR(self.extractor_optimizer, step_size=decay_epochs, gamma=0.1)
            lr_decay_classifier = optim.lr_scheduler.StepLR(self.classifier_optimizer, step_size=decay_epochs, gamma=0.1)
            
        else:
            self.extractor_optimizer = optim.Adam(self.feature_extractor_target.parameters(), lr=self.extractor_learning_rate, betas=(0.5, 0.9))
            self.classifier_optimizer = optim.Adam(self.feature_classifier_target.parameters(), lr=self.classifier_learning_rate, betas=(0.5, 0.9))

        self.criterion = nn.CrossEntropyLoss()
        
        if self.with_regularization==True:
            js = np.load(self.transpose_wj_extractor_npy)
            js = (js - np.mean(js)) / np.std(js)
            cw = torch.from_numpy(js).float().to(self.device)
            cw = F.softmax(cw / 5).detach()
            self.extractor_channel_weights = cw
            print(self.extractor_channel_weights.size())
        
        self.generator_attention = np.load(self.generator_attention_npy)
        self.generator_attention = self.generator_attention * self.feature_dimension

        for class_label in range(self.num_labels):
            self.generator_attention[class_label,:]=np.where(self.generator_attention[class_label,:]>= 0.9, self.generator_attention[class_label,:]/self.feature_dimension, 0.0)
        print(self.generator_attention)
        self.generator_attention = torch.FloatTensor(self.generator_attention)
        print(self.generator_attention.size())

        if self.with_regularization==True:
            self.channel_weights = []
            channel_wei = self.channel_weight_json
            if channel_wei:
                for js in json.load(open(channel_wei)):
                    js = np.array(js)
                    js = (js - np.mean(js)) / np.std(js)
                    cw = torch.from_numpy(js).float().to(self.device)
                    cw = F.softmax(cw / 5).detach()
                    self.channel_weights.append(cw)

        best_step_acc = 0.0
        step = 0

        self.inputs_gpu = []
        self.labels_gpu = []
        for d_step in range(num_d_iters):
            inputs, labels = next(iter(self.data_loader))
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.inputs_gpu.append(inputs)
            self.labels_gpu.append(labels)
        
        while step < generator_step:
            # 1. Train feature discriminator
            d_loss = self.train_discriminator()
            
            # 2. train generator
            g_loss = self.train_generator()

            if step % step_for_3_4 == 0:
                #3. Train generator using classifier with fake feature
                self.train_generator_with_fake_feature()

                #4. Train Extractor and Classifier by using BR with real features
                e_loss, c_loss_real, omega1_weight_extractor = self.train_extractor_classifier()

            if step % step_for_5 == 0 and step != 0:
                #5. Train Classifier by using BR with real and generated features
                c_loss, c_loss_real, c_loss_fake, omega1_weight_classifier, omega2_weight_classifier, c_loss_merged \
                                                                                     = self.train_classifier_with_fake()
            
            if step % step_for_6 == 0 and step != 0 and e_loss < e_loss_for_6:
                #6. Updating attention for feature generator
                self.update_generator_attention(self.feature_extractor_target, self.feature_classifier_target, self.layer_name_extractor)
            
            if step % 100 == 0 and step !=0:
                #7. print current step loss, write validation Log
                print ('Step: [%d/%d] d_loss: %.5f g_loss: %.5f c_loss: %.5f c_loss_real: %.5f c_loss_fake: %.5f e_loss: %.5f c_loss_merged: %.5f' \
                                        %(step, generator_step, d_loss, g_loss, c_loss, c_loss_real, c_loss_fake, e_loss, c_loss_merged))
                print ('Step: [%d/%d] Omega1 Extractor Loss: %.5f, Omega1 Classifier Loss: %.5f, Omega2 Classifier Loss: %.5f, best accuracy: %.5f' \
                                            %(step, generator_step, omega1_weight_extractor, omega1_weight_classifier, omega2_weight_classifier, best_step_acc))
                validation_loss, step_acc = self.evaluate(step, generator_step)
                test_time = time.time()
                print("time: %.3f" % (test_time - self.code_start_time))
                if step_acc > best_step_acc:
                    best_step_acc = step_acc
                    print('best accuracy: %.4f' % best_step_acc)

            if self.base_model_name=='resnet':
                lr_decay_extractor.step()
                lr_decay_classifier.step()

            step += 1
            self.inputs_gpu.pop(0)
            self.labels_gpu.pop(0)
            inputs, labels = next(iter(self.data_loader))
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.inputs_gpu.append(inputs)
            self.labels_gpu.append(labels)

        print('best accuracy: %.4f' % best_step_acc)
        torch.save(self.feature_extractor_target.state_dict(), self.feature_extractor_target_pth)
        torch.save(self.feature_classifier_target.state_dict(), self.feature_classifier_target_pth)
        torch.save(self.feature_generator.state_dict(), self.feature_generator_pth)
        torch.save(self.feature_discriminator.state_dict(), self.feature_discriminator_pth)
        optimal_attention = self.generator_attention.numpy()
        np.save(self.optimal_attention_npy, optimal_attention)

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1, help=" 0, 1, 2 or 3 ")
    parser.add_argument("--mode", type=str, default='',
                    help=" 'channel_evaluation', 'train_gan' ")

    parser.add_argument("--source_dataset", type=str, default='imagenet',
                    help=" 'emnist', 'cifar', 'imagenet' ")
                    # lenet: emnist, vgg: cifar, resnet: imagenet

    parser.add_argument("--target_dataset", type=str, default='stl',
                    help=" 'svhn', 'fashion_mnist', 'stl', 'cinic', 'caltech', 'food' ")
                    # lenet: svhn, fashion_mnist, vgg: stl, cinic, resnet: caltech, food

    parser.add_argument("--base_model_name", type=str, default='vgg', help=" 'lenet, vgg or resnet' ")
    parser.add_argument("--model_save_path", type=str, default='./model/', help='save folder name')

    parser.add_argument("--generator_step", type=int, default=40000, help='generator training step')
    parser.add_argument("--batch_size", type=int, default=64, help='batch_size')
    parser.add_argument("--minor_class_num", type=int, default=10, help='minor class num')
    parser.add_argument("--minor_class_ratio", type=float, default=0.1, help='minor class len = major_class_len * minor_class_ratio')

    parser.add_argument("--step_for_3_4", type=int, default=1, help='3, 4 per step')
    parser.add_argument("--step_for_5", type=int, default=5, help='5 per step')
    parser.add_argument("--step_for_6", type=int, default=2000, help='6 per step')
    parser.add_argument("--e_loss_for_6", type=float, default=0.05, help='minimum e loss for 6')

    parser.add_argument("--extractor_learning_rate", type=float, default=1e-4, help='extractor learning rate')
    parser.add_argument("--classifier_learning_rate", type=float, default=1e-4, help='classifier learning rate')
    parser.add_argument("--discriminator_learning_rate", type=float, default=1e-4, help='discriminator learning rate')
    parser.add_argument("--generator_learning_rate", type=float, default=1e-4, help='generator learning rate')

    # weight path
    parser.add_argument("--extractor_weight_path", type=str, default='', help='extractor weight path')
    parser.add_argument("--classifier_weight_path", type=str, default='', help='classifier weight path')
    parser.add_argument("--generator_weight_path", type=str, default='', help='generator weight path')
    parser.add_argument("--optimal_attention_path", type=str, default='', help='optimal attention path')

    parser.add_argument("--rho", type=float, default=0.75, help='rho for generator attention')
    parser.add_argument("--lambda_gp", type=float, default=10, help='lambda_gp in wgan gp')
    parser.add_argument("--alpha_extractor", type=float, default=0.01, help='alpha for extractor')
    parser.add_argument("--alpha_classifier", type=float, default=0.01, help='alpha for classifier')
    parser.add_argument("--beta_classifier", type=float, default=0.01, help='beta for classifier')
    parser.add_argument("--num_d_iters", type=int, default=5, help='num_d_iters')
    parser.add_argument("--pretrained_base_model", type=str, default='', help='pretrained_base_model')

    parser.add_argument("--with_regularization", type=str, default='with', help='with regularization')

    args = parser.parse_args()

    if args.with_regularization == 'with':
        with_regularization = True
        print('with_regularization')
    else:
        with_regularization = False
        print('no_regularization')

    if args.pretrained_base_model == '': # '' for imagenet
        pretrained_base_model = None
    else:
        pretrained_base_model = args.pretrained_base_model

    test = TrainOps(gpu_num=args.gpu, base_model_name=args.base_model_name, batch_size=args.batch_size, data_set_name=args.target_dataset, 
                    extractor_learning_rate=args.extractor_learning_rate, classifier_learning_rate=args.classifier_learning_rate, discriminator_learning_rate=args.discriminator_learning_rate,
                    generator_learning_rate=args.generator_learning_rate, minor_class_num=args.minor_class_num, minor_class_ratio=args.minor_class_ratio, with_regularization=with_regularization,
                    model_save_path=args.model_save_path)

    if args.mode == 'channel_evaluation':
        test.channel_evaluation(pretrained_base_model=pretrained_base_model)
    elif args.mode == 'train_gan':
        test.train_gan(generator_step=args.generator_step, rho=args.rho, lambda_gp=args.lambda_gp, alpha_extractor=args.alpha_extractor, alpha_classifier=args.alpha_classifier,
                        beta_classifier=args.beta_classifier, num_d_iters=args.num_d_iters, e_loss_for_6=args.e_loss_for_6, 
                        step_for_3_4=args.step_for_3_4, step_for_5=args.step_for_5, 
                        step_for_6=args.step_for_6, pretrained_base_model=pretrained_base_model)
    else:
        raise ValueError('Unknown mode')