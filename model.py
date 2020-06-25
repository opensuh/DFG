import torch
import torch.nn as nn
from torchvision.models import vgg16, resnet50


class Feature_Generator(nn.Module):

    def __init__(self, base_model_name, noise_shape):
        super(Feature_Generator, self).__init__()
        self.base_model_name = base_model_name
        if base_model_name != 'resnet':
            self.dense = nn.Sequential(
                nn.Linear(noise_shape, 128 * 4 * 4),
                nn.BatchNorm1d(128 * 4 * 4)
            )
        else:
            self.dense = nn.Sequential(
                nn.Linear(noise_shape, 512 * 4 * 4),
                nn.BatchNorm1d(512 * 4 * 4)
            )

        if base_model_name == 'vgg': # image_size 32 -> (batch, 64, 16, 16)
            self.model = nn.Sequential(
                nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(True),

                nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(True),
                
                nn.ConvTranspose2d(128, 64, 4, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(True),

                nn.ConvTranspose2d(64, 64, 4, stride=1, padding=2, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(64)
            )
        elif base_model_name == 'resnet': # image_size 224 -> (batch, 256, 56, 56)
            self.model = nn.Sequential(
                nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(True),

                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(True),
                
                nn.ConvTranspose2d(256, 256, 2, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(True),

                nn.ConvTranspose2d(256, 256, 2, stride=2, padding=2, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(256)
            )
        elif base_model_name == 'lenet': # image_size 32 -> (batch, 6, 14, 14)
            self.model = nn.Sequential(
                nn.ConvTranspose2d(128, 48, 3, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(48),
                nn.LeakyReLU(True),

                nn.ConvTranspose2d(48, 12, 3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(12),
                nn.LeakyReLU(True),

                nn.ConvTranspose2d(12, 6, 4, stride=1, padding=0, bias=False),
                nn.Tanh(),
                nn.BatchNorm2d(6)
            )
    
    def forward(self, x, einsum):
        x = self.dense(x)
        if self.base_model_name != 'resnet':
            x = x.view(x.size(0), 128, 4, 4)
        else:
            x = x.view(x.size(0), 512, 4, 4)
        x = self.model(x)
        x = torch.einsum('aijk, ai -> aijk', x, einsum) # (batch, 256, 56, 56) * (batch, 256) -> (batch, 256, 56, 56)
        return x


class Feature_Discriminator(nn.Module):

    def __init__(self, in_channels, base_model_name):
        super(Feature_Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(True), 

            nn.Conv2d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(True), 

            nn.Conv2d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(True)
        )

        if base_model_name != 'resnet':
            self.dense = nn.Linear(128 * 2 * 2, 1)
        else:
            self.dense = nn.Linear(128 * 7 * 7, 1)

    def forward(self, x):
        x = self.conv(x)       # (batch, 128, 2, 2)
        x = x.view(x.size(0), -1)   # (batch, 128 * 2 * 2)
        x = self.dense(x)       # (batch, 1)
        return x


class Feature_Extractor(nn.Module):

    def __init__(self, base_model_name, pretrained_weight=None, num_classes=10):
        super(Feature_Extractor, self).__init__()

        if base_model_name == 'vgg': # (batch, 3, 32, 32) -> (batch, 3, 16, 16)
            if pretrained_weight is None:
                vgg16_ = vgg16(pretrained=True)
                vgg16_.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, num_classes)
                )
                vgg16_list = list(vgg16_.children())
            else:
                vgg16_ = vgg16(pretrained=False, num_classes=num_classes)
                vgg16_.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, num_classes)
                )
                vgg16_.load_state_dict(torch.load(pretrained_weight, map_location='cpu'))
                vgg16_list = list(vgg16_.children())
            self.model = nn.Sequential(
                    # 'vgg16': [64, 64, 'M', /stop/ 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
                    *list( vgg16_list[0] )[:5]
                )
        elif base_model_name == 'resnet': # (batch, 3, 224, 224) -> (batch, 3, 56, 56)
            if pretrained_weight is None:
                resnet50_list = list(resnet50(pretrained=True).children())
            else:
                resnet50_ = resnet50(pretrained=False, num_classes=num_classes)
                resnet50_.load_state_dict(torch.load(pretrained_weight, map_location='cpu'))
                resnet50_list = list(resnet50_.children())
            self.model = nn.Sequential(
                    # stop at residual 1
                    *list( resnet50_list )[:5]
                )
        elif base_model_name == 'lenet': # (batch, 1, 32, 32) -> (batch, 6, 14, 14)
            if pretrained_weight is None:
                lenet5_list = list(Lenet5(num_classes=num_classes).children())
            else:
                lenet5_ = Lenet5(num_classes=num_classes)
                lenet5_.load_state_dict(torch.load(pretrained_weight, map_location='cpu'))
                lenet5_list = list(lenet5_.children())
            self.model = nn.Sequential(
                    # stop after first conv layer
                    *list( lenet5_list[0] )[:3]
                )
        
    def forward(self, x):
        x = self.model(x)
        return x


class Feature_Classifier(nn.Module):

    def __init__(self, base_model_name, pretrained_weight=None, num_classes=10):
        super(Feature_Classifier, self).__init__()
        self.base_model_name = base_model_name
        if base_model_name == 'vgg': 
            if pretrained_weight is None:
                vgg16_ = vgg16(pretrained=True)
                vgg16_.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, num_classes)
                )
                vgg16_list = list(vgg16_.children())
            else:
                vgg16_ = vgg16(pretrained=False, num_classes=num_classes)
                vgg16_.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(4096, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, num_classes)
                )
                vgg16_.load_state_dict(torch.load(pretrained_weight, map_location='cpu'))
                vgg16_list = list(vgg16_.children())
            self.feature = nn.Sequential(
                    # 'vgg16': [64, 64, 'M', /after/ 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
                    *list( vgg16_list[0] )[5:]
                )
            # (batch, 512, 7, 7)
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = vgg16_.classifier
        elif base_model_name == 'resnet':
            if pretrained_weight is None:
                resnet50_list = list(resnet50(pretrained=True).children())
            else:
                resnet50_ = resnet50(pretrained=False, num_classes=num_classes)
                resnet50_.load_state_dict(torch.load(pretrained_weight, map_location='cpu'))
                resnet50_list = list(resnet50_.children())
            self.feature = nn.Sequential(
                    # after at residual 1
                    *list( resnet50_list )[5:8]
                )
            # (batch, 2048, 1, 1)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(2048, num_classes)
        elif base_model_name == 'lenet':
            if pretrained_weight is None:
                lenet5_list = list(Lenet5(num_classes=num_classes).children())
            else:
                lenet5_ = Lenet5(num_classes=num_classes)
                lenet5_.load_state_dict(torch.load(pretrained_weight, map_location='cpu'))
                lenet5_list = list(lenet5_.children())
            self.feature = nn.Sequential(
                *list( lenet5_list[0] )[3:]
            )
            self.classifier = nn.Sequential(
                *list( lenet5_list[1] )[:]
            )

    def forward(self, x):
        x = self.feature(x)
        if self.base_model_name != 'lenet':
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Lenet5(nn.Module):

    def __init__(self, num_classes=10):
        super(Lenet5, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 6, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.dense = nn.Sequential(
            nn.Linear(16 * 5 * 5, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        return x

