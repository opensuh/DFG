import os
import os.path
import sys
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pickle
import torch

def has_file_allowed_extension(filename, extensions):
    return filename.lower().endswith(extensions)

def make_dataset(directory, class_to_idx, extensions=None):
    images = []
    directory = os.path.expanduser(directory)
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)

    images = dict()
    for idx in range(len(class_to_idx)):
        images[idx] = []

    for target in sorted(class_to_idx.keys()):
        d = os.path.join(directory, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    images[class_to_idx[target]].append(path)

    return images

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
class CinicDataset(Dataset):
    """ cinic https://github.com/BayesWatch/cinic-10 Dataset.

    Args:
        root (string): Root directory of dataset where directory
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, transform=transforms.ToTensor(), minor_class_num=0, ratio=1.0, extensions=IMG_EXTENSIONS):
        super(CinicDataset, self).__init__()
        assert 0. <= ratio + 1e-10 and ratio - 1e-10 <= 1.

        self.transform = transform
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        major_class_len = 0
        for key in samples:
            major_class_len = max(major_class_len, len(samples[key]))
        minor_class_len = int(major_class_len * ratio)
        print('major len:', major_class_len, 'minor len:', minor_class_len)

        tmp = [i for i in range(len(classes))]
        random.shuffle(tmp)
        
        self.major_class = tmp[minor_class_num:]
        minor_class = tmp[0:minor_class_num]
        print('major', self.major_class)
        print('minor', minor_class)

        self.data = []
        for key in samples:
            image_pathes = samples[key]
            if key in self.major_class:
                tmp_len = 0
                for image_path in image_pathes:
                    self.data.append( (image_path, key) )
                    tmp_len += 1
                    if tmp_len >= major_class_len:
                        break
                print(tmp_len, end=' ')
            else :
                tmp_len = 0
                for image_path in image_pathes:
                    self.data.append( (image_path, key) )
                    tmp_len += 1
                    if tmp_len >= minor_class_len:
                        break
                print(tmp_len, end=' ')
        print()
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        img_path, target = self.data[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, target

    def _find_classes(self, directory):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.data)


class Food101Dataset(Dataset):
    """ Food-101 https://www.kaggle.com/dansbecker/food-101/home Dataset.
                https://www.tensorflow.org/datasets/catalog/food101

    Args:
        root (string): Root directory of dataset where directory
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, transform=transforms.ToTensor(), minor_class_num=0, ratio=1.0, extensions=IMG_EXTENSIONS):
        super(Food101Dataset, self).__init__()
        assert 0. <= ratio + 1e-10 and ratio - 1e-10 <= 1.

        self.transform = transform
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        major_class_len = 0
        for key in samples:
            major_class_len = max(major_class_len, len(samples[key]))
        minor_class_len = int(major_class_len * ratio)
        print('major len:', major_class_len, 'minor len:', minor_class_len)

        tmp = [i for i in range(len(classes))]
        random.shuffle(tmp)
        
        self.major_class = tmp[minor_class_num:]
        minor_class = tmp[0:minor_class_num]
        print('major', self.major_class)
        print('minor', minor_class)

        self.data = []
        for key in samples:
            image_pathes = samples[key]
            if key in self.major_class:
                tmp_len = 0
                for image_path in image_pathes:
                    self.data.append( (image_path, key) )
                    tmp_len += 1
                    if tmp_len >= major_class_len:
                        break
                print(tmp_len, end=' ')
            else :
                tmp_len = 0
                for image_path in image_pathes:
                    self.data.append( (image_path, key) )
                    tmp_len += 1
                    if tmp_len >= minor_class_len:
                        break
                print(tmp_len, end=' ')
        print()
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __getitem__(self, index):
        img_path, target = self.data[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, target

    def _find_classes(self, directory):
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self):
        return len(self.data)


class SvhnDataset(Dataset):
    """ SVHN http://ufldl.stanford.edu/housenumbers/ Dataset.
            https://pytorch.org/docs/stable/torchvision/datasets.html#svhn

    Args:
        root (string): Root directory of dataset where directory
        split (string): One of {'train', 'test'}.
            Accordingly dataset is selected.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat"]
                 }

    def __init__(self, root, split='train', transform=transforms.ToTensor(),
                     major_len=5000, minor_class_num=8, ratio=0.1):
        super(SvhnDataset, self).__init__()

        root = root
        filename = self.split_list[split][1]
        self.transform = transform

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading mat file as array
        loaded_mat = sio.loadmat(os.path.join(root, filename))

        data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(labels, labels == 10, 0)
        data = np.transpose(data, (3, 2, 0, 1))

        if split == 'test':
            tmp_cnt = [0 for i in range(10)]
            for label in labels:
                tmp_cnt[label] += 1
            major_len = max(tmp_cnt)
        minor_len = int(major_len * ratio)

        print('type: %s, major len: %d, minor len: %d' % (split, major_len, minor_len))

        tmp = [i for i in range(10)]
        random.shuffle(tmp)
        
        self.major_class = tmp[minor_class_num:]
        minor_class = tmp[0:minor_class_num]
        print('major', self.major_class, 'minor:', minor_class)

        images = dict()
        classes = dict()
        label_cnt = [0 for i in range(10)]
        for data_tmp, label_tmp in zip(data, labels):
            if label_tmp in self.major_class and label_cnt[label_tmp] >= major_len:
                continue
            if label_tmp in minor_class and label_cnt[label_tmp] >= minor_len:
                continue

            if not label_tmp in images:
                images[label_tmp] = [data_tmp]
                classes[label_tmp] = [label_tmp]
                label_cnt[label_tmp] = 1
            else:
                images[label_tmp].append(data_tmp)
                classes[label_tmp].append(label_tmp)
                label_cnt[label_tmp] += 1

        print('check:', end=' ')
        for key in images:
            print(len(images[key]), end=' ')
        print()
        print('check:', end=' ')
        for key in classes:
            print(len(classes[key]), end=' ')
        print()

        self.data = []
        for key in images:
            self.data.extend(images[key])
        self.labels = []
        for key in classes:
            self.labels.extend(classes[key])

    def __getitem__(self, index):
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL gray Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0))).convert('L')
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


class FashionMnistDataset(Dataset):
    """ Fashion-MNIST https://github.com/zalandoresearch/fashion-mnist Dataset.
                    https://pytorch.org/docs/stable/torchvision/datasets.html#fashion-mnist

    Args:
        root (string): Root directory of dataset where ``Fashion-MNIST/processed/training.pt``
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, train=True, transform=transforms.ToTensor(), 
                    minor_class_num=8, ratio=0.025):
        super(FashionMnistDataset, self).__init__()

        self.transform = transform
        processed_folder = os.path.join(root, 'FashionMNIST/processed')

        if train:
            data_file = 'training.pt'
        else:
            data_file = 'test.pt'

        data, targets = torch.load(os.path.join(processed_folder, data_file))

        major_len = 6000
        minor_len = int(major_len * ratio)

        print('type: %s, major len: %d, minor len: %d' % (('train' if train else 'test'), major_len, minor_len))

        tmp = [i for i in range(10)]
        random.shuffle(tmp)
        
        self.major_class = tmp[minor_class_num:]
        minor_class = tmp[0:minor_class_num]
        print('major', self.major_class, 'minor:', minor_class)

        images = dict()
        classes = dict()
        label_cnt = [0 for i in range(10)]
        for data_tmp, label_tmp in zip(data, targets):
            idx = label_tmp.item()
            if idx in self.major_class and label_cnt[idx] >= major_len:
                continue
            if idx in minor_class and label_cnt[idx] >= minor_len:
                continue

            if not idx in images:
                images[idx] = [data_tmp]
                classes[idx] = [label_tmp]
                label_cnt[idx] = 1
            else:
                images[idx].append(data_tmp)
                classes[idx].append(label_tmp)
                label_cnt[idx] += 1

        print('check:', end=' ')
        for key in images:
            print(len(images[key]), end=' ')
        print()
        print('check:', end=' ')
        for key in classes:
            print(len(classes[key]), end=' ')
        print()

        self.data = []
        for key in images:
            self.data.extend(images[key])
        self.labels = []
        for key in classes:
            self.labels.extend(classes[key])

    def __getitem__(self, index):

        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # (3, 32, 32)
    # cinic_data_set = CinicDataset(root='/data/open_dataset/cinic10/test', minor_class_num=8, ratio=0.1)
    # for data, label in cinic_data_set:
    #     print(data.size())
    #     break

    # (3, ?, ?)
    # food101_data_set= Food101Dataset(root='/data/open_dataset/food-101/train/', minor_class_num=50, ratio=0.2)
    # for data, label in food101_data_set:
    #     print(data.size())
    #     break

    # (1, 32, 32)
    # svhn_data_set = SvhnDataset(root='/data/open_dataset/svhn/', split='train', major_len=5000, minor_class_num=0, ratio=1.0)
    # for data, label in svhn_data_set:
    #     print(data.size())
    #     break

    # (1, 28, 28)
    fashion_mnist = FashionMnistDataset(root='/data/open_dataset/fashion_mnist/', train=False, minor_class_num=8, ratio=0.025)
    for data, label in fashion_mnist:
        print(data.size())
        break