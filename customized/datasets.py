"""
Defines all Dataset objects customized to the data.
"""
__author__ = 'ryanquinnnelson'

import logging
import os
import random

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import customized.helper as helper


class DatasetHandler:
    """
    Defines object to initialize Dataset objects.
    """

    def __init__(self, data_dir):
        """
        Initialize DatasetHandler
        Args:
            data_dir (str): root directory where all training, validation, and test data exists
        """
        self.data_dir = data_dir

    def get_train_dataset(self, config):
        """
        Load training data into memory and initialize the Dataset object.

        Args:
            config (ConfigParser): configuration in ConfigParser format.

        Returns:Dataset

        """

        # parse configs
        train_dir = config['data']['train_dir']
        train_target_dir = config['data']['train_target_dir']
        train_transforms = helper.to_string_list(config['data']['transforms_list'])
        resize_height = config['data'].getint('resize_height')

        # initialize dataset
        t = _compose_transforms(train_transforms, resize_height)
        dataset = ImageDataset(train_dir, train_target_dir, 'train', t)
        logging.info(f'Loaded {len(dataset)} training images.')
        return dataset

    def get_val_dataset(self, config):
        """
        Load validation data into memory and initialize the Dataset object.

        Args:
            config (ConfigParser): configuration in ConfigParser format.

        Returns:Dataset

        """
        # parse configs
        val_dir = config['data']['val_dir']
        val_target_dir = config['data']['val_target_dir']
        train_transforms = helper.to_string_list(config['data']['transforms_list'])
        resize_height = config['data'].getint('resize_height')

        # determine whether normalize transform should also be applied to validation and test data
        self.should_normalize_val = True if 'Normalize' in train_transforms else False

        if self.should_normalize_val:
            logging.info('Normalizing validation data to match normalization of training data...')
            t = _compose_transforms(['Resize', 'ToTensor', 'Normalize'], resize_height)
        else:
            t = _compose_transforms(['Resize', 'ToTensor'], resize_height)

        # initialize dataset
        dataset = ImageDataset(val_dir, val_target_dir, 'val', t)
        logging.info(f'Loaded {len(dataset)} validation images.')
        return dataset

    def get_test_dataset(self, config):
        """
        Load test data into memory and initialize the Dataset object.

        Args:
            config (ConfigParser): configuration in ConfigParser format.

        Returns:Dataset

        """
        # parse configs
        test_dir = config['data']['test_dir']
        test_target_dir = config['data']['test_target_dir']
        train_transforms = helper.to_string_list(config['data']['transforms_list'])
        resize_height = config['data'].getint('resize_height')

        # determine whether normalize transform should also be applied to validation and test data
        self.should_normalize_test = True if 'Normalize' in train_transforms else False
        if self.should_normalize_test:
            logging.info('Normalizing validation data to match normalization of training data...')
            t = _compose_transforms(['Resize', 'ToTensor', 'Normalize'], resize_height)
        else:
            t = _compose_transforms(['Resize', 'ToTensor'], resize_height)

        # initialize dataset
        dataset = ImageDataset(test_dir, test_target_dir, 'test', t)
        logging.info(f'Loaded {len(dataset)} test images.')
        return dataset


def _apply_transformations(img, target):
    if random.random() > 0.5:
        # print('vflip')
        img = transforms.functional_pil.vflip(img)
        target = transforms.functional_pil.vflip(target)

    if random.random() > 0.5:
        # print('hflip')
        img = transforms.functional_pil.hflip(img)
        target = transforms.functional_pil.hflip(target)

    return img, target


class ImageDataset(Dataset):
    """
    Defines object that represents an image Dataset.
    """

    def __init__(self, img_dir, targets_dir, dataset_type, transform=None):
        """
        Initialize ImageDataset.

        Args:
            img_dir (str): Directory for images
            targets_dir (str): Directory for targets related to given images
            transform (transformation to perform on both images and targets):
        """
        self.img_dir = img_dir
        self.targets_dir = targets_dir
        self.transform = transform
        self.dataset_type = dataset_type

        # prepare image list
        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list.remove('.DS_Store')  # remove mac generated files
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_list[idx])

        # generate target name
        # target image name matches image but also includes suffix
        img_name = self.img_list[idx][:-4]  # strip .bmp
        target_path = os.path.join(self.targets_dir, img_name + '_anno.bmp')

        img = Image.open(img_path).convert('RGB')
        target = Image.open(target_path)  # .convert('RGB')

        # standardize image size based on original image size
        img = img.resize((775, 522), resample=Image.BILINEAR)  # standardize image size
        target = target.resize((775, 522), resample=Image.BILINEAR)  # standardize target size

        if self.dataset_type == 'train':
            # apply random transformations to image and target for training set only
            img, target = _apply_transformations(img, target)

        # resize and convert to tensors
        tensor_img = self.transform(img)
        tensor_target = self.transform(target)  # (C, H, W)

        # keep only first channel because all three channels are given the same value
        tensor_target_first_channel = tensor_target[0]  # (H,W)

        # convert all nonzero target values to 1
        # nonzero values indicate segment
        # zero values indicate background
        tensor_target_first_channel[tensor_target_first_channel != 0] = 1.0

        # convert target to long datatype to indicate classes
        tensor_target_first_channel = tensor_target_first_channel.to(torch.long)

        return tensor_img, tensor_target_first_channel


def _compose_transforms(transforms_list, resize_height):
    """
    Build a composition of transformations to perform on image data.
    Args:
        transforms_list (List): list of strings representing individual transformations,
        in the order they should be performed
        resize_height (int): Number of pixels tall that image should be after resizing
    Returns: transforms.Compose object containing all desired transformations
    """
    t_list = []

    for each in transforms_list:
        if each == 'ToTensor':
            t_list.append(transforms.ToTensor())
        elif each == 'Resize':
            t_list.append(transforms.Resize(resize_height, interpolation=Image.BILINEAR))

    composition = transforms.Compose(t_list)

    return composition
