import torch
from PIL import Image
import torchvision.transforms as transforms


def calc_channel_sums(img_path):
    img = Image.open(img_path).convert('RGB')

    # convert to tensor
    tensor_img = transforms.ToTensor()(img)

    # sum pixel values for each channel separately
    means_per_channel = torch.mean(tensor_img, dim=[1, 2])
    stds_per_channel = torch.std(tensor_img, dim=[1, 2])

    return means_per_channel, stds_per_channel


def calc_images_means_stds(img_path_list):
    """
    Calculate mean and std dev for each channel for an image set.
    :param img_path_list:
    :return:
    """
    means = []
    stds = []
    for img_path in img_path_list:
        means_per_channel, stds_per_channel = calc_channel_sums(img_path)
        means.append(means_per_channel)
        stds.append(stds_per_channel)

    # create single tensor of each result
    m = torch.vstack(means)
    s = torch.vstack(stds)

    # divide by number of results
    channel_means = m / len(means)
    channel_stds = s / len(stds)
    return channel_means, channel_stds
