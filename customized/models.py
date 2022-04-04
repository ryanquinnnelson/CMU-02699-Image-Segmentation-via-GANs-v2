import logging

import torch.nn as nn
import torch


class ModelHandler:
    def __init__(self):
        pass

    # TODO: revise models to modular, accept argument for which model type to use
    def get_models(self, wandb_config):
        sn = None
        en = None

        if wandb_config.sn_model_type == 'SNLite':
            sn = SNLite()
        elif wandb_config.sn_model_type == 'ConcatenationFCN':
            input_block_depth = wandb_config.sn_input_block_depth
            num_fcn_blocks = wandb_config.sn_num_fcn_blocks
            fcn_block_depth = wandb_config.sn_fcn_block_depth
            input_channels = wandb_config.sn_input_channels
            output_channels = wandb_config.sn_output_channels
            first_layer_out_channels = wandb_config.sn_first_layer_out_channels
            block_pattern = wandb_config.sn_block_pattern
            upsampling_pattern = wandb_config.sn_upsampling_pattern
            original_height = wandb_config.original_height
            original_width = wandb_config.original_width

            sn = ConcatenationFCN(input_block_depth, num_fcn_blocks, fcn_block_depth, input_channels, output_channels,
                                  first_layer_out_channels, block_pattern, upsampling_pattern, original_height,
                                  original_width)

        elif wandb_config.sn_model_type == 'ZhengSN':

            input_channels = wandb_config.sn_input_channels
            sn = ZhengSN(input_channels)

        if wandb_config.en_model_type == 'ENLite':
            en = ENLite()
        elif wandb_config.en_model_type == 'ZhengEN':
            en = ZhengEN()
        elif wandb_config.en_model_type == 'FlexVGG':

            input_block_depth = wandb_config.en_input_block_depth
            num_fcn_blocks = wandb_config.en_num_fcn_blocks
            depth_fcn_block = wandb_config.en_depth_fcn_block
            input_channels = wandb_config.en_input_channels
            first_layer_out_channels = wandb_config.en_first_layer_out_channels
            block_pattern = wandb_config.en_fcn_block_pattern
            depth_linear_block = wandb_config.en_depth_linear_block
            linear_block_pattern = wandb_config.en_linear_block_pattern
            first_linear_layer_out_features = wandb_config.en_first_linear_layer_out_features
            out_features = wandb_config.en_out_features

            en = FlexVGG(input_block_depth, num_fcn_blocks, depth_fcn_block, input_channels, first_layer_out_channels,
                         block_pattern,
                         depth_linear_block, linear_block_pattern, first_linear_layer_out_features, out_features)

        logging.info(f'Generator model initialized:\n{sn}')
        logging.info(f'Discriminator model initialized:\n{en}')

        return [sn, en], ['sn_model', 'en_model']


def _init_weights(layer):
    """
    Perform initialization of layer weights if layer is a Conv2d layer.
    Args:
        layer: layer under consideration
    Returns: None
    """
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
    elif isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)


class CnnBlock(nn.Module):

    def __init__(self, block_number, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cnn_block = nn.Sequential()
        self.cnn_block.add_module('cnn' + str(block_number) + '_0',
                                  nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, bias=False))
        self.cnn_block.add_module('bn' + str(block_number) + '_0', nn.BatchNorm2d(out_channels))
        self.cnn_block.add_module('relu' + str(block_number) + '_0', nn.ReLU(inplace=True))

        # initialize weights
        self.cnn_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'cnn_block_input:{x.shape}')
        x = self.cnn_block(x)
        # logging.info(f'cnn_block:{x.shape}')
        return x


class UpConvBlock(nn.Module):
    # bi-linear interpolation, or learned up-sampling filters
    # nn.functional.interpolate(input, size=None, scale_factor=None, mode='bilinear')
    # https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html

    def __init__(self, in_channels, out_channels, size, mode='bilinear'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_block = nn.Sequential(
            nn.Upsample(size=size, mode=mode),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # initialize weights
        self.up_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'up_block_input:{x.shape}')
        x = self.up_block(x)
        # logging.info(f'up_block:{x.shape}')
        return x


class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, batchnorm=True, activation='relu'):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear_block = nn.Sequential()
        self.linear_block.add_module('linear', nn.Linear(in_features, out_features))

        if batchnorm:
            self.linear_block.add_module('bn', nn.BatchNorm1d(out_features))

        if activation == 'relu':
            self.linear_block.add_module('activation', nn.ReLU(inplace=True))
        elif activation == 'sigmoid':
            self.linear_block.add_module('activation', nn.Sigmoid())

        self.linear_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'linear_block_input:{x.shape}')
        x = self.linear_block(x)
        # logging.info(f'linear_block:{x.shape}')
        return x


# Simplified SN model from paper "Deep Adversarial Networks for Biomedical Image Segmentation..."
# layers are reduced to run fast for testing purposes
class SNLite(nn.Module):
    def __init__(self):
        super().__init__()

        self.block7 = nn.Sequential(
            CnnBlock(0, 3, 3),
            nn.Conv2d(3, 2, kernel_size=1, stride=1, padding=0, bias=False),  # 2 classes
            nn.Softmax2d()
        )

    def forward(self, x, i):
        block7out = self.block7(x)

        if i == 0:
            logging.info(f'block7out.shape:{block7out.shape}')

        return block7out


# Simplified EN model from paper "Deep Adversarial Networks for Biomedical Image Segmentation..."
# layers are reduced to run fast for testing purposes
class ENLite(nn.Module):
    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(

            # conv1
            CnnBlock(0, 4, 4)

        )

        self.block2 = nn.Sequential(

            nn.Flatten(),  # need to convert 2d to 1d

        )

        self.block3 = nn.Sequential(
            LinearBlock(297472, 256),  # 4*224*332
            LinearBlock(256, 128),
            LinearBlock(128, 64),
            nn.Linear(64, 1),  # binary classes
            nn.Sigmoid()
        )

    def forward(self, x, i):
        if i == 0:
            logging.info(f'x:{x.shape}')

        block1out = self.block1(x)

        if i == 0:
            logging.info(f'block1out:{block1out.shape}')
        block2out = self.block2(block1out)

        if i == 0:
            logging.info(f'block2out:{block2out.shape}')

        block3out = self.block3(block2out)

        if i == 0:
            logging.info(f'block3out:{block3out.shape}')

        return block3out


# SN model from paper "Deep Adversarial Networks for Biomedical Image Segmentation Utilizing Unannotated Images"
class ZhengSN(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block1 = nn.Sequential(
            # conv1
            CnnBlock(1, in_features, 64),
            CnnBlock(2, 64, 64),

            # pool1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv2
            CnnBlock(3, 64, 128),

            # pool2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv3
            CnnBlock(4, 128, 128),
            CnnBlock(5, 128, 256),

            # pool3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv4
            CnnBlock(6, 256, 512),
            CnnBlock(7, 512, 512)  # shortcut to up-conv1

        )

        self.block2 = nn.Sequential(

            # pool4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv5
            CnnBlock(8, 512, 512),
            CnnBlock(9, 512, 512)  # shortcut to up-conv2

        )

        self.block3 = nn.Sequential(

            # pool5
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv6
            CnnBlock(10, 512, 1024),
            CnnBlock(11, 1024, 1024)  # shortcut to up-conv3
        )

        self.block4 = UpConvBlock(1024, 1024, (224, 332))

        self.block5 = UpConvBlock(512, 512, (224, 332))

        self.block6 = UpConvBlock(512, 512, (224, 332))

        self.block7 = nn.Sequential(
            CnnBlock(12, 2048, 1024),
            nn.Conv2d(1024, 2, kernel_size=1, stride=1, padding=0, bias=False),  # 2 classes
            nn.Softmax2d()
        )

    def forward(self, x, i):
        block1out = self.block1(x)
        block2out = self.block2(block1out)
        block3out = self.block3(block2out)

        # upconvolution
        block4out = self.block4(block3out)
        block5out = self.block5(block2out)
        block6out = self.block6(block1out)

        # concatenate results
        concatenated = torch.cat((block4out, block5out, block6out), dim=1)  # channels are the second dimension

        block7out = self.block7(concatenated)

        if i == 0:
            logging.info(f'block1out.shape:{block1out.shape}')
            logging.info(f'block2out.shape:{block2out.shape}')
            logging.info(f'block3out.shape:{block3out.shape}')
            logging.info(f'block4out.shape:{block4out.shape}')
            logging.info(f'block5out.shape:{block5out.shape}')
            logging.info(f'block6out.shape:{block6out.shape}')
            logging.info(f'concatenated.shape:{concatenated.shape}')
            logging.info(f'block7out.shape:{block7out.shape}')

        return block7out


# EN model from paper "Deep Adversarial Networks for Biomedical Image Segmentation Utilizing Unannotated Images"
class ZhengEN(nn.Module):
    def __init__(self, en_input_features=4):
        super().__init__()

        self.block1 = nn.Sequential(

            # conv1
            CnnBlock(1, en_input_features, 64),
            CnnBlock(2, 64, 64),

            # pool1
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv2
            CnnBlock(3, 64, 128),

            # pool2
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv3
            CnnBlock(4, 128, 256),
            CnnBlock(5, 256, 256),

            # pool3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv4
            CnnBlock(6, 256, 512),
            CnnBlock(7, 512, 512),

            # pool4
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),

            # conv5
            CnnBlock(8, 512, 512),
            CnnBlock(9, 512, 512)

        )

        self.block2 = nn.Sequential(

            nn.Flatten(),  # need to convert 2d to 1d

        )

        self.block3 = nn.Sequential(
            LinearBlock(168960, 256),  # 512*15*22
            LinearBlock(256, 128),
            LinearBlock(128, 64),
            nn.Linear(64, 1),  # binary classes
            nn.Sigmoid()
        )

    def forward(self, x, i):
        if i == 0:
            logging.info(f'x:{x.shape}')

        block1out = self.block1(x)

        if i == 0:
            logging.info(f'block1out:{block1out.shape}')
        block2out = self.block2(block1out)

        if i == 0:
            logging.info(f'block2out:{block2out.shape}')

        block3out = self.block3(block2out)

        if i == 0:
            logging.info(f'block3out:{block3out.shape}')

        return block3out


def _generate_channels_lists(in_channels, block_pattern, block_depth):
    in_channels_list = []
    out_channels_list = []

    # calculate channels sizes for block based on block pattern
    current_in_channels = in_channels
    current_out_channels = None
    if block_pattern == 'single_run':

        # out_channel is 2x the in_channel of that layer
        # for block_depth = 3
        # in_channels_list  [ 64, 128, 256]
        # out_channels_list [128, 256, 512]
        for each in range(block_depth):
            # in channels
            in_channels_list.append(current_in_channels)  # match output channels of previous layer
            current_out_channels = current_in_channels * 2  # output is 2x the input

            # out channels
            out_channels_list.append(current_out_channels)
            current_in_channels = current_out_channels

    elif block_pattern == 'double_run':

        # odd layers have in_channels and out_channels that are the same value
        # even layers have out_channel = 2 * in_channel
        # for block_depth = 4
        # in_channels_list  [64,  64, 128, 128]
        # out_channels_list [64, 128, 128, 256]
        current_in_channels = in_channels
        current_out_channels = current_in_channels
        is_symmetrical_layer = True
        for each in range(block_depth):

            # in channels
            in_channels_list.append(current_in_channels)  # match output channels of previous layer

            if is_symmetrical_layer:
                current_out_channels = current_in_channels
            else:
                current_out_channels = current_in_channels * 2

            # toggle opposite rule for next layer
            is_symmetrical_layer = not is_symmetrical_layer

            # out channels
            out_channels_list.append(current_out_channels)
            current_in_channels = current_out_channels

    return in_channels_list, out_channels_list


class FcnBlock(nn.Module):

    def __init__(self, block_number, in_channels, block_depth, block_pattern,
                 kernel_size=3, stride=1, padding=1):
        super().__init__()

        # calculate sizes for input and output channels for all cnn layers in this block
        in_channels_list, out_channels_list = _generate_channels_lists(in_channels, block_pattern, block_depth)
        #         print(in_channels_list, out_channels_list)

        self.in_channels = in_channels
        self.block_number = block_number
        self.out_channels = out_channels_list[-1]  # last out channel of block

        # build block
        self.fcn_block = nn.Sequential()
        self.fcn_block.add_module('pool' + str(block_number), nn.MaxPool2d(kernel_size=2, stride=2, padding=1))

        # build block one group at a time
        for i, curr_in_channels in enumerate(in_channels_list):
            curr_out_channels = out_channels_list[i]
            self.fcn_block.add_module('cnn' + str(block_number) + '_' + str(i),
                                      nn.Conv2d(curr_in_channels, curr_out_channels,
                                                kernel_size=kernel_size, stride=stride,
                                                padding=padding, bias=False))

            self.fcn_block.add_module('bn' + str(block_number) + '_' + str(i), nn.BatchNorm2d(curr_out_channels))
            self.fcn_block.add_module('relu' + str(block_number) + '_' + str(i), nn.ReLU(inplace=True))

        # initialize weights
        self.fcn_block.apply(_init_weights)

    def forward(self, x):
        # logging.info(f'fcn_block_inputx:{x.shape}')
        x = self.fcn_block(x)
        # logging.info(f'fcn_block_input:{x.shape}')
        return x


# TODO: allow prob block to have custom out channels for first cnn layer
# TODO: upsampling allow different input and output channel sizes
# blocks start with pool so output of block can be concatenated, first block doesn't have pool
# DCAN block_pattern: if 6 blocks, blocks 4 and 5 are the same channels

# ZhengFCN: concatenates upsampling channels before calculating the 2 class label probabilities
# DCAN_FCN: calculates 2 class label probabilities for each upsampling channel, then sums probabilities
class ConcatenationFCN(nn.Module):
    def __init__(self, input_block_depth=1, num_fcn_blocks=3, fcn_block_depth=1, input_channels=3, output_channels=2,
                 first_layer_out_channels=64,
                 block_pattern='single_run', upsampling_pattern='last_three', original_height=224, original_width=332):
        """

        Args:
            num_fcn_blocks:  4,5,6 number of cnn blocks in network
            fcn_block_depth: 1, 2, 3, 4 number of cnn layers in a block
            input_channels: 3 channels in original image
            output_channels: 2 number of classes to softmax
            first_layer_out_channels: 64
            block_pattern: single_run, double_run, dcan_run,
            upsampling_pattern: last_three, last_two
            original_height: 224 upsampling to restore image to this size
            original_width: 332 upsampling to restore image to this size
        """
        super().__init__()

        self.block_pattern = block_pattern
        self.upsampling_pattern = upsampling_pattern

        # add input block
        block_number = 0
        self.input_block = nn.Sequential()
        curr_in_channels = input_channels
        curr_out_channels = first_layer_out_channels

        for i in range(input_block_depth):
            cnn_block = CnnBlock(block_number, curr_in_channels, curr_out_channels)
            self.input_block.add_module('cnn' + str(i), cnn_block)

            # all remaining cnn blocks in the input block have matching input and output channels
            curr_in_channels = cnn_block.out_channels

        # add FCN blocks
        fcn_blocks = []
        curr_in_channels = first_layer_out_channels

        for n in range(1, num_fcn_blocks + 1):
            # create block
            block_number = n
            block = FcnBlock(block_number, curr_in_channels, fcn_block_depth, block_pattern)
            fcn_blocks.append(block)

            # update settings for next block
            curr_in_channels = block.out_channels

        # subdivide fcn blocks based on connections to upsampling blocks
        self.fcn1 = None
        self.fcn2 = None
        self.fcn3 = None

        if upsampling_pattern in ['last_three']:
            # three fcn blocks
            self.fcn1 = nn.ModuleList(fcn_blocks[:num_fcn_blocks - 2])
            self.fcn2 = nn.ModuleList([fcn_blocks[-2]])
            self.fcn3 = nn.ModuleList([fcn_blocks[-1]])

        # add upsampling blocks
        self.up1 = None
        self.up2 = None
        self.up3 = None

        block_number += 1
        size = (original_height, original_width)
        concatenated_channels = 0

        if upsampling_pattern == 'last_three':
            self.up1 = UpConvBlock(self.fcn1[-1].out_channels, self.fcn1[-1].out_channels, size)
            self.up2 = UpConvBlock(self.fcn2[-1].out_channels, self.fcn2[-1].out_channels, size)
            self.up3 = UpConvBlock(self.fcn3[-1].out_channels, self.fcn3[-1].out_channels, size)

            # calculate concatenated_channels
            concatenated_channels += self.up1.out_channels
            concatenated_channels += self.up2.out_channels
            concatenated_channels += self.up3.out_channels

        # create probability block
        block_number += 1
        self.map_block = nn.Sequential(
            CnnBlock(block_number, concatenated_channels, concatenated_channels),
            nn.Conv2d(concatenated_channels, output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            # 2 classes
            nn.Softmax2d()
        )

    def forward(self, x, i):

        x = self.input_block(x)

        # perform forward pass on fcn and up blocks at the same time to avoid needing extra copies of fcn output
        if self.upsampling_pattern in ['last_three']:

            for fcn in self.fcn1:  # ModuleList requires iteration, can't do forward pass directly
                x = fcn(x)
            up1 = self.up1(x)

            for fcn in self.fcn2:  # ModuleList requires iteration, can't do forward pass directly
                x = fcn(x)
            up2 = self.up2(x)

            for fcn in self.fcn3:  # ModuleList requires iteration, can't do forward pass directly
                x = fcn(x)
            up3 = self.up3(x)

            up_tuple = (up1, up2, up3)

        else:
            raise NotImplementedError

        # concatenate upsampling output
        x = torch.cat(up_tuple, dim=1)  # channels are the second dimension

        x = self.map_block(x)

        return x


def _generate_linear_size_lists(in_size, linear_pattern, block_depth):
    in_sizes_list = []
    out_sizes_list = []

    # calculate channels sizes for block based on block pattern
    current_in_size = in_size
    current_out_size = current_in_size
    if linear_pattern == 'single_run':

        # out size is half the in size of that layer
        # for block_depth = 3
        # in_channels_list  [ 64, 32, 16]
        # out_channels_list [ 32, 16, 8]
        for each in range(block_depth):
            # in size
            in_sizes_list.append(current_in_size)  # match output channels of previous layer
            current_out_size = int(current_out_size / 2)  # output is half the input

            # out size
            out_sizes_list.append(current_out_size)
            current_in_size = current_out_size

    return in_sizes_list, out_sizes_list


def _calc_conversion_feature_size(num_fcn_blocks, depth_fcn_block, first_layer_out_channels, fcn_block_pattern):
    """
    This is a temporary workaround because LazyLinear allocates way too much memory when performing the dummy forward() pass to determine in_features.
    Args:
        num_fcn_blocks:
        depth_fcn_block:
        first_layer_out_channels:
        fcn_block_pattern:

    Returns:

    """
    width_times_height = 0
    if first_layer_out_channels == 64:
        if fcn_block_pattern == 'single_run':
            if depth_fcn_block == 1:
                if num_fcn_blocks == 1:
                    width_times_height = 113 * 167
                elif num_fcn_blocks == 2:
                    width_times_height = 57 * 84
                elif num_fcn_blocks == 3:
                    width_times_height = 29 * 43
        elif fcn_block_pattern == 'double_run':
            if depth_fcn_block == 2:
                if num_fcn_blocks == 4:
                    width_times_height = 15 * 22
                elif num_fcn_blocks == 5:
                    width_times_height = 8 * 12
                elif num_fcn_blocks == 6:
                    width_times_height = 5 * 7

    channels = 0
    if first_layer_out_channels == 64:
        if fcn_block_pattern == 'single_run':
            if depth_fcn_block == 1:
                if num_fcn_blocks == 1:
                    channels = 128
                elif num_fcn_blocks == 2:
                    channels = 256
                elif num_fcn_blocks == 3:
                    channels = 512
            elif depth_fcn_block == 2:
                if num_fcn_blocks == 1:
                    channels = 256
                elif num_fcn_blocks == 2:
                    channels = 1024
                elif num_fcn_blocks == 3:
                    channels = 4096
        elif fcn_block_pattern == 'double_run':
            if depth_fcn_block == 2:
                if num_fcn_blocks == 4:
                    channels = 1024
                elif num_fcn_blocks == 5:
                    channels = 2048
                elif num_fcn_blocks == 6:
                    channels = 4096

    return channels * width_times_height


class FlexVGG(nn.Module):
    def __init__(self, input_block_depth=1, num_fcn_blocks=1, depth_fcn_block=1, input_channels=4,
                 first_layer_out_channels=64,
                 fcn_block_pattern='single_run', depth_linear_block=1, linear_block_pattern='single_run',
                 first_linear_layer_out_features=64, out_features=1):
        super().__init__()

        self.fcn_block_pattern = fcn_block_pattern

        # add input block
        block_number = 0
        self.input_block = nn.Sequential()
        curr_in_channels = input_channels
        curr_out_channels = first_layer_out_channels

        for i in range(input_block_depth):
            cnn_block = CnnBlock(block_number, curr_in_channels, curr_out_channels)
            self.input_block.add_module('cnn' + str(i), cnn_block)

            # all remaining cnn blocks in the input block have matching input and output channels
            curr_in_channels = cnn_block.out_channels

        # add fcn block
        fcn_blocks = []
        curr_in_channels = first_layer_out_channels

        for i in range(1, num_fcn_blocks + 1):
            # create block
            block_number = i
            block = FcnBlock(block_number, curr_in_channels, depth_fcn_block, fcn_block_pattern)
            fcn_blocks.append(block)

            # update settings for next block
            curr_in_channels = block.out_channels

        self.fcn_block = nn.ModuleList(fcn_blocks)

        self.flatten_block = nn.Flatten()

        # add linear block to convert between 2d fcn output and 1d classifier block
        # input size after flattening is difficult to calculate, so use a lazy linear layer to calculate for you
        conversion_features = _calc_conversion_feature_size(num_fcn_blocks, depth_fcn_block, first_layer_out_channels,
                                                            fcn_block_pattern)
        logging.info(f'conversion_features:{conversion_features}')
        self.conversion_block = nn.Sequential()
        self.conversion_block.add_module('linear', nn.Linear(conversion_features, first_linear_layer_out_features))
        self.conversion_block.add_module('bn', nn.BatchNorm1d(first_linear_layer_out_features))
        self.conversion_block.add_module('activation', nn.ReLU(inplace=True))

        # build classifier block
        # calculate linear in and out sizes according to the
        in_sizes_list, out_sizes_list = _generate_linear_size_lists(first_linear_layer_out_features,
                                                                    linear_block_pattern, depth_linear_block)

        classifier_block = []
        for i in range(depth_linear_block):
            in_size = in_sizes_list[i]
            out_size = out_sizes_list[i]
            lb = LinearBlock(in_size, out_size)
            classifier_block.append(lb)

        # add final linear layer that classifies input into binary classes
        # final linear layer does not have batch normalization
        # final linear layer uses sigmoid activation function
        lb = LinearBlock(out_sizes_list[-1], out_features, batchnorm=False, activation='sigmoid')
        classifier_block.append(lb)

        self.classifier_block = nn.ModuleList(classifier_block)

    def forward(self, x, i):
        if i == 0:
            logging.info(f'x:{x.shape}')

        # input block
        x = self.input_block(x)

        if i == 0:
            logging.info(f'input_block:{x.shape}')

        # fcn blocks
        for fcn in self.fcn_block:
            x = fcn(x)

            if i == 0:
                logging.info(f'fcn:{x.shape}')

        x = self.flatten_block(x)

        if i == 0:
            logging.info(f'flatten_block:{x.shape}')

        x = self.conversion_block(x)

        if i == 0:
            logging.info(f'conversion_block:{x.shape}')

        for lin in self.classifier_block:
            x = lin(x)

            if i == 0:
                logging.info(f'lin:{x.shape}')

        return x
