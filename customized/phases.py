"""
All things related to training, validation, and testing phases.
"""

import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from customized.losses import TripletLoss
from customized.optimizers import OptimizerHandler


class PhaseHandler:
    """
    Defines object to initialize phases.
    """

    def get_train_phase(self, devicehandler, train_loader, wandb_config):
        """
        Obtain a Training object based on given parameters.
        Args:
            devicehandler (DeviceHandler): defines torch.device
            train_loader (DataLoader): DataLoader object for the training dataset
            wandb_config (wandb.config): Object which contains configuration in a key.value format.

        Returns: Training object

        """
        training_phase = Training(devicehandler, train_loader, wandb_config)

        return training_phase

    def get_val_phase(self, devicehandler, val_loader, wandb_config):
        """
        Obtain a Validation object based on given parameters.
        Args:
            devicehandler (DeviceHandler): defines torch.device
            val_loader (DataLoader): DataLoader object for the validation dataset
            wandb_config (wandb.config): Object which contains configuration in a key.value format.

        Returns: Validation object

        """
        validation_phase = Validation(devicehandler, val_loader, wandb_config)

        return validation_phase

    def get_test_phase(self, devicehandler, test_loader, wandb_config, output_dir):
        """
        Obtain a Testing object based on given parameters.
        Args:
            devicehandler (DeviceHandler): defines torch.device
            test_loader (DataLoader): DataLoader object for the testing dataset
            wandb_config (wandb.config): Object which contains configuration in a key.value format.

        Returns: Testing object

        """
        testing_phase = Testing(devicehandler, test_loader, wandb_config, output_dir)
        return testing_phase


def _get_criterion(criterion_type):
    criterion = None
    if criterion_type == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    elif criterion_type == 'BCELoss':
        criterion = nn.BCELoss()
    return criterion


def _calculate_num_hits(i, targets, out):
    # convert out to class labels
    labels_out = out.argmax(axis=1)
    if i == 0:
        logging.info(f'labels_out.shape:{labels_out.shape}')

    # count the total number of matches between predictions and actual
    compare = targets == labels_out
    n_hits = np.sum(compare.cpu().detach().numpy())

    if i == 0:
        logging.info(f'n_hits:{n_hits}')

    return n_hits


# https://towardsdatascience.com/intersection-over-union-iou-calculation-for-evaluating-an-image-segmentation-model-8b22e2e84686
def _calculate_iou_score(i, targets, out):
    targets = targets.cpu().detach().numpy()

    # convert to class labels
    # convert out to class labels
    labels_out = out.argmax(axis=1)
    labels_out = labels_out.cpu().detach().numpy()

    intersection = np.logical_and(targets, labels_out)
    union = np.logical_or(targets, labels_out)

    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def _combine_input_and_map(input, map):
    combined = torch.cat((input, map), dim=1)
    return combined


def _en_loss(pred, criterion, annotated=True):
    n = len(pred)

    if annotated:
        targets = torch.ones(n)  # targets should be 1.0
    else:
        targets = torch.zeros(n)  # targets should be 0.0

    loss = criterion(pred.squeeze(-1).cpu(), targets)  # make same dimensions

    return loss


class Training:
    """
    Defines object to run a training phase.
    """

    def __init__(self, devicehandler, dataloader, wandb_config):
        """
        Initialize Training object.

        Args:
            devicehandler (DeviceHandler): defines torch.device
            dataloader (DataLoader): DataLoader object for the training dataset
            wandb_config (wandb.config): Object which contains configuration in a key.value format.
        """
        self.devicehandler = devicehandler
        self.dataloader = dataloader

        self.sn_criterion = _get_criterion(wandb_config.sn_criterion)
        self.en_criterion = _get_criterion(wandb_config.en_criterion)
        self.use_gan = wandb_config.use_gan
        self.en_loss_weight = wandb_config.en_loss_weight
        self.en_loss_increment = wandb_config.en_loss_increment
        self.gan_start_epoch = wandb_config.gan_start_epoch

        self.n_triplets = wandb_config.n_triplets
        self.margin = wandb_config.margin
        self.pretrain_with_triplet_loss = wandb_config.pretrain_with_triplet_loss
        self.n_pretraining_epochs = wandb_config.n_pretraining_epochs
        self.wandbconfig = wandb_config

        if self.pretrain_with_triplet_loss:
            logging.info(f'Criterion for training phase:' +
                         f'\ngenerator:{self.sn_criterion} + triplet_loss\ndiscriminator:{self.en_criterion}')
        else:
            logging.info(f'Criterion for training phase:' +
                         f'\ngenerator:{self.sn_criterion}\ndiscriminator:{self.en_criterion}')

    def run_epoch(self, epoch, num_epochs, models, optimizers):
        """
        Run training epoch.

        Args:
            epoch (int): Epoch being trained
            num_epochs (int): Total number of epochs to be trained
            models (Collection[nn.Module]): models being trained
            optimizers (Collection[nn.optim]): optimizers matching each model

        Returns: Dict of training stats

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of training...')

        # track losses
        total_sn_train_loss = 0
        total_sn_train_mce_loss = 0
        total_sn_train_triplet_loss = 0
        total_en_train_loss_unannotated = 0
        total_en_train_loss_annotated = 0
        total_en_train_loss = 0

        # extract models and optimizers
        sn_model = models[0]
        en_model = models[1]
        sn_optimizer = optimizers[0]
        en_optimizer = optimizers[1]

        # Set model in 'Training mode'
        sn_model.train()
        en_model.train()

        # reset optimizers after pretraining
        if self.pretrain_with_triplet_loss and epoch == self.n_pretraining_epochs + 1:
            optimizers, optimizer_names = OptimizerHandler().get_optimizers(models, self.wandbconfig)
            sn_optimizer = optimizers[0]
            en_optimizer = optimizers[1]

        # process mini-batches
        for i, (inputs, targets) in enumerate(self.dataloader):

            # prep
            sn_optimizer.zero_grad()
            torch.cuda.empty_cache()
            inputs, targets = self.devicehandler.move_data_to_device(sn_model, inputs, targets)

            # compute forward pass on generator
            out = sn_model.forward(inputs, i)

            if i == 0:
                logging.info(f'inputs.shape:{inputs.shape}')
                logging.info(f'targets.shape:{targets.shape}')
                logging.info(f'out.shape:{out.shape}')

            # calculate generator loss
            if self.pretrain_with_triplet_loss and epoch <= self.n_pretraining_epochs:
                out_1d = out[:, 0, :, :]  # keep only class that indicates segment label
                if i == 0:
                    logging.info('Calculating triplet loss...')
                    logging.info(f'out_1d.shape:{out_1d.shape}')
                # out_1d = torch.unsqueeze(out_1d, dim=1)  # triplet loss expects (B,F,H,W) dimensions
                sn_triplet_loss = TripletLoss().calculate_loss(out_1d, targets, self.margin, self.n_triplets)

                # track total
                total_sn_train_triplet_loss += sn_triplet_loss.item()

                # only triplet loss is used
                sn_loss = sn_triplet_loss

            else:
                if i == 0:
                    logging.info('Calculating mce loss...')
                # cross entropy loss
                sn_mce_loss = self.sn_criterion(out, targets)
                total_sn_train_mce_loss += sn_mce_loss.item()

                # cross entropy loss is the only SN loss
                sn_loss = sn_mce_loss
                # sn_loss = sn_mce_loss + self.triplet_loss_weight * sn_triplet_loss

                # check if gan process should be run
                if self.use_gan and epoch >= self.gan_start_epoch:
                    # run gan process
                    losses = self.run_gan(i, epoch, inputs, out, targets, en_model, en_optimizer, sn_loss)

                    # unpack losses
                    sn_loss, en_loss_unannotated, en_loss_annotated, en_loss = losses

                    # append discriminator losses to running totals
                    total_en_train_loss_unannotated += en_loss_unannotated.item()
                    total_en_train_loss_annotated += en_loss_annotated.item()
                    total_en_train_loss += en_loss.item()

            # compute backward pass of generator
            sn_loss.backward()

            # update generator weights
            sn_optimizer.step()

            # delete mini-batch data from device
            del inputs
            del targets

            # append generator losses to running totals
            total_sn_train_loss += sn_loss.item()

        # calculate average loss across all mini-batches
        n_mini_batches = len(self.dataloader)

        # generator
        total_sn_train_loss /= n_mini_batches
        total_sn_train_triplet_loss /= n_mini_batches
        total_sn_train_mce_loss /= n_mini_batches

        # discriminator
        total_en_train_loss /= n_mini_batches
        total_en_train_loss_unannotated /= n_mini_batches
        total_en_train_loss_annotated /= n_mini_batches

        # build stat dictionary
        sn_lr = sn_optimizer.state_dict()["param_groups"][0]["lr"]
        en_lr = en_optimizer.state_dict()["param_groups"][0]["lr"]
        stats = {'g_train_loss': total_sn_train_loss,
                 'g_train_triplet_loss': total_sn_train_triplet_loss,
                 'g_train_mce_loss': total_sn_train_mce_loss,
                 'd_train_loss': total_en_train_loss,
                 'd_train_loss_unannotated': total_en_train_loss_unannotated,
                 'd_train_loss_annotated': total_en_train_loss_annotated,
                 'g_lr': sn_lr,
                 'd_lr': en_lr}

        return stats

    def run_gan(self, i, epoch, inputs, out, targets, en_model, en_optimizer, sn_loss):

        # select subset of mini-batch to be unannotated vs annotated at random
        unannotated_idx = np.random.choice(len(inputs), size=int(len(inputs) / 2), replace=False)
        annotated_idx = np.delete(np.array([k for k in range(len(inputs))]), unannotated_idx)

        # 1 - compute forward pass on discriminator using unannotated data
        # combine inputs and probability map
        unannotated_inputs = inputs[unannotated_idx]  # (B, C, H, W)
        unannotated_out = out[unannotated_idx, 0, :, :]  # keep 1 class to match inputs + targets shape, get (B, H, W)
        en_input = _combine_input_and_map(unannotated_inputs, unannotated_out.unsqueeze(1))  # unsqueeze to match inputs

        # forward pass
        pred = en_model(en_input.detach(), i)  # detach to not affect generator?

        # calculate loss
        en_loss_unannotated = _en_loss(pred, self.en_criterion, annotated=False)

        # 2 - compute forward pass on discriminator using annotated data
        # combine inputs and probability map
        annotated_inputs = inputs[annotated_idx]  # (B, C, H, W)
        annotated_targets = targets[annotated_idx]  # (B, H, W) targets only has a single class
        en_input = _combine_input_and_map(annotated_inputs, annotated_targets.unsqueeze(1))  # unsqueeze to match inputs

        # forward pass
        pred = en_model(en_input.detach(), i)  # detach to not affect generator?

        # calculate loss
        en_loss_annotated = _en_loss(pred, self.en_criterion, annotated=True)

        # 3 - update discriminator based on loss
        # calculate total discriminator loss for unannotated and annotated data
        en_loss = en_loss_unannotated + en_loss_annotated
        en_loss.backward()
        en_optimizer.step()

        # 4 - compute forward pass on updated discriminator using only unannotated data for calculating generator loss
        # combine inputs and probability map
        # can I use all output here or only the ones selected for unannotation?
        unannotated_out = out[:, 0, :, :]  # keep 1 class to match inputs + targets shape, get (B, H, W)
        en_input = _combine_input_and_map(inputs, unannotated_out.unsqueeze(1))  # unsqueeze to match inputs

        # forward pass
        pred = en_model(en_input, i)  # leave attached so backpropagation through discriminator affects generator

        # calculate generator loss based on discriminator predictions
        # if discriminator predicts unannotated correctly, generator not doing good enough job
        en_loss_weight = self.en_loss_weight
        en_loss_weight += (epoch / self.en_loss_increment)  # add more weight to EN each time
        en_loss = _en_loss(pred, self.en_criterion, annotated=True)
        total_sn_loss = sn_loss + en_loss_weight * en_loss

        return total_sn_loss, en_loss_unannotated, en_loss_annotated, en_loss


class Validation:
    """
    Defines object to run a validation phase.
    """

    def __init__(self, devicehandler, dataloader, wandb_config):
        """
        Initialize Validation object.

        Args:
            devicehandler (DeviceHandler): defines torch.device
            dataloader (DataLoader): DataLoader object for the validation dataset
            wandb_config (wandb.config): Object which contains configuration in a key.value format.
        """

        self.devicehandler = devicehandler
        self.dataloader = dataloader
        self.sn_criterion = _get_criterion(wandb_config.sn_criterion)
        self.use_gan = wandb_config.use_gan
        self.pretrain_with_triplet_loss = wandb_config.pretrain_with_triplet_loss
        self.n_pretraining_epochs = wandb_config.n_pretraining_epochs

        logging.info(f'Criterion for validation phase:\ngenerator:{self.sn_criterion}')

    def run_epoch(self, epoch, num_epochs, models):
        """
        Run validation epoch.

        Args:
            epoch (int): Epoch being validated
            num_epochs (int): Total number of epochs to be validated
            models (Collection[nn.Module]): models being validated

        Returns: Dict of validation stats

        """
        logging.info(f'Running epoch {epoch}/{num_epochs} of evaluation...')

        total_val_loss = 0
        val_mce_loss = 0
        total_hits = 0
        total_iou_score = 0
        n_correct_predictions = 0
        out_shape = None  # save for calculating total number of pixels per image
        total_inputs = 0

        # unpack models
        sn_model = models[0]
        en_model = models[1]

        with torch.no_grad():  # deactivate autograd engine to improve efficiency

            # Set model in validation mode
            sn_model.eval()
            en_model.eval()

            # process mini-batches
            for i, (inputs, targets) in enumerate(self.dataloader):
                # logging.info(f'validation batch:{i}')
                total_inputs += len(inputs)

                # prep
                inputs, targets = self.devicehandler.move_data_to_device(sn_model, inputs, targets)

                # compute forward pass on generator
                out = sn_model.forward(inputs, i)
                out_shape = out.shape

                if i == 0:
                    logging.info(f'inputs.shape:{inputs.shape}')
                    logging.info(f'targets.shape:{targets.shape}')
                    logging.info(f'out.shape:{out.shape}')

                if epoch > self.n_pretraining_epochs:

                    # cross entropy loss
                    sn_mce_loss = self.sn_criterion(out, targets)
                    val_mce_loss += sn_mce_loss.item()

                    # combine losses
                    sn_loss = sn_mce_loss
                    total_val_loss += sn_loss.item()

                    if self.use_gan:
                        # compute forward pass on discriminator
                        # select subset of mini-batch to be unannotated vs annotated at random
                        unannotated_idx = np.random.choice(len(inputs), size=int(len(inputs) / 2), replace=False)
                        annotated_idx = np.delete(np.array([k for k in range(len(inputs))]), unannotated_idx)

                        # 1 - compute forward pass on discriminator using unannotated data
                        # combine inputs and probability map
                        unannotated_inputs = inputs[unannotated_idx]  # (B, C, H, W)
                        unannotated_out = out[unannotated_idx, 0, :,
                                          :]  # 1 class to match inputs + targets shape, (B, H, W)
                        en_input = _combine_input_and_map(unannotated_inputs,
                                                          unannotated_out.unsqueeze(1))  # unsqueeze to match inputs
                        # forward pass
                        pred = en_model(en_input, i)

                        # count number of predictions that accurately predict unannotated
                        n_correct_predictions += torch.sum(
                            pred < 0.5).item()  # en_model should predict 0 for unannotated

                        if i == 0:
                            logging.info(f'fake pred:{pred.detach()}')
                            logging.info(f'n_correct_predictions:{n_correct_predictions}')

                        # 2 - compute forward pass on discriminator using annotated data
                        # combine inputs and probability map
                        annotated_inputs = inputs[annotated_idx]  # (B, C, H, W)
                        annotated_targets = targets[annotated_idx]  # (B, H, W) targets only has a single class
                        en_input = _combine_input_and_map(annotated_inputs,
                                                          annotated_targets.unsqueeze(1))  # unsqueeze to match inputs

                        # forward pass
                        pred = en_model(en_input, i)

                        # count number of predictions that accurately predict unannotated
                        n_correct_predictions += torch.sum(pred >= 0.5).item()  # d_model should predict 1 for annotated

                        if i == 0:
                            logging.info(f'real pred:{pred.detach()}')
                            logging.info(f'n_correct_predictions:{n_correct_predictions}')

                # calculate accuracy
                total_hits += _calculate_num_hits(i, targets, out)
                total_iou_score += _calculate_iou_score(i, targets, out)

                # delete mini-batch from device
                del inputs
                del targets

            # calculate average generator evaluation metrics per mini-batch
            n_mini_batches = len(self.dataloader)
            pixels_per_image = out_shape[2] * out_shape[3]
            possible_hits = total_inputs * pixels_per_image
            val_acc = total_hits / possible_hits
            total_iou_score /= n_mini_batches
            discriminator_acc = n_correct_predictions / total_inputs

            total_val_loss /= n_mini_batches
            val_mce_loss /= n_mini_batches

            # build stats dictionary
            stats = {'val_loss': total_val_loss,
                     'val_mce_loss': val_mce_loss,
                     'val_acc': val_acc,
                     'val_iou_score': total_iou_score,
                     'discriminator_acc': discriminator_acc}

            return stats


# TODO: format and save output
class Testing:
    def __init__(self, devicehandler, dataloader, wandb_config, output_dir):
        """
        Initialize Validation object.

        Args:
            devicehandler (DeviceHandler): defines torch.device
            dataloader (DataLoader): DataLoader object for the validation dataset
            wandb_config (wandb.config): Object which contains configuration in a key.value format.
            output_dir (str): Directory where output should be written
        """
        self.devicehandler = devicehandler
        self.output_dir = output_dir
        self.dataloader = dataloader
        self.run_name = wandb_config.run_name

    def run_epoch(self, epoch, num_epochs, models):
        """
        Run test epoch.

        Args:
            epoch (int): Epoch being tested
            num_epochs (int): Total number of epochs to be tested
            models (Collection[nn.Module]): models being tested

        Returns: Dict of test stats

        """

        logging.info(f'Running epoch {epoch}/{num_epochs} of testing...')

        if epoch == num_epochs:  # perform this step on the last epoch only

            sn_model = models[0]
            count = 0
            with torch.no_grad():  # deactivate autograd engine to improve efficiency

                # Set model in validation mode
                sn_model.eval()

                # process mini-batches
                for i, (inputs, targets) in enumerate(self.dataloader):
                    # prep
                    inputs, targets = self.devicehandler.move_data_to_device(sn_model, inputs, targets)

                    # compute forward pass
                    out = sn_model.forward(inputs, i)

                    # convert two channels into single output label
                    # convert datatype to a type that pillow can use
                    labels_out = out.argmax(axis=1).to(torch.float32)

                    # format and save output
                    for j, each in enumerate(labels_out):
                        # convert tensor to image
                        t = transforms.ToPILImage()
                        img = t(each)

                        # save image
                        filepath = os.path.join(self.output_dir,
                                                'output.' + self.run_name + '.epoch.' + str(epoch).zfill(
                                                    3) + '.img.' + str(
                                                    count).zfill(2) + '.bmp')
                        img.save(filepath)
                        count += 1

        return {}  # empty dictionary
