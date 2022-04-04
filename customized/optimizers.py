"""
All things related to optimizers.
"""
__author__ = 'ryanquinnnelson'

import logging

import torch.optim as optim


# TODO: Add additional hyperparameters to optimizer options
# TODO: Add additional optimizers as options
class OptimizerHandler:
    """
    Defines object to handle initializing optimizers.
    """

    def __init__(self):
        """
        Initialize OptimizerHandler.
        """
        logging.info('Initializing optimizer handler...')

    def get_optimizers(self, models, wandb_config):
        """
        For each model, define an optimizer based on parameters.

        Args:
            models (Collection[torch.nn.Module]): Models for which optimizers will be defined
            wandb_config (wandb.config):  Object which contains configuration in a key.value format.

        Returns: (Collection[torch.optim], Collection[String]) representing (optimizers, optimizer names)

        """

        optimizers = []
        optimizer_names = ['sn_optimizer', 'en_optimizer']
        model_type = None
        for i, model in enumerate(models):
            if i == 0:
                model_type = 'sn'
            elif i == 1:
                model_type = 'en'
            opt = self.get_optimizer(model, wandb_config, model_type)
            optimizers.append(opt)

        return optimizers, optimizer_names

    def get_optimizer(self, model, wandb_config, model_type):
        """
        Obtain the optimizer based on parameters.
        Args:
            model (nn.Module): model optimizer will manage
            wandb_config (wandb.config):  Object which contains configuration in a key.value format.

        Returns: nn.optim optimizer
        """
        opt = None
        lr = None
        if model_type == 'sn':
            lr = wandb_config.sn_lr
        elif model_type == 'en':
            lr = wandb_config.en_lr

        if wandb_config.optimizer_type == 'adam':
            opt = optim.Adam(model.parameters(), lr=lr)

        elif wandb_config.optimizer_type == 'sgd':
            opt = optim.SGD(model.parameters(), lr=lr)

        logging.info(f'Optimizer initialized:\n{opt}')
        logging.info(f'LR={opt.state_dict()["param_groups"][0]["lr"]}')  # to ensure function works during training
        return opt
