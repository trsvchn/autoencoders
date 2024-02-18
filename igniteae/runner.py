import logging
import random

import torch

import models
from dataloaders import prepare_loaders
from train import train
from valid import validate
from utils import set_device

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main(data_dir: str = './data',
         epochs: int = 100,
         print_every: int = 1,
         gpu: bool = False,
         tbs: int = 64,
         vbs: int = 64,
         reshuffle_data: bool = False,
         nworkers: int = 2,
         lr: float = 0.001,
         wd: float = 1e-05,
         seed: int = 42,
         **kwargs):

    # Planting random seeds
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Setting device
    logging.info('Setting device...')
    device = set_device(gpu)
    logging.info(f'Training on {str(device).upper()}.')

    # Getting data ready..
    logging.info('Preparing dataloaders...')
    train_loader, val_loader = prepare_loaders(data_dir, tbs, vbs, reshuffle_data, nworkers)
    logging.info('Data is ready!')

    # Initilize model, loss, optimizer
    logging.info(f'Initializing model FooBar...')
    model = models.FooBar().to(device)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Train/val loop
    logging.info('Starting training...')
    for epoch in range(0, epochs):
        train_loss = train(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        # Print some info on training process
        if (epoch % print_every == print_every - 1) or (epoch == 0):
            print(f'[{epoch + 1}/{epochs}] lr: {lr} | train_loss: {train_loss:.3f} | val_loss: {val_loss:.3f}')
    logging.debug('Training has finished!')
