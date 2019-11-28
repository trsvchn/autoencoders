import logging
import random
import argparse

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


def args_parser() -> dict:
    parser = argparse.ArgumentParser(description='Autoencoder Training.')

    parser.add_argument('--data_dir', default='./data', type=str, metavar='PATH',
                        help='Path to data. Default: ./data')

    parser.add_argument('-s', '--seed', default=42, type=int, metavar='SEED',
                        help='Random seed. Default: 42')

    parser.add_argument('-j', '--nworkers', default=2, type=int, metavar='N',
                        help='Number of data loading workers. Default: 2')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='Number of epochs to run. Default: 100')
    parser.add_argument('-p', '--print-every', default=1, type=int, metavar='N',
                        help='Print frequency (in number of epoch). Default: 1')

    parser.add_argument('-b', '--tbs', default=64, type=int, metavar='N',
                        help='Training batch size. Default: 64')
    parser.add_argument('-vb', '--vbs', default=64, type=int, metavar='N',
                        help='Validation batch size. Default: 64')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR',
                        help='Initial learning rate. Default: 0.001')
    parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float, metavar='WD',
                        help='Weight decay. Default: 1e-5')

    parser.add_argument('-g', '--gpu', action='store_true',
                        help='Use GPU. Default: False')

    parser.add_argument('--reshuffle_data', action='store_true',
                        help='Reshuffle data at every epoch. Default: False')

    args = vars(parser.parse_args())

    return args


if __name__ == '__main__':
    args_ = args_parser()
    logging.info('INPUT PARAMS:\n' + '\n'.join([': '.join([k, str(v)]) for k, v in args_.items()]))
    main(**args_)
