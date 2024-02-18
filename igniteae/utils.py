"""Utils.
"""
import logging
import torch


def loader_test(loader):
    """For testing a loader.
    """

    for i, data in enumerate(loader):
        batch, label = data
        print('batch size', batch.shape)
        print(label)

        if i == 0:
            break


def set_device(gpu: bool):
    """Sets torch device to gpu if possible.
    """
    if gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            logging.error('CUDA is not available! Switching to "CPU"')
            device = torch.device('cpu')
    else:
        # cpu by default
        device = torch.device('cpu')
    return device
