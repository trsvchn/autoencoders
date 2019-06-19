"""Utils.
"""


def loader_test(loader):
    """For testing a loader.
    """

    for i, data in enumerate(loader):
        batch, label = data
        print('batch size', batch.shape)
        print(label)

        if i == 0:
            break
