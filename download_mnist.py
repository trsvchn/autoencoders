from torchvision.datasets import MNIST


data_path = './data'


def main():
    MNIST(data_path, train=True, download=True)


if __name__ == '__main__':
    main()
