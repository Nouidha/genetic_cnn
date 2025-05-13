import conv
import conv_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def main(dataset:conv_utils.DatasetName=conv_utils.DatasetName.CIFAR10, train_size=0.2, test_size=0.1):
    """

    :param dataset: enum value designing the name of the dataset
    :param train_size: should be between 0 and 1, representing the percentage of data to be used for training
    :param test_size: should be between 0 and 1, representing the percentage of data to be used for testing
    :return:
    """
    #ensure deterministic behavior
    conv_utils.set_seed(42)
    # use cuda when possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_dataset, test_dataset, num_classes, img_shape = conv_utils.load_train_test_dataset(dataset, train_size=train_size, test_size=test_size,
                                                                                  random_state=42, force_download=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # create model
    model = conv.CNN(num_classes, img_shape[0], img_shape[1], img_shape[2]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    accuracy = conv_utils.train_model(train_loader, test_loader, model, optimizer, device, n_epochs=10)
    print(accuracy)


if __name__ == '__main__':
    main()