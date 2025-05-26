import conv
import conv_utils
import genetic_utils
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def build_model_optimizer(num_classes, img_shape, chromosome, device):
    img_channels = 1 if len(img_shape) == 2 else img_shape[2]
    model = conv.CNN(num_classes=num_classes, img_rows=img_shape[0], img_cols=img_shape[1], img_channels=img_channels,
                     num_conv_layers=chromosome.num_conv_layers, conv_dropout=chromosome.conv_dropout,
                     classifier_dropout=chromosome.classifier_dropout).to(device)
    optimizer = chromosome.return_optimizer(model)
    return model, optimizer


def main(dataset:conv_utils.DatasetName=conv_utils.DatasetName.MINST, train_size=0.4, test_size=0.2):
    """

    :param dataset: enum value designing the name of the dataset
    :param train_size: should be between 0 and 1, representing the percentage of data to be used for training
    :param test_size: should be between 0 and 1, representing the percentage of data to be used for testing
    :return:
    """
    #ensure deterministic behavior
    conv_utils.set_seed(42)
    # use cuda when possible
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device("mps") # stands for Metal Performance Shaders (on Apple Silicon)
    else:
        device = torch.device("cpu")

    # load data
    train_dataset, test_dataset, num_classes, img_shape = conv_utils.load_train_test_dataset(dataset, train_size=train_size, test_size=test_size,
                                                                                  random_state=42, force_download=False)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # create initial random chromosomes
    random_chromosomes = genetic_utils.build_random_chromosomes(10)
    # create and train a model for each config and add it to the models history
    model_history = genetic_utils.PopulationHistory()
    for chromosome in random_chromosomes:
        print(f"chromosome: {chromosome}")
        model, optimizer = build_model_optimizer(num_classes=num_classes, img_shape=img_shape, chromosome=chromosome, device=device)
        print(f"model: {model}")
        accuracy = conv_utils.train_model(train_loader, test_loader, model, optimizer, device, n_epochs=5)
        print(f"model accuracy: {accuracy}")
        model_history.add_instance(chromosome=chromosome, model=model, accuracy=accuracy)

    # run the genetic algorithme for 5 epochs
    # for _ in range(5):
    #     instance1, instance2 = model_history.return_couple(how_far_back=1.0, top=0.3)
    #     print(f"chromosome 1: {instance1['chromosome']} \nchromosome 2: {instance2['chromosome']}")
    #     crossover_chromosome = instance1['chromosome'].crossover(instance2['chromosome'])
    #     print(f"crossover chromosome: {crossover_chromosome}")
    #     mutated_chromosome = crossover_chromosome.mutate()
    #     print(f"mutated chromosome: {mutated_chromosome}")
    #     model, optimizer = build_model_optimizer(num_classes=num_classes, img_shape=img_shape, chromosome=mutated_chromosome,
    #                                              device=device)
    #     accuracy = conv_utils.train_model(train_loader, test_loader, model, optimizer, device, n_epochs=10)
    #     print(f"model accuracy: {accuracy}")
    #     model_history.add_instance(chromosome=mutated_chromosome, model=model, accuracy=accuracy)




if __name__ == '__main__':
    main()