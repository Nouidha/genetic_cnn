import conv
import conv_utils
import genetic_utils
import torch
import random
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

def build_model_optimizer(num_classes, img_shape, chromosome, device):
    img_channels = 1 if len(img_shape) == 2 else img_shape[2]
    model = conv.CNN(num_classes=num_classes, img_rows=img_shape[0], img_cols=img_shape[1], img_channels=img_channels,
                     num_conv_layers=chromosome.num_conv_layers, conv_dropout=chromosome.conv_dropout,
                     classifier_dropout=chromosome.classifier_dropout).to(device)
    optimizer = chromosome.return_optimizer(model)
    return model, optimizer

def get_dataset_name(dataset:conv_utils.DatasetName):
    if dataset == conv_utils.DatasetName.MINST:
        return "MINST"
    elif dataset == conv_utils.DatasetName.CIFAR10:
        return "CIFAR10"
    elif dataset == conv_utils.DatasetName.CIFAR100:
        return "CIFAR100"
    else:
        raise ValueError("Unknown dataset")



def main(dataset:conv_utils.DatasetName=conv_utils.DatasetName.CIFAR100, train_size=0.3, test_size=0.1):
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


    #building initial chromosomes // ajouter 
    if False:#dataset == conv_utils.DatasetName.MINST:
        random_chromosomes = genetic_utils.build_suboptimal_chromosomes(10)
    else:
        random_chromosomes = genetic_utils.build_random_chromosomes(15)

    # load data
    train_dataset, test_dataset, num_classes, img_shape = conv_utils.load_train_test_dataset(dataset, train_size=train_size, test_size=test_size,
                                                                                  random_state=42, force_download=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


    # create and train a model for each config and add it to the models history
    model_history = genetic_utils.PopulationHistory()
    for chromosome in random_chromosomes:
        model, optimizer = build_model_optimizer(num_classes=num_classes, img_shape=img_shape, chromosome=chromosome, device=device)
        accuracy = conv_utils.train_model(train_loader, test_loader, model, optimizer, device, n_epochs=10)
        print(f"accuracy: {accuracy}, {chromosome}")
        model_history.add_instance(chromosome=chromosome, model=model, accuracy=accuracy)

    # generate Histogram of the obtained accuracies
    plt.hist(model_history.get_accuracies(), bins=5, edgecolor='black')
    plt.title('Quality of the initial population')
    plt.xlabel('Accuracy')
    plt.ylabel('Number of chromosomes')
    plt.savefig(f"quality_of_initial_population_{get_dataset_name(dataset)}.png")
    plt.show()


    # run the genetic algorithme for 20 epochs
    offsprings_accuracies = []
    top = 0.3
    for epoch in range(20):

        instance1, instance2 = model_history.return_couple(how_far_back=1.0, top=top)

        crossover_chromosome = instance1['chromosome'].crossover(instance2['chromosome'])
        mutated_chromosome = crossover_chromosome.mutate(rate=0.5)

        # Calculate diversity of the population
        population = [instance["chromosome"] for instance in model_history.history]
        diversity_score = genetic_utils.calculate_diversity(population)
        #mutation_rate = genetic_utils.get_mutation_rate(diversity_score)
        #
        print(f"Diversity Score: {diversity_score})") #, Mutation Rate: {mutation_rate}")

        ###
        
        model, optimizer = build_model_optimizer(num_classes=num_classes, img_shape=img_shape, chromosome=mutated_chromosome, device=device)
        accuracy = conv_utils.train_model(train_loader, test_loader, model, optimizer, device, n_epochs=10)

        print(f"epoch: {epoch}")
        print(f"instance 1: accuracy: {instance1['accuracy']}, {instance1['chromosome']}")
        print(f"instance 2: accuracy: {instance2['accuracy']}, {instance2['chromosome']}")
        print(f"crossover: accuracy: N.A., {crossover_chromosome}")
        print(f"mutated: accuracy: {accuracy}, {mutated_chromosome}")

        offsprings_accuracies.append(accuracy)


        # jai ajouté ca pour remplacer le pire de la pop avec le bb si il est meilleur
        worst = min(model_history.history, key=lambda x: x["accuracy"])
        if accuracy > worst["accuracy"]:
            model_history.history.remove(worst)
            model_history.add_instance(chromosome=mutated_chromosome, model=model, accuracy=accuracy)
            #print("Replaced worst model with new child.")
        #else:
            #print("Child was not better than worst in population — discarded.")
        
    # tu peux  l'enlever si tu veux
    best_model = max(model_history.history, key=lambda x: x["accuracy"])
    print(f"Best chromosome: accuracy: {best_model['accuracy']:.4f}, {best_model['chromosome']}")

    # display the evolution of the offsprings
    plt.plot(range(len(offsprings_accuracies)), offsprings_accuracies, marker='o')
    plt.title("Evolution of offspring accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"offsprings_accuracy_{get_dataset_name(dataset)}.png")
    plt.show()

    # generate Histogram of the obtained accuracies
    plt.hist(model_history.get_accuracies(), bins=5, edgecolor='black')
    plt.title('Quality of the final population')
    plt.xlabel('Accuracy')
    plt.ylabel('Number of chromosomes')
    plt.savefig(f"quality_of_final_population_{get_dataset_name(dataset)}.png")
    plt.show()




  




if __name__ == '__main__':
    main()
