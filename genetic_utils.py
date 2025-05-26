import random
import torch.optim as optim


class Chromosome:
    def __init__(self, optimizer_name, learning_rate, momentum, weight_decay, num_conv_layers, conv_dropout, classifier_dropout):
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_conv_layers = num_conv_layers
        self.conv_dropout = conv_dropout
        self.classifier_dropout = classifier_dropout

    def __str__(self):
        return f"optimizer : '{self.optimizer_name}', learning rate : {self.learning_rate}, momentum : {self.momentum}, weight decay : {self.weight_decay}, conv layers : {self.num_conv_layers}, conv dropout : {self.conv_dropout}, classification dropout : {self.classifier_dropout}"


    def return_optimizer(self, model):
        lowercase_name = self.optimizer_name.lower()
        if lowercase_name == "sgd":
            return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif lowercase_name == "adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif lowercase_name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
             raise ValueError(f"optimizer '{self.optimizer_name}' is not supported")


class PopulationHistory:
    def __init__(self):
        self.history = []

    def add_instance(self, chromosome, model, accuracy):
        model_instance = {"model": model, "accuracy": accuracy, "chromosome": chromosome}
        self.history.append(model_instance)


    def return_couple(self, how_far_back=1.0, top=0.3):
        """

        :param how_far_back: 1.0 to consider the whole population, 0.25 to consider the last 25% of the population
        :param top: if equal to 0.5 it will return two instances in the top 50% of the selected population
        :return: return two instances in the top {top} percent of the last {how_far_back} percent of the population
        """
        assert 0 < how_far_back <= 1, "how_far_back must be between 0 and 1"
        assert 0 < top <= 1, "top must be must be between 0 and 1"
        starting_index = int(len(self.history) * (1-how_far_back))
        sorted_population_0f_interest =  sorted(self.history[starting_index:], key=lambda x: (x["accuracy"], x))
        start_range = int(len(sorted_population_0f_interest) * (1-top))
        end_range = len(sorted_population_0f_interest)
        assert (end_range-start_range)>2, "Not enough data"
        a, b = random.sample(range(start_range, end_range), 2)
        return sorted_population_0f_interest[a], sorted_population_0f_interest[b]

def build_random_chromosomes(number_of_instances=10):
    list_of_optimizers = ['SGD', 'Adam', 'RMSprop']
    list_of_learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    list_of_momentums = [0.9, 0.95, 0.99]
    list_of_weight_decays = [1e-5, 1e-6, 1e-7]
    list_of_num_conv_layers = [1, 2]
    list_of_conv_dropouts = [0.1, 0.2, 0.3]
    list_of_classifier_dropouts = [0.3, 0.4, 0.5]

    chromosomes = []
    for index in range(number_of_instances):
        optimizer_name = random.choice(list_of_optimizers)
        lr = random.choice(list_of_learning_rates)
        momentum = random.choice(list_of_momentums)
        weight_decay = random.choice(list_of_weight_decays)
        num_conv_layers = random.choice(list_of_num_conv_layers)
        conv_dropout = random.choice(list_of_conv_dropouts)
        classifier_dropout = random.choice(list_of_classifier_dropouts)
        chromosomes.append(Chromosome(optimizer_name=optimizer_name, learning_rate=lr, momentum=momentum, weight_decay=weight_decay,
                                      num_conv_layers=num_conv_layers, conv_dropout=conv_dropout, classifier_dropout=classifier_dropout))
    return chromosomes




#### adding crossover and mutation 

def crossover(self, second_instance, rate=0.5):
    """
    Perform crossover between two parent chromosomes to produce an offspring.

    :param second_instance: The second parent chromosome.
    :param rate: The probability of inheriting a gene from the first parent.
    :return: offspring.
    """
    # here we randomly combine genes from both parents
    optimizer_name = self.optimizer_name if random.random() < rate else second_instance.optimizer_name
    learning_rate = self.learning_rate if random.random() < rate else second_instance.learning_rate
    momentum = self.momentum if random.random() < rate else second_instance.momentum
    weight_decay = self.weight_decay if random.random() < rate else second_instance.weight_decay
    num_conv_layers = self.num_conv_layers if random.random() < rate else second_instance.num_conv_layers
    conv_dropout = self.conv_dropout if random.random() < rate else second_instance.conv_dropout
    classifier_dropout = self.classifier_dropout if random.random() < rate else second_instance.classifier_dropout

    # we reeturn the babies offspring chromosome 

    return Chromosome(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        num_conv_layers=num_conv_layers,
        conv_dropout=conv_dropout,
        classifier_dropout=classifier_dropout
    )
    

def mutation(self, rate=0.1):
    """
    Perform mutation on the chromosome by randomly altering its genes.

    :param mutation_rate: The probability of mutating each gene.
    :return: A new Chromosome instance (mutated version).
    """
    # Define the possible values for each gene
    # g gardé ta grille d'avant pour les parametres utilisé pr les random change
    list_of_optimizers = ['SGD', 'Adam', 'RMSprop']
    list_of_learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    list_of_momentums = [0.9, 0.95, 0.99]
    list_of_weight_decays = [1e-5, 1e-6, 1e-7]
    list_of_num_conv_layers = [1, 2]
    list_of_conv_dropouts = [0.1, 0.2, 0.3]
    list_of_classifier_dropouts = [0.3, 0.4, 0.5]

    # to mutate each gene with a probability set to 10% (each chromosome gene has a 10% chance of being mutated) 
     #50% was too high ? dont know tu peux changer la proba
    optimizer_name = random.choice(list_of_optimizers) if random.random() < rate else self.optimizer_name
    learning_rate = random.choice(list_of_learning_rates) if random.random() < rate else self.learning_rate
    momentum = random.choice(list_of_momentums) if random.random() < rate else self.momentum
    weight_decay = random.choice(list_of_weight_decays) if random.random() < rate else self.weight_decay
    num_conv_layers = random.choice(list_of_num_conv_layers) if random.random() < rate else self.num_conv_layers
    conv_dropout = random.choice(list_of_conv_dropouts) if random.random() < rate else self.conv_dropout
    classifier_dropout = random.choice(list_of_classifier_dropouts) if random.random() < rate else self.classifier_dropout

    
    return Chromosome(
        optimizer_name=optimizer_name,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        num_conv_layers=num_conv_layers,
        conv_dropout=conv_dropout,
        classifier_dropout=classifier_dropout
    )
