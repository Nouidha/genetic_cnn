import random
import torch.optim as optim
from sympy.physics.units import momentum


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

    # def crossover(self, second_instance, rate=0.5):
    #
    # def mutate(self, rate=0.5):

    def return_optimizer(self, model):
        lowercase_name = self.optimizer_name.lower()
        if lowercase_name == "sgd":
            return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        elif lowercase_name == "adam":
            return optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif lowercase_name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
             raise ValueError(f"optimizer '{self.optimizer_name}' not supported")


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

def build_random_chromosomes(number_of_instances=10):
    list_of_optimizers = ['SGD', 'Adam', 'RMSprop']
    list_of_learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
    list_of_momentums = [0.9, 0.95, 0.99]
    list_of_weight_decays = [1e-5, 1e-6, 1e-7]
    list_of_num_conv_layers = [2, 3]
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