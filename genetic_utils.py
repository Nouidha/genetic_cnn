import random
import math
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

    
    def mutate(self, rate):
        mutated = False 
        #making sure that at least one parameter is mutated
        while not mutated:
            optimizer_name = random.choice(['SGD', 'Adam', 'RMSprop']) if random.random() < rate else self.optimizer_name
            learning_rate = random.uniform(1e-5, 1e-2) if random.random() < rate else self.learning_rate
            momentum = random.uniform(0.8, 0.99) if random.random() < rate else self.momentum
            weight_decay = random.uniform(1e-7, 1e-4) if random.random() < rate else self.weight_decay
            num_conv_layers = random.randint(1, 3) if random.random() < rate else self.num_conv_layers
            conv_dropout = random.uniform(0.0, 0.5) if random.random() < rate else self.conv_dropout
            classifier_dropout = random.uniform(0.2, 0.6) if random.random() < rate else self.classifier_dropout

            mutated = any([
                optimizer_name != self.optimizer_name,
                learning_rate != self.learning_rate,
                momentum != self.momentum,
                weight_decay != self.weight_decay,
                num_conv_layers != self.num_conv_layers,
                conv_dropout != self.conv_dropout,
                classifier_dropout != self.classifier_dropout,
            ])

        return Chromosome(
            optimizer_name, 
            learning_rate, 
            momentum, 
            weight_decay,
            num_conv_layers, 
            conv_dropout, 
            classifier_dropout
        )


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
        starting_index = math.floor(len(self.history) * (1-how_far_back))
        sorted_population_0f_interest =  sorted(self.history[starting_index:], key=lambda x: x["accuracy"])
        start_range = math.floor(len(sorted_population_0f_interest) * (1-top))
        end_range = len(sorted_population_0f_interest)
        assert (end_range-start_range)>2, "Not enough data"
        a, b = random.sample(range(start_range, end_range), 2)
        return sorted_population_0f_interest[a], sorted_population_0f_interest[b]

def build_random_chromosomes(number_of_instances=10): #for initial population 
    chromosomes = []
    for _ in range(number_of_instances):
        optimizer_name = random.choice(['SGD', 'Adam', 'RMSprop'])
        learning_rate = random.uniform(1e-5, 1e-2)
        momentum = random.uniform(0.8, 0.99)
        weight_decay = random.uniform(1e-7, 1e-4)
        num_conv_layers = random.randint(1, 3)
        conv_dropout = random.uniform(0.0, 0.5)
        classifier_dropout = random.uniform(0.2, 0.6)
        chromosomes.append(Chromosome(
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            num_conv_layers=num_conv_layers,
            conv_dropout=conv_dropout,
            classifier_dropout=classifier_dropout
        ))
    return chromosomes


OPTIMIZER_DISTANCE = {
    ('SGD', 'Adam'): 1.0,
    ('SGD', 'RMSprop'): 0.8,
    ('Adam', 'RMSprop'): 0.5,
}

def optimizer_distance(opt1, opt2):
    if opt1 == opt2:
        return 0.0
    return OPTIMIZER_DISTANCE.get((opt1, opt2), OPTIMIZER_DISTANCE.get((opt2, opt1), 1.0))

def compare_chromosomes(ch1, ch2):
    score = 0
    score += optimizer_distance(ch1.optimizer_name, ch2.optimizer_name)
    score += abs(ch1.learning_rate - ch2.learning_rate) / 0.01  # normalize by max range
    score += abs(ch1.momentum - ch2.momentum) / 0.19
    score += abs(ch1.weight_decay - ch2.weight_decay) / 0.0001
    score += abs(ch1.num_conv_layers - ch2.num_conv_layers) / 2  # max difference is 2
    score += abs(ch1.conv_dropout - ch2.conv_dropout) / 0.5
    score += abs(ch1.classifier_dropout - ch2.classifier_dropout) / 0.4
    return score


def calculate_diversity(population):
    total_difference = 0
    n = len(population)
    for i in range(n):
        for j in range(i + 1, n):
            difference = compare_chromosomes(population[i], population[j])
            print(f"Difference between chromosome {i} and {j}: {difference}")
            total_difference += difference
    num_pairs = n * (n - 1) / 2
    diversity_score = total_difference / num_pairs if num_pairs > 0 else 0
    print(f"Calculated Diversity Score: {diversity_score}")
    return diversity_score



def get_mutation_rate(diversity, low_threshold=1.0, high_threshold=2.5):
    """
    Adjust the mutation rate based on the diversity of the population.

    :param diversity: The diversity score of the population.
    :param low_threshold: The threshold below which mutation rate is increased.
    :param high_threshold: The threshold above which mutation rate is decreased.
    :return: The mutation rate.
    """
    if diversity < low_threshold:
        return 0.8
    elif diversity > high_threshold:
        return 0.15
    else:
        return 0.3


















