from random import random

class Population:
    """
    A population consists of a type of individual that has certain genetic composition. Each individual in a population
    has the same number of genes, and each gene is within a range specific to that gene.
    """
    def __init__(self, pop_size, genome):
        self.population_size = pop_size
        self.genome = genome
        self.individuals = self.__create_individuals()

    def __create_individuals(self):
        """
        Create a population based of a given size, where the genome of its Individuals is randomly generated.
        Return: List of Individual
        """
        pop = []
        for individual in range(self.population_size):
            g = [random() for i in range(self.genome)]
            pop.append(Individual(g))
        return pop


class Individual:
    """
    Individuals consist of a genotype and a phenotype.
    The genotype is the genetic encoding of the individual.
    The phenotype is the part of the individual that is tested in the environment.
    The fitness is a score or a result of a fitness function.
    """
    def __init__(self, g_type):
        self.genotype = g_type
        self.phenotype = None
        self.fitness = None
        self.model = None
