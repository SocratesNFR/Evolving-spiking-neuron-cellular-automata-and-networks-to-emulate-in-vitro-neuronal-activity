from random import random, choice, shuffle
# import CellularAutomataModel
from src import Fitness, Population, Data
# import NetworkModel
from src.Model import Model



class Evolution:
    """
    Creates an evolution object with the defined parameters.
    Parameters must be given as a dictionary, and must contain the following:
    MODEL_TYPE, DIMENSION, POPULATION_SIZE, SIMULATION_DURATION, TIME_STEP_RESOLUTION, REFERENCE_PHENOTYPE,
    PARENTS_P, RETAINED_ADULTS_P AND MUTATION_P.
    See additional documentation for details on these parameters.
    """
    def __init__(self, parameters):
        self.model_type = parameters["MODEL_TYPE"][0]
        self.dimension = parameters["DIMENSION"]
        self.population_size = parameters["POPULATION_SIZE"]
        self.recording_start = parameters["RECORDING_START"]
        self.simulation_duration = parameters["SIMULATION_DURATION"]
        self.resolution = parameters["TIME_STEP_RESOLUTION"]
        self.reference_file = parameters["REFERENCE_PHENOTYPE"]
        self.reference_phenotype = Data.get_spikes_file(
            parameters["REFERENCE_PHENOTYPE"],
            recording_len=self.simulation_duration, recording_start=self.recording_start
            )
        self.reference_spikes = Data.get_spikerate(self.reference_phenotype, self.simulation_duration, recording_start=self.recording_start)
        self.parents_p = parameters["PARENTS_P"]
        self.retained_adults_p = parameters["RETAINED_ADULTS_P"]
        self.mutation_p = parameters["MUTATION_P"]
        self.best_individual_overall = False

    def select_parents(self, individuals):
        """
        Sorts the array of individuals by decreasing fitness.
        Returns a number of the best individuals according to PARENTS_P
        """
        individuals.sort(key=lambda x: x.fitness, reverse=True)
        best_individuals = individuals[:round(len(individuals) * self.parents_p)]
        return best_individuals

    def reproduce(self, parents, individuals):
        """
        Selects a number of the best individuals according to RETAINED_ADULTS_P and adds them to return-list.
        Shuffles array, then matches two-and-two individuals until return-list is full.
        If RETAINED_ADULTS_P is present then certain matches might occur more often.
        """
        individuals.sort(key=lambda x: x.fitness, reverse=True)
        retained_adults = individuals[:round(len(individuals) * self.retained_adults_p)]
        shuffle(parents)
        next_generation = retained_adults if retained_adults else []
        i = 0
        while len(next_generation) < self.population_size:
            genes = []
            for gene1, gene2 in zip(parents[i].genotype, parents[i + 1].genotype):
                genes.append(random() if random() < self.mutation_p else choice((gene1,gene2)))
            next_generation.append(Population.Individual(g_type=genes))
            i = (i + 2) % (len(parents) - 1)
        return next_generation

    def generate_phenotype(self, individual):
        """
        Runs the simulation and adds phenotype list to individual.
        Gets the fitness score and appends it to the individual.
        """
        # print(current_process().name, end=" ")
        if individual.model == None:
            model = Model(
                individual=individual,
                dimension=self.dimension,
                duration=self.simulation_duration,
                resolution=self.resolution
            )
            individual.model = model
        phenotype = individual.model.run_simulation()
        #   Calculate the fitness of the phenotype
        fitness = Fitness.get_fitness(phenotype, 0, self.reference_phenotype, self.recording_start, self.simulation_duration)
        #   Append results to the individual
        individual.phenotype = phenotype
        individual.fitness = fitness
        return individual
