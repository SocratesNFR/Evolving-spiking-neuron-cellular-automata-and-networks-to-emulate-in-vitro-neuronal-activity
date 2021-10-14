import os
import time
import sys
import getopt
import csv
from multiprocessing import Pool
import numpy as np
import Population, Summary, Evolution, Data

"""
PARAMETERS
"""

# TYPE parameter sets properties specific for each model type
TYPE = {
    # Set properties for CA model
    "ca": (
        #   Name of model
        "ca",
        #   Size of genome (number of parameters in genotype)
        7,
        #   Labels
        (
            "Firing threshold",
            "Random fire probability",
            "Refractory period",
            "Inhibition percentage",
            "Leak constant",
            "Integration constant",
            "Density constant"
        )),
    # Set properties for network model
    "network": (
        #   Name of model
        "network",
        #   Size of genome (number of parameters in genotype)
        7,
        #   Labels
        (
            "Firing threshold",
            "Random fire probability",
            "Refractory period",
            "Inhibition percentage",
            "Leak constant",
            "Integration constant",
            "Density constant"
        )),
}

#   Use these parameters if no .CSV file is given
default_parameters = {
    #   Choose between CA and Network by commenting out the other.
    # "MODEL_TYPE": TYPE["CA"],
    "MODEL_TYPE": TYPE["ca"],
    # Size of one dimension in the array / grid / matrix
    "DIMENSION": 10,
    #   Number of individuals in the population
    "POPULATION_SIZE": 6,
    #   Number of generations to run.
    #   Each generation will run one simulation of the model for every individual in the population
    "NUM_GENERATIONS": 3,
    #   Start of recording
    "RECORDING_START": 1000,
    #   Simulation duration in seconds
    "SIMULATION_DURATION": 100,
    #   Number of simulation iterations per second
    "TIME_STEP_RESOLUTION": 40,
    #   The probability of mutation in any gene
    "MUTATION_P": 0.1,
    #   The percentage of the current population that will create offspring
    "PARENTS_P": 0.5,
    #   The percentage of the current population that will carry over to the next generation
    "RETAINED_ADULTS_P": 0.05,
    #   Name of the file of experimental data used as reference for the fitness function and raster plot
    "REFERENCE_PHENOTYPE": "2-1-31.spk.txt"
}

""" 
FUNCTIONS
"""


def run_threads(individuals):
    """
    Creates threads of run_thread method.
    Pool-size = threads - 1.
    Each thread result is mapped to a variable that is returned when all processes are finished
    """
    with Pool(os.cpu_count() - 1) as p:
        new_individuals = p.map(evo.generate_phenotype, individuals)
        p.close()
    return new_individuals


def process_arguments():
    """
    Will process arguments given from terminal.
    If no arguments are given, script will run with defaults
    """
    input_file = ""
    help_string = f"{sys.argv[0]} -i <inputfile.csv>"
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:", ["input="])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print(help_string)
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file = arg
            print(input_file)
    return input_file


"""
PROGRAM
"""
if __name__ == "__main__":
    # initialize the evolution
    t_simulation_start = time.time()
    evolution_parameters = list()
    input_file = process_arguments()
    # run with default parameters if no .CSV file is given
    if input_file == "":
        evolution_parameters.append(default_parameters)
    else:
        with open(input_file, newline="") as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            reader_iter = iter(reader)
            header = next(reader_iter)
            for row in reader_iter:
                evolution_parameters.append({
                    "MODEL_TYPE": TYPE[str(row[0])],
                    "DIMENSION": int(row[1]),
                    "POPULATION_SIZE": int(row[2]),
                    "NUM_GENERATIONS": int(row[3]) if max(Data.get_spikerate(
                        Data.get_spikes_file(row[10], recording_start=int(row[4]), recording_len=int(row[5])), recording_len=int(row[5]), recording_start=int(row[4]))) < 1000 else int(int(row[3]) * 1.5),
                    "RECORDING_START": int(row[4]),
                    "SIMULATION_DURATION": int(row[5]),
                    "TIME_STEP_RESOLUTION": int(row[6]),
                    "MUTATION_P": float(row[7]),
                    "PARENTS_P": float(row[8]),
                    "RETAINED_ADULTS_P": float(row[9]),
                    "REFERENCE_PHENOTYPE": row[10]
                })
    for evo_i, params in enumerate(evolution_parameters):
        t_evo_start = time.time()
        # print summary of the running simulation parameters
        print("\n-------------------------------------")
        print(f"Running simulation {evo_i + 1}/{len(evolution_parameters)}:")
        iterator = iter(params)
        key = next(iterator)
        print(f"{key}: {params[key][0]} (genome size = {params[key][1]})")
        for key in iterator:
            print(f"{key}: {params[key]}")
        print()
        #   Creates population object with POPULATION_SIZE.
        #   Creates the set of genes that apply to this specific population
        pop = Population.Population(
            params["POPULATION_SIZE"],
            params["MODEL_TYPE"][1]
        )
        # Creates Evolution object with parameters set
        evo = Evolution.Evolution(params)
        # Initialize datasets
        fitness_trend = []
        average_fitness_trend = []
        parameter_trend = []
        generation_summary = {}
        # Start the evolution. Runs loop for NUM_GENERATIONS
        # print("Running simulation...")
        est_time = "?"
        for i in range(params["NUM_GENERATIONS"]):
            t_generation_start = time.time()
            # print("\nGeneration:", i)
            # print("Workers: ", end="")
            print(f"Simulating generation {i + 1}/{params['NUM_GENERATIONS']} ({est_time} s per generation)", end="\r")
            #   Run the evolutionary algorithm on the population
            pop_with_phenotypes = run_threads(pop.individuals)
            #   Record data of the population
            fitness_trend.append([i.fitness for i in pop_with_phenotypes])
            average_fitness_trend.append(sum(
                [i.fitness for i in pop_with_phenotypes]
            ) / params["POPULATION_SIZE"])
            parameter_trend.append(np.sum(
                [i.genotype for i in pop_with_phenotypes], 0) / params["POPULATION_SIZE"])
            # Sort the population in order to pick out the best fitness
            sorted_pop = pop_with_phenotypes
            sorted_pop.sort(key=lambda x: x.fitness, reverse=True)
            #   If there is no recorded best individual, choose the best one of this generation
            if not evo.best_individual_overall:
                evo.best_individual_overall = (i, sorted_pop[0])
            #   If the fitness of the best individual in this generation is better than the best recorded individual,
            #   let the new one take its place as the best.
            elif evo.best_individual_overall[1].fitness < sorted_pop[0].fitness:
                evo.best_individual_overall = (i, sorted_pop[0])
            #   Reproduction
            if i < params["NUM_GENERATIONS"] - 1:
                parents = evo.select_parents(pop_with_phenotypes)
                new_gen = evo.reproduce(parents, pop_with_phenotypes)
                pop.individuals = new_gen
            else:
                pop.individuals = pop_with_phenotypes
            # estimate generation run time
            if est_time == "?":
                est_time = round(time.time() - t_generation_start)
            else:
                est_time = round(est_time * 0.8 + (time.time() - t_generation_start) * 0.2)
            # record the phenotype top individuals every 5th generation
            gen_summary = {}
            if i % 5 == 0 or i + 1 == params["NUM_GENERATIONS"]:
                for j in range(5):
                    gen_summary[f"rank {j + 1}"] = {
                        # "generation" : i,
                        # "rank" : j+1,
                        "genotype": sorted_pop[j].genotype,
                        "phenotype": sorted_pop[j].phenotype.tolist(),
                        "fitness": sorted_pop[j].fitness,
                    }
            # record fitness and genotype of all individuals every generation
            gen_summary["all"] = [{"fitness": indiv.fitness,"genotype": indiv.genotype} for indiv in sorted_pop]
            gen_summary["time"] = round(time.time() - t_generation_start, 3)
            generation_summary[i + 1] = gen_summary
        #   Save the running time of the script
        end_time = time.time()
        t_evo_total = time.time() - t_evo_start
        # Save a summary of the evolution
        summary = Summary.Summary(pop, params, evo)
        summary.raster_plot()
        summary.fitness_trend_plot((fitness_trend, average_fitness_trend))
        summary.parameter_trend_plot(parameter_trend)
        summary.average_distance_plot()
        summary.output_text(t_evo_total)
        summary.save_model(evo.best_individual_overall[1].model)
        summary.save_stats(generation_summary)
        print(f"\nEA simulated in {t_evo_total:.2f} seconds")
    t_simulation_total = round(time.time() - t_simulation_start)
    print(f"Simulation completed in {t_simulation_total // 60} minutes, {t_simulation_total % 60} seconds")
