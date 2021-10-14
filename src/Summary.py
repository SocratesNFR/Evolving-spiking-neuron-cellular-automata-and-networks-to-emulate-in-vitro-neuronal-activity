import csv
import os
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src import Data


class Summary:
    """
    Creates summaries of relevant results and trends after the last generation of Evolution.
    """
    def __init__(self, population, evolution_parameters, evo):
        self.population = population
        self.evolution_parameters = evolution_parameters
        self.bin_size = self.evolution_parameters["SIMULATION_DURATION"]
        self.population.individuals.sort(key=lambda x: x.fitness, reverse=True)
        self.median_individual = self.population.individuals[int(round(len(self.population.individuals) / 2))]
        self.best_individual = self.population.individuals[0]
        self.best_individual_overall = evo.best_individual_overall
        self.top_five = self.population.individuals[0:5] if len(self.population.individuals) >= 5 else False
        self.reference_spikes = Data.get_spikerate(
            Data.get_spikes_file(self.evolution_parameters["REFERENCE_PHENOTYPE"], recording_start=self.evolution_parameters["RECORDING_START"], recording_len=self.evolution_parameters["SIMULATION_DURATION"]),
            self.evolution_parameters["SIMULATION_DURATION"], recording_start=self.evolution_parameters["RECORDING_START"])
        self.simulation_spikes = Data.get_spikerate(
            self.best_individual.phenotype, recording_len=self.evolution_parameters["SIMULATION_DURATION"], recording_start=0)
        reference_name = str(self.evolution_parameters["REFERENCE_PHENOTYPE"]).replace(".spk.txt", "")
        now = datetime.now().strftime("%Y%m%d%H%M%S")
        self.dir_path = "../Output/" + self.evolution_parameters["MODEL_TYPE"][0] + \
                        "_dim" + str(self.evolution_parameters["DIMENSION"]) + \
                        "_pop" + str(self.evolution_parameters["POPULATION_SIZE"]) + \
                        "_gen" + str(self.evolution_parameters["NUM_GENERATIONS"]) + \
                        "_dur" + str(self.evolution_parameters["SIMULATION_DURATION"]) + \
                        "_res" + str(self.evolution_parameters["TIME_STEP_RESOLUTION"]) + \
                        "_mut" + str(self.evolution_parameters["MUTATION_P"]) + \
                        "_par" + str(self.evolution_parameters["PARENTS_P"]) + \
                        "_ret" + str(self.evolution_parameters["RETAINED_ADULTS_P"]) + \
                        "_" + reference_name + \
                        "_" + now
        try:
            os.makedirs(self.dir_path)
        except:
            print(f"Failed to create folder {self.dir_path}")

    def raster_plot(self):
        """
        Takes two phenotypes as input and plot them side-by-side as raster plot and histogram

        Assumes phenotype is a list of lists or 2D numpy array similar to:
        [[.00396, 56],
        [0.05284, 16],
        [0.05800, 15],
        ...,
        [A, B]]
        Where A is a timestamp and B is electrode ID (must be a integer between 0-63)

        To create histogram it is necessary to specify bin-size. For spikes per second "bin_size" = simulation length [seconds]
        """
        self.phenotype_reference = Data.read_recording(
            self.evolution_parameters["REFERENCE_PHENOTYPE"], recording_start=self.evolution_parameters["RECORDING_START"],
            recording_len=self.evolution_parameters["SIMULATION_DURATION"],
               #Where to start reading experimental data [s]
        )
        #   Check if input is in the correct format
        self.best_individual.phenotype = np.array(
            [(row[0], row[1]) for row in self.best_individual.phenotype],
            dtype=[("t", "float64"), ("electrode", "int64")])
        self.phenotype_reference = np.array(
            [(row[0] - self.evolution_parameters["RECORDING_START"], row[1]) for row in self.phenotype_reference],
            dtype=[("t", "float64"), ("electrode", "int64")])
        #   Sort spikes by electrode
        self.A_spikes_per_array = [[] for _ in range(60)]
        for row in self.best_individual.phenotype:
            self.A_spikes_per_array[row[1]].append(row[0])
        self.B_spikes_per_array = [[] for _ in range(60)]
        for row in self.phenotype_reference:
            self.B_spikes_per_array[row[1]].append(row[0])
        #   Initialize plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharey="row")
        #   Make raster plots
        ax1.eventplot(
            self.A_spikes_per_array,
            linewidths=0.5
        )
        ax1.set_ylabel("Electrode ID")
        ax1.set_title("Best model")
        ax2.eventplot(
            self.B_spikes_per_array,
            linewidths=0.5,
            color="black"
        )
        ax2.set_title("Neural culture")
        #   Make histograms
        ax3.bar(range(len(self.simulation_spikes)), self.simulation_spikes, align="edge", width=1)
        ax3.set_xlabel("Seconds")
        ax3.set_ylabel("Spikes per second")
        ax4.bar(range(len(self.reference_spikes)), self.reference_spikes, color="black", align="edge", width=1)
        ax4.set_xlabel("Seconds")
        fig.savefig(self.dir_path + "/Best_individual.png")
        plt.close()

    def parameter_trend_plot(self, parameter_data):
        """
        Plot parameter trend
        """
        par, ax_par = plt.subplots()
        for param, label in zip(list(map(list, zip(*parameter_data))), self.evolution_parameters["MODEL_TYPE"][2]):
            ax_par.plot(param, label=label)
        ax_par.legend(loc="upper left")
        ax_par.set_title("Parameter trend")
        ax_par.set_xlabel("Generation")
        ax_par.set_ylabel("Normalized genome value")
        par.savefig(self.dir_path + "/Parameter_trend.png")
        plt.close()

    def fitness_trend_plot(self, fitness_data):
        """
        Plot Fitness trend
        """
        avg_fit, ax_avg_fit = plt.subplots()
        ax_avg_fit.plot(fitness_data[0], linestyle="", marker=".", color="red")
        ax_avg_fit.plot(fitness_data[1], label="Average fitness", color="blue")
        ax_avg_fit.legend(loc="upper left")
        ax_avg_fit.set_title("Fitness trend")
        ax_avg_fit.set_xlabel("Generation")
        ax_avg_fit.set_ylabel("Fitness score")
        ax_avg_fit.set_ylim(ymin=0, ymax=1)
        avg_fit.savefig(self.dir_path + "/Fitness_trend.png")
        plt.close()

    def average_distance_plot(self):
        """
        Plot average distance
        """
        simulation_s = sorted(self.simulation_spikes)
        reference_s = sorted(self.reference_spikes)
        simulation = self.simulation_spikes
        reference = self.reference_spikes
        fig, ax = plt.subplots(2, sharex="all")
        ax[0].set_xlabel("Sorted time [s]")
        ax[0].set_ylabel("Spikes per second")
        ax[0].plot(simulation_s, 'b', label="Simulation")
        ax[0].plot(reference_s, 'black', label="Reference")
        ax[0].plot([abs(sim - ref) for ref, sim in zip(simulation_s, reference_s)], label="Difference")
        ax[0].legend()
        ax[0].fill_between(range(len(simulation_s)), simulation_s, reference_s, color='red', alpha=0.2,
                           where=[_y2 < _y1 for _y2, _y1 in zip(simulation_s, reference_s)])
        ax[0].fill_between(range(len(simulation_s)), simulation_s, reference_s, color='green', alpha=0.2,
                           where=[_y2 > _y1 for _y2, _y1 in zip(simulation_s, reference_s)])
        for i in range(0, len(simulation_s), int(len(simulation_s) / 10)):
            ax[0].text(i, min(simulation_s[i], reference_s[i]) + 30, simulation_s[i] - reference_s[i])

        ax[1].set_xlabel("Time [s]")
        ax[1].set_ylabel("Spikes per second")
        ax[1].plot(simulation, 'b', label="Simulation")
        ax[1].plot(reference, 'black', label="Reference")
        ax[1].legend()
        ax[1].fill_between(range(len(simulation)), simulation, reference, color='red', alpha=0.2,
                           where=[_y2 < _y1 for _y2, _y1 in zip(simulation, reference)])
        ax[1].fill_between(range(len(simulation)), simulation, reference, color='green', alpha=0.2,
                           where=[_y2 > _y1 for _y2, _y1 in zip(simulation, reference)])
        fig.savefig(self.dir_path + "/Average_distance.png")
        plt.close()

    def output_text(self, simulation_time):
        """
        Write a text file with relevant information about the population when the Evolution is over.
        """
        if self.top_five:
            top_five_string = "| INDIVIDUAL 2 | " + "Parameters: " + str(
                self.top_five[1].genotype) + " Fitness score: " + str(
                self.top_five[1].fitness) + "\n" + "| INDIVIDUAL 3 | " + "Parameters: " + str(
                self.top_five[2].genotype) + " Fitness score: " + str(
                self.top_five[2].fitness) + "\n" + "| INDIVIDUAL 4 | " + "Parameters: " + str(
                self.top_five[3].genotype) + " Fitness score: " + str(
                self.top_five[3].fitness) + "\n" + "| INDIVIDUAL 5 | " + "Parameters: " + str(
                self.top_five[4].genotype) + " Fitness score: " + str(
                self.top_five[4].fitness) + "\n" + "TOP 5 AVERAGE: " + str(
                (sum([self.top_five[i].fitness for i in range(1, 5)]) + self.best_individual.fitness) / 5)
        else:
            top_five_string = ""
        text_file = open(self.dir_path + "/Info.txt", "wt")
        n = text_file.write(
            "EVOLUTION PARAMETERS: " + str(self.evolution_parameters) + " Simulation time [min]: " + str(
                simulation_time / 60) + "\n" + "*LAST GENERATION*" + "\n| INDIVIDUAL 1 | " + "Parameters: " + str(
                self.best_individual.genotype) + " Fitness score: " + str(
                self.best_individual.fitness) + "\n" + top_five_string + "\n" + "| MEDIAN INDIVIDUAL |" + " Fitness score: " + str(
                self.median_individual.fitness) + "\n\nBEST OVERALL\n" + "| TOP INDIVIDUAL | " + "Generation: " + str(
                self.best_individual_overall[0]) + " Parameters: " + str(
                self.best_individual_overall[1].genotype) + " Fitness score: " + str(
                self.best_individual_overall[1].fitness))
        text_file.close()

    def write_csv(self, fitness_data):
        """
        Write a CSV file with average fitness score per generation.
        """
        with open(self.dir_path + "/fitness.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(["generation", "avg_fitness"])
            for i, score in enumerate(fitness_data):
                writer.writerow([str(i), str(score)])

    def save_model(self, model):
        '''
        Saved a GraphML document that stores the model configuration.
        '''
        nx.write_graphml(model.config, self.dir_path + "/model.gml")

    def save_stats(self, generation_summary):
        '''
        Saves simulation data to JSON-file.
        '''
        self.evolution_parameters["generations"] = generation_summary
        # save output
        data_path = Path(f"{self.dir_path}/evolution_data.json")
        with data_path.open("w") as file:
            json.dump(self.evolution_parameters, file)


"""
Helper functions for testing models independently 

"""
def read_neural_recording(filename, recording_start=0, recording_len=30 * 60):
    '''
    Reads neural recordings as txt files and returns spike times and electrode spike-rates.
    '''
    # cleaning data, making array
    f = open(filename, "r")
    data_points = [line.split(" ") for line in f]
    data_points = np.array(
        [(row[0].rstrip(), row[1].rstrip()) for row in data_points],
        dtype=[("t", "float64"), ("electrode", "int64")])
    # edit to requested recording length
    start_index, stop_index = np.searchsorted(data_points["t"], [recording_start, recording_start + recording_len])
    data_points = data_points[start_index:stop_index]
    # bin the data spikes
    spikes = np.array(data_points["t"])
    # sort data by electrode ID
    data_by_electrode = [[] for _ in range(64)]
    for row in data_points:
        data_by_electrode[row["electrode"]].append(row["t"])
    # convert to array
    spikes_per_array = np.array(data_by_electrode, dtype="object")
    # count average fire rate per electrode (spikes per second)
    spike_rates = {key: [] for key in range(64)}
    for key, item in enumerate(data_by_electrode):
        f_c = len(data_by_electrode[key]) / recording_len
        spike_rates[key] = f_c  # spike rate
    f.close
    return (spikes, spikes_per_array)


def make_raster_plot(neural_data_filepath, phenotype, simulation_length):
    # read reference neural data
    neuron_spikes, neuron_spikes_per_array = read_neural_recording(
        neural_data_filepath,  # starting point in seconds
        recording_len=simulation_length  # simlation length in seconds, set to match simulation
    )
    sim_spikes_per_array = [[] for _ in range(60)]
    for row in phenotype:
        sim_spikes_per_array[row[1]].append(row[0])
    # plot neural spikes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, sharex="all", sharey="row")
    # plot simulation spikes
    ax1.eventplot(
        sim_spikes_per_array,
        linewidths=0.5,
        color="black"
    )
    ax1.set_xlabel("Seconds")
    ax1.set_title("Simulation raster plot")
    ax2.eventplot(
        neuron_spikes_per_array,
        linewidths=0.5
    )
    ax2.set_xlabel("Seconds")
    ax2.set_ylabel("Electrode ID")
    ax2.set_title("Neural raster plot")
    ax3.hist(phenotype["t"], bins=simulation_length, color="black")
    ax3.set_ylabel("Spikes per second")
    ax3.set_xlabel("Seconds")
    ax4.hist(neuron_spikes, bins=simulation_length)
    ax4.set_xlabel("Seconds")
    plt.show()