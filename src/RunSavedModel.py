from pylab import *
import networkx as nx
import numpy as np
import time
import random
from src import Data
import getopt
import sys
import json

RESOLUTION = 40
RESTING_POTENTIAL = 0

'''
This version of the model code requires running from terminal with arguments input file and duration.
Completes a single run with these parameters. Batch files can be made to do multiple runs.
'''
class Single_Model:
    """
    Creates a 2D grid network using NetworkX.
    The nodes in the network emulate the behaviour of neurons.
    The model iterates over itself and updates the nodes based on a set of rules.
    Takes an individual's genotype as input, and returns its phenotype.
    """
    def __init__(
            self,
            duration,
            model,
            resolution=RESOLUTION
    ):
        self.model, self.genotype, dimension, model_type = Data.load_model(model)
        self.model_type = model_type
        #   Firing Threshold in the membrane.
        #   (Default: 1) (Range: ~0.1-5.1)
        self.firing_threshold = (self.genotype[0] * 5) + 0.1
        #   Chance to randomly fire (Default: 0.05 (5%)) (Range: ~0-0.15)
        self.random_fire_prob = self.genotype[1] * 0.15
        #   Refractory period: time to recharge after firing.
        #   Subtracts this constant from the membrane potential when a neuron fires.
        #   (Default: 1) (Range: ~0-10)
        self.refractory_period = round(self.genotype[2] * 10)
        #   The distribution of inhibiting and exciting neurons.
        #   Determines the likelihood of setting a neuron's type to inhibitory.
        #   (Default: 0.25) (Range: ~0-0.5)
        self.inhibition_percentage = self.genotype[3] * 0.5
        #   By which ratio does the membrane potential passively move towards the
        #   resting potential every iteration. (Default: 0.1) (Range: ~0-0.2)
        self.leak_constant = self.genotype[4] * 0.2
        #   By which ratio does the input from the neighborhood integrate with the neuron
        #   (Default: 0.5) (Range: ~0-0.5)
        self.integ_constant = self.genotype[5] * 0.5
        #   For CA it determines the radius of connections. For the network model it determines number of conenctions.
        #   (Default: 2.1 (Network))
        #   Resting potential in the membrane (Default: 0.5)
        #   Currently not controlled by the algorithm
        self.rest_pot = RESTING_POTENTIAL
        self.step = 0
        self.duration = duration
        self.dimension = dimension
        #   How many iterations make up 1 second (Default: 50)
        self.resolution = resolution
        self.steps = self.duration * self.resolution
        self.electrodes = self.get_electrodes(dimension)
        #  Initialize Dataset
        self.spikes = []
        #  Initialize Network
        if model:
            self.config = self.model
        #  Copy Network
        self.next_config = self.config.copy()

    def alter_state(self, neuron, inp):
        """
        Return new state and membrane potential for a node/neuron.
        """
        dV = (self.leak_constant * (self.rest_pot - neuron['mem_pot']) + (self.integ_constant * inp))
        membrane_potential = neuron["mem_pot"] + dV
        if (membrane_potential >= self.firing_threshold or random.random() < self.random_fire_prob) and neuron[
            "refractory"] <= 0:
            return 1, self.rest_pot, self.refractory_period
        else:
            return 0, membrane_potential, max(0, neuron["refractory"] - 1)

    def update(self):
        """
        Apply the ruleset to the current Network and update the next iteration.
        """
        for node in self.config.nodes:
            in_potential = 0
            if self.config.nodes[node]["refractory"] == 0:
                neighbor_list = self.config.in_edges(node, data=True)
                for conn in neighbor_list:
                    state = self.config.nodes[conn[0]]["state"]
                    weight = conn[2]["weight"]
                    type = self.config.nodes[conn[0]]["type"]
                    in_potential += state * weight * type
                self.next_config.nodes[node]['state'], self.next_config.nodes[node]['mem_pot'], \
                self.next_config.nodes[node]["refractory"] = self.alter_state(self.config.nodes[node], in_potential)
            else:
                self.next_config.nodes[node]['state'], self.next_config.nodes[node]['mem_pot'], \
                self.next_config.nodes[node]["refractory"] = self.alter_state(self.config.nodes[node], 0)
        #  Update the configuration for the next iteration
        self.config, self.next_config = self.next_config, self.config
        #  Get the spikes from this iteration and append them to the list of spikes if there were any
        current_spikes = self.get_spikes()
        if current_spikes:
            self.spikes += current_spikes

    def get_spikes(self):
        """
        Get spikes in the current iteration.
        Return: List with spikes on electrodes in the network.
        """
        s = []
        for x, y in self.electrodes:
            if self.config.nodes[str((x, y))]["state"] == 1:
                s.append((0 + (self.step / self.resolution), self.electrodes.index((x, y))))
        return s if s else 0

    def get_electrodes(self, dimension):
        """
    	Return a list of electrode positions based on the size of the network.
    	The index of an electrode can be used as its ID.
    	"""
        el_list = []
        r = 0
        target = 8

        low = round((dimension % target) / 2)
        high = round(dimension - (dimension % target) / 2)

        for row in range(low, high, dimension // target):
            c = 0
            for col in range(low, high, dimension // target):
                if (r == 0 or r == 7) and (c == 0 or c == 7):
                    c += 1
                    continue
                else:
                    el_list.append((row, col))
                    c += 1
            r += 1
        return el_list

    def print_weights(self):
        '''
        Prints the weights of the network.
        '''
        for n, nbrs_dict in self.config.adjacency():
            for nbr, e_attr in nbrs_dict.items():
                if "weight" in e_attr:
                    print(e_attr)

    def show_network(self, grid=False):
        '''
        Uses the networkx library to display the network. Red nodes are inhibitory, while green are excitatory.
        '''
        edge_weights = []
        for e in self.config.edges(data=True):
            edge_weights.append(e[2]["weight"])
        node_colors = []
        for n in self.config.nodes(data=True):
            if n[1]["type"] == 1:
                node_colors.append("green")
            else:
                node_colors.append("red")
        plt.figure(figsize=(10, 10))
        if grid:
            p = {}
            for pos, node in zip(self.position, self.config.nodes):
                p[node] = pos
            nx.draw(self.config, p, edge_color=edge_weights, edge_cmap=plt.cm.Greys, node_color=node_colors,
                    node_size=50, width=edge_weights)
        else:
            nx.draw(self.config, edge_color=edge_weights, edge_cmap=plt.cm.Greys, node_color=node_colors, node_size=50,
                    width=edge_weights)
        plt.show()

    def run_simulation(self, plot=False):
        """
        Simulation loop.
        Return: Numpy array with spikes on electrode ID's.
        """
        while self.step < self.steps:
            self.update()
            self.step += 1
        if plot:
            self.show_network()
        #   Return phenotype
        return np.array(self.spikes, dtype=[("t", "float64"), ("electrode", "int64")])


def process_arguments():
    """
    Will process arguments given from terminal.
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
            print("Input file: ", input_file)
            duration = int(sys.argv[3])
            print("Duration: ", duration)
    return input_file, duration


#   Requires running from terminal with arguments input file and simulation duration.
if __name__ == "__main__":
    input, duration = process_arguments()
    # use model to generate a phenotype
    model = Single_Model(duration=duration, model=input)
    s = time.time()
    output = model.run_simulation()
    data = {"duration": duration, "spike_times": [float(x[0]) for x in output],
            "electrod_id": [int(x[1]) for x in output]}
    dir = "../Output/" + input + "/"
    with open(dir + "single_run_" + str(duration) + ".json", "w") as f:
        json.dump(data, f)
    print(len(output))
    print(f"{time.time() - s:.2f} seconds")
