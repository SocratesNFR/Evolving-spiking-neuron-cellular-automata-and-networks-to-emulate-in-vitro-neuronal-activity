from src import Population
from pylab import *
import networkx as nx
import math as m
import numpy as np
import time
import itertools
import random

'''
Default values for Model class
'''
# Model-type. "network" or "ca".
MODEL = "network"
# Duration of simulation in seconds.
DURATION = 60
# Dimensions of model in number of neurons.
DIMENSION = 10
# Number of simulation updates per second.
RESOLUTION = 40
# Whether bidirectional connections should be allowed or not.
BIDIRECTIONAL = False
# Potential at rest.
RESTING_POTENTIAL = 0
# Threshold for firing spike.
FIRING_THRESHOLD = 1
# Probability for a neuron to spontaneously spike.
RANDOM_FIRE_PROBABILITY = 0.05
# Refractory period in simulation time-steps.
REFRACTORY_PERIOD = 1
# Leak constant of the LIF model.
LEAK_CONSTANT = 0.05
# integration constant of the LIF model.
INTEGRATION_CONSTANT = 0.25
# Determines connection radius for CA and density constant for Network.
DENSITY_CONSTANT = 2.1
# Percentage of neurons that are inhibitory.
INHIBITION_PERCENTAGE = 0.25
#   The Default Individual for testing. Normalizes values.
INDIVIDUAL = Population.Individual(
    [(FIRING_THRESHOLD - 0.1) / 5,
     RANDOM_FIRE_PROBABILITY / 0.15,
     REFRACTORY_PERIOD / 10,
     INHIBITION_PERCENTAGE / 0.5,
     LEAK_CONSTANT / 0.2,
     INTEGRATION_CONSTANT / 0.5,
     (DENSITY_CONSTANT - 0.1) / 4
     ]
)


def test_class():
    """
    Run the model/simulation with defaults and plot the results.
    """
    from src.Summary import make_raster_plot
    # use model to generate a phenotype
    model = Model()
    s = time.time()
    output = model.run_simulation()
    print(f"{time.time() - s:.2f} seconds")
    # generate reference phenotype from experimental data
    reference_file = {
        "small": "../Resources/Small - 7-1-35.spk.txt",
        "dense": "../Resources/Dense - 2-1-20.spk.txt"
    }
    #  Compare model output with experimental data
    make_raster_plot(reference_file["small"], output, DURATION)
    # Plot the network topology
    model.show_network(grid=True)


class Model:
    """
    Creates a 2D grid network using NetworkX.
    The nodes in the network emulate the behaviour of neurons.
    The model iterates over itself and updates the nodes based on a set of rules.
    Takes an individual's genotype as input, and returns its phenotype.
    """

    def __init__(
            self,
            individual=INDIVIDUAL,
            model=MODEL,
            dimension=DIMENSION,
            duration=DURATION,
            resolution=RESOLUTION
    ):
        self.model = model
        #   Firing Threshold in the membrane.
        #   (Default: 1) (Range: ~0.1-5.1)
        self.firing_threshold = (individual.genotype[0] * 5) + 0.1
        #   Chance to randomly fire (Default: 0.05 (5%)) (Range: ~0-0.15)
        self.random_fire_prob = individual.genotype[1] * 0.15
        #   # Refractory period in simulation time-steps.
        #   (Default: 1) (Range: ~0-10)
        self.refractory_period = round(individual.genotype[2] * 10)
        #   The distribution of inhibiting and exciting neurons.
        #   Determines the likelihood of setting a neuron's type to inhibitory.
        #   (Default: 0.25) (Range: ~0-0.5)
        self.inhibition_percentage = individual.genotype[3] * 0.5
        #   By which ratio does the membrane potential passively move towards the
        #   resting potential every iteration. (Default: 0.1) (Range: ~0-0.2)
        self.leak_constant = individual.genotype[4] * 0.2
        #   By which ratio does the input from the neighborhood integrate with the neuron
        #   (Default: 0.5) (Range: ~0-0.5)
        self.integ_constant = individual.genotype[5] * 0.5
        #   For CA it determines the radius of connections. For the network model it determines the density of connections.
        #   (Default: 2.1 (Network))
        if self.model == "ca":
            self.density_constant = round(individual.genotype[6] * 5) + 1
        elif self.model == "network":
            self.density_constant = (individual.genotype[6] * 4) + 0.1
        else:
            raise Exception("Invalid model chosen.")
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
        self.config = nx.DiGraph()
        self.create_nodes()
        self.node_list = list(self.config.nodes)
        if self.model == "network":
            for node in range(len(self.node_list)):
                self.create_distance_connections(node)
        elif self.model == "ca":
            for node in self.config.nodes:
                self.create_grid_connections(node)
        else:
            raise Exception("Invalid model chosen...")
        #  Copy Network for next simulation step.
        self.next_config = self.config.copy()

    def create_nodes(self):
        '''
        Creates the nodes of the network with state values based on parameters.
        '''
        self.position = list(itertools.product(range(self.dimension), range(self.dimension)))
        for pos in self.position:
            self.config.add_node(pos)
            node = self.config.nodes[pos]
            node['mem_pot'] = self.rest_pot
            if random.random() > self.inhibition_percentage:
                node["type"] = 1
            else:
                node["type"] = -1
            if random.random() < self.random_fire_prob:
                node['state'] = 1
                node['refractory'] = self.refractory_period
            else:
                node['state'] = 0
                node['refractory'] = 0

    def create_distance_connections(self, node):
        '''
        Method for creating connections for network based on distance-equation described further in the report.
        '''
        pos = self.node_list[node]
        for n in range(0 if BIDIRECTIONAL else node + 1, len(self.node_list)):
            distance = m.sqrt(((pos[0] - self.node_list[n][0]) ** 2) + ((pos[1] - self.node_list[n][1]) ** 2))
            p = m.exp(-((distance / self.density_constant) ** 2))
            if p >= random.random():
                weight = 1
                order = random.choice([(pos, self.node_list[n]), (self.node_list[n], pos)])
                self.config.add_edge(order[0], order[1], weight=weight)

    def create_grid_connections(self, node):
        '''
        Method for creating the grid connections of the CA.
        '''
        for x in range(node[0] - round(self.density_constant), node[0] + round(self.density_constant) + 1):
            for y in range(node[1] - round(self.density_constant), node[1] + round(self.density_constant) + 1):
                if 0 <= x < self.dimension and 0 <= y < self.dimension and (x != node[0] or y != node[1]):
                    self.config.add_edge(node, (x, y), weight=1)
                else:
                    continue

    def create_grid_random_connections(self):
        '''
        Method for creating grid connections for network model. Currently not used.
        '''
        max_coordinate = self.dimension - 1
        max_distance = m.sqrt(max_coordinate ** 2 + max_coordinate ** 2)
        for node in self.config.nodes:
            for g in [(node[0], node[1] + 1), (node[0], node[1] - 1), (node[0] + 1, node[1]), (node[0] - 1, node[1])]:
                if g in self.config.nodes:
                    self.config.add_edge(node, g, weight=1)
            current_neighbors = self.config.out_edges(node, data=True)
            potential_neighbors = list(self.config.nodes)
            potential_neighbors.remove(node)
            for re in current_neighbors:
                if re[1] in potential_neighbors:
                    potential_neighbors.remove(re[1])
            new_neighbors = np.random.choice(range(len(potential_neighbors)), self.density_constant)
            for n in new_neighbors:
                neighbor = potential_neighbors[n]
                distance = m.sqrt(((node[0] - neighbor[0]) ** 2) + ((node[1] - neighbor[1]) ** 2))
                weight = round((max_distance - distance) / max_distance, 2)
                self.config.add_edge(node, neighbor, weight=weight)

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
            if self.config.nodes[(x, y)]["state"] == 1:
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
            nx.draw(self.config, p, edge_cmap=plt.cm.Greys, node_color=node_colors, node_size=50, width=edge_weights)
        else:
            nx.draw(self.config, edge_cmap=plt.cm.Greys, node_color=node_colors, node_size=50, width=edge_weights)
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


#   Run the class test and print the result when the script is run standalone.
if __name__ == "__main__":
    test_class()
