"""
Methods for fitness calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from src import Data
from collections import Counter


def get_fitness_corr(spike_rate_X, spike_rate_control, plot_graph=False):
    """
    https://www.mathworks.com/help/matlab/ref/xcorr.html
    Takes two phenotypes as input at returns the best normalized cross-correlation score
    Can also plot the xcorr graph if flag is set
    Calls xcorr function in matplotlib which returns an array
    """
    #   Check if array is only zeros
    if spike_rate_X.any():
        corr = plt.xcorr(spike_rate_X, spike_rate_control, usevlines=True, normed=True, lw=2, maxlags=1)
        fitness_score = np.amax(corr[1])
    else:
        fitness_score = 0.0
    #   Debugging purposes
    if plot_graph:
        # # lag of best fit
        lag = corr[0][np.where(corr[1] == fitness_score)]
        print(lag)
        plt.show()
    return "NA", "NA", fitness_score


def get_fitness_spike_dist(spike_rate_x, spike_rate_control, order=6, threshold=0.6):
    """
    Calculates fitness from a combination of burst correlation
    and average distance between the spikes per second data points.
    Counts the number of bursts by checking if the number of spikes exceed a threshold relative to the maximum value.
    """
    #   Calculate the maximum value in the reference data
    max_val = max([spike_rate_control[i] for i in signal.argrelextrema(spike_rate_control, np.greater, order=order)[0]])
    #   Look for bursts in the experiment data
    control_bursts = signal.argrelextrema(spike_rate_control, np.greater, order=order)[0]
    #   Look for bursts in the model data
    x_bursts = signal.argrelextrema(spike_rate_x, np.greater, order=order)[0]
    #   Count the number of bursts, values that exceed the threshold.
    adj_control_bursts = len(
        [spike_rate_control[i] for i in control_bursts if spike_rate_control[i] >= max_val * threshold])
    adj_x_bursts = len(
        [spike_rate_x[i] for i in x_bursts if spike_rate_x[i] >= max_val * threshold])
    #   Calculate correlation between bursts in the reference data and the model data
    burst_corr = (adj_control_bursts - abs(adj_control_bursts - adj_x_bursts)) / adj_control_bursts
    if burst_corr < 0:
        burst_corr = 0
    #   Sort spike rate lists
    a = sorted(spike_rate_control)
    b = sorted(spike_rate_x)
    dist = []
    #   Append the distance between each data point in the lists to a new list.
    for i, j in zip(a, b):
        dist.append(abs(i - j))
    #   Calculate the average distance
    avg_dist = (max(spike_rate_control) - (sum(dist) / len(dist))) / max(spike_rate_control)
    if avg_dist < 0:
        avg_dist = 0
    fitness = (burst_corr + avg_dist) / 2
    return burst_corr, avg_dist, fitness


def get_fitness_sorted_dist_ned(phenotype, phenotype_control, duration):
    """
    Sorts the lists of spike data from both the experiment and the model.
    The average distance between them is returned as the fitness score.
    """
    #   Sort spike rate lists
    spike_rate_x = Data.get_spikerate(phenotype, duration, recording_start=0)
    spike_rate_control = Data.get_spikerate(phenotype_control, duration, recording_start=900)
    a = np.array(sorted(spike_rate_x))
    b = np.array(sorted(spike_rate_control))
    return normalized_euclidean_distance(a, b)


def get_fitness_normalized_dist(phenotype, phenotype_control, duration, resolution):
    '''
    Calculates average normalized distance between two arrays of spike rates.
    Normalized based on the average value of the average spikerate of the dataset and the distance from this average
    to the theoretical maximum spikerate for one second simulation.
    '''
    spike_rate_x = Data.get_spikerate(phenotype, duration, recording_start=0)
    spike_rate_control = Data.get_spikerate(phenotype_control, duration, recording_start=900)
    dist = np.mean(abs(spike_rate_x - spike_rate_control))
    mean_control = np.mean(spike_rate_control)
    normalized_dist = 0.5 * (mean_control + abs(mean_control - resolution * 60))
    fitness = (normalized_dist - dist) / normalized_dist
    return fitness


def get_fitness_electrode_dist_ned(phenotype, phenotype_control):
    '''
    Calculates the normalized euclidean distance of the sorted lists of total number of spikes per electrode.
    '''
    cnt_x = Counter(elem[1] for elem in phenotype)
    cnt_control = Counter(elem[1] for elem in phenotype_control)
    sorted_cnt_x = sorted(cnt_x.values())
    sorted_cnt_control = sorted(cnt_control.values())
    sorted_cnt_x = np.pad(np.array(sorted_cnt_x), (60 - len(sorted_cnt_x), 0))
    sorted_cnt_control = np.pad(np.array(sorted_cnt_control), (60 - len(sorted_cnt_control), 0))
    return normalized_euclidean_distance(sorted_cnt_x, sorted_cnt_control)


def get_fitness_electrode_dist(phenotype, phenotype_control, duration):
    '''
    Calculates the normalized average distance between two sorted lists of spikerates per electrode.
    Normalized based on the average electrode spike rate of the dataset.
    '''
    cnt_x = Counter(elem[1] for elem in phenotype)
    cnt_control = Counter(elem[1] for elem in phenotype_control)
    x_avg = [x / duration for x in cnt_x.values()]
    control_avg = [x / duration for x in cnt_control.values()]
    sorted_x = np.array(sorted(x_avg))
    sorted_control = np.array(sorted(control_avg))
    sorted_cnt_x = np.pad(np.array(sorted_x), (60 - len(sorted_x), 0))
    sorted_cnt_control = np.pad(np.array(sorted_control), (60 - len(sorted_control), 0))
    dist = abs(sorted_cnt_x - sorted_cnt_control)
    mean_control = np.mean(sorted_cnt_control)
    mean_dist = np.mean(dist)
    fitness = (mean_control - mean_dist) / mean_control
    if fitness < 0:
        fitness = 0
    return fitness


def get_fitness_sorted_dist(phenotype, phenotype_start, phenotype_control, phenotype_control_start, duration):
    '''
    Calculates the normalized average distance of two sorted list of spikerates.
    Normalized based on mean spikerate of datafile.
    '''
    spike_rate = Data.get_spikerate(phenotype, recording_len=duration, recording_start=phenotype_start)
    spike_rate_control = Data.get_spikerate(phenotype_control, recording_len=duration,
                                            recording_start=phenotype_control_start)
    a = np.array(sorted(spike_rate))
    b = np.array(sorted(spike_rate_control))
    dist = abs(a - b)
    mean_control = np.mean(spike_rate_control)
    fitness = (mean_control - np.mean(dist)) / mean_control
    if fitness < 0:
        fitness = 0
    return fitness


def normalized_euclidean_distance(x, y):
    '''
    Calculates the normalized euclidean distance.
    '''
    return 1 - (np.std(x - y) ** 2) / (np.std(x) ** 2 + np.std(y) ** 2)


def get_fitness(phenotype, phenotype_start, phenotype_control, phenotype_control_start, duration):
    '''
    Returns the fitness of the phenotype.
    '''
    fitness = get_fitness_sorted_dist(phenotype, phenotype_start, phenotype_control, phenotype_control_start, duration)
    return fitness
