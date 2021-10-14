"""
Methods for data processing.
"""
import networkx as nx
import numpy as np
import pandas as pd
import re
import json


def get_spikes_file(filename, recording_start, recording_len, fullpath=False):
    """
    Get spikes per electrode data from a file.
    Uses helper method get_spikes_pheno.
    Return: Numpy Array
    """
    #   Clean up the data and create a numpy array from it
    if fullpath:
        f = open(filename, "r")
    else:
        f = open("../Resources/" + filename, "r")
    data_points = [line.split(" ") for line in f]
    data_points = np.array(
        [(row[0].rstrip(), row[1].rstrip()) for row in data_points], 
        dtype=[("t", "float64"), ("electrode", "int64")])
    
    #   Edit according to specified recording length
    start_index, stop_index = np.searchsorted(data_points["t"], [recording_start, recording_start+recording_len])
    data_points = data_points[start_index:stop_index]
    return data_points


def get_spikerate(phenotype, recording_len, recording_start):
    """
    Get spikes per electrode data from input phenotype.
    Return: Numpy Array
    """
    array_wide_spikes_per_second = pd.cut(
        phenotype["t"],
        bins=pd.interval_range(start=recording_start, end=recording_start + recording_len),
        precision=0
    )
    return np.array(array_wide_spikes_per_second.value_counts().tolist(), dtype="float64")



def read_recording(filename, recording_start, recording_len):
    """
    Takes as input recording of experimental data as text file
    Returns a phenotype
    """
    #   Clean up the data and create a numpy array from it
    f = open("../Resources/" + filename, "r")
    data_points = [line.split(" ") for line in f]
    data_points = np.array(
        [(row[0].rstrip(), row[1].rstrip()) for row in data_points],
        dtype=[("t", "float64"), ("electrode", "int64")])
    #   Edit according to specified recording length
    start_index, stop_index = np.searchsorted(data_points["t"], [recording_start, recording_start+recording_len])
    data_points = data_points[start_index:stop_index]
    f.close()
    return data_points

def load_model(filename):
    '''
    Loads a graph object with nodes, edges and properties.
    '''
    model = nx.read_graphml("../Output/" + filename + "/model.gml")
    file = open("../Output/" + filename + "/evolution_data.json")
    data = json.load(file)
    dimension = data["DIMENSION"]
    model_type = data["MODEL_TYPE"][0]
    info = open("../Output/" + filename + "/Info.txt", "r")
    top_ind = ""
    for i, line in enumerate(info):
        if line.startswith("| TOP INDIVIDUAL |"):
            top_ind = line
            break
    info.close()
    genome = [float(i) for i in re.findall("\[(.+)\]", top_ind)[0].split(", ")]
    return model, genome, dimension, model_type


