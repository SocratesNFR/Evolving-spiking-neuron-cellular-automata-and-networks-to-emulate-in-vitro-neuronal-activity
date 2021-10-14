# Evolving-spiking-neuron-cellular-automata-and-networks-to-emulate-in-vitro-neuronal-activity
Code accompanying "Evolving spiking neuron cellular automata and networks to emulate in vitro neuronal activity" [1], 
accepted to the International Conference on Evolvable Systems (IEEE SSCI 2021).

ICES page:
https://attend.ieee.org/ssci-2021/international-conference-on-evolvable-systems-ices/

STRUCTURE:\
There are two folders in the main directory.

Resources contains the neural data used in this study as .txt files. The data were collected by Wagenaar et al. [2], 
and the full open dataset can be found here: http://neurodatasharing.bme.gatech.edu/development-data/html/index.html

Each file contains the time (column 1) and recording channel (column 2) of each spike detected in the data.

The project code is found in the src-folder. The code to run the models and evolutionary algorithm is found here.
Additionally there is a separate folder for plotting results.

RUNNING SINGLE MODEL:\
A single model with desired parameters can be run with the Model.py file. Parameters are set at the top of this file.

RUNNING EVOLUTIONARY ALGORITHM:\
To run the evolutionary algorithm, the Main.py file is run and parameters are set in the default_parameters dict.

RUNNING SAVED MODEL:\
To run a saved model, the RunSavedModel.py files is run from terminal with the first argument being the GraphML file
and the second argument being simulation duration in seconds.

RUNNING BATCH FILES:\
Multiple simulations can be run by passing batch files as arguments when running Main.py. Batch files must be .csv
files. An example can be seen in batch_example.csv. Each row is a separate run.

EXTERNAL LIBRARIES:
- Pandas
- Numpy
- NetworkX
- Scipy
- Matplotlib
- Pylab
- Seaborn
- Pandas

[1] J Jensen Farner, H Weydahl, CR Jahren, O Huse Ramstad, S Nichele, and K Heiney. "Evolving spiking neuron cellular
automata and networks to emulate in vitro neuronal activity," International Conference on Evolvable Systems
(IEEE Symposium Series on Computational Intelligence 2021), 2021. 

[2] DA Wagenaar, J Pine, and SM Potter, "An extremely rich repertoire of bursting patterns during teh development of 
cortical cultures," BMC Neuroscience, 7(1):11, 2006.