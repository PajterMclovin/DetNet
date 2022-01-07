## Detector reconstruction of gamma-rays
This directory contains much of the Python code used in the bachelors thesis (https://odr.chalmers.se/handle/20.500.12380/301461?locale=sv), which was used as an interface for easily setting up custom designed neural networks, preprocessing training data and defining any kind loss function, all this to make lengthy parameter sweeps of any relevant hyper parameters (see eg. depth_width.py).

## Requirements
Python packages:
- tensorflow-gpu
- numpy
- matplotlib
- scipy

## Quick tutorial
The script example.py creates a simple fully connected neural network and trains it on GEANT4 generated data. As input it receives 162 nonnegative values (detected energy deposit in the Crystal ball detector array). Since the dataset is generated using a maximimum of two photons (maximum multiplicity), the network is set up with this as an assumption and the output therefore consists of six values, the xyz-components of each photon's mommentum vector. There is an ambiguity in this way of defining the output layer: for a maximum multiplicity m, there's m! number of ways for the network to list the momenta of the m photons. The way we solved this issue was to consider all m! possible combinations that the network could have produced and use the combination that minimized the loss function. A obvious drawback with this method, is that it scales as a factorial (if I remember correctly, we could barely train networks with m=6 or higher).

## Questions
If there's any questions about the code or our thesis, you could mail them to me at: pethalld@student.chalmers.se :)

/ Peter Halldestam 7/1/2022
