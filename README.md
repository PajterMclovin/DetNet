## Detector reconstruction of gamma-rays
This directory contains much of the Python code used in the bachelors thesis (https://odr.chalmers.se/handle/20.500.12380/301461?locale=sv), which was used as an interface for easily setting up custom designed neural networks, preprocessing training data and defining any kind loss function, all this to make lengthy parameter sweeps of any relevant hyper parameters (see eg. depth_width.py).

## Requirements
Python packages:
- tensorflow-gpu
- numpy
- matplotlib
- scipy

## Quick tutorial
The script example.py is a simple illustration of a neural network which can be trained on GEANT4 generated data, if provided (!). In the script, the network receives 162 nonnegative values of data (detected energy deposit in the Crystal ball detector array) as input. The number of output nodes is determined from the dimensions of the training labels, assuming they consist at most of N photons (maximum multiplicity) and have the following form: Y = (p1x, p1y, p1z, ... pNx, pNy, pNz) where p1x is the reconstructed photon momentum in the x-direction and so on yielding 3N nodes. If the network thinks there is fewer than N photons, say only 2 out of 3 for instance, it should return one of the following as output:

(p1x, p1y, p1z, p2x, p2y, p2z, 0, 0, 0),  
(p1x, p1y, p1z, 0, 0, 0, p3x, p3y, p3z),  
(0, 0, 0, p2x, p2y, p2z, p3x, p3y, p3z).

If this reconstruction is accurate in that there's 2/3 photons, then the corresponding training label will be similar to one of these. The loss function L is evaluated for all N! possible permutations and the one minimizing L is used while training the network. (For more details, see section 4.3.3 in our thesis)

After the training is over, the learning curve, the event reconstruction of evaluation data is saved along with a .h5 file with the current state of the network parameters.

## Questions
If there's any questions about the code/our thesis, you are more than welcome to mail them to me at: pethalld@student.chalmers.se and I'll do my best to answer! :)

/ Peter Halldestam 7/1/2022
