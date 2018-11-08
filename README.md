# Single-Pass Tensor Decomposition with Sketching

## Background

Given limited storage resources, many higher order datasets such as video, PDE, and weather benefit from being modeled as a streaming case, where data is in the form of linear updates. Even with these linear updates, storing sketches of the data may be needed to combat the problem of limited storage. This paper proposes a new low-rank tensor approximation sketch algorithm that only passes through the original tensor once during calculation. This algorithm also provides a theoretical approximation guarantee, as well as computational speed comparable to existing non-streaming algorithms. Simulations as well as experiments on real weather and PDE data show the efficiency of this algorithm. 

## Objective
- Minimize the communication cost by reducing the access to the original large tensor into a single pass 
- Enable a stream of input data stored in a distributed setting 

## Prerequisite

We implemented our algorithms in python3. Please first install the dependency packages by running the following commands in the terminal. 
```
pip install numpy
pip install matplotlib 
pip install scipy
pip install sklearn 
pip install tensorflow 
pip install -U tensorly
```

## Simulations 

Please refer to [simulation_weather.ipynb](https://github.com/udellgroup/tensorsketch/blob/master/examples/weather/simulation_weather.ipynb) for simulation on synthetic data to understand the theoretical properties of tensor sketching. 

To apply tensor sketching to real-world example, please refer to [simulation.ipynb](https://github.com/udellgroup/tensorsketch/blob/master/examples/simulation/simulation.ipynb) for the experiments on real-world weather data. In particular, [`run_realdata_frk`] evaluates the performance for HOOI (Higher Order Orthogonal Iteration), One-Pass sketching, Two-Pass sketching algorithms given the rank of the real data and the desired compression level. 


## Local Package installation 
Change the directory to the root repository (tensorsketch), and then run the following code in terminal. 
```
pip install -e .
```
You can use the package in python by loading
```
import tensorsketch
```
