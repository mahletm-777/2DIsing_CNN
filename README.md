# 2D Ising Model applied to Deep Neural Networks

For my graduate computational physics (PHYS 514) final project, I tried to answer the following question:
> *Can a neural network predict physical properties of a system using only the spatial features and without knowledge of the physics needed to understand the problem*

The short answer is yes! This repository demonstrates how deep learning can be used to generalize and predict physical patterns from simulated data. Specifically, I trained a Convolutional Neural Network (CNN) to distinguish between the *ordered* and *disordered* states in a simple 2D magnetic system, and tested the accuracy in transferring this knowledge to a **different underlying structure** it has never seen before. 

## Repo Structure
2DIsing_CNN/
- notebooks/
  - MCMC.ipynb
  - Triangular_MCMC.ipynb
  - CNN.ipynb
  - figures (*.png)
- data/
  - ordered_configs.npy
  - disordered_configs.npy
  - tri_ordered_configs.npy
  - tri_disordered_configs.npy
- reports/
  - Final_project.pptx
  - PHYS_514_Final_Project_Report.pdf
- README.md

## High-Level Overview of Problem

The system consists of 2D grids where each site (representing a spin up/down) interacts with its neighbors. Depending on a global control parameter (e.g. Temperature), the grid exhibits either:
>*Ordered* behavior: Large regions of aligned values

>*Disordered* behavior: Near-random patterns

For the context of this project, I only considered an **$N\times N$ square lattice** and **$N\timesN triangular lattice**. 

### Metropolis-Hastings algorithm

I utilized this algorithm to simulate the lattice systems in accordance with the 2D Ising Model. The algorithm produces equilibrium configurations across a range of temperatures, yielding labeled examples of ordered and disordered states.

To ensure physical and numerical consistency, the simulations were validated using:
* Physical order parameters (e.g. Magnetization)
* Finite-size scaling across different grid sizes
* Visualization of spin configurations for qualitative sanity checks

### Deep Learning Approach

I constructed a CNN to be trained to classify configurations from the square lattice into ordered or disordered phases using raw grid data as input. The trained model was then evaluated on triangular lattice configurations to assess whether the learned representation captures general spatial structure rather than lattice-specific details.

To analyze how learning rate choices affect convergence and generalization, I performed a comparitive study of Adam and Stochastic Gradient Descent (SGD) optimizers when using a Cross-Entropy loss function.
