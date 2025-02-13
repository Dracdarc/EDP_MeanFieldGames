# Project: Mean Field Game with Congestion Effects

## Overview
This project explores numerical solutions of a Mean Field problem with congestion effects (see the [subject](subject.pdf) for more details). Using finite difference schemes, we solve forward-backward PDE systems to characterize optimal strategies of agents.

## Features
- Implementation of finite-difference schemes for HJB and KFP equations.
- Resolution of the MFG system using fixed-point iterations.
- Visualizations: Contour plots and animations of solutions.

## Requirements
- Python libraries: `numpy`, `matplotlib`, `scipy`, `tqdm`.
- Video framework: `ffmpeg` enables to produce `.mp4` videos, see [this documentation](https://example.com) for set up.

## Running the Simulation
Open the file [notebook.ipynb](notebook.ipynb) and follow these steps:
1. Set up parameters and run the solver using `MF_solver`.
2. Visualize the results with `contour_plot` and `evolution_video`.
3. By default, all configurations will be run and numerical results and videos are automatically saved in the [results](results/) folder.

## To go Further
Look at the [report](report.pdf) for a detailed description of the problem and numerical methods.

### Authors
- [Nathan Sanglier](https://github.com/Nathan-Sanglier)
- [Ronan Pécheul](https://github.com/Dracdarc)