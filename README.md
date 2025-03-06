# Project: Mean Field Game with Congestion Effects

## Overview
This project explores numerical solutions of a Mean Field problem with congestion effects (see the [subject](subject.pdf) for more details). Using finite difference schemes, we solve forward-backward PDE systems to characterize optimal strategies of agents. A short summary of our work, used for our project defense, is available in the file [slides](slides.pdf).

## Features
- Implementation of finite-difference schemes for HJB and KFP equations.
- Resolution of the MFG system using fixed-point iterations.
- Visualizations: Contour plots and animations of solutions.
- Analysis of algorithms statistics.

## Requirements
- Python libraries: `numpy`, `matplotlib`, `scipy`, `tqdm`, `pandas`, `seaborn`.
- Video framework: `ffmpeg` enables to produce `.mp4` videos, see [this documentation](https://example.com) for set up.

## Running the Simulation
Open the file [notebook.ipynb](notebook.ipynb) and follow these steps:
1. Set up parameters and run the solver using `MF_solver`.
2. Visualize the results with `contour_plot` and `evolution_video`.
3. All configurations can be automatically run (uncomment the corresponding cells) and numerical results, videos, and algorithms statistics are directly saved in the [results](results/) and [statistics](statistics/) folders.

## To go Further
Look at the [report](report.pdf) for a detailed description of the problem and numerical methods.

### Authors
- [Nathan Sanglier](https://github.com/Nathan-Sanglier)
- [Ronan Pécheul](https://github.com/Dracdarc)
