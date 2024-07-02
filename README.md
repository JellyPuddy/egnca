# E(*n*)-equivariant Graph Cellular Automata

This repository is a fork of https://github.com/gengala/egnca attempting to improve model expressiveness and remove fixed edges during training

**N.B.** Sometimes GIFs are not properly loaded in `README.md`. Please refresh the page.  

## Angles

<img src="./result_figures/angles-vs-original-animation.gif" width="300" height="300">

    python -m trainers.geometric_graph -ds Grid2d -sdg 0.05 -rdg 1.0 -bsc 0 8 2000 16 4000 32 -pats 800 -an --seed 42
    python -m trainers.geometric_graph -ds Grid2d -sdg 0.05 -rdg 1.0 -bsc 0 8 2000 16 4000 32 -pats 800     --seed 42

## Robustness
E(*n*) Convergence to 2D-Grid           |  Adapting to rotated structured seeds
:-------------------------:|:-------------------------:
 <img src="./notebooks/-ds Grid2d -bsc 0 8 2000 16 4000 32 -l ot_p -pd 0.3 -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn-animation.gif" width="300" height="300"> | <img src="./result_figures/grid-robust-rotate-animation.gif" width="300" height="300">


    python -m trainers.geometric_graph -ds Grid2d -sdg 0.05 -rdg 1.0 -bsc 0 8 2000 16 4000 32 -pats 800
    python -m trainers.geometric_graph -ds Torus  -sdg 0.05 -rdg 1.0 -bsc 0 6 1000 8 2000 16 4000 32 -pats 800
    python -m trainers.geometric_graph -ds Cube   -sdg 0.05 -rdg 1.0 -bsc 0 16
    python -m trainers.geometric_graph -ds Bunny  -sdg 0.05 -rdg 1.0 -bsc 0 4 1000 8 2000 16 4000 32

For testing, play with `notebooks/test_geometric_graph.ipynb` and `notebooks/visualise_results.ipynb`.

