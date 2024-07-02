# E(*n*)-equivariant Graph Cellular Automata

This repository is a fork of https://github.com/gengala/egnca attempting to improve model expressiveness and remove fixed edges during training

**N.B.** Sometimes GIFs are not properly loaded in `README.md`. Please refresh the page.  

## Angles

<img src="./result_figures/angles-vs-original-animation.gif" width="600" height="300">

    python -m trainers.geometric_graph -ds Grid2d -sdg 0.05 -rdg 1.0 -bsc 0 8 2000 16 4000 32 -pats 800 -an --seed 42
    python -m trainers.geometric_graph -ds Grid2d -sdg 0.05 -rdg 1.0 -bsc 0 8 2000 16 4000 32 -pats 800     --seed 42

## Local loss

Non-persistent           |  Persistent
:-------------------------:|:-------------------------:
<img src="./result_figures/simple-local-50-animation.gif" width="300" height="300"> | <img src="./result_figures/simple-local-pers-500-animation.gif" width="300" height="300">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l local -an --seed 42
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 8 2000 16 4000 32 -pats 800 -l local -an --seed 42

### Anchors

<img src="./result_figures/local-no_penalty-ss-comparison-50.gif" width="900" height="600">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l local -ss -as corners -an --seed 42
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l local -ss -as corners -asc 1.5 -an --seed 42
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l local -ss -as simplex -an --seed 42

### Penalty

<img src="./result_figures/grid-local-enp-ss_comparison.gif" width="600" height="300">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 4 -l local_enp -an --seed 42
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 8 2000 16 4000 32 -l local_enp -pd 0.3 -ss -as corners -asc 1.5 -an --seed 42

### Different targets

<!-- <img src="./result_figures/local-diff-shapes-animation.gif" width="300" height="300"> -->
<img src="https://drive.google.com/uc?export=view&id=1plY13LX7eWIIyKf01JtWft24_8qHUZPa" width="900" height="300">

IF the gif does not load, check it out here: https://drive.google.com/uc?export=view&id=1plY13LX7eWIIyKf01JtWft24_8qHUZPa

    python -m trainers.geometric_graph -ds Cube -sdg 0.05 -rdg 1.0 -bsc 0 2 -l local_enp -pd 0.2 -ss -as corners -ad 0.15 -asc 0.5 -an --seed 42
    python -m trainers.geometric_graph -ds Torus -bsc 0 8 2000 16 -l local_enp -an -at undirected -ss -as corners -ad 0.5
    python -m trainers.geometric_graph -ds x_small -bsc 0 8 2000 16 4000 32 -l local_enp -pd 0.2 -ss -as corners --seed 42

## OT-based loss
### Anchors

<img src="./result_figures/bare_ot-t50-animation.gif" width="600" height="300">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l ot -ss -as simplex --seed 42

### Penalty

<img src="./notebooks/grid otp ss simplex nodenorm.gif" width="600" height="300">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l ot_p -ss -as simplex --seed 42 -nt nn

### Static and dynamic relative edges
#### Static

<img src="./notebooks/grid otp ss simplex re nodenorm.gif" width="600" height="300">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l ot_p -ss -as simplex -re -en 4 --seed 42 -nt nn

#### Dynamic, every step

<img src="./result_figures/ot-p-des1-traj-animation.gif" width="300" height="300">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l ot_p -ss -as simplex -re -de -en 4 --seed 42

#### Dynamic, every 5 steps

<img src="./result_figures/grid otp ss simplex dyn animation.gif" width="600" height="600">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 8 2000 16 -l ot_p -pd 0.3 -an -ss -as corners -re -de -en 4 -des 5 -ad 0.3 --seed 42
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 8 2000 16 -l ot_p -pd 0.3     -ss -as simplex -re -de -en 4 -des 5 -ad 0.3 --seed 42

#### Dynamic, no penalty

<img src="./result_figures/grid_ot_divergent_animation.gif" width="300" height="300">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l ot -at undirected -an -ffa -ss -as corners -asc 1.5 -uaf -re -de -des 5 -en 5 --seed 123

### Beacons
#### No persistency

<img src="./result_figures/grid-otp-bc-pd_comp-edges-animation.gif" width="600" height="600">

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l ot_p -apd -an -bc -as simplex -re -de -en 4 -des 5 --seed 42'
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l ot_p -pd 0.3 -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l ot_p -pd 0.4 -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 1 -l ot_p -pd 0.5 -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn

#### Persistency

<!-- <img src="./result_figures/grid-pers-otp-bc-pd_comp-edges-250_steps.gif" width="300" height="300"> -->
<img src="https://drive.google.com/uc?export=view&id=1v0uKqbdVi8sYXpQh4ufRqNDLKZV7iQM7" width="600" height="600">

IF the gif does not load, check it out here: https://drive.google.com/uc?export=view&id=1v0uKqbdVi8sYXpQh4ufRqNDLKZV7iQM7

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 8 2000 16 4000 32 -l ot_p -apd -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 8 2000 16 4000 32 -l ot_p -pd 0.3 -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 8 2000 16 4000 32 -l ot_p -pd 0.4 -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn
    python -m trainers.geometric_graph -ds Grid2d -bsc 0 8 2000 16 4000 32 -l ot_p -pd 0.5 -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn

### Robustness
All robustness experiments are performed with the same model:

    python -m trainers.geometric_graph -ds Grid2d -bsc 0 8 2000 16 4000 32 -l ot_p -pd 0.3 -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn

#### Random initial coordinates

<img src="./result_figures/grid-robust-ric-animation.gif" width="300" height="300">

#### Damage

Local damage               | Global damage
:-------------------------:|:-------------------------:
<img src="./result_figures/grid-robust-damage-Local-animation.gif" width="300" height="300"> | <img src="./result_figures/grid-robust-damage-Global-animation.gif" width="300" height="300">

#### Increased or decreased node count

<!-- <img src="./result_figures/grid-robust-node_count_factor-edges-animation.gif" width="1500" height="600"> -->
<img src="https://drive.google.com/uc?export=view&id=140wQ-GTYflekb7xVrBeOxdZHsupSkGIJ" width="1500" height="600">

IF the gif does not load, check it out here: https://drive.google.com/uc?export=view&id=140wQ-GTYflekb7xVrBeOxdZHsupSkGIJ

#### Decreased fire rate

<!-- <img src="./result_figures/grid-robust-fire_rate-animation.gif" width="300" height="300"> -->
<img src="https://drive.google.com/uc?export=view&id=1qUGOCaR8KIUvhIQ5yuampMZqxfGTuVsr" width="900" height="300">

IF the gif does not load, check it out here: https://drive.google.com/uc?export=view&id=1qUGOCaR8KIUvhIQ5yuampMZqxfGTuVsr

#### Rotated structured seeds
E(*n*) Convergence to 2D-Grid           |  Adapting to rotated structured seeds
:-------------------------:|:-------------------------:
 <img src="./notebooks/-ds Grid2d -bsc 0 8 2000 16 4000 32 -l ot_p -pd 0.3 -bc -as simplex -re -de -en 4 -des 5 --seed 42 -nt nn-animation.gif" width="300" height="300"> | <img src="./result_figures/grid-robust-rotate-animation.gif" width="300" height="300">

### Different targets
#### X and a

<img src="./result_figures/x-a-comparison-50.gif" width="600" height="600">

#### Cube and Torus

<img src="./result_figures/torus-cube-animation.gif" width="600" height="300">

For testing, play with `notebooks/test_geometric_graph.ipynb` and `notebooks/visualise_results.ipynb`.

