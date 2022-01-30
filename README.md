# Motion Planning Project

[![build](https://github.com/earasteh/Python-Motionplanning/actions/workflows/build.yml/badge.svg?branch=master)](https://github.com/earasteh/Python-Motionplanning/actions/workflows/build.yml)


<div align="center">
	<img src="resources/animation.gif" />
</div>


The animation part of this project is taken from https://github.com/winstxnhdw/KinematicBicycleModel 
and the motion planning part is taken from Coursera's class on motion planning.



Added features:
- Conformal Lattice Planner
- A 7 DoF vehicle model with Pacejka magic tire formula with algebraic normal forces
- Longitudinal PID controller
- 4th-Order Runge-Kutta (RK4) is added for the integration of the vehicle model instead of Euler

To Do:
- Motion controllers need more tuning (not working in a higher speeds)

## Abstract

## Advantages

## Limitations

## Requirements
Numpy, pandas, scipy, multiprocessing
## Demo

Install the requirements

```bash
pip install -r requirements.txt
```

Play the animation

```bash
python animation.py
```

## Folder structure
```bash
├── animate.py
├── data
│   ├── waypoints.csv
├── libs
│   ├── controllers
│   │   └── stanley_controller.py
│   ├── motionplanner
│   │   ├── collision_checker.py
│   │   ├── local_planner.py
│   │   ├── path_optimizer.py
│   │   └── velocity_planner.py
│   └── utils
│       ├── car_description.py
│       ├── cubic_spline_interpolator.py
│       ├── env.py
│       ├── normalise_angle.py
│       ├── plots.py
├── README.md
├── requirements.txt
├── resources
│   └── animation.gif
└── vehicle_model.py
```

## Concepts
- Conformal Lattice Planner
- Planar vehicle model
- Stanley Controller as a lateral control
- Longitudinal Controller