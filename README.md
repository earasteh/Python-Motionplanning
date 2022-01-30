# Motion Planning Project

<div align="center">
	<img src="resources/animation.gif" />
</div>


The animation part of this project is taken from https://github.com/winstxnhdw/KinematicBicycleModel 
and the motion planning part is taken from Coursera's class on motion planning.



Added features:
- Conformal Lattice Planner
- A 7 DoF vehicle model with Pacejka magic tire formula (Constant normal forces)
- Longitudinal PID controller
- 4th-Order Runge-Kutta (RK4) is added for the integration of the vehicle model instead of Euler

To Do:
- Motion controllers need more tuning (not working in a higher speeds)
- Change the chassis model to have algebraic normal forces instead of constant normal forces

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

## Concepts
- Conformal Lattice Planner
- Planar vehicle model
- Stanley Controller as a lateral control
- Longitudinal Controller