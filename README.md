# Motion Planning Project

<div align="center">
	<img src="resources/animation.gif" />
</div>


Some part of this project is taken from https://github.com/winstxnhdw/KinematicBicycleModel .

Added features:
- A 7 DoF vehicle model with Pacejka magic tire formula (Constant normal forces)
- Longitudinal PID controller
- 4th-Order Runge-Kutta (RK4) is added for the integration of the vehicle model instead of Euler

To Do:
- Change the chassis model to have algebraic normal forces instead of constant normal forces
- Motion planning features
  - Conformal Lattice Planner
  - RRT*
- Motion controllers need more tuning

## Abstract

## Advantages

## Limitations

## Requirements

```bash
pip install numpy
```

## Demo

Install the requirements

```bash
pip install -r requirements.txt
```

Play the animation

```bash
python animation.py
```

## Concept
- Planar vehicle model
- Stanley Controller as a lateral control
- Longitudinal Controller