# Motion Planning Project

<div align="center">
	<img src="resources/animation.gif" />
</div>


Some part of this project is taken from https://github.com/winstxnhdw/KinematicBicycleModel .

Added features:
- A 7 DoF vehicle model with Pacejka magic tire formula (Constant normal forces)
- Longitudinal PID controller

To Do:
- Add Runge-Kutta (RK4) to the integration step
- Change the chassis model to have algebraic normal forces instead of constant normal forces
- Motion planning features
  - Conformal Lattice Planner
  - RRT*
- Motion controllers need more tuning

## Abstract

```yaml
At initialisation
:param wheelbase:       (float) vehicle's wheelbase [m]
:param max_steer:       (float) vehicle's steering limits [rad]
:param dt:              (float) discrete time period [s]
:param c_r:             (float) vehicle's coefficient of resistance 
:param c_a:             (float) vehicle's aerodynamic coefficient

At every time step  
:param x:               (float) vehicle's x-coordinate [m]
:param y:               (float) vehicle's y-coordinate [m]
:param yaw:             (float) vehicle's heading [rad]
:param velocity:        (float) vehicle's velocity in the x-axis [m/s]
:param throttle:        (float) vehicle's accleration [m/s^2]
:param delta:           (float) vehicle's steering angle [rad]

:return x:              (float) vehicle's x-coordinate [m]
:return y:              (float) vehicle's y-coordinate [m]
:return yaw:            (float) vehicle's heading [rad]
:return velocity:       (float) vehicle's velocity in the x-axis [m/s]
:return delta:          (float) vehicle's steering angle [rad]
:return omega:          (float) vehicle's angular velocity [rad/s]
```

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