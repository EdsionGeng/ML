import numpy as np


def sgd_step(value_and_grad, x, itr, state=None, step_size=0.1, mass=0.9):
    velocity = state if state is not None else np.zeros(len(x))
    val, g = value_and_grad(x)
    velocity = mass * velocity - (1.0 - mass) * g
    x = x + step_size * velocity
    return x, val, g, velocity
