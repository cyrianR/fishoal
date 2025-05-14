import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from aquarium import RandomPositionsAquarium2D
from behaviors import StraightToroidalBehavior
from behaviors import StraightReboundBehavior
from fishes import Fish


## Simulation parameters
frame_delay = 10            # delay between frames in ms
dt_simu = 2                 # time step of simulation in ms
width = 1000                # width of aquarium
height = 1000               # height of aquarium 
nb_fish = 100                # number of fishes in simulation
fish_color = "orange"       # color of fishes  


## Simulation setup
# Create aquarium with fishes
behavior = StraightToroidalBehavior(None)
aquarium = RandomPositionsAquarium2D(width, height, nb_fish, fish_color, behavior, dt_simu)
behavior.set_aquarium(aquarium)

# Launch simulation
fig, ax = plt.subplots()
xs = [fish.position[0] for fish in aquarium.fishes]
ys = [fish.position[1] for fish in aquarium.fishes]
points, = ax.plot(xs, ys, 'o', color=fish_color)
ax.set_xlim(0, width)
ax.set_ylim(0, height)

def update(data):
    ys, xs = data
    points.set_ydata(ys)
    points.set_xdata(xs)
    return points

def generate_points():
    while True:
        aquarium.update_all()
        xs = [fish.position[0] for fish in aquarium.fishes]
        ys = [fish.position[1] for fish in aquarium.fishes]
        yield (ys, xs)

ani = animation.FuncAnimation(fig, update, generate_points, interval=frame_delay)
plt.show()