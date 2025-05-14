import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from aquarium import RandomPositionsAquarium2D, Aquarium2D
from behaviors import StraightToroidalBehavior
from behaviors import StraightReboundBehavior
from behaviors import RandomBehavior
from behaviors import TrafalgarBehavior
from fishes import Fish


## Simulation parameters
frame_delay = 10                            # delay between frames in ms
dt_simu = 2                                 # time step of simulation in ms
width = 1000                                # width of aquarium
height = 1000                               # height of aquarium 
nb_fish = 100                                # number of fishes in simulation
fish_color = "orange"                       # color of fishes  


## Simulation setup
# Create aquarium with fishes

# Partie 1
#behavior = StraightReboundBehavior(None)
#aquarium = RandomPositionsAquarium2D(width, height, nb_fish, fish_color, behavior, dt_simu)
#behavior.set_aquarium(aquarium)

# Partie 2
aquarium = Aquarium2D(width, height, nb_fish, dt_simu)

leader_behavior = StraightReboundBehavior(aquarium)
velocity = (np.random.rand(2) * 2 - 1)
velocity /= np.linalg.norm(velocity)
leader_fish = Fish(np.array([width / 2, height / 2]), velocity, "red", leader_behavior)
aquarium.put_fish(0, leader_fish)

for i in range(1, nb_fish):
    position = np.random.rand(2) * [width, height]
    velocity = (np.random.rand(2) * 2 - 1)
    velocity /= np.linalg.norm(velocity)
    trafalgar_behavior = TrafalgarBehavior(aquarium, leader_fish, contamination_dist=50, random_variation=0.1, delay_random_variation=10)
    fish = Fish(position, velocity, fish_color, trafalgar_behavior)
    aquarium.put_fish(i, fish)



## Launch simulation
fig, ax = plt.subplots()
xs = [fish.position[0] for fish in aquarium.fishes]
ys = [fish.position[1] for fish in aquarium.fishes]
points = ax.scatter(xs, ys, marker='o', c=fish_color)
ax.set_xlim(0, width)
ax.set_ylim(0, height)

def update(data):
    ys, xs, fish_colors = data
    points.set_offsets(np.c_[xs, ys])
    points.set_color(fish_colors)
    return points

def generate_points():
    while True:
        aquarium.update_all()
        xs = [fish.position[0] for fish in aquarium.fishes]
        ys = [fish.position[1] for fish in aquarium.fishes]
        fish_colors = [fish.color for fish in aquarium.fishes]
        yield (ys, xs, fish_colors)

ani = animation.FuncAnimation(fig, update, generate_points, interval=frame_delay, cache_frame_data=False)
plt.show()