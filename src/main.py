import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from aquarium import RandomPositionsAquarium2D, Aquarium2D, AquariumKDTree2D
from behaviors import StraightToroidalBehavior
from behaviors import StraightReboundBehavior
from behaviors import StraightVariableReboundBehavior
from behaviors import RandomBehavior
from behaviors import TrafalgarBehavior
from behaviors import AokiBehavior
from fishes import Fish


## SIMULATION PARAMETERS
frame_delay = 10                            # delay between frames in ms
dt_simu = 2                                 # time step of simulation in ms
width = 1000                                # width of aquarium
height = 1000                               # height of aquarium 
nb_fish = 100                                # number of fishes in simulation
fish_color = "orange"                       # color of fishes  


## SIMULATION SETUP
# Create aquarium with fishes

# Partie 1
#behavior = StraightReboundBehavior(None)
#aquarium = RandomPositionsAquarium2D(width, height, nb_fish, fish_color, behavior, dt_simu)
#behavior.set_aquarium(aquarium)

# Partie 2
aquarium = Aquarium2D(width, height, nb_fish, dt_simu)

leader_behavior = StraightVariableReboundBehavior(
    aquarium,
    max_angle_rand_variation=np.pi/12,
    delay_rand_variation=50)
velocity = (np.random.rand(2) * 2 - 1)
velocity /= np.linalg.norm(velocity)
position = np.random.rand(2) * [width, height]
leader_fish = Fish(position, velocity, "red", leader_behavior)
aquarium.put_fish(0, leader_fish)

for i in range(1, nb_fish):
    position = np.random.rand(2) * [width, height]
    velocity = (np.random.rand(2) * 2 - 1)
    velocity /= np.linalg.norm(velocity)
    trafalgar_behavior = TrafalgarBehavior(
        aquarium,
        fish_leader=leader_fish,
        contamination_dist=30,
        max_angle_rand_variation=np.pi/12,
        delay_rand_variation=50,
        delay_change_behavior=5)
    fish = Fish(position, velocity, fish_color, trafalgar_behavior)
    aquarium.put_fish(i, fish)

# Partie 3
""" aquarium = AquariumKDTree2D(width, height, nb_fish, dt_simu)
for i in range(0, nb_fish):
    position = np.random.rand(2) * [width, height]
    velocity = (np.random.rand(2) * 2 - 1)
    velocity /= np.linalg.norm(velocity)
    aoki_behavior = AokiBehavior(aquarium, 10, 20, 75, 5, 10)
    fish = Fish(position, velocity, fish_color, aoki_behavior)
    aquarium.put_fish(i, fish) """




## LAUNCH SIMULATION
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