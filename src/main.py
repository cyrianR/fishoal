import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np

from aquarium import Aquarium2D, Aquarium3D, AquariumKDTree2D
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
depth = 1000                                 # depth of aquarium
nb_fish = 100                                # number of fishes in simulation
fish_color = "orange"                       # color of fishes  
steve_orange_img = plt.imread("./ressources/steve_orange.png")
steve_red_img = plt.imread("./ressources/steve_red.png")
steve_green_img = plt.imread("./ressources/steve_green.png")


## SIMULATION SETUP
# Create aquarium with fishes

# Partie 1
#behavior = StraightReboundBehavior(None)
#aquarium = RandomPositionsAquarium2D(width, height, nb_fish, fish_color, behavior, dt_simu)
#behavior.set_aquarium(aquarium)

# Partie 2
""" aquarium = Aquarium2D(width, height, nb_fish, dt_simu)

leader_behavior = StraightVariableReboundBehavior(
    aquarium,
    max_angle_rand_variation=np.pi/12,
    delay_rand_variation=50)
velocity = (np.random.rand(2) * 2 - 1)
velocity /= np.linalg.norm(velocity)
position = np.random.rand(2) * [width, height]
leader_fish = Fish(position, velocity, "red", leader_behavior, steve_red_img)
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
        delay_contamination=5,
        contaminated_color="green",
        contaminated_img=steve_green_img)
    fish = Fish(position, velocity, fish_color, trafalgar_behavior, steve_orange_img)
    aquarium.put_fish(i, fish) """

r_repulsion = 25
r_alignement = 30
r_attraction = 40
k_repulsion = 1
k_attraction = 0.005


# Partie 3
aquarium = AquariumKDTree2D(width, height, nb_fish, dt_simu)
for i in range(0, nb_fish):
    position = np.random.rand(2) * [width, height]
    velocity = (np.random.rand(2) * 2 - 1)
    velocity /= np.linalg.norm(velocity)
    aoki_behavior = AokiBehavior(aquarium, r_repulsion, r_alignement, r_attraction, k_repulsion, k_attraction)
    fish = Fish(position, velocity, fish_color, aoki_behavior)
    aquarium.put_fish(i, fish)




## LAUNCH SIMULATION

# 2D visualization with points
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


# 2D visualization with images for partie 2
""" fig, ax = plt.subplots()
ax.set_xlim(0, width)
ax.set_ylim(0, height)

fish_boxes = []
for fish in aquarium.fishes:
    image_box = OffsetImage(fish.image, zoom=0.05)
    ab = AnnotationBbox(image_box, (fish.position[0], fish.position[1]), frameon=False)
    ax.add_artist(ab)
    fish_boxes.append(ab)

def update(data):
    xs, ys, fish_images = data
    for i, ab in enumerate(fish_boxes):
        ab.offsetbox = OffsetImage(fish_images[i], zoom=0.05)
        ab.xy = (xs[i], ys[i])
        ab.xybox = (xs[i], ys[i])
    return fish_boxes

def generate_points():
    while True:
        aquarium.update_all()
        xs = [fish.position[0] for fish in aquarium.fishes]
        ys = [fish.position[1] for fish in aquarium.fishes]
        fish_images = [fish.image for fish in aquarium.fishes]
        yield (xs, ys, fish_images)

ani = animation.FuncAnimation(fig, update, generate_points, interval=frame_delay, cache_frame_data=False, blit=False)
plt.show() """

# Create the fishes in a 3D aquarium 
""" aquarium = Aquarium3D(width, height, depth, nb_fish, dt_simu)
for i in range(0, nb_fish):
    position = np.random.rand(3) * [width, height, depth]
    velocity = (np.random.rand(3) * 2 - 1)
    velocity /= np.linalg.norm(velocity)
    behavior = StraightReboundBehavior(aquarium)
    fish = Fish(position, velocity, fish_color, behavior)
    aquarium.put_fish(i, fish)

# Launch the simulation in 3D
xs = [fish.position[0] for fish in aquarium.fishes]
ys = [fish.position[1] for fish in aquarium.fishes]
zs = [fish.position[2] for fish in aquarium.fishes]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_zlim(0, depth)

points = ax.scatter(xs, ys, zs, marker='o', c=fish_color, s=50)

def update(data):
    ys, xs, zs, fish_colors = data
    points._offsets3d = (xs, ys, zs)
    points.set_color(fish_colors)
    return points

def generate_points():
    while True:
        aquarium.update_all()
        xs = [fish.position[0] for fish in aquarium.fishes]
        ys = [fish.position[1] for fish in aquarium.fishes]
        zs = [fish.position[2] for fish in aquarium.fishes]
        fish_colors = [fish.color for fish in aquarium.fishes]
        yield (ys, xs, zs, fish_colors)

anim = animation.FuncAnimation(fig, update, generate_points, interval=frame_delay, cache_frame_data=False)
plt.show() """