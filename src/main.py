import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from utils import *


## CONSTANTS
dt_spe = 10
dt_aff = 100
width = 1000
height = 1000
nb_fish = 10
fishes = np.array([Fish(width, height) for _ in range(nb_fish)])
print(fishes[0].position)

fig, ax = plt.subplots()
xs = [fish.position[0] for fish in fishes]
ys = [fish.position[1] for fish in fishes]
points, = ax.plot(xs, ys, 'o')
ax.set_xlim(0, width)
ax.set_ylim(0, height)

def update(data):
    ys, xs = data
    points.set_ydata(ys)
    points.set_xdata(xs)
    return points

def generate_points():
    while True:
        for fish in fishes:
            fish.position += fish.speed * dt_spe
            if fish.position[0] > width:
                fish.position[0] = fish.position[0] % width
            elif fish.position[0] <= 0:
                fish.position[0] = (-fish.position[0]) % width
            if fish.position[1] > height:
                fish.position[1] = fish.position[1] % height
            elif fish.position[1] <= 0:
                fish.position[1] = (-fish.position[1] % height)
        
        xs = [fish.position[0] for fish in fishes]
        ys = [fish.position[1] for fish in fishes]
        yield (ys, xs)

ani = animation.FuncAnimation(fig, update, generate_points, interval=dt_aff)
plt.show()