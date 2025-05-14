from abc import ABC
from turtle import up
import numpy as np
import scipy as sp
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from behaviors import Behavior

from fishes import Fish

class Aquarium(ABC):
    def __init__(self, size: np.ndarray, capacity: int, dt: float):
        self.size = np.array(size, dtype=int)
        self.fishes = np.empty(capacity, dtype=object)
        self.dt = dt

    def update_all(self):
        for fish in self.fishes:
            fish.update_state()



class Aquarium2D(Aquarium):
    def __init__(self, width: int, height: int, n_fishes: int, dt: float):
        super().__init__(np.array([width, height]), n_fishes, dt)

    def put_fish(self, ind: int, fish: "Fish"):
        self.fishes[ind] = fish


class AquariumKDTree2D(Aquarium2D):
    def __init__(self, width: int, height: int, n_fishes: int, dt: float):
        super().__init__(width, height, n_fishes, dt)
        self.kdtree = None

    @override
    def update_all(self):
        self.kdtree = sp.spatial.KDTree([fish.position for fish in self.fishes])
        super().update_all()



class RandomPositionsAquarium2D(Aquarium2D):
    def __init__(self, width: int, height: int, n_fishes: int, color: str, behavior: "Behavior", dt: float):
        super().__init__(width, height, n_fishes, dt)
        self.populate_fishes(n_fishes, color, behavior, width, height)

    def populate_fishes(self, n_fishes: int, color: str, behavior: "Behavior", width: int, height: int):
        for i in range(n_fishes):
            position = np.random.rand(2) * [width, height]
            velocity = (np.random.rand(2) * 2 - 1)
            velocity /= np.linalg.norm(velocity)
            self.fishes[i] = Fish(position, velocity, color, behavior)

    def change_fish_colors(self, new_color: str):
        for fish in self.fishes:
            fish.color = new_color

    def change_fish_behaviors(self, new_behavior: "Behavior"):
        for fish in self.fishes:
            fish.behavior = new_behavior