from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aquarium import Aquarium
    from fishes import Fish

class Behavior(ABC):
    def __init__(self, aquarium: "Aquarium") -> None:
        self.aquarium = aquarium

    def set_aquarium(self, aquarium: "Aquarium") -> None:
        self.aquarium = aquarium

    @abstractmethod
    def behave(self, fish: "Fish") -> None:
        pass



class StraightToroidalBehavior(Behavior):
    def behave(self, fish: "Fish") -> None:
        # Fish has constant velocity and moves straight 
        fish.position += fish.velocity * self.aquarium.dt

        # Toroidal boundaries : if the fish goes out of bounds, it reappears on the other side
        for i in range(len(fish.position)):
            if fish.position[i] < 0:
                fish.position[i] = fish.position[i] % self.aquarium.size[i]
            elif fish.position[i] > self.aquarium.size[i]:
                fish.position[i] = fish.position[i] % self.aquarium.size[i]

class StraightReboundBehavior(Behavior):
    def behave(self, fish: "Fish") -> None:
        # Fish has constant velocity and moves straight 
        fish.position += fish.velocity * self.aquarium.dt

        # Rebound boundaries : if the fish goes out of bounds, it bounces back
        for i in range(len(fish.position)):
            if fish.position[i] < 0:
                fish.position[i] = -fish.position[i]
                fish.velocity[i] *= -1
            elif fish.position[i] > self.aquarium.size[i]:
                fish.position[i] = self.aquarium.size[i] - (fish.position[i] - self.aquarium.size[i])
                fish.velocity[i] *= -1