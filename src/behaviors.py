from abc import ABC, abstractmethod
import random
import numpy as np
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


class RandomBehavior(Behavior):
    def __init__(self, aquarium: "Aquarium", direction_change_prob: float = 0.1) -> None:
        super().__init__(aquarium)
        self.direction_change_prob = direction_change_prob

    def behave(self, fish: "Fish") -> None:
        # Randomly change direction
        if np.random.rand() < self.direction_change_prob:  # direction_change_prob chance to change direction
            fish.velocity = (np.random.rand(2) * 2 - 1)
            fish.velocity /= np.linalg.norm(fish.velocity)

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


class TrafalgarBehavior(Behavior):
    def __init__(self, aquarium: "Aquarium", fish_leader: "Fish", contamination_dist: float, random_variation: float, delay_random_variation: int) -> None:
        super().__init__(aquarium)
        self.contaminated = False
        self.fish_leader = fish_leader
        self.contamination_dist = contamination_dist
        self.random_variation = random_variation
        self.delay_random_variation = delay_random_variation
        self.iteration = 0

    def behave(self, fish: "Fish") -> None:
        self.iteration += 1
        if not self.contaminated:
            if fish.distance(self.fish_leader) < self.contamination_dist:
                # Fish is now contaminated because it is close to the leader
                self.contaminated = True
                fish.color = "green"
            else:
                for f in self.aquarium.fishes:
                    if isinstance(f.behavior, TrafalgarBehavior) and f.behavior.contaminated and f != self.fish_leader:
                        # Check if the fish is close to any contaminated fish
                        d = fish.distance(f)
                        if (d < self.contamination_dist):
                            self.contaminated = True
                            fish.color = "green"
                            break
            if self.contaminated:
                # Fish now follows the leader
                fish.velocity = self.fish_leader.velocity.copy()
        # Fish has constant velocity and moves straight
        if (self.iteration % self.delay_random_variation == 0):
            fish.velocity += np.random.rand(2) * self.random_variation - self.random_variation / 2
            fish.velocity /= np.linalg.norm(fish.velocity)
        fish.position += fish.velocity * self.aquarium.dt


        # Rebound boundaries : if the fish goes out of bounds, it bounces back
        for i in range(len(fish.position)):
            if fish.position[i] < 0:
                fish.position[i] = -fish.position[i]
                fish.velocity[i] *= -1
            elif fish.position[i] > self.aquarium.size[i]:
                fish.position[i] = self.aquarium.size[i] - (fish.position[i] - self.aquarium.size[i])
                fish.velocity[i] *= -1
        