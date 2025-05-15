from abc import ABC, abstractmethod
import random
import numpy as np
from typing import TYPE_CHECKING, override, Optional

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



class StraightVariableReboundBehavior(Behavior):
    def __init__(self, aquarium: "Aquarium", max_angle_rand_variation: float, delay_rand_variation: int) -> None:
        super().__init__(aquarium)
        self.max_angle_rand_variation = max_angle_rand_variation
        self.delay_rand_variation = delay_rand_variation
        self.iteration = 0
        self.base_velocity = None

    def behave(self, fish: "Fish") -> None:
        self.iteration += 1

        # Initial base velocity
        if self.iteration == 1:
            self.base_velocity = fish.velocity.copy()
            self.base_velocity /= np.linalg.norm(self.base_velocity)

        # Fish has constant velocity norm and moves straight with a random small variation of angle each delay
        if (self.iteration % self.delay_rand_variation == 0):
            # Case 2D
            if len(fish.velocity) == 2:
                # Randomly add a small angle to the fish's base velocity
                angle = np.random.rand() * 2 * self.max_angle_rand_variation - self.max_angle_rand_variation
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                             [np.sin(angle), np.cos(angle)]])
                fish.velocity = rotation_matrix @ self.base_velocity
                fish.velocity /= np.linalg.norm(fish.velocity)
            # Case 3D
            elif len(fish.velocity) == 3:
                # Randomly add a small angle to the fish's base velocity
                angle = np.random.rand() * 2 * self.max_angle_rand_variation - self.max_angle_rand_variation
                circular_angle = np.random.rand() * 2 * np.pi
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, np.cos(angle), -np.sin(angle)],
                                            [0, np.sin(angle), np.cos(angle)]])
                v_inter = self.base_velocity.copy()
                v_inter = v_inter/np.linalg.norm(v_inter)
                # Rotate velocity to add a small angle
                v = rotation_matrix @ v_inter
                # Rodrigues' rotation formula to rotate around initial fish velocity
                fish.velocity = v*np.cos(circular_angle) + np.cross(v_inter,v)*np.sin(circular_angle) + v_inter*(np.transpose(v_inter)@v)*(1-np.cos(circular_angle))
        fish.position += fish.velocity * self.aquarium.dt

        # Rebound boundaries : if the fish goes out of bounds, it bounces back
        for i in range(len(fish.position)):
            if fish.position[i] < 0:
                fish.position[i] = -fish.position[i]
                fish.velocity[i] *= -1
                self.base_velocity[i] *= -1
            elif fish.position[i] > self.aquarium.size[i]:
                fish.position[i] = self.aquarium.size[i] - (fish.position[i] - self.aquarium.size[i])
                fish.velocity[i] *= -1
                self.base_velocity[i] *= -1
        



class RandomBehavior(Behavior):
    def __init__(self, aquarium: "Aquarium", direction_change_prob: float = 0.1) -> None:
        super().__init__(aquarium)
        self.direction_change_prob = direction_change_prob

    def behave(self, fish: "Fish") -> None:
        # Randomly change direction
        if np.random.rand() < self.direction_change_prob:  # direction_change_prob chance to change direction
            fish.velocity = (np.random.rand(len(fish.velocity)) * 2 - 1)
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
    def __init__(self,
            aquarium: "Aquarium",
            fish_leader: "Fish",
            contamination_dist: float,
            max_angle_rand_variation: float,
            delay_rand_variation: int,
            delay_contamination: int,
            contaminated_color: str,
            contaminated_img: Optional[np.ndarray]) -> None:
        super().__init__(aquarium)
        self.contaminated = False
        self.fish_leader = fish_leader
        self.contamination_dist = contamination_dist
        self.max_angle_rand_variation = max_angle_rand_variation
        self.delay_rand_variation = delay_rand_variation
        self.delay_contamination = delay_contamination
        self.iteration = 0
        self.base_velocity = fish_leader.velocity.copy()
        self.contaminated_color = contaminated_color
        self.contaminated_img = contaminated_img

    def behave(self, fish: "Fish") -> None:
        self.iteration += 1

        # Initial base velocity
        if self.iteration == 1:
            self.base_velocity = fish.velocity.copy()
            self.base_velocity /= np.linalg.norm(self.base_velocity)

        # Contamination
        if not self.contaminated and self.iteration % self.delay_contamination == 0:
            if fish.distance(self.fish_leader) < self.contamination_dist:
                # Fish is now contaminated because it is close to the leader
                self.contaminated = True
                fish.color = self.contaminated_color
                fish.image = self.contaminated_img
            else:
                for f in self.aquarium.fishes:
                    if isinstance(f.behavior, TrafalgarBehavior) and f.behavior.contaminated and f != self.fish_leader:
                        # Check if the fish is close to any contaminated fish
                        d = fish.distance(f)
                        if (d < self.contamination_dist):
                            # Fish is now contaminated because it is close to a contaminated fish
                            self.contaminated = True
                            fish.color = self.contaminated_color
                            fish.image = self.contaminated_img
                            break
            if self.contaminated:
                # Fish now follows the leader
                fish.velocity = self.fish_leader.behavior.base_velocity.copy()
                self.base_velocity = fish.velocity.copy()

        # Fish has constant velocity norm and moves straight with a random small variation of angle each delay
        if (self.iteration % self.delay_rand_variation == 0):
            # Case 2D
            if len(fish.velocity) == 2:
                # Randomly add a small angle to the fish's base velocity
                angle = np.random.rand() * 2 * self.max_angle_rand_variation - self.max_angle_rand_variation
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                             [np.sin(angle), np.cos(angle)]])
                fish.velocity = rotation_matrix @ self.base_velocity
                fish.velocity /= np.linalg.norm(fish.velocity)
            # Case 3D
            elif len(fish.velocity) == 3:
                # Randomly add a small angle to the fish's base velocity
                angle = np.random.rand() * 2 * self.max_angle_rand_variation - self.max_angle_rand_variation
                circular_angle = np.random.rand() * 2 * np.pi
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, np.cos(angle), -np.sin(angle)],
                                            [0, np.sin(angle), np.cos(angle)]])
                v_inter = self.base_velocity.copy()
                v_inter = v_inter/np.linalg.norm(v_inter)
                # Rotate velocity to add a small angle
                v = rotation_matrix @ v_inter
                # Rodrigues' rotation formula to rotate around initial fish velocity
                fish.velocity = v*np.cos(circular_angle) + np.cross(v_inter,v)*np.sin(circular_angle) + v_inter*(np.transpose(v_inter)@v)*(1-np.cos(circular_angle))
        fish.position += fish.velocity * self.aquarium.dt

        # Rebound boundaries : if the fish goes out of bounds, it bounces back
        for i in range(len(fish.position)):
            if fish.position[i] < 0:
                fish.position[i] = -fish.position[i]
                fish.velocity[i] *= -1
                self.base_velocity[i] *= -1
            elif fish.position[i] > self.aquarium.size[i]:
                fish.position[i] = self.aquarium.size[i] - (fish.position[i] - self.aquarium.size[i])
                fish.velocity[i] *= -1
                self.base_velocity[i] *= -1



class AokiBehavior(Behavior):
    def __init__(self, aquarium: "Aquarium", r_repulsion: float, r_alignement: float, r_attraction : float, k_repulsion: float, k_attraction: float) -> None:
        super().__init__(aquarium)
        self.r_repulsion = r_repulsion
        self.r_alignement = r_alignement
        self.r_attraction = r_attraction
        self.k_repulsion = k_repulsion
        self.k_attraction = k_attraction
        

    def behave(self, fish: "Fish") -> None:
        v_repulsion = self.aquarium.kdtree.query_ball_point(fish.position, self.r_repulsion)

        v_alignement = self.aquarium.kdtree.query_ball_point(fish.position, self.r_alignement)
        v_alignement = [v for v in v_alignement if v not in v_repulsion]

        v_attraction = self.aquarium.kdtree.query_ball_point(fish.position, self.r_attraction)
        v_attraction = [v for v in v_attraction if v not in v_repulsion and v not in v_alignement]

        f_repulsion = [self.aquarium.fishes[i] for i in v_repulsion if self.aquarium.fishes[i] != fish]
        f_alignement = [self.aquarium.fishes[i] for i in v_alignement]
        f_attraction = [self.aquarium.fishes[i] for i in v_attraction]

        # Repulsion
        F_repulsion = -self.k_repulsion * np.sum([(fish.position - f.position)/np.linalg.norm((fish.position - f.position)) for f in f_repulsion], axis=0) if len(f_repulsion) > 0 else 0

        # Alignement
        F_alignement = 1 / len(f_alignement) * np.sum([f.velocity for f in f_alignement], axis=0) if len(f_alignement) > 0 else 0

        # Attraction
        F_attraction = self.k_attraction * np.sum([(f.position - fish.position)/np.linalg.norm((f.position - fish.position)) for f in f_attraction], axis=0) if len(f_attraction) > 0 else 0 

        # Update velocity
        fish.velocity += F_repulsion + F_alignement + F_attraction
        fish.velocity /= np.linalg.norm(fish.velocity)

        # Fish has constant velocity and moves straight
        fish.position += fish.velocity * self.aquarium.dt
