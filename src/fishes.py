from abc import ABC
import numpy as np
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from behaviors import Behavior

class Fish:
    def __init__(self, position: np.ndarray, velocity: np.ndarray, color: str, behavior: "Behavior", image: Optional[np.ndarray] = None):
        self.behavior = behavior
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.color = color
        self.image = image
    
    def update_state(self):
        self.behavior.behave(self)

    def distance(self, other: "Fish") -> float:
        return np.linalg.norm(self.position - other.position)
