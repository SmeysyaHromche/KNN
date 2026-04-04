import random
from abc import ABC, abstractmethod


class Transform(ABC):
    """
    Abstract base class for image transformations.

    Each transformation is applied with a given probability.

    :param probability: Probability of applying the transformation.
    """

    def __init__(self, probability: float = 1.0):
        """
        Initialize the transformation.

        :param probability: Probability of applying the transformation.
        """
        self.p = probability

    def __call__(self, image):
        """
        Apply the transformation to the input image with a given probability.

        The transformation is applied only if a randomly sampled valued is
        lower than ``self.p``. Otherwise, the input image is returned unchanged.

        :param image: Input image to apply the transformation to.
        """
        if random.random() < self.p:
            return self.apply(image)
        return image
    
    @abstractmethod
    def apply(self, image):
        """
        Abstract method for applying the transformation.

        :param image: Input image to apply the transformation to.
        """
        pass
