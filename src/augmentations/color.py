import cv2 as cv
import random
import numpy as np

from augmentations.base import Transform

class RandomBrightness(Transform):
    """
    Randomly adjust the brightness of the image with a given probability.

    Simulates variations in lighting or a scan exposure.
    """

    def __init__(self, probability: float = 1.0, brightness: float = 70.0) -> None:
        """
        Initialize the RandomBrightness transformation.

        :param probability: Probability of applying the brightness change.
        :param brightness: Maximum brightness change (positive of negative).
        """
        
        super().__init__(probability)
        self.brightness = brightness

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random brightness adjustment.

        :param image: Input image (NumPy array) to adjust.
        :return: Brightness-adjusted image as a NumPy array.
        """

        beta = random.uniform(-self.brightness, self.brightness)
        return cv.convertScaleAbs(image, alpha=1.0, beta=beta)