import cv2 as cv
import random
import numpy as np

from src.learn.augmentations.base import Transform

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
    

class RandomContrast(Transform):
    """
    Randomly adjust the contrast of the image with a given probability.

    Helps simulate variations in per darkness, scan quality, or camera exposure.
    """

    def __init__(self, probability: float = 1.0, contrast: float = 70.0) -> None:
        """
        Initialize the RandomContrast transformation.

        :param probability: Probability of applying the contrast change.
        :param contrast: Maximum contrast change as a percentage.
        """
        
        super().__init__(probability)
        self.contrast = contrast

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random contrast adjustment.

        :param image: Input image (NumPy array) to adjust.
        :return: Contrast-adjusted image as a NumPy array.
        """
        
        alpha = 1.0 + random.uniform(-self.contrast / 100, self.contrast / 100)
        return cv.convertScaleAbs(image, alpha=alpha, beta=0)
