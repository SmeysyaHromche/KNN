import random
import numpy as np
import cv2 as cv

from augmentations.base import Transform


class RandomSkew(Transform):
    """
    Apply a random skew transformation to an input image.

    This transformation randomly tilts the input image to the right
    or to the left. The magnitude of the skew is chosen randomly from
    provided range.
    """

    def __init__(self, probability: float = 1.0, skew_range: tuple[int, int] = (-0.5, 0.5)) -> None:
        """
        Initialize the random skew transformation.

        :param probability: Probability that the transformation will be applied.
        :param skew_range: Tuple containing the min and max magnitude (in that order).
        """
        super().__init__(probability)
        self.skew_range = skew_range

    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply a random skew transformation (tilt to the right or left) to an input image.

        :param image: Input image to apply the skew to.
        """
        height, width = image.shape[:2]
        skew_factor = random.uniform(self.skew_range[0], self.skew_range[1])
        # Once the image gets skewed, the width gets bigger, but the output image's width remain the same.
        # If the skew is too big, it may push some letters off the screen -> resize the image to accommodate the skew.
        new_width = int(width + abs(skew_factor) * height)
        # Center the skewed image on the x-axis
        translate_x = max(0, -skew_factor * height)

        # Affine matrix for horizontal skew
        skew_matrix = np.array([[1, skew_factor, translate_x],
                                [0, 1, 0]], dtype=np.float32)
        
        return cv.warpAffine(image, skew_matrix, (new_width, height), borderMode=cv.BORDER_REPLICATE)
