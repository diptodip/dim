from jaxtyping import Float
import numpy as np

class RollingBall:
    def __init__(self, radius: float, downsample_factor: int, arc_trim_percentage: float) -> None:
        """
        Constructs a rolling ball kernel that can be used to estimate the background of an
        input image.
        """
        pass

    @property
    def radius(self) -> float:
        """The radius of the rolling ball kernel."""
        pass

    @property
    def downsample_factor(self) -> int:
        """
        The downsample factor of the rolling ball kernel, used when the input
        image is very large to accelerate background estimation.
        """
        pass

    @property
    def kernel_width(self) -> int:
        """The width of the rolling ball kernel in pixels."""
        pass

    def estimate_background(self, image: Float[np.ndarray, "y x"]) -> Float[np.ndarray, "y x"]:
        """
        Estimates the background of the image using the rolling ball algorithm.
        """
        pass
