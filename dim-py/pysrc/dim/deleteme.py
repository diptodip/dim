from dim import dim

class RollingBall:
    def __init__(self, radius: float, downsample: int, arc_trim_percentage: float):
        self._rolling_ball = dim.RollingBall(radius, downsample, arc_trim_percentage)

    @property
    def radius(self) -> float:
        return self._rolling_ball.radius

    def python(self) -> bool:
        return True
