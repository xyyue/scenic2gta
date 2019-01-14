from gta import *

spot = OrientedPoint on curb
badAngle = Options([1.0, -1.0]) * (10, 20) deg
parkedCar = Car left of (spot offset by -0.5 @ 0), facing badAngle relative to roadDirection

ego = Car at parkedCar offset by (-20, 0) @ (-30, 30)