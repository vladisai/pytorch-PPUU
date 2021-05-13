LANE_WIDTH_METRES = 3.7
LANE_WIDTH_PIXELS = 24  # pixels / 3.7 m, lane width
# SCALE = 1 / 4
PIXELS_IN_METRE = LANE_WIDTH_PIXELS / LANE_WIDTH_METRES
MAX_SPEED_MS = 130 / 3.6  # m/s
LOOK_AHEAD_M = MAX_SPEED_MS  # meters
LOOK_SIDEWAYS_M = 2 * LANE_WIDTH_METRES  # meters
METRES_IN_FOOT = 0.3048
TIMESTEP = 0.1


class UnitConverter:
    """
    LOOKAHEAD is 36 meters
    LOOK sideways is 2 lane widths, which is 7.2 meters
    One 6.5 pixels per s is about 1 m/s
    """

    @classmethod
    def feet_to_m(cls, x):
        return x * METRES_IN_FOOT

    @classmethod
    def m_to_feet(cls, x):
        return x / METRES_IN_FOOT

    @classmethod
    def pixels_to_m(cls, x):
        return x / PIXELS_IN_METRE

    @classmethod
    def m_to_pixels(cls, x):
        return x * PIXELS_IN_METRE

    @classmethod
    def feet_to_pixels(cls, x):
        return cls.m_to_pixels(cls.feet_to_m(x))

    @classmethod
    def pixels_to_feet(cls, x):
        return cls.m_to_feet(cls.pixels_to_m(x))

    @classmethod
    def pixels_per_s_to_kmph(cls, x):
        return cls.pixels_to_m(x) / 1000 * 60 * 60

    @classmethod
    def kmph_to_pixels_per_s(cls, x):
        return cls.m_to_pixels(x * 1000) / (60 * 60)
