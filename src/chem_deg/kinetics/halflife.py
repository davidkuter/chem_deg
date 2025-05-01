class HalfLife:
    def __init__(self, min: int | float, max: int | float):
        """
        Initialize the HalfLife class with minimum and maximum half-life values. These values
        should be in hour units

        :param min: Minimum half-life value.
        :param max: Maximum half-life value.
        """
        self.min = float(min)
        self.max = float(max)

    @property
    def midpoint(self) -> float:
        """
        Calculate the midpoint of the half-life range.

        :return: Midpoint of the half-life range.
        """
        return (self.min + self.max) / 2

    @staticmethod
    def rate(value: float) -> float:
        """
        Calculate the rate constant based on the half-life value.

        :param value: Half-life value.
        :return: Rate constant.
        """
        return 0.693 / value


# Rank 7: t < 30 mins
HALFLIFE7 = HalfLife(
    min=0,
    max=0.5,
)

# Rank 6: 30 mins < t < 3.33 hours
HALFLIFE6 = HalfLife(
    min=0.5,
    max=3.33,
)

# Rank 5: 3.33 hours < t < 24 hours
HALFLIFE5 = HalfLife(
    min=3.33,
    max=24,
)

# Rank 4: 24 hours < t < 7 days
HALFLIFE4 = HalfLife(
    min=24,
    max=168,
)

# Rank 3: 7 days < t < 60 days
HALFLIFE3 = HalfLife(
    min=168,
    max=1_440,
)

# Rank 2: 60 days < t < 1 year
HALFLIFE2 = HalfLife(
    min=1_440,
    max=8_760,
)

# Rank 1: t > 1 year
HALFLIFE1 = HalfLife(
    min=8_760,
    # 3 years but can be extended as needed
    max=26_280,
)

