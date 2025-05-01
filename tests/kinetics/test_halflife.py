from chem_deg.kinetics.halflife import HALFLIFE7


def test_halflife():
    """
    Test the HalfLife class.
    """
    assert HALFLIFE7.midpoint == 0.25
    assert HALFLIFE7.rate(0.693) == 1.0
