# -*- coding: utf-8 -*-
"""
This file contains several basic functional forms which are useful for the
construction of star formation rate, infall rate, and delay time distribution
models.
"""

import math as m

class Exponential:
    """
    A generic normalized exponentially-declining function of time.

    Attributes
    ----------
    timescale : float
        The exponential timescale in units of time.
    coeff : float
        The post-normalization coefficient.

    """
    def __init__(self, timescale=1, coeff=1):
        self.timescale = timescale
        self.coeff = coeff

    def __call__(self, time):
        return self.coeff / self.timescale * m.exp(-time / self.timescale)


class LinearExponential(Exponential):
    """
    A normalized linear-exponential function. Attributes are inherited from the
    Exponential class.

    """
    def __init__(self, timescale=1, coeff=1, peak=1):
        super().__init__(timescale, coeff)

    def __call__(self, time):
        return (time / self.timescale) * super().__call__(time)


class PowerLaw:
    """
    A generic normalized power-law function of time.

    Attributes
    ----------
    slope : float
        The slope of the power-law.
    coeff : float
        The post-normalization coefficient.
    norm : float
        The normalization coefficient.

    """
    def __init__(self, slope=-1, coeff=1, tmin=0.1, tmax=1):
        self.slope = slope
        self.coeff = coeff
        self.norm = self.normalize(tmin, tmax)

    def __call__(self, time):
        return self.coeff * self.norm * (time ** self.slope)

    def normalize(self, tmin, tmax):
        intslope = self.slope + 1 # The slope of the integral
        return intslope / (tmax ** intslope - tmin ** intslope)


class Gaussian:
    """
    A generic normalized Gaussian function of time.

    Attributes
    ----------
    center : float
        The location of the peak of the Gaussian function.
    stdev : float
        The standard deviation of the Gaussian function.
    norm : float
        The normalization of the Gaussian function.

    """
    def __init__(self, center=1, stdev=1, coeff=1):
        """
        Initialize the Gaussian.

        Parameters
        ----------
        center : float [default: 1]
            The location of the peak of the Gaussian function.
        stdev : float [default: 1]
            The standard deviation of the Gaussian function.
        norm : float [default: 1]
            The normalization pre-factor.

        """
        self.center = center
        self.stdev = stdev
        self.coeff = coeff
        self.norm = 1 / (stdev * m.sqrt(2 * m.pi))

    def __call__(self, time):
        C = self.coeff * self.norm
        return C * m.exp(-(time-self.center)**2 / (2*self.stdev**2))