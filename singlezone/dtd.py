# -*- coding: utf-8 -*-
"""
This file contains the Type Ia supernova delay time distributions and related
functions.
"""

from _globals import END_TIME
from functions import Exponential, LinearExponential, PowerLaw, Gaussian


class BrokenPowerLaw:
    """
    A two-part broken power-law delay-time distribution of SNe Ia. The default
    setting is a flat distribution (slope of 0) before the time of separation
    and the standard -1.1 slope after.

    Attributes
    ----------
    tsplit : float
        Time in Gyr separating the two power-law components.
    coeff : float
        The post-normalization coefficient.
    plaw1 : <function>
        The first power-law function, called when t < tsplit.
    plaw2 : <function>
        The second power-law function, called when t >= tsplit.

    """
    def __init__(self, tsplit=0.2, slope1=0, slope2=-1.1, coeff=1,
                 tmin=0.04, tmax=END_TIME):
        self.tsplit = tsplit
        self.coeff = coeff
        # Initialize both power-law functions
        plaw1 = PowerLaw(slope=slope1, coeff=coeff, tmin=tmin, tmax=tsplit)
        plaw2 = PowerLaw(slope=slope2, coeff=coeff, tmin=tsplit, tmax=tmax)
        # Calculate new normalization coefficients
        norm1 = (plaw1.norm**-1 + tsplit**(slope1-slope2) * plaw2.norm**-1)**-1
        plaw1.norm = norm1
        norm2 = tsplit**(slope1-slope2) * norm1
        plaw2.norm = norm2
        self.plaw1 = plaw1
        self.plaw2 = plaw2

    def __call__(self, time):
        if time < self.tsplit:
            return self.plaw1(time)
        else:
            return self.plaw2(time)


class Bimodal:

    """
    The bimodal delay-time distribution of SNe Ia. This assumes 50% of SNe Ia
    belong to a prompt component with the form of a narrow Gaussian, and the
    remaining 50% form an exponential DTD.

    Parameters
    ----------
    tsplit : float [default: 0.1]
        Time in Gyr separating the first 50% of SNe Ia in the prompt component
        from the second 50% in the tardy component.
    center : float [default: 0.05]
        Center of the prompt Gaussian component in Gyr.
    stdev : float [default: 0.01]
        Spread of the prompt Gaussian component in Gyr.
    timescale : float [default: 3]
        Timescale of the tardy exponential in Gyr.
    tmax : float [default: 13.2]
        Maximum simulation time in Gyr.
    """

    def __init__(self, center=0.05, stdev=0.01, timescale=3, tmin=0.04,
                 tmax=END_TIME):
        self.prompt = Gaussian(center=center, stdev=stdev, coeff=0.5)
        self.tardy = LinearExponential(timescale=timescale, coeff=0.5)
        self.tsplit = self.solve_tsplit()
        self.norm = 1
        self.norm *= self.normalize(tmin, tmax)

    def __call__(self, time):
        if time < self.tsplit:
            return self.norm * self.prompt(time)
        else:
            return self.norm * self.tardy(time)

    def solve_tsplit(self, dt=1e-3):
        """
        Numerically solve for the time at which the two components are equal.

        Parameters
        ----------
        dt : float [default: 1e-3]
            Numerical timestep.

        Returns
        -------
        tsplit : float
            Time at which the prompt component and tardy component rates equal.

        """
        tsplit = self.prompt.center
        diff = self.prompt(tsplit) - self.tardy(tsplit)
        while diff > 0:
            tsplit += dt
            diff = self.prompt(tsplit) - self.tardy(tsplit)
        return tsplit

    def normalize(self, tmin, tmax, dt=1e-3):
        integral = 0
        time = tmin
        while time < tmax:
            integral += self.__call__(time) * dt
            time += dt
        return 1 / integral


def test_plot():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    dtds = {
            'exponential': Exponential(timescale=1.5),
            'exponential_long': Exponential(timescale=3),
            'powerlaw': PowerLaw(slope=-1.1, tmin=0.04, tmax=END_TIME),
            'powerlaw_broken': BrokenPowerLaw(),
            'bimodal': Bimodal()
    }
    time = [0.001*i for i in range(40, 13201)]
    for dist in list(dtds.keys()):
        func = dtds[dist]
        ax.plot(time, [func(t) for t in time], label=dist)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time [Gyr]')
    ax.set_ylabel('Normalized SN Ia Rate [Msun^-1 Gyr^-1]')
    ax.legend()
    plt.show()
    integral = 0
    dt = 1e-3
    print(time[60])
    for t in time[:60]:
        integral += dtds['bimodal'](t) * dt
    print(integral)

if __name__ == '__main__':
    test_plot()
