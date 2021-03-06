# -*- coding: utf-8 -*-
"""
This file contains the Type Ia supernova delay time distributions and related
functions.
"""

from _globals import END_TIME
from utils import ExponentialFunction, PowerLawFunction, GaussianFunction


class Bimodal:
    """
    The bimodal delay-time distribution of SNe Ia. This assumes 50% of SNe Ia
    belong to a prompt component with the form of a narrow Gaussian, and the
    remaining 50% form a linear-exponential DTD.

    Attributes
    ----------
    tsplit : float
        Time in Gyr separating the first 50% of SNe Ia in the prompt component
        from the second 50% in the tardy component.
    prompt : <function>
        Prompt component as a function of time.
    tardy : <function>
        Tardy component as a function of time.
    norm : float
        Normalization coefficient scaled so the total integral is unity.
    """

    def __init__(self, center=0.05, stdev=0.01, timescale=3, tardy_peak=0.5,
                 tmin=0.04, tmax=END_TIME):
        """
        Initialize the bimodal model.

        Parameters
        ----------
        center : float [default: 0.05]
            Center of the prompt Gaussian component in Gyr.
        stdev : float [default: 0.01]
            Standard deviation of the prompt Gaussian component in Gyr.
        timescale : float [default: 3]
            Exponential timescale of the tardy component in Gyr.
        tardy_peak : float [default: 0.2]
            Peak of the tardy (linear-exponential) component in Gyr.
        tmin : float [default: 0.04]
            Minimum delay time in Gyr for integration purposes.
        tmax : float [default: 13.2]
            Maximum delay time in Gyr for integration purposes

        """
        self.prompt = GaussianFunction(center=center, stdev=stdev,
                                       coeff=0.5e-9)
        self.tardy = ExponentialFunction(timescale=timescale, coeff=0.5e-9)
        self.tsplit = self.solve_tsplit()
        self.norm = 1
        self.norm *= 1e-9 * self.normalize(tmin, tmax)

    def __call__(self, time):
        """
        Calculate the normalized SN Ia rate at the given time.

        Parameters
        ----------
        time : float
            Time in Gyr since the starburst.

        Returns
        -------
        RIa : float
            Normalized SN Ia rate per solar mass per year.

        """
        if time < self.tsplit:
            RIa = self.norm * self.prompt(time)
        else:
            RIa = self.norm * self.tardy(time)
        return RIa

    def solve_tsplit(self, dt=1e-4):
        """
        Numerically solve for the time at which the two components are equal.

        Parameters
        ----------
        dt : float [default: 1e-4]
            The numerical timestep in Gyr.

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
        """
        Calculate the normalization coefficient over the whole distribution.

        Parameters
        ----------
        tmin : float
            Lower limit of integration in Gyr.
        tmax : float
            Upper limit of integration in Gyr.
        dt : float [default: 1e-3]
            The numerical integration timestep in Gyr.

        Returns
        -------
        norm : float
            Normalization coefficient for which the total integral is unity.

        """
        integral = 0
        time = tmin
        while time < tmax:
            integral += self.__call__(time) * dt
            time += dt
        return 1 / integral


class BrokenPowerLaw:
    """
    A two-part broken power-law delay-time distribution of SNe Ia. The default
    setting is a flat distribution (slope of 0) before the time of separation
    and the standard -1.1 slope after.

    Attributes
    ----------
    tsplit : float
        Time in Gyr separating the two power-law components.
    plaw1 : <function>
        The first power-law function, called when t < tsplit.
    plaw2 : <function>
        The second power-law function, called when t >= tsplit.

    """
    def __init__(self, tsplit=0.2, slope1=0, slope2=-1.1, tmin=0.04,
                 tmax=END_TIME):
        """
        Initialize the broken power-law.

        Parameters
        ----------
        tsplit : float [default: 0.2]
            Time in Gyr separating the two components.
        slope1 : float [default: 0]
            The slope of the first power-law.
        slope2 : float [default: -1.1]
            The slope of the second power-law.
        tmin : float [default: 0.04]
            The minimum delay time and lower limit of integration.
        tmax : float [default: 13.2]
            The maximum simulation time and upper limimt of integration.

        """
        self.tsplit = tsplit
        # Initialize both power-law functions
        plaw1 = PowerLawFunction(slope=slope1, tmin=tmin, tmax=tsplit,
                                 coeff=1e-9)
        plaw2 = PowerLawFunction(slope=slope2, tmin=tsplit, tmax=tmax,
                                 coeff=1e-9)
        # Calculate new normalization coefficients
        norm1 = (plaw1.norm**-1 + tsplit**(slope1-slope2) * plaw2.norm**-1)**-1
        plaw1.norm = norm1
        norm2 = tsplit**(slope1-slope2) * norm1
        plaw2.norm = norm2
        self.plaw1 = plaw1
        self.plaw2 = plaw2

    def __call__(self, time):
        """
        Calculate the normalized SN Ia rate at a given time.

        Parameters
        ----------
        time : float
            Time since starburst in Gyr.

        Returns
        -------
        RIa : float
            Normalized SN Ia rate per solar mass per year.

        """
        if time < self.tsplit:
            RIa = self.plaw1(time)
        else:
            RIa = self.plaw2(time)
        return RIa


class Exponential(ExponentialFunction):
    def __init__(self, timescale=1.5):
        super().__init__(timescale=timescale, coeff=1e-9)


class PowerLaw(PowerLawFunction):
    def __init__(self, slope=-1.1, tmin=0.04, tmax=END_TIME):
        super().__init__(slope=slope, coeff=1e-9, tmin=tmin, tmax=tmax)


def test_plot():
    """
    Plot all normalized delay time distributions as a function of time.

    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    dtds = {
            'Power-Law': PowerLaw(),
            'Broken Power-Law': BrokenPowerLaw(),
            'Exponential (1.5 Gyr)': Exponential(timescale=1.5),
            'Exponential (3 Gyr)': Exponential(timescale=3),
            'Bimodal': Bimodal()
    }
    time = [0.001*i for i in range(40, 13201)]
    for dist in list(dtds.keys()):
        func = dtds[dist]
        ax.plot(time, [func(t) for t in time], label=dist)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim((1e-12, 1e-7))
    ax.set_xlabel('Time [Gyr]')
    ax.set_ylabel(r'Normalized SN Ia Rate [$M_\odot^{-1}$ yr$^{-1}$]')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    test_plot()
