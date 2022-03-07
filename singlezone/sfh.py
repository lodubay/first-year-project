# -*- coding: utf-8 -*-
"""
This file contains various star formation histories and related functions.
"""

import math as m
from _globals import END_TIME, M_STAR_MW
from utils import ExponentialFunction, GaussianFunction


def normalize(func, dt=1e-3, tmax=END_TIME):
    """
    Compute normalization pre-factor for star formation function.
    """
    integral = sum([func(i * dt) * dt * 1e9 for i in range(int(tmax / dt))])

    return 1 / integral


class Constant:
    """
    A constant star formation history as a function of time.

    """
    def __init__(self, Mstar=M_STAR_MW, tmax=END_TIME, recycling=0.4):
        self.scale = Mstar / (1 - recycling)
        self.tmax = tmax

    def __call__(self, time):
        return 1e-9 * self.scale / self.tmax


class Exponential(ExponentialFunction):
    def __init__(self, sfh_timescale=6, Mstar=M_STAR_MW, recycling=0.4,
                 tmax=END_TIME):
        super().__init__(timescale=sfh_timescale)
        self.coeff = Mstar * 1/(1-recycling) * normalize(self, tmax=tmax)


class InsideOut:
    def __init__(self, rise_timescale=2, sfh_timescale=15, Mstar=M_STAR_MW,
                 tmax=END_TIME, recycling=0.4):
        self.scale = 1
        self.rise_timescale = rise_timescale
        self.sfh_timescale = sfh_timescale
        self.norm = 1
        self.norm = normalize(self, tmax=tmax)
        self.scale = Mstar / (1 - recycling)

    def __call__(self, time):
        rise = m.exp(-time/self.rise_timescale)
        decline = m.exp(-time/self.sfh_timescale)
        return self.scale * self.norm * (1 - rise) * decline


class LateBurst(InsideOut):
    def __init__(self, burst_strength=1.5, burst_time=10.5, burst_stdev=1,
                 **kwargs):
        self.burst = GaussianFunction(center=burst_time, stdev=burst_stdev,
                                      coeff=burst_strength)
        self.burst.norm = 1
        super().__init__(**kwargs)

    def __call__(self, time):
        return (1 + self.burst(time)) * super().__call__(time)


def test_plot():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    sfhs = {
            'Inside-Out': InsideOut(),
            'Late-Burst': LateBurst(),
            'Exponential': Exponential(),
            'Constant': Constant()
    }
    time = [0.001*i for i in range(40, 13201)]
    for sfh in list(sfhs.keys()):
        func = sfhs[sfh]
        ax.plot(time, [func(t) for t in time], label=sfh)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_ylim((1e-12, 1e-7))
    ax.set_xlabel('Time [Gyr]')
    # ax.set_ylabel('Normalized SN Ia Rate [Msun^-1 yr^-1]')
    ax.set_ylabel('Normalized SFR [$M_\odot$ yr$^{-1}$]')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    test_plot()
