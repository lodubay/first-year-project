# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 11:15:46 2022

@author: dubay.11
"""

import math as m
from .._globals import END_TIME

class exponential:
	
	"""
	The exponential delay-time distribution of SNe Ia.
	
	Parameters
	----------
	timescale : float [default: 3]
		The e-folding timescale in Gyr of the exponential.
	norm : float [default: 1]
		The normalization of the exponential, i.e., the value at t=0.
	"""
	
	def __init__(self, timescale=3, norm=1):
		self.timescale = timescale
		self.norm = norm
	
	def __call__(self, time):
		return self.norm * 1/self.timescale * m.exp(-time/self.timescale)


class powerlaw:
	
	"""
	The power-law delay-time distribution of SNe Ia.
	
	Parameters
	----------
	slope : float [default: -1.1]
		The slope of the power law.
	norm : float [default: 1]
		The normalization of the power law, i.e., the value at t=1.
	"""
	
	def __init__(self, slope=-1.1, norm=1):
		self.slope = slope
		self.norm = norm
	
	def __call__(self, time):
		return self.norm * time ** self.slope


class broken_powerlaw:
	"""
	A two-part broken power-law delay-time distribution of SNe Ia. The default
	setting is a flat distribution (slope of 0) before the time of separation
	and the standard -1.1 slope after.
	    
	Parameters
	----------
	tsplit : float [default: 0.2]
		Time in Gyr separating the two power-law components.
	slope1 : float [default: 0]
		Slope of the early power-law component.
	slope2 : float [default: -1.1]
		Slope of the late power-law component.
	norm : float [default: 1]
		Normalization of the second power-law component.

	"""
	def __init__(self, tsplit=0.2, slope1=0, slope2=-1.1, norm=1):
		self.tsplit = tsplit
		self.plaw2 = powerlaw(slope=slope2, norm=norm)
		norm1 = norm * tsplit ** (slope2 - slope1)
		self.plaw1 = powerlaw(slope=slope1, norm=norm1)

	def __call__(self, time):
		if time > self.tsplit:
			return self.plaw2(time)
		else:
			return self.plaw1(time)


class gaussian:
	
	"""
	A Gaussian distribution in time.
	
	Parameters
	----------
	center : float [default: 1]
		The location of the peak of the Gaussian function.
	stdev : float [default: 1]
		The standard deviation of the Gaussian function.
	norm : float [default: 1]
		The normalization of the Gaussian function, i.e., the value of the peak.
	"""
	
	def __init__(self, center=1, stdev=1, norm=1):
		self.center = center
		self.stdev = stdev
		self.norm = norm
	
	def __call__(self, time):
		return self.norm * m.exp(-(time-self.center)**2 / (2*self.stdev**2))


class bimodal(gaussian, exponential):
	
	"""
	The bimodal delay-time distribution of SNe Ia. This assumes 50% of SNe Ia
	belong to a prompt component with the form of a narrow Gaussian, and the
	remaining 50% form an exponential DTD.
	
	Parameters
	----------
	t50 : float [default: 0.1]
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
	
	def __init__(self, t50=0.1, center=0.05, stdev=0.01, timescale=3):
		self.t50 = t50
		gaussian.__init__(self, center=center, stdev=stdev)
		exponential.__init__(self, timescale=timescale)
		# Set Gaussian area = exponential area
		gauss_sum = sum([gaussian.__call__(self, t50/100 * t) * t50/100 \
				   for t in range(100)])
		exp_sum = sum([exponential.__call__(self, (END_TIME-t50)/100*t + t50) \
				 * (END_TIME-t50)/100 for t in range(100)])
		self.scale_gaussian = exp_sum / gauss_sum
	
	def __call__(self, time):
		if time < self.t50:
			return self.scale_gaussian * gaussian.__call__(self, time)
		else:
			return exponential.__call__(self, time)


def test_plot():
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	dtds = {
			'exponential': exponential,
			'powerlaw': powerlaw,
			'bimodal': bimodal
	}
	time = [0.001*i for i in range(40, 13201)]
	for dist in list(dtds.keys()):
		func = dtds[dist]()
		ax.plot(time, [func(t) for t in time], label=dist)
	ax.set_xscale('log')
	ax.set_yscale('log')
	ax.set_xlabel('Time [Gyr]')
	ax.set_ylabel('DTD')
	ax.legend()
	plt.show()
