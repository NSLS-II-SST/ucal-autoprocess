from scipy.optimize import curve_fit
import numpy as np

import matplotlib.pyplot as plt
plt.ion()

def fitPeak(sd, energy, channels=None, plot=True, delta=10):
    y, x = sd.getEmission(energy-2*delta, energy + 2*delta, 0.1, channels=channels)
    xcenter = x[np.argmax(y)]
    xidx = ( x < energy + delta ) & ( x > energy - delta )

    xwin = x[xidx]
    ywin = y[xidx]

    def gaussian(x, amplitude, mean, standard_deviation):
        return amplitude * np.exp(- ((x - mean)**2)/(2*standard_deviation **2))
    
    popt, pcov = curve_fit(gaussian, xwin, ywin, p0=[25000, xcenter, 3])
    optimal_amplitude = popt[0]
    optimal_mean = popt[1]
    optimal_std = np.abs(popt[2])

    fitted_y = gaussian(xwin, *popt)
    nchan = len(channels) if channels is not None else len(sd.data.chanlist)

    if plot:
        plt.figure(figsize = (8,4))
        plt.plot(xwin, ywin, "b.", label=f"{nchan} coadded channels")
        plt.plot(xwin, fitted_y, "r-", label = f"Fit with FWHM: {2.355*optimal_std:.3f}")
        plt.title("Fit to elastic scattering line")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.legend()
        plt.show()
        
    return optimal_amplitude, optimal_mean, optimal_std


def fitAllChannels(sd, energy, cutoff = 5):
    statslist = []
    chanlist = []
    for chan in sd.data.chanlist:
        try:
            stats = fitPeak(sd, energy, [chan], False)
            if stats[2] > cutoff:
                continue
            chanlist.append(chan)
            statslist.append(stats)
        except RuntimeError:
            continue
    statsarray = np.array(statslist)
    return statsarray, chanlist
