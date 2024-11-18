import mass
from mass.calibration.algorithms import line_names_and_energies
import os
from os import path
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import h5py


cal_line_master = {
    "ck": 278.21,
    "nk": 392.25,
    "tila": 452,
    "ok": 524.45,
    "fell": 614.84,
    "coll": 675.98,
    "fk": 677,
    "fela": 705.01,
    "felb": 717.45,
    "cola": 775.31,
    "colb": 790.21,
    "nill": 742.3,
    "nila": 848.85,
    "nilb": 866.11,
    "cema": 883,
    "cula": 926.98,
    "culb": 947.52,
    "znla": 1009.39,
    "znlb": 1032.46,
}

# mass.line_models.VALIDATE_BIN_SIZE = False


def get_line_energies(line_names):
    """
    Takes a list of strings or floats, and returns the line energies in
    cal_line_master.
    """
    line_energies = [cal_line_master.get(n, n) for n in line_names]
    return line_energies


def assignPeaks(
    peak_positions,
    line_names,
    nextra=2,
    nincrement=2,
    nextramax=4,
    rms_cutoff=0.2,
    polyorder=2,
    autoinclude=1,
    curvename="gain",
    debug=False,
):
    """Tries to find an assignment of peaks to line names that is reasonably self consistent and smooth

    Args:
        peak_positions (np.array(dtype=float)): a list of peak locations in arb units,
            e.g. p_filt_value units
        line_names (list[str or float)]): a list of calibration lines either as number (which is
            energies in eV), or name to be looked up in STANDARD_FEATURES
        nextra (int): the algorithm starts with the first len(line_names) + nextra peak_positions
        nincrement (int): each the algorithm fails to find a satisfactory peak assignment, it uses
            nincrement more lines
        nextramax (int): the algorithm stops incrementint nextra past this value, instead
            failing with a ValueError saying "no peak assignment succeeded"
        rms_cutoff (float): an empirical number that determines if an assignment is good enough.
            The default number works reasonably well for NSLS-II data
        autoinclude (int): Number of tallest peaks to include in all combinations
    """

    name_e, e_e = line_names_and_energies(line_names)
    energies = np.asarray(e_e, dtype="float")
    n_sel = len(line_names) + nextra  # number of peaks to consider for fitting
    nmax = len(line_names) + nextramax
    while True:
        sel_positions = np.asarray(peak_positions[:n_sel], dtype="float")

        assign = getPeakCombinations(sel_positions, len(energies), autoinclude)
        bestPeaks, bestRMS, allRMS = getAccuracyEstimates(
            energies, assign, curvename, polyorder
        )

        if bestRMS > rms_cutoff:
            n_sel += nincrement
            if n_sel > nmax:
                print(
                    f"no peak assignment succeeded: Best RMS: {bestRMS}, RMS Cutoff: {rms_cutoff}"
                )
                if debug:
                    return name_e, energies, assign, allRMS
                else:
                    return name_e, energies, bestPeaks, bestRMS
            else:
                continue
        else:
            if debug:
                return name_e, energies, assign, allRMS
            else:
                return name_e, energies, bestPeaks, bestRMS


def getAccuracyEstimates(energies, assignments, curvename="gain", maxPolyOrder=5):
    """
    energies : Physical energies of peaks
    assignments : Array of possible peak combinations
    curvename : input to find_poly_residual, assumed form of TES gain curve
    maxPolyOrder : The maximum order of polynomial to be used to fit the peaks
    """
    polyorder = min(len(energies) - 2, maxPolyOrder)
    allRMS = []
    for peaks in assignments[:, ...]:
        _, _, rms = find_poly_residual(energies, peaks, polyorder, curvename)
        allRMS.append(rms)
    bestRMSIndex = np.argmin(allRMS)
    bestRMS = allRMS[bestRMSIndex]
    bestPeaks = assignments[bestRMSIndex, :]

    return bestPeaks, bestRMS, allRMS


def getPeakCombinations(positions, npeaks, autoinclude=1):
    peakCombos = []
    if autoinclude == npeaks:
        peakCombos.append(list(positions[:autoinclude]))
    else:
        for combo in combinations(positions[autoinclude:], npeaks - autoinclude):
            tmp = list(positions[:autoinclude])
            tmp.extend(combo)
            peakCombos.append(tmp)
    peakCombos = np.array(peakCombos)
    peakCombos.sort(axis=1)
    return peakCombos


def debugAssignment(ds, attr, states, ph_fwhm, line_names, assignment="nsls", **kwargs):
    peak_ph_vals, _peak_heights = mass.algorithms.find_local_maxima(
        ds.getAttr(attr, indsOrStates=states), ph_fwhm
    )


def ds_learnCalibrationPlanFromEnergiesAndPeaks(
    self, attr, states, ph_fwhm, line_names, assignment="nsls", **kwargs
):
    peak_positions, _peak_heights = mass.algorithms.find_local_maxima(
        self.getAttr(attr, indsOrStates=states), ph_fwhm
    )
    if assignment == "nsls":
        name_or_e, e_out, assignment, rms = assignPeaks(
            peak_positions, line_names, rms_cutoff=1, **kwargs
        )
    else:
        name_or_e, e_out, assignment = mass.algorithms.find_opt_assignment(
            peak_positions, line_names, maxacc=0.1, **kwargs
        )
        rms = None

    self.calibrationPlanInit(attr)
    for ph, name in zip(assignment, name_or_e):
        if type(name) is str:
            self.calibrationPlanAddPoint(ph, name, states=states)
        else:
            energy = name
            name = str(energy)
            self.calibrationPlanAddPoint(ph, name, states=states, energy=energy)
    return e_out, assignment, rms


mass.off.Channel.learnCalibrationPlanFromEnergiesAndPeaks = (
    ds_learnCalibrationPlanFromEnergiesAndPeaks
)


def data_calibrationLoadFromHDF5Simple(self, h5name, recipeName="energy"):
    print(f"loading calibration from {h5name}")
    with h5py.File(h5name, "r") as h5:
        nchans = len(list(h5.keys()))
        print(f"Calibration for {nchans} channels found")
        calibrationAttr = h5.attrs.get("calAttr", "filtValue")
        for channum_str in h5.keys():
            cal = mass.calibration.EnergyCalibration.load_from_hdf5(h5, channum_str)
            channum = int(channum_str)
            if channum in self:
                ds = self[channum]
                ds.recipes.add(recipeName, cal, [calibrationAttr], overwrite=True)
    # set other channels bad
    for ds in self.values():
        if recipeName not in ds.recipes.keys():
            ds.markBad("no loaded calibration")


mass.off.ChannelGroup.calibrationLoadFromHDF5Simple = data_calibrationLoadFromHDF5Simple


def data_calibrationSaveToHDF5Simple(self, h5name, recipeName="energy"):
    print(f"writing calibration to {h5name}")
    with h5py.File(h5name, "w") as h5:
        for ds in self.values():
            cal = ds.recipes[recipeName].f
            cal.save_to_hdf5(h5, f"{ds.channum}")
        h5.attrs["calAttr"] = ds.calibrationPlanAttr


mass.off.ChannelGroup.calibrationSaveToHDF5Simple = data_calibrationSaveToHDF5Simple


def find_poly_residual(cal_energies, opt_assignment, degree, curvename="gain"):
    if curvename == "gain":
        x = opt_assignment
        y = opt_assignment / cal_energies
    elif curvename == "loglog":
        y = np.log(opt_assignment)
        x = np.log(cal_energies)
    elif curvename == "loggain":
        x = opt_assignment
        y = np.log(opt_assignment / cal_energies)
    elif curvename == "linear":
        x = np.insert(opt_assignment, 0, 0.0)
        y = np.insert(cal_energies, 0, 0.0)
    coeff = np.polyfit(x, y, degree)
    poly = np.poly1d(coeff)
    residual = poly(x) - y
    residual_rms = np.sqrt(sum(np.square(residual)) / len(cal_energies))
    return coeff, residual, residual_rms


def data_calibrate(
    self,
    cal_state,
    line_names,
    fv="filtValueDC",
    rms_cutoff=0.2,
    assignment="nsls",
    recipeName="energy",
    **kwargs,
):
    """
    Calibrate the data using specified calibration lines.

    Parameters
    ----------
    cal_state : str
        State to calibrate
    line_names : list
        Names of calibration lines to use
    fv : str, optional
        Name of filtered value attribute
    rms_cutoff : float, optional
        Maximum RMS value for acceptable calibration
    assignment : str, optional
        Peak assignment method
    recipeName : str, optional
        Name for calibration recipe
    **kwargs : dict
        Additional arguments passed to learnCalibrationPlanFromEnergiesAndPeaks

    Returns
    -------
    dict
        Processing information including:
        - 'calibration_status': dict mapping channel numbers to status
        - 'rms_per_channel': dict mapping channel numbers to RMS values
        - 'calibrated_channels': int count of successfully calibrated channels
        - 'total_channels': int total number of channels
        - 'calibration_histograms': dict containing histogram data per channel:
            - 'bin_centers': array of energy bin centers
            - 'counts': dict mapping channel numbers to count arrays
            - 'energy_range': tuple of (min_energy, max_energy)
    """
    self.setDefaultBinsize(0.2)
    line_energies = get_line_energies(line_names)

    # Initialize processing info
    processing_info = {
        "calibration_status": {},
        "rms_per_channel": {},
        "calibrated_channels": 0,
        "total_channels": len(self),
        "calibration_histograms": {
            "counts": {},
            "energy_range": (min(line_energies) - 50, max(line_energies) + 50),
        },
    }

    # Set up histogram bins
    bins = np.arange(
        processing_info["calibration_histograms"]["energy_range"][0],
        processing_info["calibration_histograms"]["energy_range"][1],
        1,
    )
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    processing_info["calibration_histograms"]["bin_centers"] = bin_centers
    processing_info["calibration_lines"] = line_energies

    for ds in self.values():
        try:
            e_out, peaks, rms = ds.learnCalibrationPlanFromEnergiesAndPeaks(
                attr=fv,
                ph_fwhm=50,
                states=cal_state,
                line_names=line_energies,
                assignment=assignment,
                **kwargs,
            )
            if rms < rms_cutoff:
                processing_info["calibrated_channels"] += 1
                processing_info["rms_per_channel"][ds.channum] = rms
                processing_info["calibration_status"][
                    ds.channum
                ] = f"Calibrated (RMS: {rms:.3f})"

                print(f"Calibrating {ds.channum} succeeded with rms: {rms}")
                calibration = mass.EnergyCalibration(
                    curvetype="gain", approximate=False
                )
                calibration.uncalibratedName = fv
                for e, ph, line_name in zip(e_out, peaks, line_names):
                    calibration.add_cal_point(ph, e, str(line_name))
                ds.recipes.add(
                    recipeName,
                    calibration,
                    [calibration.uncalibratedName],
                    overwrite=True,
                )

                # Save calibration histogram
                try:
                    energies = ds.getAttr("energy", cal_state)
                    counts, _ = np.histogram(energies, bins)
                    processing_info["calibration_histograms"]["counts"][
                        ds.channum
                    ] = counts
                except Exception as e:
                    print(
                        f"Failed to save histogram for channel {ds.channum}: {str(e)}"
                    )

            else:
                msg = f"Failed RMS cut: {rms:.3f} > {rms_cutoff}"
                processing_info["calibration_status"][ds.channum] = msg
                print(f"Chan {ds.channum}: {msg}")
                ds.markBad(msg)
        except ValueError as e:
            msg = f"Failed peak assignment: {str(e)}"
            processing_info["calibration_status"][ds.channum] = msg
            print(f"Chan {ds.channum}: {msg}")
            ds.markBad(msg)

    return processing_info


"""    self.calibrateFollowingPlan(
        fv, calibratedName=recipeName, dlo=7, dhi=7, overwriteRecipe=True
    )
    for ds in self.values():
        # ds.calibrateFollowingPlan(fv, overwriteRecipe=True, dlo=7, dhi=7)

        ecal = ds.recipes[recipeName].f
        degree = min(len(ecal._ph) - 1, 2)
        _, _, rms = find_poly_residual(ecal._energies, ecal._ph, degree, "gain")
        if np.any(ecal._ph < 0):
            msg = "Failed calibration with ph < 0"
            print(msg)
            ds.markBad(msg)
            continue
        if rms > rms_cutoff:
            msg = f"Failed calibration cut with RMS: {rms}, cutoff: {rms_cutoff}"
            print(msg)
            ds.markBad(msg)
            continue
        try:
            ds.getAttr(recipeName, cal_state)[:10]
        except ValueError:
            ds.markBad(
                "ValueError on energy access, calibration curve is probably broken"
            )
"""

mass.off.ChannelGroup.calibrate = data_calibrate


def should_make_new_calibration(cal_file_name, overwrite):
    """
    Returns True if we should make a new calibration
    Returns False if a calibration exists and we are not overwriting it
    """
    if cal_file_name is not None and path.exists(cal_file_name) and not overwrite:
        return False
    else:
        return True


from matplotlib.gridspec import GridSpec


class CalFigure:
    def __init__(
        self, line_names, line_energies, figsize=None, title="Stacked calibration"
    ):
        naxes = len(line_names)
        self.line_names = line_names
        self.line_energies = line_energies
        if figsize is None:
            figsize = (2 * naxes, 8)
        self.fig = plt.figure(figsize=figsize)
        self.fig.subplots_adjust(wspace=0)
        gs = GridSpec(2, naxes)
        self.panel = self.fig.add_subplot(gs[0, :])
        self.axlist = [self.fig.add_subplot(gs[1, 0])]
        for n in range(1, naxes):
            ax = self.fig.add_subplot(gs[1, n])
            ax.set_yticks([])
            self.axlist.append(ax)
        for i in range(naxes):
            name = line_names[i]
            energy = line_energies[i]
            self.axlist[i].set_xlim(energy - 20, energy + 20)
            self.axlist[i].set_title(name)
            self.axlist[i].axvline(energy)
            self.panel.axvline(energy)
        self.fig.suptitle(title)

    def plot_ds_calibration(self, ds, state, legend=True):
        bins = np.arange(
            np.min(self.line_energies) - 50, np.max(self.line_energies) + 50, 1
        )
        centers = 0.5 * (bins[1:] + bins[:-1])
        energies = ds.getAttr("energy", state)
        counts, _ = np.histogram(energies, bins)
        max_ylim = 0
        for ax in self.axlist:
            ax.plot(centers, counts, label=f"Chan {ds.channum}")
            max_ylim = max(max_ylim, ax.get_ylim()[1])
        for ax in self.axlist:
            ax.set_ylim(0, max_ylim)
        self.panel.plot(centers, counts, label=f"Chan {ds.channum}")
        if legend:
            self.panel.legend()

    def save(self, savename, close=True):
        self.fig.savefig(savename)
        if close:
            self.close()

    def close(self):
        plt.close(self.fig)


def _make_panel_figure(
    line_names, line_energies, figsize=None, title="Stacked calibration"
):
    naxes = len(line_names)
    if figsize is None:
        figsize = (2 * naxes, 4)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0)
    axlist = fig.subplots(1, naxes, sharey=True)
    for i in range(naxes):
        name = line_names[i]
        energy = line_energies[i]
        axlist[i].set_xlim(energy - 20, energy + 20)
        axlist[i].set_title(name)
        axlist[i].axvline(energy)
    fig.suptitle(title)
    return fig, axlist


def _make_single_figure(
    line_names, line_energies, figsize=None, title="Stacked calibration"
):
    naxes = len(line_names)
    if figsize is None:
        figsize = (2 * naxes, 4)
    fig = plt.figure(figsize=figsize)
    fig.subplots_adjust(wspace=0)
    axlist = fig.subplots(1, naxes, sharey=True)
    for i in range(naxes):
        name = line_names[i]
        energy = line_energies[i]
        axlist[i].set_xlim(energy - 20, energy + 20)
        axlist[i].set_title(name)
        axlist[i].axvline(energy)
    fig.suptitle(title)
    return fig, axlist


def plot_ds_calibration(ds, state, line_energies, axlist, legend=True):
    bins = np.arange(np.min(line_energies) - 50, np.max(line_energies) + 50, 1)
    centers = 0.5 * (bins[1:] + bins[:-1])
    energies = ds.getAttr("energy", state)
    counts, _ = np.histogram(energies, bins)

    for ax in axlist:
        ax.plot(centers, counts, label=f"Chan {ds.channum}")
    if legend:
        ax.legend()


def summarize_calibration(data, state, line_names, savedir, overwrite=False):
    """
    Should try to produce an overall summary
    Also, splitting into panels sometimes makes it hard to figure out if we are globally misaligned
    """
    print(f"Saving summaries to {savedir}")
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    line_energies = get_line_energies(line_names)
    nstack = 8
    naxes = len(line_names)
    bigfig = CalFigure(
        line_names,
        line_energies,
        figsize=(3 * naxes, 6),
        title="All ds calibration stacked",
    )
    fig = CalFigure(line_names, line_energies)
    startchan = 1
    for n, chan in enumerate(data):
        if chan > startchan + nstack - 1:
            filename = f"cal_{startchan}_to_{startchan + nstack - 1}.png"
            savename = os.path.join(savedir, filename)
            if not os.path.exists(savename) or overwrite:
                fig.save(savename)
            else:
                fig.close()
            fig = CalFigure(line_names, line_energies)
            startchan = startchan + nstack

        ds = data[chan]
        bigfig.plot_ds_calibration(ds, state, legend=False)
        fig.plot_ds_calibration(ds, state)
        lastchan = chan
        # work in progress
    bigfig.save(os.path.join(savedir, "cal_summary_all_chan.png"))

    filename = f"cal_{startchan}_to_{lastchan}.png"
    savename = os.path.join(savedir, filename)
    if not os.path.exists(savename) or overwrite:
        fig.save(savename)
    else:
        fig.close()
