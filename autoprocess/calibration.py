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
    peak_heights,
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
    peak_positions = peak_positions[peak_heights.argsort()[::-1]]
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
    allRMS = np.array(allRMS)

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


def compute_closeness(peaks, heights):
    peaks = np.array(peaks)
    heights = np.array(heights)
    sorted_peak_args = peaks.argsort()
    sorted_peaks = peaks[sorted_peak_args]
    sorted_heights = heights[sorted_peak_args]
    closeness = (sorted_peaks[1:] - sorted_peaks[:-1]) / (
        0.5 * (sorted_peaks[1:] + sorted_peaks[:-1])
    )
    return closeness, sorted_peaks, sorted_heights


def remove_close_peaks(peaks, heights, cutoff=0.02):
    closeness, sorted_peaks, sorted_heights = compute_closeness(peaks, heights)
    if min(closeness) > cutoff:
        return sorted_peaks, sorted_heights

    isolated_peaks = []
    isolated_heights = []
    skipidx = set()
    skipnext = False
    for i, c in enumerate(closeness):
        if skipnext:
            skipnext = False
            continue
        if c < cutoff:
            h1 = sorted_heights[i]
            h2 = sorted_heights[1 + i]
            if h1 > h2:
                skipidx.add(i + 1)
            else:
                skipidx.add(i)
            skipnext = True

    isolated_peaks = np.array(
        [p for i, p in enumerate(sorted_peaks) if i not in skipidx]
    )
    isolated_heights = np.array(
        [h for i, h in enumerate(sorted_heights) if i not in skipidx]
    )
    return remove_close_peaks(isolated_peaks, isolated_heights, cutoff)


def get_peaks(ds, attr, states, ph_fwhm, max_peak_ratio=15, min_closeness=0.02):
    peak_positions, peak_heights = mass.algorithms.find_local_maxima(
        ds.getAttr(attr, indsOrStates=states), ph_fwhm
    )
    second_highest_peak = peak_heights[
        1
    ]  # compute ratio from second highest peak for robustness against outliers
    peak_cutoff = second_highest_peak / max_peak_ratio
    filtered_peaks = peak_positions[peak_heights > peak_cutoff]
    filtered_heights = peak_heights[peak_heights > peak_cutoff]
    isolated_peaks, isolated_heights = remove_close_peaks(
        filtered_peaks, filtered_heights, cutoff=min_closeness
    )
    return isolated_peaks, isolated_heights


def make_test_calibration(lines, peaks, curvetype="gain", approximate=False):
    cal = mass.EnergyCalibration(curvetype=curvetype, approximate=approximate)
    for ph, e in zip(peaks, lines):
        cal.add_cal_point(ph, e)
    return cal


def drop_one_rms(cal):
    _, err = cal.drop_one_errors()
    return np.sqrt(np.sum(err**2) / len(err))


def compare_drop_ones(channum, lines, assignments, rms, nmax=5, **kwargs):
    rms = np.array(rms)
    idx = np.argsort(rms)
    sorted_peaks = assignments[idx]
    sorted_rms = rms[idx]
    drop_ones = []
    cal_list = []
    for assignment, r in zip(sorted_peaks[:nmax], sorted_rms[:nmax]):
        cal = make_test_calibration(lines, assignment, **kwargs)
        try:
            drms = drop_one_rms(cal)
        except:
            drms = np.inf
        drop_ones.append(drms)
        cal_list.append(cal)
    didx = np.argmin(drop_ones)
    if didx == 0:
        print(f"Chan {channum} has no drop-one improvement")
        return sorted_peaks[0], sorted_rms[0]
    else:
        print(
            f"Chan {channum} has drop-one improvement, RMS degradation from {sorted_rms[0]} to {sorted_rms[didx]}"
        )
        return sorted_peaks[didx], sorted_rms[didx]


def nsls_assignment(
    ds,
    attr,
    states,
    ph_fwhm,
    line_names,
    max_peak_ratio=15,
    min_closeness=0.02,
    **kwargs,
):
    peak_positions, peak_heights = get_peaks(
        ds, attr, states, ph_fwhm, max_peak_ratio, min_closeness
    )
    name_or_e, e_out, assignment_list, rms_list = assignPeaks(
        peak_positions, peak_heights, line_names, rms_cutoff=1, debug=True, **kwargs
    )
    rms_idx = np.argsort(rms_list)
    rms_sort = rms_list[rms_idx]
    if len(rms_sort) > 1:
        if rms_sort[1] < rms_sort[0] * 1.5:
            print(
                f"Chan {ds.channum} has close assignments with RMS {rms_sort[0]} and {rms_sort[1]}, checking drop-one"
            )
            drop_one_assignment, drop_one_rms = compare_drop_ones(
                ds.channum, e_out, assignment_list, rms_list, nmax=3
            )
            return name_or_e, e_out, drop_one_assignment, drop_one_rms
        else:
            return name_or_e, e_out, assignment_list[rms_idx[0]], rms_list[rms_idx[0]]
    else:
        return name_or_e, e_out, assignment_list[rms_idx[0]], rms_list[rms_idx[0]]


def ds_learnCalibrationPlanFromEnergiesAndPeaks(
    self,
    attr,
    states,
    ph_fwhm,
    line_energies,
    assignment="nsls",
    max_peak_ratio=15,
    min_closeness=0.02,
    **kwargs,
):

    if assignment == "nsls":
        name_or_e, e_out, assignment, rms = nsls_assignment(
            self,
            attr,
            states,
            ph_fwhm,
            line_energies,
            max_peak_ratio,
            min_closeness,
            **kwargs,
        )
    else:
        peak_positions, _peak_heights = mass.algorithms.find_local_maxima(
            self.getAttr(attr, indsOrStates=states), ph_fwhm
        )
        name_or_e, e_out, assignment = mass.algorithms.find_opt_assignment(
            peak_positions, line_energies, maxacc=0.1, **kwargs
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
        ds = self.firstGoodChannel()
        h5.attrs["calAttr"] = ds.calibrationPlanAttr

        with self.includeBad():
            for ds in self.values():
                if recipeName in ds.recipes.keys():
                    try:
                        cal = ds.recipes[recipeName].f
                        cal.save_to_hdf5(h5, f"{ds.channum}")
                    except:
                        print(f"Failed to save calibration for channel {ds.channum}")


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


def rms_histograms(allcounts):
    medcounts = np.median(allcounts, axis=0)
    medcounts = medcounts / np.sum(medcounts)
    n = len(medcounts)
    allcounts = allcounts / np.sum(allcounts, axis=1)[:, np.newaxis]
    rms = np.sqrt(np.sum((medcounts[np.newaxis, :] - allcounts) ** 2 / n, axis=1))
    return rms


def cut_dissimilar_histograms(calibration_histograms, stddev_cutoff=2):
    """
    Identify bad channels based on similarity to the median calibration histogram.

    stddev_cutoff : i.e, if 2, cut channels with rms error more than 2 std dev away from the average

    Returns list of channel keys with rms error outside the given standard deviation cutoff
    """
    allkeys = list(calibration_histograms["counts"].keys())
    allcounts = np.array([calibration_histograms["counts"][i] for i in allkeys])
    all_rms = rms_histograms(allcounts)
    bad_idx = np.arange(len(all_rms))[
        all_rms > np.mean(all_rms) + stddev_cutoff * np.std(all_rms)
    ]
    bad_keys = [allkeys[i] for i in bad_idx]
    bad_rms = [all_rms[i] for i in bad_idx]

    return bad_keys, bad_rms, np.mean(all_rms)


def calibrate_channel(
    ds,
    attr,
    cal_state,
    line_names,
    rms_cutoff=0.2,
    assignment="nsls",
    recipeName="energy",
    processing_info=None,
    **kwargs,
):

    if processing_info is None:
        processing_info = initialize_processing_info(line_names, total_channels=1)
    line_energies = processing_info["line_energies"]

    e_out, peaks, rms = ds.learnCalibrationPlanFromEnergiesAndPeaks(
        attr=attr,
        ph_fwhm=50,
        states=cal_state,
        line_energies=line_energies,
        assignment=assignment,
        **kwargs,
    )

    processing_info["rms_per_channel"][ds.channum] = rms
    processing_info["status"][ds.channum] = {
        "message": f"Calibrated (RMS: {rms:.3f})",
        "success": True,
    }

    print(f"Calibrating {ds.channum} succeeded with rms: {rms}")
    calibration = mass.EnergyCalibration(curvetype="gain", approximate=False)
    calibration.uncalibratedName = attr
    for e, ph, line_name in zip(e_out, peaks, line_names):
        calibration.add_cal_point(ph, e, str(line_name))
    ds.recipes.add(
        recipeName,
        calibration,
        [calibration.uncalibratedName],
        overwrite=True,
    )

    # Save calibration histogram
    bins = processing_info["histograms"]["bin_centers"]
    try:
        energies = ds.getAttr("energy", cal_state)
        counts, _ = np.histogram(energies, bins)
        processing_info["histograms"]["counts"][ds.channum] = counts
    except Exception as e:
        print(f"Failed to save histogram for channel {ds.channum}: {str(e)}")

    if rms > rms_cutoff:
        msg = f"Failed Calibration: Failed RMS cut ({rms:.3f} > {rms_cutoff})"
        processing_info["status"][ds.channum]["message"] = msg
        processing_info["status"][ds.channum]["success"] = False
        print(f"Chan {ds.channum}: {msg}")
        ds.markBad(msg)
    else:
        processing_info["calibrated_channels"] += 1
        print(f"Chan {ds.channum}: Calibrated with RMS: {rms:.3f}")

    return processing_info


def initialize_processing_info(line_names, total_channels=1):
    line_energies = get_line_energies(line_names)  # Initialize processing info

    processing_info = {
        "status": {},
        "rms_per_channel": {},
        "calibrated_channels": 0,
        "total_channels": total_channels,
        "histograms": {
            "counts": {},
            "energy_range": (min(line_energies) - 50, max(line_energies) + 50),
        },
    }

    # Set up histogram bins
    bins = np.arange(
        processing_info["histograms"]["energy_range"][0],
        processing_info["histograms"]["energy_range"][1],
        1,
    )
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    processing_info["histograms"]["bin_centers"] = bin_centers
    processing_info["line_energies"] = line_energies
    processing_info["line_names"] = line_names
    return processing_info


def data_calibrate(
    self,
    cal_state,
    line_names,
    fv="filtValueDC",
    rms_cutoff=0.2,
    assignment="nsls",
    recipeName="energy",
    stddev_cutoff=2,
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
    processing_info = initialize_processing_info(line_names, total_channels=len(self))

    for ds in self.values():
        try:
            ds_info = calibrate_channel(
                ds,
                attr=fv,
                cal_state=cal_state,
                line_names=line_names,
                rms_cutoff=rms_cutoff,
                assignment=assignment,
                recipeName=recipeName,
                processing_info=processing_info,
                **kwargs,
            )
            processing_info.update(ds_info)
        except ValueError as e:
            msg = f"Failed Calibration: Failed peak assignment: {str(e)}"
            processing_info["status"][ds.channum] = {"message": msg, "success": False}
            print(f"Chan {ds.channum}: {msg}")
            ds.markBad(msg)

    bad_keys, bad_rms, mean_rms = cut_dissimilar_histograms(
        processing_info["histograms"], stddev_cutoff
    )
    for i, key in enumerate(bad_keys):
        msg = f"Failed Calibration: Bad histogram (RMS: {bad_rms[i]:.3e} > {stddev_cutoff}*{mean_rms:.3e})"
        processing_info["status"][key] = {"message": msg, "success": False}
        print(f"Chan {key}: {msg}")
        self[key].markBad(msg)

    return processing_info


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


def plot_calibration_failure(
    ds,
    state,
    reason="",
    savedir=None,
    close=True,
    overwrite=True,
    default_attr="filtValueDC",
):
    """
    Plot the RMS and peak failure for a given channel
    """
    fig = plt.figure()
    if "energy" in ds.recipes.keys():
        ax = fig.add_subplot(212)
        ax2 = fig.add_subplot(211)
        ecal = ds.recipes["energy"].f
        attr = ecal.uncalibratedName
        ph_min = np.min(ecal._ph) * 0.9
        ph_max = np.max(ecal._ph) * 1.1
        e_min = np.min(ecal._energies) * 0.9
        e_max = np.max(ecal._energies) * 1.1
        e_range = np.linspace(e_min, e_max, 1000)
        ds.plotHist(e_range, "energy", axis=ax2, states=[state])
    elif "energyRough" in ds.recipes.keys():
        ax = fig.add_subplot(212)
        ax2 = fig.add_subplot(211)
        ecal = ds.recipes["energyRough"].f
        attr = ecal.uncalibratedName
        ph_min = np.min(ecal._ph) * 0.9
        ph_max = np.max(ecal._ph) * 1.1
        e_min = np.min(ecal._energies) * 0.9
        e_max = np.max(ecal._energies) * 1.1
        e_range = np.linspace(e_min, e_max, 1000)
        ds.plotHist(e_range, "energyRough", axis=ax2, states=[state])
    else:
        print("No energy recipe found for channel {ds.channum}")
        ax = fig.add_subplot(111)
        attr = default_attr
        ph_min = 0
        ph_max = 20000

    ph_range = np.linspace(ph_min, ph_max, 1000)

    ds.plotHist(ph_range, attr, axis=ax, states=[state])
    fig.suptitle(f"Chan {ds.channum}: {reason}")
    if savedir is not None:
        fig.savefig(os.path.join(savedir, f"cal_failure_{ds.channum}_{attr}.png"))
    if close:
        plt.close(fig)


def plot_calibration_channel(ds, state, line_names):
    line_energies = get_line_energies(line_names)
    fig = CalFigure(line_names, line_energies)
    fig.plot_ds_calibration(ds, state)


def summarize_calibration(
    data, state, line_names, savedir, close=True, overwrite=False
):
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
    offset = data.keys()[0]
    startchan = offset
    for n, chan in enumerate(data):
        if (chan - (chan - offset) % nstack) > startchan:
            filename = f"cal_{startchan}_to_{startchan + nstack - 1}.png"
            savename = os.path.join(savedir, filename)
            if not os.path.exists(savename) or overwrite:
                fig.save(savename)
            else:
                fig.close()
            startchan = chan - (chan - offset) % nstack

            fig = CalFigure(line_names, line_energies)

        ds = data[chan]
        bigfig.plot_ds_calibration(ds, state, legend=False)
        fig.plot_ds_calibration(ds, state)
        lastchan = chan
        # work in progress
    if savedir is not None:
        bigfig.save(os.path.join(savedir, "cal_summary_all_chan.png"), close=close)

        filename = f"cal_{startchan}_to_{lastchan}.png"
        savename = os.path.join(savedir, filename)
        if not os.path.exists(savename) or overwrite:
            fig.save(savename, close=close)
    elif close:
        fig.close()

    for channum, reason in data.whyChanBad.items():
        print(f"Channel {channum}: {reason}")
        if reason.startswith("Failed Calibration"):
            print(f"Plotting calibration failure for channel {channum}")
            try:
                plot_calibration_failure(
                    data[channum],
                    state,
                    reason,
                    savedir=savedir,
                    close=close,
                    overwrite=overwrite,
                )
            except Exception as e:
                print(f"Failed to plot calibration failure for channel {channum}: {e}")
