import numpy as np
import json
from .utils import get_savename, get_logname

"""
Module for loading and working with processed energy/timestamp data
"""


class ProcessedData:
    def __init__(self, timestamps, energies, channels):
        self.timestamps = timestamps
        self.energies = energies
        self.channels = channels
        self.chanlist = sorted(list(set(channels)))

    def index_between_times(self, start, stop):
        idx1 = np.searchsorted(self.timestamps, start)
        idx2 = np.searchsorted(self.timestamps, stop)
        return idx1, idx2

    def select_between_times(self, start, stop, channels=None):
        idx1, idx2 = self.index_between_times(start, stop)
        e = self.energies[idx1:idx2]
        if channels is not None:
            chans = self.channels[idx1:idx2]
            chansel = np.array(channels)
            if len(chansel) > 1:
                chans = chans[np.newaxis, :]
                chansel = chansel[:, np.newaxis]
                chan_idx = np.any(chans == chansel, axis=0)
            else:
                chan_idx = chans == chansel
            e = e[chan_idx]
        return e

    def sum_roi_between_times(self, start, stop, llim, ulim, channels=None):
        energies = self.select_between_times(start, stop, channels=channels)
        return np.sum((energies < ulim) & (energies > llim))

    def histogram_between_times(self, start, stop, e_bins, channels=None, offset=0):
        energies = self.select_between_times(start, stop, channels=channels)
        ehist, _ = np.histogram(energies - offset, e_bins)
        return ehist


class LogData:
    def __init__(self, start_times, stop_times, motor_name, motor_vals):
        self.start_times = start_times
        self.stop_times = stop_times
        self.motor_name = motor_name
        self.motor_vals = motor_vals


class ScanData:
    def __init__(self, data, log):
        self.data = data
        self.log = log

    def getScan1d(self, llim, ulim, channels=None):
        counts = np.zeros_like(self.log.start_times)
        for n in range(len(counts)):
            counts[n] = self.data.sum_roi_between_times(
                self.log.start_times[n],
                self.log.stop_times[n],
                llim,
                ulim,
                channels=channels,
            )
        return counts, self.log.motor_vals

    def getScan2d(self, llim, ulim, eres=0.3, channels=None, eloss=False):
        mono_list = self.log.motor_vals
        mono_grid, energy_grid, e_bins = getScan2dGrids(mono_list, llim, ulim, eres)
        counts = np.zeros_like(mono_grid)
        for n, m in enumerate(mono_list):
            offset = m if eloss else 0
            counts[:, n] = self.data.histogram_between_times(
                self.log.start_times[n],
                self.log.stop_times[n],
                e_bins,
                channels=channels,
                offset=offset,
            )
        return counts, mono_grid, energy_grid

    def getEmission(
        self, llim, ulim, eres=0.3, strictTimebins=False, channels=None, **kwargs
    ):
        n_e_pts = int((ulim - llim) // eres)
        e_bins = np.linspace(llim, ulim, n_e_pts)
        e_centers = (e_bins[1:] + e_bins[:-1]) / 2
        emission = self.data.histogram_between_times(
            self.log.start_times[0],
            self.log.stop_times[-1],
            e_bins,
            channels=channels,
            **kwargs,
        )
        return emission, e_centers

    def getArrays1d(self, llim, ulim, channels=None):
        mono_list = self.log.motor_vals
        mono_arr = []
        emission_arr = []
        for n in range(len(mono_list)):
            e = self.data.select_between_times(
                self.log.start_times[n], self.log.stop_times[n], channels=channels
            )
            e = e[(e < ulim) & (e > llim)]
            m = np.zeros_like(e) + mono_list[n]
            mono_arr.append(m)
            emission_arr.append(e)
        mono_arr = np.hstack(mono_arr)
        emission_arr = np.hstack(emission_arr)
        return mono_arr, emission_arr


def getScan2dGrids(mono_list, llim, ulim, eres=0.3):
    n_e_pts = int((ulim - llim) // eres)
    e_bins = np.linspace(llim, ulim, n_e_pts)
    e_centers = (e_bins[1:] + e_bins[:-1]) / 2

    mono_grid, energy_grid = np.meshgrid(mono_list, e_centers)
    return mono_grid, energy_grid, e_bins


def data_from_file(filename):
    data = np.load(filename)
    timestamps = data["timestamps"] * 1e-9  # data is stored as nanoseconds
    energies = data["energies"]
    channels = data["channels"]
    return ProcessedData(timestamps, energies, channels)


def log_from_json(run):
    logname = get_logname(run)
    with open(logname, "r") as f:
        log = json.load(f)
    start_time = log["epoch_time_start_s"]
    acquire_time = run.primary["config"]["tes"]["tes_acquire_time"].read()[0]
    stop_time = start_time + acquire_time
    # stop_time = log['epoch_time_end_s']
    motor_name = log["var_name"]
    motor_vals = log["var_values"]
    return LogData(start_time, stop_time, motor_name, motor_vals)


def log_from_run(run):
    try:
        start_time = run.primary.data["tes_scan_point_start"].read()
    except KeyError:
        start_time = run.primary["timestamps"]["time"].read()
    # acquire_time = run.primary.descriptors[0]['configuration']['tes']['data']['tes_acquire_time']
    acquire_time = run.primary["config"]["tes"]["tes_acquire_time"].read()[0]
    stop_time = start_time + acquire_time
    if run.metadata["start"].get("scantype", "") in [
        "calibration",
        "xes",
        "darkcounts",
    ]:
        motor_name = "time"
        motor_vals = start_time
    else:
        try:
            motor_name = run.metadata["start"]["motors"][0]
            motor_vals = run.metadata["start"]["plan_args"]["args"][1]
        except KeyError:
            motor_name = "time"
            motor_vals = start_time
    return LogData(start_time, stop_time, motor_name, motor_vals)


def scandata_from_run(run, save_directory, logtype="json"):
    filename = get_savename(run, save_directory)
    data = data_from_file(filename)
    if logtype == "run":
        log = log_from_run(run)
    else:
        log = log_from_json(run)
    return ScanData(data, log)
