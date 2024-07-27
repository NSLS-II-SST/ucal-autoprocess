import mass
import numpy as np
import matplotlib.pyplot as plt
from ucalpost.databroker.run import get_tes_state, get_filename, get_logname, get_line_names, get_config_dict
from ucalpost.databroker.catalog import WrappedDatabroker
from ucalpost.tes.process_classes import log_from_run, ProcessedData, ScanData, LogData, data_from_file
from ucalpost.tes.noise import get_noise_and_projectors, load_mass
from tiled.client import from_uri
from os.path import dirname, join, exists, basename
import os
import calibration
import json
from mass.off import ChannelGroup, getOffFileListFromOneFile, Channel, labelPeak, labelPeaks

"""
Todo:
* Run summary function in SA
* Time filtering of catalog (lookback based on selected run)
* Server mode that automatically processes data
* Automatic lookback for most recent cal from data (needs TES cal uid signal)
* Data export to formats TBD

"""

plt.close("all")
plt.ion()

class SerialAnalysis():
    def __init__(self, start_index=None):
        self.raw_catalog = WrappedDatabroker(from_uri("http://172.31.133.41:8000"))
        self.catalog = self.raw_catalog.filter_by_stop()
        self._data = None
        self._save_directory = "/home/decker/processed"
        self._clear_run()
        if start_index is not None:
            try:
                self.run = self.catalog[start_index]
            except:
                self.run = self.catalog[-1]
        else:
            self.run = self.catalog[-1]

    def refresh(self):
        self.catalog = self.raw_catalog.filter_by_stop()
        if self._data is not None:
            self._data.refreshFromFiles()

    def _clear_run(self):
        self.run = None
        self.sd = None
        
    def _close_data(self):
        for ds in self._data.values():
            ds.offFile.close()
        self._data = None
    
    def get_data(self):
        filename = get_filename(self.run, convert_local=False)
        files = getOffFileListFromOneFile(filename, maxChans=400)
        if self._data is None:
            self._data = ChannelGroup(files)
            return self._data
        elif self._data.offFileNames[0] == files[0]:
            return self._data
        else:
            self._close_data()
            self._data = ChannelGroup(files)
            return self._data

    def run_is_corrected(self):
        try:
            data = self.get_data()
            ds = data.firstGoodChannel()
            if not hasattr(ds, "filtValue5LagDC"):
                return False
            else:
                return True
        except:
            return False

    def run_is_calibrated(self):
        try:
            data = self.get_data()
            ds = data.firstGoodChannel()
            if not hasattr(ds, "energy"):
                return False
            else:
                return True
        except:
            return False
            
    def correct_run(self):
        if self.run is None:
            raise ValueError("run is None!")
        data = self.get_data()
        ds = data.firstGoodChannel()
        state = get_tes_state(self.run)
        model_path = get_model_file(self.run, self.catalog)
        data.add5LagRecipes(model_path)
        data.learnDriftCorrection(indicatorName="pretriggerMean", uncorrectedName=f"filtValue5Lag", correctedName=f"filtValue5LagDC", states=state)

    def calibrate_run(self, line_names=None):
        if not self.run_is_corrected():
            self.correct_run()
        state = get_tes_state(self.run)
        data = self.get_data()
        if line_names is None:
            line_names = get_line_names(self.run)
        data.calibrate(state, line_names, "filtValue5LagDC")
        return data

    def handle_run(self):
        if "tes" not in self.run.start.get('detectors', []):
            print("No TES in run, skipping!")
            return True
        if self.run.start.get('scantype', '') == "calibration":
            self.calibrate_run()
        if self.run_is_calibrated():
            self.save_run()
            self.load_scandata()
            return True
        else:
            print("No Calibration Exists! Something went wrong")
            return False

    def handle_all_runs(self):
        success = True
        while success:
            success = self.handle_run() & bool(self.advance_one_run())
        
    def save_run(self):
        state = get_tes_state(self.run)
        savename = get_savename(self.run, self._save_directory)
        os.makedirs(dirname(savename), exist_ok=True)
        ts, e, ch = get_tes_arrays(self._data, state)
        print(f"Saving {savename}")
        np.savez(savename, timestamps=ts, energies=e, channels=ch)
        
    def load_scandata(self):
        savename = get_savename(self.run, self._save_directory)
        if exists(savename):
            data = data_from_file(savename)
            log = log_from_run(self.run)
            self.sd = ScanData(data, log)
        else:
            self.sd = sd_from_run(self._data, self.run)

    def get_scandata(self):
        if self.sd is not None:
            return self.sd
        else:
            self.load_scandata()
            return self.sd
        
    def advance_one_run(self):
        uid = self.run.start['uid']
        uids = list(self.catalog._catalog.keys())
        idx = uids.index(uid)

        if idx == len(uids) - 1:
            print("Already at last run! Staying put")
            return None
        else:
            self._clear_run()
            new_uid = uids[idx+1]
            self.run = self.catalog[new_uid]
            return self.run

    def go_back_one_run(self):
        uid = self.run.start['uid']
        uids = list(self.catalog._catalog.keys())
        idx = uids.index(uid)

        if idx == 0:
            print("Already at first run! Staying put")
            return None
        else:
            self._clear_run()
            new_uid = uids[idx-1]
            self.run = self.catalog[new_uid]
            return self.run
        
    def advance_to_latest_run(self):
        self._clear_run()
        self.run = self.catalog[-1]
        return self.run

    def advance_to_index(self, index):
        self._clear_run()
        self.run = self.catalog[index]
        return self.run

    def get_emission(self, llim, ulim, eres=0.3, channels=None):
        """
        Returns emission energy, counts
        """
        sd = self.get_scandata()
        y, x = sd.getEmission(llim, ulim, eres=0.3, channels=channels)
        return x, y
    
    def plot_emission(self, llim, ulim, eres=0.3, channels=None):
        x, y = self.get_emission(llim, ulim, eres, channels)
        if channels is None:
            title = "XES of all channels"
        else:
            title = f"XES of {channels}"
        plt.figure()
        plt.title(title)
        plt.plot(x, y)
        plt.xlabel("Emission energy (eV)")
        plt.ylabel("Counts")

    def get_scan1d(self, llim, ulim, channels=None):
        """
        Returns motor values, counts
        """
        sd = self.get_scandata()
        y, x = sd.getScan1d(llim, ulim, channels=channels)
        return x, y
        
    def plot_scan1d(self, llim, ulim, channels=None):
        x, y = self.get_scan1d(llim, ulim, channels=channels)
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Probably mono energy (eV)")
        plt.ylabel("Counts")

    def get_scan2d(self, llim, ulim, eres=0.3, channels=None):
        """
        Returns Mono, Emission, Counts
        """
        sd = self.get_scandata()
        zz, xx, yy = sd.getScan2d(llim, ulim, eres=0.3, channels=channels)
        return xx, yy, zz
        
    def plot_scan2d(self,  llim, ulim, eres=0.3, channels=None):
        xx, yy, zz = self.get_scan2d(llim, ulim, eres, channels)
        plt.figure()
        plt.contourf(xx, yy, zz, 50)
        
        
def get_model_file(run, catalog):
    projector_uid = run.primary.descriptors[0]['configuration']['tes']['data']['tes_projector_uid']
    projector_run = catalog[projector_uid]
    projector_base = dirname(projector_run.primary.descriptors[0]['configuration']['tes']['data']['tes_filename'])
    projector_filename = join(projector_base, "projectors.hdf5")
    return projector_filename

def get_savename(run, save_directory="/home/decker/processed"):
    filename = get_filename(run, convert_local=False)
    date = filename.split('/')[-3]
    tes_prefix = "_".join(basename(filename).split("_")[:2])
    state = get_tes_state(run)
    scanid = run.start['scan_id']
    savename = join(save_directory, date, f"{tes_prefix}_{state}_scan_{scanid}.npz")
    return savename

def get_tes_arrays(data, state, attr="energy"):
    timestamps = []
    energies = []
    channels = []
    for ds in data.values():
        try:
            uns, es = ds.getAttr(["unixnano", attr], state)
        except:
            print(f"{ds.channum} failed")
            ds.markBad("Failed to get energy")
            continue
        ch = np.zeros_like(uns) + ds.channum
        timestamps.append(uns)
        energies.append(es)
        channels.append(ch)
    ts_arr = np.concatenate(timestamps)
    en_arr = np.concatenate(energies)
    ch_arr = np.concatenate(channels)
    sort_idx = np.argsort(ts_arr)
    
    timestamps=ts_arr[sort_idx]
    energies=en_arr[sort_idx]
    channels=ch_arr[sort_idx]
    
    return timestamps, energies, channels
