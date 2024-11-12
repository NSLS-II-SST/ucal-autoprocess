import numpy as np
from mass.off import ChannelGroup, getOffFileListFromOneFile
from os.path import dirname
import os
from .utils import (
    get_tes_state,
    get_filename,
    get_tes_arrays,
    get_savename,
    get_calibration,
)

from .processing import (
    correct_run,
    calibrate_run,
    load_calibration,
    load_correction,
    data_is_calibrated,
    data_is_corrected,
)

from .scanData import scandata_from_run


def get_data(run):
    filename = get_filename(run)
    files = getOffFileListFromOneFile(filename, maxChans=400)
    data = ChannelGroup(files)
    return data


def handle_run(uid, catalog, save_directory):
    """
    Process a single run given its UID.

    Parameters
    ----------
    uid : str
        Unique identifier for the run to process
    catalog : WrappedDatabroker, optional
        Data catalog instance. If None, creates new connection
    save_directory : str, optional
        Directory to save processed data

    Returns
    -------
    bool
        True if processing succeeded, False otherwise
    """

    run = catalog[uid]

    # Check if run contains TES data
    if "tes" not in run.start.get("detectors", []):
        print("No TES in run, skipping!")
        return False
    if run.start.get("scantype", "") in ["noise", "projectors"]:
        print("Nothing to be done for Noise or Projectors")
        return False

    # Get data files
    try:
        data = get_data(run)
    except:
        print(f"Could not find or load TES .off files for {run.start['scan_id']}")
        return False
    # Handle calibration runs first
    if run.start.get("scantype", "") == "calibration":
        return handle_calibration_run(run, data, catalog, save_directory)
    else:
        return handle_science_run(run, data, catalog, save_directory)


def handle_calibration_run(run, data, catalog, save_directory):
    """
    Process a calibration run.

    Parameters
    ----------
    run : DataBroker run
        Run to process
    data : ChannelGroup
        TES data
    catalog : WrappedDatabroker
        Data catalog
    save_directory : str
        Directory to save processed data

    Returns
    -------
    bool
        True if processing succeeded
    """
    scan_id = run.start.get("scan_id", "")

    print(f"Handling Calibration Run for scan {scan_id}")
    print("Correcting data")
    correct_run(run, data, save_directory)
    print(f"Calibrating Scan {scan_id}")
    calibrate_run(run, data, save_directory)
    save_processed_data(run, data, save_directory)

    return data


def handle_science_run(run, data, catalog, save_directory):
    """
    Process a science run.

    Parameters
    ----------
    run : DataBroker run
        Run to process
    data : ChannelGroup
        TES data
    catalog : WrappedDatabroker
        Data catalog
    save_directory : str
        Directory to save processed data

    Returns
    -------
    bool
        True if processing succeeded
    """
    # Find the last calibration run

    scan_id = run.start.get("scan_id", "")
    cal_run = get_calibration(run, catalog)
    cal_id = cal_run.start.get("scan_id", "")
    print(f"Handling science data for scan {scan_id}, with cal from scan {cal_id}")
    if load_correction(cal_run, data, save_directory) and load_calibration(
        cal_run, data, save_directory
    ):
        print("Correction and Calibration loaded successfully")
    else:
        print("Loading Calibration Data")
        handle_calibration_run(cal_run, data, catalog, save_directory)

    if data_is_corrected(data) and data_is_calibrated(data):
        save_processed_data(run, data, save_directory)
        return True
    else:
        print(f"Data was not fully processed for scan {scan_id}")
        return False


def save_processed_data(run, data, save_directory):
    """Save processed calibration data"""
    state = get_tes_state(run)
    savename = get_savename(run, save_directory)
    scan_id = run.start.get("scan_id", "")
    print(f"Saving data for scan {scan_id} to {save_directory}")
    os.makedirs(dirname(savename), exist_ok=True)

    ts, e, ch = get_tes_arrays(data, state)
    np.savez(savename, timestamps=ts, energies=e, channels=ch)


def get_tes_data(run, save_directory, logtype="run"):
    """
    rois : dictionary of {roi_name: (llim, ulim)}
    """
    d = scandata_from_run(run, save_directory, logtype)
    rois = {}
    for key in run.primary.descriptors[0]["object_keys"]["tes"]:
        if "tes_mca" in key and key not in rois and key != "tes_mca_spectrum":
            llim = run.primary.descriptors[0]["data_keys"][key].get("llim", 200)
            ulim = run.primary.descriptors[0]["data_keys"][key].get("ulim", 2000)
            rois[key] = [llim, ulim]

    tes_data = {}
    for roi in rois:
        llim, ulim = rois[roi]
        y, x = d.getScan1d(llim, ulim)
        tes_data[roi] = y
    # Kludge for certain older data, and non-XAS data that was still scanning the energy
    if "tes_mca_pfy" not in rois:
        if d.log.motor_name == "en_energy":
            llim = min(d.log.motor_vals)
            ulim = max(d.log.motor_vals)
            y, x = d.getScan1d(llim, ulim)
            rois["tes_mca_pfy"] = [llim, ulim]
            tes_data["tes_mca_pfy"] = y
    return rois, tes_data
