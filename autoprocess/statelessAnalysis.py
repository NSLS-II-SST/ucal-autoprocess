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

    # Get data files
    data = get_data(run)
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
