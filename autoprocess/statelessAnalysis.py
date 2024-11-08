import numpy as np
from mass.off import ChannelGroup, getOffFileListFromOneFile
from os.path import dirname
import os
from .utils import get_tes_state, get_filename, get_tes_arrays, get_savename
from .processing import (
    drift_correct_run,
    calibrate_run,
    run_is_calibrated,
    run_is_corrected,
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
    drift_correct_run(run, data, save_directory)
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
    # TODO: Find corresponding calibration run
    # cal_run = find_calibration_run(run, catalog)

    # TODO: Check if calibration is processed, process if needed
    # if not is_calibration_processed(cal_run):
    #     handle_run(cal_run.start['uid'], catalog, save_directory)

    # TODO: Apply calibration to science data
    # apply_calibration(data, cal_run)

    # Save processed data
    state = get_tes_state(run)
    save_processed_data(run, data, state, save_directory)

    return True


def save_processed_data(run, data, state, save_directory):
    """Save processed calibration data"""
    state = get_tes_state(run)
    savename = get_savename(run, save_directory)
    os.makedirs(dirname(savename), exist_ok=True)

    ts, e, ch = get_tes_arrays(data, state)
    np.savez(savename, timestamps=ts, energies=e, channels=ch)