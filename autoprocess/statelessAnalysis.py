import numpy as np
from mass.off import ChannelGroup, getOffFileListFromOneFile
from os.path import dirname, exists
import os
import h5py

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


def handle_run(uid, catalog, save_directory, reprocess=False, verbose=True):
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

    if not reprocess:
        try:
            savename = get_savename(run, save_directory)
        except Exception as e:
            print(f"Could not get TES Filename: {e}")
            return False
        if exists(savename):
            print(f"TES Already processed to {savename}, will not reprocess")
            return True
    # Get data files
    try:
        print(f"Loading TES Data from {get_filename(run)}")
        data = get_data(run)
        print("TES Data loaded")
    except Exception as e:
        print(f"Error {e} for .off files for {run.start['scan_id']}")
        return False
    # Handle calibration runs first
    data.verbose = verbose
    try:
        if run.start.get("scantype", "") == "calibration":
            r = handle_calibration_run(run, data, catalog, save_directory)
        else:
            r = handle_science_run(run, data, catalog, save_directory)
    except Exception as e:
        print(f"Error {e} while handling {run.start['uid']}")
        r = False
    for ds in data.values():
        ds.offFile.close()
    return r


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
    scan_data = scandata_from_run(run, save_directory, logtype)
    rois = {}
    for key in run.primary.descriptors[0]["object_keys"]["tes"]:
        if "tes_mca" in key and key not in rois and key != "tes_mca_spectrum":
            llim = run.primary.descriptors[0]["data_keys"][key].get("llim", 200)
            ulim = run.primary.descriptors[0]["data_keys"][key].get("ulim", 2000)
            rois[key] = [llim, ulim]

    tes_data = {}
    for roi in rois:
        llim, ulim = rois[roi]
        y, x = scan_data.getScan1d(llim, ulim)
        tes_data[roi] = y
    # Kludge for certain older data, and non-XAS data that was still scanning the energy
    if "tes_mca_pfy" not in rois:
        if scan_data.log.motor_name == "en_energy":
            llim = min(scan_data.log.motor_vals)
            ulim = max(scan_data.log.motor_vals)
            y, x = scan_data.getScan1d(llim, ulim)
            rois["tes_mca_pfy"] = [llim, ulim]
            tes_data["tes_mca_pfy"] = y
    return rois, tes_data


# Need to actually test/debug
def save_2d_data(
    run, save_directory, energy_range=(200, 2000), energy_step=0.3, logtype="run"
):
    """
    Save 2D data (motor energy vs detector energy) to an HDF5 file.

    Parameters
    ----------
    run : DataBroker run
        Run to process
    save_directory : str
        Directory to save processed data
    energy_range : tuple
        (lower_limit, upper_limit) for detector energy range in eV
    energy_step : float
        Step size for detector energy binning in eV
    logtype : str
        Type of log data to use ('run' or 'json')

    Returns
    -------
    str
        Path to saved HDF5 file
    """

    scan_data = scandata_from_run(run, save_directory, logtype)
    llim, ulim = energy_range

    # Get 2D data
    counts, mono_grid, energy_grid = scan_data.getScan2d(
        llim=llim, ulim=ulim, eres=energy_step
    )

    # Create HDF5 filename
    base_name = get_savename(run, save_directory)
    h5_name = base_name.replace(".npz", "_rixs.h5")

    # Save to HDF5
    with h5py.File(h5_name, "w") as f:
        f.create_dataset("counts", data=counts)
        f.create_dataset("mono_energy", data=mono_grid)
        f.create_dataset("detector_energy", data=energy_grid)

        # Add metadata
        f.attrs["scan_id"] = run.start.get("scan_id", "")
        f.attrs["uid"] = run.start.get("uid", "")
        f.attrs["sample_name"] = run.start.get("sample", "")
        f.attrs["motor_name"] = scan_data.log.motor_name

    return h5_name
