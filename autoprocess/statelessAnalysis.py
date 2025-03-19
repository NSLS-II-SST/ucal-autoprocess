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


def handle_run(
    uid,
    catalog,
    save_directory,
    reprocess=False,
    verbose=True,
    data=None,
    return_data=False,
    processing_settings=None,
):
    """
    Process a single run given its UID.

    Parameters
    ----------
    uid : str
        Unique identifier for the run to process
    catalog : WrappedDatabroker
        Data catalog instance.
    save_directory : str
        Directory to save processed data
    reprocess : bool, optional
        If True, force reprocessing even if data already exists
    verbose : bool, optional
        If True, prints processing status messages

    Returns
    -------
    dict
        Dictionary containing processing information
    """

    run = catalog[uid]
    if data is None and not return_data:
        should_close_data = True
    else:
        should_close_data = False

    processing_info = {
        "processed": False,
        "reason": "",
        "calibrated_channels": 0,
        "total_channels": 0,
        "success": False,
        "details": {},
        "run_info": {
            "uid": run.start.get("uid", ""),
            "scan_id": run.start.get("scan_id", ""),
            "sample_name": run.start.get("sample", ""),
            "scantype": run.start.get("scantype", ""),
            "timestamp": run.start.get("time", ""),
        },
    }

    if return_data:
        processing_info["data"] = data

    # Check if run contains TES data
    if "tes" not in run.start.get("detectors", []):
        processing_info["reason"] = "No TES in run"
        if verbose:
            print("No TES in run, skipping!")
        return processing_info

    scantype = run.start.get("scantype", "")
    if scantype in ["noise", "projectors"]:
        processing_info["reason"] = f"Scantype '{scantype}' does not require processing"
        if verbose:
            print("Nothing to be done for Noise or Projectors")
        return processing_info

    if not reprocess:
        try:
            savename = get_savename(run, save_directory)
        except Exception as e:
            processing_info["reason"] = f"Could not get TES Filename: {e}"
            if verbose:
                print(f"Could not get TES Filename: {e}")
            return processing_info
        if exists(savename):
            processing_info["processed"] = True
            processing_info["reason"] = f"TES already processed to {savename}"
            if verbose:
                print(f"TES Already processed to {savename}, will not reprocess")
            # Check calibrated channels
            processing_info["success"] = True
            return processing_info

    # Get data files
    try:
        if data is None:
            if verbose:
                print(f"Loading TES Data from {get_filename(run)}")
            data = get_data(run)
            if verbose:
                print("TES Data loaded")
        else:
            if verbose:
                print("Using provided TES Data")
        processing_info["data"] = data
    except Exception as e:
        processing_info["reason"] = f"Error loading TES data: {e}"
        if verbose:
            print(f"Error {e} for .off files for {run.start.get('scan_id', '')}")
        return processing_info

    # Handle calibration runs first
    data.verbose = verbose

    try:
        if scantype == "calibration":
            _processing_info = handle_calibration_run(
                run, data, catalog, save_directory, processing_settings
            )
        else:
            _processing_info = handle_science_run(
                run, data, catalog, save_directory, processing_settings
            )

        # Save and remove run info before update
        run_info = processing_info.pop("run_info")
        # Get calibration source if it exists before update
        calibration_source = _processing_info.pop("calibration_source", None)

        processing_info.update(_processing_info)

        # Restore run info and calibration source after update
        processing_info["run_info"] = run_info
        if calibration_source:
            processing_info["calibration_source"] = calibration_source

    except Exception as e:
        processing_info["reason"] = f"Error while handling run: {e}"
        if verbose:
            print(f"Error {e} while handling {run.start['uid']}")
        return processing_info

    if should_close_data:
        print("Offloading TES Data Files")
        for ds in data.values():
            ds.offFile.close()
    else:
        print("Not offloading TES Data Files")

    # Collect processing information
    processing_info["processed"] = True
    processing_info["success"] = True

    return processing_info


def handle_calibration_run(
    run, data, catalog, save_directory, processing_settings=None
):
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
    dict
        Dictionary containing data_calibration_info with calibration processing information
    """

    scan_id = run.start.get("scan_id", "")

    print(f"Handling Calibration Run for scan {scan_id}")
    print("Correcting data")
    correct_run(run, data, save_directory)
    print(f"Calibrating Scan {scan_id}")
    cal_info = calibrate_run(run, data, save_directory)
    save_processed_data(run, data, save_directory)

    processing_info = {
        "data_calibration_info": {
            "run_info": {
                "uid": run.start.get("uid", ""),
                "scan_id": run.start.get("scan_id", ""),
                "sample_name": run.start.get("sample", ""),
                "scantype": run.start.get("scantype", ""),
                "timestamp": run.start.get("time", ""),
            },
            **cal_info,  # Include all calibration information
        }
    }
    processing_info["calibration_applied"] = True

    return processing_info


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
    dict
        Dictionary containing either data_processing_info or both data_processing_info
        and data_calibration_info if new calibration was performed
    """
    scan_id = run.start.get("scan_id", "")
    cal_run = get_calibration(run, catalog)
    cal_id = cal_run.start.get("scan_id", "")

    processing_info = {
        "data_processing_info": {
            "run_info": {
                "uid": run.start.get("uid", ""),
                "scan_id": run.start.get("scan_id", ""),
                "sample_name": run.start.get("sample", ""),
                "scantype": run.start.get("scantype", ""),
                "timestamp": run.start.get("time", ""),
            },
            "calibration_source": {
                "uid": cal_run.start.get("uid", ""),
                "scan_id": cal_id,
                "sample_name": cal_run.start.get("sample", ""),
                "scantype": cal_run.start.get("scantype", ""),
                "timestamp": cal_run.start.get("time", ""),
            },
            "total_channels": len(data),
            "calibrated_channels": 0,
            "calibration_status": {},
        }
    }

    print(f"Processing scan {scan_id} using calibration from scan {cal_id}")

    if load_correction(cal_run, data, save_directory) and load_calibration(
        cal_run, data, save_directory
    ):
        print("Loaded existing calibration")
        for ds in data.values():
            if "energy" in ds.recipes.keys():
                processing_info["data_processing_info"]["calibrated_channels"] += 1
                processing_info["data_processing_info"]["calibration_status"][
                    ds.channum
                ] = "Loaded from file"
            else:
                processing_info["data_processing_info"]["calibration_status"][
                    ds.channum
                ] = "Load failed"
    else:
        print("Performing new calibration")
        # Get calibration info and merge it with processing info
        cal_info = handle_calibration_run(cal_run, data, catalog, save_directory)
        processing_info.update(cal_info)  # This adds data_calibration_info

        # Update processing info with calibration results
        for channum, status in cal_info["data_calibration_info"][
            "calibration_status"
        ].items():
            if "Calibrated" in status:
                processing_info["data_processing_info"]["calibrated_channels"] += 1
                processing_info["data_processing_info"]["calibration_status"][
                    channum
                ] = "Newly calibrated"
            else:
                processing_info["data_processing_info"]["calibration_status"][
                    channum
                ] = status

    if data_is_corrected(data) and data_is_calibrated(data):
        save_processed_data(run, data, save_directory)
        processing_info["data_processing_info"]["data_saved"] = True

    return processing_info


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
    h5_name = base_name.replace(".npz", "_2d.h5")

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
