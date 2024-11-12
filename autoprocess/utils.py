"""Utility functions for TES data processing."""

import numpy as np
from os.path import dirname, join, basename, exists
import datetime
import os


def get_proposal_path(run):
    proposal = run.start.get("proposal", {}).get("proposal_id", None)
    is_commissioning = (
        "commissioning" in run.start.get("proposal", {}).get("type", "").lower()
    )
    cycle = run.start.get("cycle", None)
    if proposal is None or cycle is None:
        raise ValueError("Proposal Metadata not Loaded")
    if is_commissioning:
        proposal_path = f"/nsls2/data/sst/proposals/commissioning/pass-{proposal}/"
    else:
        proposal_path = f"/nsls2/data/sst/proposals/{cycle}/pass-{proposal}/"
    return proposal_path


def get_processed_path(run):
    proposal_path = get_proposal_path(run)

    export_path = join(proposal_path, "processing", "ucal-1")
    return export_path


def get_config_dict(run):
    return run.primary.descriptors[0]["configuration"]["tes"]["data"]


def get_noise_uid(run):
    config = get_config_dict(run)
    if "tes_noise_uid" not in config:
        raise KeyError("No noise UID found in run metadata")
    return config["tes_noise_uid"]


def get_noise(run, catalog):
    uid = get_noise_uid(run)
    try:
        return catalog[uid]
    except KeyError:
        raise KeyError(f"Noise run with UID {uid} not found in catalog")


def get_projector_uid(run):
    config = get_config_dict(run)
    if "tes_projector_uid" not in config:
        raise KeyError("No projector UID found in run metadata")
    return config["tes_projector_uid"]


def get_projector(run, catalog):
    uid = get_projector_uid(run)
    try:
        return catalog[uid]
    except KeyError:
        raise KeyError(f"Projector run with UID {uid} not found in catalog")


def get_calibration_uid(run):
    config = get_config_dict(run)
    if "tes_calibration_uid" not in config:
        raise KeyError("No calibration UID found in run metadata")
    return config["tes_calibration_uid"]


def get_calibration(run, catalog):
    uid = get_calibration_uid(run)
    try:
        return catalog[uid]
    except KeyError:
        raise KeyError(f"Calibration run with UID {uid} not found in catalog")


def get_filename(run):
    filename = get_config_dict(run)["tes_filename"]
    return filename


def get_logname(run):
    config = get_config_dict(run)
    state_str = "calibration" if config["tes_cal_flag"] else "scan"
    scan_num = config["tes_scan_num"]
    filebase = dirname(get_filename(run))
    filename = f"{state_str}{scan_num:0=4d}.json"
    return join(filebase, "logs", filename)


def get_tes_state(run):
    config = get_config_dict(run)
    if "tes_scan_str" in config:
        state = config["tes_scan_str"]
    else:
        state_str = "CAL" if config["tes_cal_flag"] else "SCAN"
        scan_num = config["tes_scan_num"]
        state = f"{state_str}{scan_num}"
    return state


def get_samplename(run):
    return run.start.get("sample_name", "")


def get_model_file(run, catalog):
    """
    Get projector model file path from run metadata.

    Parameters
    ----------
    run : DataBroker run
        Run containing projector metadata
    catalog : WrappedDatabroker
        Data catalog instance

    Returns
    -------
    str
        Path to projector model file
    """
    projector_uid = run.primary.descriptors[0]["configuration"]["tes"]["data"][
        "tes_projector_uid"
    ]
    projector_run = catalog[projector_uid]
    projector_base = dirname(
        projector_run.primary.descriptors[0]["configuration"]["tes"]["data"][
            "tes_filename"
        ]
    )
    return join(projector_base, "projectors.hdf5")


def get_line_names(run):
    """
    Get calibration line names based on sample and energy.

    Parameters
    ----------
    run : DataBroker run
        Run containing calibration metadata

    Returns
    -------
    list
        List of calibration line names
    """
    if "cal_lines" in run.start:
        return run.start["cal_lines"]

    samplename = get_samplename(run)
    energy = run.start.get("calibration_energy", 980)

    if samplename == "mixv1":
        line_energies = np.array([300, 400, 525, 715, 840, 930])
        line_names = np.array(["ck", "nk", "ok", "fela", "nila", "cula"])
    else:
        line_energies = np.array([300, 400, 525, 715, 840, 930])
        line_names = np.array(["ck", "nk", "ok", "fela", "nila", "cula"])

    return list(line_names[line_energies < energy])


def get_savename(run, save_directory):
    """
    Generate save filename from run metadata.

    Parameters
    ----------
    run : DataBroker run
        Run to generate filename for
    save_directory : str
        Base directory for saving data

    Returns
    -------
    str
        Full path for saving data
    """
    filename = get_filename(run)
    date = basename(filename).split("_")[0]
    tes_prefix = "_".join(basename(filename).split("_")[:2])
    state = get_tes_state(run)
    scanid = run.start["scan_id"]
    return join(save_directory, date, f"{tes_prefix}_{state}_scan_{scanid}.npz")


def run_is_processed(run, save_directory):
    try:
        filename = get_savename(run, save_directory)
    except:
        print(
            f"Could not find any TES file in {save_directory} for Run {run.start['scan_id']}"
        )
        return False
    return exists(filename)


def get_processing_directory(run, save_directory):
    savebase = get_savename(run, save_directory)[:-4]
    savename = f"{savebase}_processing"
    return savename


def get_correction_file(run, save_directory, make_dirs=False):
    dc_dir = get_processing_directory(run, save_directory)
    if make_dirs:
        os.makedirs(dc_dir, exist_ok=True)
    dc_name = join(dc_dir, "drift_correction.pkl")
    return dc_name


def get_calibration_file(run, save_directory, make_dirs=False):
    cal_dir = get_processing_directory(run, save_directory)
    if make_dirs:
        os.makedirs(cal_dir, exist_ok=True)
    cal_name = join(cal_dir, "calibration.hdf5")
    return cal_name


def get_tes_arrays(data, state, attr="energy"):
    """
    Extract timestamp, energy and channel arrays from data.

    Parameters
    ----------
    data : ChannelGroup
        TES data container
    state : str
        Data state to extract
    attr : str, optional
        Attribute to extract, defaults to "energy"

    Returns
    -------
    tuple
        (timestamps, energies, channels) arrays sorted by timestamp
    """
    timestamps = []
    energies = []
    channels = []

    for ds in data.values():
        try:
            uns, es = ds.getAttr(["unixnano", attr], state)
        except Exception as e:
            print(f"Channel {ds.channum} failed: {str(e)}")
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

    return ts_arr[sort_idx], en_arr[sort_idx], ch_arr[sort_idx]
