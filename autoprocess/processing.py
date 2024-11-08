from .utils import (
    get_tes_state,
    get_correction_file,
    get_calibration_file,
    get_line_names,
)
from .calibration import summarize_calibration
from os.path import join, dirname, exists
import os


def correct_run(run, data, save_directory=None):
    state = get_tes_state(run)
    data.learnResidualStdDevCut(states=[state])
    data.learnDriftCorrection(
        indicatorName="pretriggerMean",
        uncorrectedName="filtValue",
        correctedName="filtValueDC",
        states=state,
    )
    if save_directory is not None:
        dc_name = get_correction_file(run, save_directory, make_dirs=True)
        data.saveRecipeBooks(dc_name)
    return data


def load_correction(run, data, save_directory):
    dc_name = get_correction_file(run, save_directory)
    if exists(dc_name):
        try:
            data.loadRecipeBooks(dc_name)
            return True
        except:
            return False
    else:
        return False


def calibrate_run(run, data, save_directory=None):
    # Phase correct and calibrate
    line_names = get_line_names(run)
    state = get_tes_state(run)
    data.calibrate(state, line_names, "filtValueDC")

    if save_directory is not None:
        h5name = get_calibration_file(run, save_directory, make_dirs=True)
        cal_dir = dirname(h5name)
        data.calibrationSaveToHDF5Simple(h5name, recipeName="energy")
        summarize_calibration(data, state, line_names, cal_dir, overwrite=True)
    return data


def load_calibration(run, data, save_directory):
    h5name = get_calibration_file(run, save_directory)
    if exists(h5name):
        try:
            data.calibrationLoadFromHDF5Simple(h5name, recipeName="energy")
            return True
        except:
            return False
    else:
        return False


def data_is_corrected(data):
    """
    Check if run data has drift corrections applied.

    Parameters
    ----------
    data : ChannelGroup
        TES data to check

    Returns
    -------
    bool
        True if corrections are applied
    """
    try:
        ds = data.firstGoodChannel()
        return hasattr(ds, "filtValueDC")
    except Exception as e:
        print(f"Failed to check corrections: {str(e)}")
        return False


def data_is_calibrated(data):
    """
    Check if run data is calibrated.

    Parameters
    ----------
    data : ChannelGroup
        TES data to check

    Returns
    -------
    bool
        True if data is calibrated
    """
    try:
        ds = data.firstGoodChannel()
        return hasattr(ds, "energy")
    except Exception as e:
        print(f"Failed to check calibration: {str(e)}")
        return False
