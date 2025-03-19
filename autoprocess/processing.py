from .utils import (
    get_tes_state,
    get_correction_file,
    get_calibration_file,
    get_line_names,
)
from .calibration import summarize_calibration
from os.path import join, dirname, exists


def correct_run(run, data, save_directory=None):
    state = get_tes_state(run)
    print(f"Correcting data for {state}")
    data.learnResidualStdDevCut(states=[state], overwriteRecipe=True)
    data.learnDriftCorrection(
        indicatorName="pretriggerMean",
        uncorrectedName="filtValue",
        correctedName="filtValueDC",
        states=state,
        overwriteRecipe=True,
    )
    if save_directory is not None:
        dc_name = get_correction_file(run, save_directory, make_dirs=True)
        print(f"Saving corrections to {dc_name}")
        data.saveRecipeBooks(dc_name)
    return data


def load_correction(run, data, save_directory):
    dc_name = get_correction_file(run, save_directory)
    print(f"Loading correction from {dc_name}")
    if exists(dc_name):
        try:
            data.loadRecipeBooks(dc_name)
            return True
        except Exception as e:
            print(f"Got error loading correction: {e}")
            return False
    else:
        return False


def calibrate_run(run, data, save_directory=None):
    """
    Calibrate the run data and save calibration results.

    Parameters
    ----------
    run : DataBroker run
        The run to calibrate
    data : ChannelGroup
        TES data associated with the run
    save_directory : str, optional
        Directory to save calibration results

    Returns
    -------
    dict
        Processing information from calibration
    """
    line_names = get_line_names(run)
    state = get_tes_state(run)
    print(f"Calibrating {state} with lines {line_names}")

    processing_info = data.calibrate(state, line_names, "filtValueDC")

    if save_directory is not None:
        h5name = get_calibration_file(run, save_directory, make_dirs=True)
        cal_dir = dirname(h5name)
        print(f"Saving calibration to {h5name}")
        data.calibrationSaveToHDF5Simple(h5name, recipeName="energy")
        summarize_calibration(data, state, line_names, cal_dir, overwrite=True)
        processing_info["calibration_saved"] = True
        processing_info["calibration_file"] = h5name

    return processing_info


def load_calibration(run, data, save_directory):
    h5name = get_calibration_file(run, save_directory)

    if exists(h5name):
        print(f"Loading correction from {h5name}")
        try:
            data.calibrationLoadFromHDF5Simple(h5name, recipeName="energy")
            return True
        except Exception as e:
            print(f"Got error loading calibration: {e}")
            return False
    else:
        print(f"No calibration exists at {h5name} yet")
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
        return "filtValueDC" in ds.recipes.keys()
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
