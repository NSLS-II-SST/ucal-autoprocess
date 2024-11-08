from .utils import get_tes_state, get_processing_directory, get_line_names
from .calibration import summarize_calibration
from os.path import join


def drift_correct_run(run, data, save_directory=None):
    state = get_tes_state(run)

    data.learnDriftCorrection(
        indicatorName="pretriggerMean",
        uncorrectedName="filtValue",
        correctedName="filtValueDC",
        states=state,
    )
    if save_directory is not None:
        dc_dir = get_processing_directory(run, save_directory)
        dc_name = join(dc_dir, "drift_correction.hdf5")
        data.saveRecipeBooks(dc_name)
    return data


def calibrate_run(run, data, save_directory=None):
    # Phase correct and calibrate
    line_names = get_line_names(run)
    state = get_tes_state(run)
    data.calibrate(state, line_names, "filtValueDC")

    if save_directory is not None:
        cal_dir = get_processing_directory(run, save_directory)
        h5name = join(cal_dir, "calibration.hdf5")
        data.calibrationSaveToHDF5Simple(h5name, recipeName="energy")
        summarize_calibration(data, state, line_names, cal_dir, overwrite=True)
    return data


def run_is_corrected(data):
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


def run_is_calibrated(data):
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
