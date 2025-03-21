from .utils import (
    get_tes_state,
    get_correction_file,
    get_calibration_file,
    get_line_names,
)
from .calibration import summarize_calibration
from os.path import join, dirname, exists


def correct_run(
    run, data, save_directory=None, correction_dict={}, calibration_dict={}
):
    """
    Correct the run data and save correction results.

    Parameters
    ----------
    run : Tiled run
        The run to correct
    data : ChannelGroup
        TES data associated with the run
    save_directory : str, optional
        Directory to save correction results
    correction_dict : dict, optional
        Dictionary containing correction parameters

    Returns
    -------
    dict
        Correction information
    """
    state = get_tes_state(run)
    correction_info = {"state": state}
    dcIndicatorName = correction_dict.get("dcIndicatorName", "pretriggerMean")
    dcUncorrectedName = correction_dict.get("dcUncorrectedName", "filtValue")
    dcCorrectedName = correction_dict.get("dcCorrectedName", dcUncorrectedName + "DC")
    if correction_dict.get("5lagModelFile", False):
        model_path = correction_dict["5lagModelFile"]
        data.add5LagRecipes(model_path)
        correction_info["5lagModelFile"] = model_path
        dcUncorrectedName = dcUncorrectedName + "5Lag"
        dcCorrectedName = dcUncorrectedName + "5LagDC"

    print(f"Correcting data for {state}")
    correction_info["dcIndicatorName"] = dcIndicatorName
    correction_info["dcUncorrectedName"] = dcUncorrectedName
    correction_info["dcCorrectedName"] = dcCorrectedName
    data.learnResidualStdDevCut(states=[state], overwriteRecipe=True)
    correction_info["cutType"] = "residualStdDev"
    data.learnDriftCorrection(
        indicatorName=dcIndicatorName,
        uncorrectedName=dcUncorrectedName,
        correctedName=dcCorrectedName,
        states=state,
        overwriteRecipe=True,
    )
    correction_info["correctedName"] = dcCorrectedName
    correction_info["driftCorrected"] = True

    if correction_dict.get("phaseCorrect", False):
        prelim_cal_dict = {}
        prelim_cal_dict.update(calibration_dict)
        prelim_cal_dict["fvAttr"] = dcCorrectedName
        calibrate_run(run, data, save_directory=None, calibration_dict=prelim_cal_dict)
        pcIndicatorName = correction_dict.get("pcIndicatorName", "filtPhase")
        pcUncorrectedName = dcCorrectedName
        pcCorrectedName = dcCorrectedName + "PC"
        data.learnPhaseCorrection(
            pcIndicatorName,
            pcUncorrectedName,
            pcCorrectedName,
            states=state,
            overwriteRecipe=True,
        )
        correction_info["phaseCorrected"] = True
        correction_info["pcIndicatorName"] = pcIndicatorName
        correction_info["pcUncorrectedName"] = pcUncorrectedName
        correction_info["pcCorrectedName"] = pcCorrectedName
        correction_info["correctedName"] = pcCorrectedName

    if save_directory is not None:
        dc_name = get_correction_file(run, save_directory, make_dirs=True)
        print(f"Saving corrections to {dc_name}")
        data.saveRecipeBooks(dc_name)
        correction_info["correction_file"] = dc_name
    return correction_info


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


def calibrate_run(run, data, save_directory=None, calibration_dict={}):
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
    _cal_dict = {}
    _cal_dict.update(calibration_dict)
    fvAttr = _cal_dict.pop("fvAttr", "filtValueDC")
    processing_info = data.calibrate(state, line_names, fvAttr, **_cal_dict)

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
        recipes = ds.recipes.keys()
        possible_dc_recipes = ["filtValueDC", "filtValue5LagDC", "filtValue5LagDCPC"]
        return any(recipe in recipes for recipe in possible_dc_recipes)
    except Exception as e:
        print(f"Failed to check corrections: {str(e)}")
        return False


def data_is_phase_corrected(data):
    """
    Check if run data has phase corrections applied.

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
        recipes = ds.recipes.keys()
        possible_pc_recipes = ["filtValue5LagDCPC"]
        return any(recipe in recipes for recipe in possible_pc_recipes)
    except Exception as e:
        print(f"Failed to check phase corrections: {str(e)}")
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
