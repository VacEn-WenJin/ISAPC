"""
Input/output utility functions for ISAPC
Enhanced with error propagation support
"""

import os
import numpy as np
import logging
from astropy.io import fits
from pathlib import Path
from typing import Optional, Dict, Union, List, Any

logger = logging.getLogger(__name__)


def save_results_to_npz(
    output_file: Union[str, Path], 
    data_dict: Dict[str, Any],
    include_errors: bool = True
) -> None:
    """
    Save results dictionary to NPZ file with enhanced handling of variable-length arrays
    and error data.

    Parameters
    ----------
    output_file : str or Path
        Path to save the NPZ file
    data_dict : dict
        Dictionary of results to save
    include_errors : bool, default=True
        Whether to include error arrays in the output
    """
    # Convert Path to string
    output_file = str(output_file)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Filter out objects that can't be saved
    filtered_dict = {}
    
    # Track error fields for logging
    error_fields_saved = []
    
    for key, value in data_dict.items():
        try:
            # Skip error fields if not requested
            if not include_errors and ('error' in key.lower() or key.endswith('_err')):
                continue
                
            # Track error fields
            if 'error' in key.lower() or key.endswith('_err'):
                error_fields_saved.append(key)
            
            # Convert pandas DataFrame to dictionary
            if hasattr(value, "to_dict"):
                filtered_dict[key] = value.to_dict()
            # Handle lists of arrays (like bin_indices) that can cause issues
            elif key == "bin_indices" and isinstance(value, list):
                # For bin_indices, store as object array to handle variable lengths
                indices_array = np.empty(len(value), dtype=object)
                for i, indices in enumerate(value):
                    indices_array[i] = indices
                filtered_dict[key] = indices_array
            # Process nested dictionaries (including error dictionaries)
            elif isinstance(value, dict):
                nested_dict = {}
                for nested_key, nested_value in value.items():
                    # Check for error fields in nested dicts
                    if not include_errors and ('error' in nested_key.lower() or nested_key.endswith('_err')):
                        continue
                    if 'error' in nested_key.lower() or nested_key.endswith('_err'):
                        error_fields_saved.append(f"{key}.{nested_key}")
                        
                    if hasattr(nested_value, "to_dict"):
                        nested_dict[nested_key] = nested_value.to_dict()
                    else:
                        nested_dict[nested_key] = nested_value
                filtered_dict[key] = nested_dict
            else:
                filtered_dict[key] = value
        except Exception as e:
            logger.warning(f"Could not save '{key}': {e}")

    # Log error fields being saved
    if error_fields_saved:
        logger.info(f"Saving {len(error_fields_saved)} error fields: {', '.join(error_fields_saved[:5])}...")

    try:
        np.savez_compressed(output_file, **filtered_dict)
        logger.info(f"Saved results to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results to {output_file}: {e}")

        # Try to save with pickle protocol for object arrays (fallback method)
        try:
            # Use allow_pickle=True explicitly
            np.savez(output_file, allow_pickle=True, **filtered_dict)
            logger.info(f"Saved results to {output_file} using pickle protocol")
        except Exception as e2:
            logger.error(f"Error saving results even with pickle: {e2}")


def load_results_from_npz(input_file: Union[str, Path], load_errors: bool = True) -> Dict[str, Any]:
    """
    Load results dictionary from NPZ file with error support.

    Parameters
    ----------
    input_file : str or Path
        Path to the NPZ file
    load_errors : bool, default=True
        Whether to load error arrays

    Returns
    -------
    dict
        Dictionary of results
    """
    # Convert Path to string
    input_file = str(input_file)

    try:
        data = np.load(input_file, allow_pickle=True)

        # Convert to regular dictionary
        result_dict = {}
        error_fields_loaded = []
        
        for key in data.files:
            # Skip error fields if not requested
            if not load_errors and ('error' in key.lower() or key.endswith('_err')):
                continue
                
            # Track error fields
            if 'error' in key.lower() or key.endswith('_err'):
                error_fields_loaded.append(key)
                
            result_dict[key] = (
                data[key].item() if data[key].dtype == np.dtype("O") else data[key]
            )

        # Log error fields loaded
        if error_fields_loaded:
            logger.info(f"Loaded {len(error_fields_loaded)} error fields from {input_file}")

        logger.info(f"Loaded results from {input_file}")
        return result_dict
    except Exception as e:
        logger.error(f"Error loading results from {input_file}: {e}")
        return {}


def save_standardized_results(galaxy_name, analysis_type, results, base_dir):
    """
    Save results in a standardized format and directory structure with error support.

    Parameters
    ----------
    galaxy_name : str
        Name of the galaxy
    analysis_type : str
        Analysis type ('P2P', 'VNB', or 'RDB')
    results : dict
        Results dictionary
    base_dir : Path or str
        Base output directory
    """
    base_dir = Path(base_dir)

    # Create standardized directory structure
    galaxy_dir = base_dir / galaxy_name
    data_dir = galaxy_dir / "Data"
    error_dir = galaxy_dir / "Errors"  # New directory for error data

    galaxy_dir.mkdir(exist_ok=True, parents=True)
    data_dir.mkdir(exist_ok=True)

    # Check if we have error data
    has_errors = any(
        'error' in str(k).lower() or str(k).endswith('_err') 
        for k in _get_all_keys(results)
    )
    
    if has_errors:
        error_dir.mkdir(exist_ok=True)
        logger.info(f"Error data detected for {galaxy_name} - will save error information")

    # Main results file (including errors)
    main_file = data_dir / f"{galaxy_name}_{analysis_type}_results.npz"
    save_results_to_npz(main_file, results, include_errors=True)
    
    # Also save a version without errors for compatibility
    main_file_no_errors = data_dir / f"{galaxy_name}_{analysis_type}_results_no_errors.npz"
    save_results_to_npz(main_file_no_errors, results, include_errors=False)

    # Save standardized results file
    std_file = data_dir / f"{galaxy_name}_{analysis_type}_standardized.npz"
    save_results_to_npz(std_file, results, include_errors=True)

    # Enhanced components with error fields
    components = {
        "stellar_kinematics": {
            "fields": ["velocity_field", "dispersion_field"],
            "errors": ["velocity_error", "dispersion_error"]
        },
        "emission": {
            "fields": ["velocity_field", "dispersion_field", "emission_flux"],
            "errors": ["velocity_error", "dispersion_error", "flux_error"]
        },
        "indices": {
            "fields": None,  # Save all indices
            "errors": None   # Save all error fields
        },
        "stellar_population": {
            "fields": ["log_age", "age", "metallicity"],
            "errors": ["log_age_error", "age_error", "metallicity_error"]
        },
    }

    # Save components with their errors
    for component, config in components.items():
        if component in results:
            component_data = {}
            fields = config["fields"]
            errors = config["errors"]

            if fields is None:
                # Save entire component
                component_data = results[component]
            else:
                # Extract specified fields
                for field in fields:
                    if field in results[component]:
                        component_data[field] = results[component][field]
                
                # Extract error fields if they exist
                if errors and f"{component}_errors" in results:
                    error_component = results[f"{component}_errors"]
                    for error_field in errors:
                        if error_field in error_component:
                            component_data[error_field] = error_component[error_field]
                elif errors:
                    # Check if errors are in the main component
                    for error_field in errors:
                        if error_field in results[component]:
                            component_data[error_field] = results[component][error_field]

            if component_data:
                component_file = (
                    data_dir / f"{galaxy_name}_{analysis_type}_{component}.npz"
                )
                save_results_to_npz(component_file, component_data)

    # Save error-only files if we have errors
    if has_errors:
        error_data = {}
        
        # Extract all error fields
        for key, value in results.items():
            if 'error' in key.lower() or key.endswith('_errors'):
                error_data[key] = value
            elif isinstance(value, dict):
                # Check nested dictionaries for error fields
                error_subdict = {}
                for subkey, subvalue in value.items():
                    if 'error' in subkey.lower() or subkey.endswith('_err'):
                        error_subdict[subkey] = subvalue
                if error_subdict:
                    error_data[f"{key}_errors"] = error_subdict
        
        if error_data:
            error_file = error_dir / f"{galaxy_name}_{analysis_type}_errors.npz"
            save_results_to_npz(error_file, error_data)
            logger.info(f"Saved error data separately to {error_file}")

    # Save binning info for VNB and RDB
    if analysis_type in ["VNB", "RDB"]:
        if "binning" in results:
            binning_file = data_dir / f"{galaxy_name}_{analysis_type}_binning.npz"
            save_results_to_npz(binning_file, results["binning"])
        elif "bin_info" in results:
            binning_file = data_dir / f"{galaxy_name}_{analysis_type}_binning.npz"
            save_results_to_npz(binning_file, results["bin_info"])

    logger.info(f"Saved standardized {analysis_type} results for {galaxy_name}")


def _get_all_keys(d, parent_key=''):
    """Helper function to get all keys including nested ones"""
    keys = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        keys.append(new_key)
        if isinstance(v, dict):
            keys.extend(_get_all_keys(v, new_key))
    return keys


def load_fits_data(input_file: Union[str, Path], hdu_index: int = 0) -> np.ndarray:
    """
    Load data from FITS file.

    Parameters
    ----------
    input_file : str or Path
        Path to the FITS file
    hdu_index : int, default=0
        HDU index to load

    Returns
    -------
    ndarray
        Data array
    """
    # Convert Path to string
    input_file = str(input_file)

    try:
        with fits.open(input_file) as hdul:
            data = hdul[hdu_index].data
        logger.info(f"Loaded FITS data from {input_file}, HDU {hdu_index}")
        return data
    except Exception as e:
        logger.error(f"Error loading FITS data from {input_file}: {e}")
        return None


def save_fits_data(
    output_file: Union[str, Path],
    data: np.ndarray,
    header: Optional[fits.Header] = None,
) -> None:
    """
    Save data to FITS file.

    Parameters
    ----------
    output_file : str or Path
        Path to save the FITS file
    data : ndarray
        Data array to save
    header : fits.Header, optional
        Header to include in the FITS file
    """
    # Convert Path to string
    output_file = str(output_file)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    try:
        hdu = fits.PrimaryHDU(data, header=header)
        hdu.writeto(output_file, overwrite=True)
        logger.info(f"Saved FITS data to {output_file}")
    except Exception as e:
        logger.error(f"Error saving FITS data to {output_file}: {e}")


def find_template(search_paths: Optional[List[str]] = None) -> Optional[str]:
    """
    Find a suitable stellar template file.

    Parameters
    ----------
    search_paths : list of str, optional
        Paths to search for template files

    Returns
    -------
    str or None
        Path to the template file, or None if not found
    """
    if search_paths is None:
        search_paths = [
            ".",
            "./templates",
            "../templates",
            os.path.expanduser("~/templates"),
            os.path.expanduser("~/.isapc/templates"),
        ]

    # Add ISAPC_TEMPLATES env var if it exists
    if "ISAPC_TEMPLATES" in os.environ:
        search_paths.insert(0, os.environ["ISAPC_TEMPLATES"])

    # File patterns to search for
    patterns = ["*.fits", "*.npz", "templates*.npz", "miles*.npz", "MILES*.npz"]

    # Search for template files
    for path in search_paths:
        if not os.path.exists(path):
            continue

        for pattern in patterns:
            files = list(Path(path).glob(pattern))
            if files:
                return str(files[0])

    return None


def save_error_summary(results, output_file: Union[str, Path]):
    """
    Save a summary of all error estimates in the results
    
    Parameters
    ----------
    results : dict
        Analysis results with errors
    output_file : str or Path
        Output file path for error summary
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    summary = {}
    
    # Extract error statistics
    for key, value in results.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if 'error' in subkey.lower() and isinstance(subvalue, np.ndarray):
                    stats = {
                        'mean': np.nanmean(subvalue),
                        'median': np.nanmedian(subvalue),
                        'std': np.nanstd(subvalue),
                        'min': np.nanmin(subvalue),
                        'max': np.nanmax(subvalue),
                        'percent_valid': np.sum(np.isfinite(subvalue)) / subvalue.size * 100
                    }
                    summary[f"{key}.{subkey}"] = stats
                    
    # Save summary
    with open(output_file, 'w') as f:
        f.write("Error Summary\n")
        f.write("=" * 60 + "\n")
        for field, stats in summary.items():
            f.write(f"\n{field}:\n")
            for stat, value in stats.items():
                f.write(f"  {stat}: {value:.4f}\n")
                
    logger.info(f"Saved error summary to {output_file}")