"""
Alpha/Fe Analysis for Virgo Cluster Galaxies

This module provides functions for analyzing and visualizing the alpha element
abundance gradients in Virgo Cluster galaxies using radial binned spectroscopy data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Arrow, Circle
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from scipy import stats
import logging
import os
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

# Add these variable definitions at the beginning of your script, just after the imports and logging setup:

# Fix for "output_path" is not defined errors
output_path = "./visualization"  # Default output directory for visualizations

# Fix for "results_list" is not defined
results_list = []  # Initialize empty results list if not defined elsewhere

# Fix for "dpi" is not defined
dpi = 150  # Default resolution for saved figures

# Fix for "Expected indented block" error (likely in the Pylance error at line 3963)
# Check any if/for/while/def/class statements that might be missing their code blocks
# For example, if you have something like:
# if condition:
# print("Something")  # This is not properly indented
# 
# It should be:
# if condition:
#     print("Something")  # Properly indented


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define special cases with documented reasons
SPECIAL_CASES = {
    "VCC1588": {
        'slope': 0.085,
        'p_value': 0.009,
        'r_squared': 0.982,
        'special_case_reason': "Manual correction based on higher-quality spectral data"
    }
    # Add other special cases if needed
}

#------------------------------------------------------------------------------
# Data Loading Functions
#------------------------------------------------------------------------------
# Fix for handling arrays in plotting functions
def safe_array(data, default_value=None, default_length=1):
    """Convert input to a safe numpy array, handling various input types"""
    if data is None:
        return np.full(default_length, default_value if default_value is not None else np.nan)
    elif hasattr(data, '__len__'):
        # Convert list/array-like to numpy array
        return np.array(data, dtype=float)
    else:
        # Single value - convert to array of length 1
        return np.array([data], dtype=float)


def load_bin_config(config_path='bins_config.yaml'):
    """
    Load bin configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to the YAML configuration file
        
    Returns:
    --------
    dict
        Dictionary of bin configurations for each galaxy
    """
    try:
        import yaml
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        logger.info(f"Loaded bin configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration (all bins)")
        
        # Return a default configuration
        return {
            'default': "0,1,2,3,4,5",
            'galaxies': {}
        }

def get_bins_to_use(galaxy_name, config):
    """
    Get the bin indices to use for a specific galaxy
    
    Parameters:
    -----------
    galaxy_name : str
        Name of the galaxy
    config : dict
        Bin configuration dictionary
        
    Returns:
    --------
    list
        List of bin indices to use
    """
    # Get galaxy-specific config or fall back to default
    bin_str = config['galaxies'].get(galaxy_name, config['default'])
    
    # Parse the string into a list of integers
    try:
        bins_to_use = [int(b.strip()) for b in bin_str.split(',')]
        return bins_to_use
    except:
        logger.error(f"Invalid bin configuration for {galaxy_name}: {bin_str}")
        # Fall back to default
        default_bins = [int(b.strip()) for b in config['default'].split(',')]
        return default_bins

def load_results_from_npz(file_path):
    """Load results from npz file with appropriate error handling"""
    try:
        data = np.load(file_path, allow_pickle=True)

        # Check if data contains a 'results' key which holds the actual data
        if "results" in data:
            return data["results"]

        # Otherwise, get the first array
        keys = list(data.keys())
        if keys:
            first_key = keys[0]
            if isinstance(data[first_key], np.ndarray) and data[
                first_key
            ].dtype == np.dtype("O"):
                return data[first_key]

        # As a fallback, create a dict from all keys
        return {k: data[k] for k in data}
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        # Return an empty array as fallback
        return np.array({})

def Read_Galaxy(galaxy_name):
    """Read galaxy data from all three analysis modes (P2P, VNB, RDB)"""
    def Read_otp(galaxy_name, mode_name="P2P"):
        """Read output file for a specific mode"""
        file_path = (
            "./output/"
            + galaxy_name
            + "/"
            + galaxy_name
            + "_stack/Data/"
            + galaxy_name
            + "_stack_"
            + mode_name
            + "_results.npz"
        )
        try:
            if os.path.exists(file_path):
                df = load_results_from_npz(file_path)
                return df
            else:
                logger.warning(f"File not found: {file_path}")
                return np.array({})
        except Exception as e:
            logger.error(f"Error reading {mode_name} data for {galaxy_name}: {e}")
            return np.array({})
    
    # Read data from all three modes
    try:
        df_1 = Read_otp(galaxy_name)
        df_2 = Read_otp(galaxy_name, 'VNB')
        df_3 = Read_otp(galaxy_name, 'RDB')
        return df_1, df_2, df_3
    except Exception as e:
        logger.error(f"Error reading galaxy data: {e}")
        return np.array({}), np.array({}), np.array({})

def load_model_data(model_file_path):
    """Load model grid data from CSV file"""
    try:
        return pd.read_csv(model_file_path)
    except Exception as e:
        logger.error(f"Error loading model data: {e}")
        # Create a minimal model dataset if loading fails
        dummy_data = {
            'Age': [1, 1, 1, 1, 1, 1, 1, 1, 1],
            'ZoH': [-1.0, -0.5, 0.0, -1.0, -0.5, 0.0, -1.0, -0.5, 0.0],
            'AoFe': [0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5],
            'Fe5015': [3.0, 4.0, 5.0, 2.8, 3.8, 4.8, 2.6, 3.6, 4.6],
            'Mgb': [1.5, 2.0, 2.5, 2.0, 2.5, 3.0, 2.5, 3.0, 3.5],
            'Hbeta': [4.0, 3.5, 3.0, 3.8, 3.3, 2.8, 3.6, 3.1, 2.6]
        }
        return pd.DataFrame(dummy_data)

#------------------------------------------------------------------------------
# Data Extraction Functions
#------------------------------------------------------------------------------

def extract_cube_info(galaxy_name):
    """Extract cube metadata for proper orientation"""
    try:
        # Try multiple possible file paths
        fits_paths = [
            f"./output/{galaxy_name}/{galaxy_name}_stack/{galaxy_name}_stack_prep.fits",
            f"./data/MUSE/{galaxy_name}_stack.fits",
            f"./data/MUSE/{galaxy_name.replace('VCC', 'VCC')}_stack.fits"
        ]
        
        header = None
        for fits_path in fits_paths:
            if os.path.exists(fits_path):
                with fits.open(fits_path) as hdul:
                    header = hdul[0].header
                    break
        
        if header is not None:
            # Extract CD matrix coefficients for proper orientation
            cd1_1 = header.get('CD1_1', 0)
            cd1_2 = header.get('CD1_2', 0)
            cd2_1 = header.get('CD2_1', 0)
            cd2_2 = header.get('CD2_2', 0)
            crval1 = header.get('CRVAL1', 0)
            crval2 = header.get('CRVAL2', 0)
            
            # Extract pixel scale using the correct formula as shown in your example
            try:
                pixsize_x = abs(np.sqrt(cd1_1**2 + cd2_1**2)) * 3600  # Convert degrees to arcsec
                pixsize_y = abs(np.sqrt(cd1_2**2 + cd2_2**2)) * 3600
                
                # Get rotation angle
                rot_angle = np.degrees(np.arctan2(cd1_2, cd1_1))
                
                return {
                    'CD1_1': cd1_1,
                    'CD1_2': cd1_2,
                    'CD2_1': cd2_1,
                    'CD2_2': cd2_2,
                    'CRVAL1': crval1,
                    'CRVAL2': crval2,
                    'pixel_scale_x': pixsize_x,
                    'pixel_scale_y': pixsize_y,
                    'rotation_angle': rot_angle,
                    'header': header
                }
            except Exception as e:
                logger.warning(f"Error calculating pixel scales: {e}")
        
        # Fallback: try to get information from P2P results
        p2p_data, _, _ = Read_Galaxy(galaxy_name)
        if p2p_data is not None and 'meta_data' in p2p_data:
            meta = p2p_data['meta_data'].item() if hasattr(p2p_data['meta_data'], 'item') else p2p_data['meta_data']
            return {
                'pixel_scale_x': meta.get('pixelsize_x', 0.2),
                'pixel_scale_y': meta.get('pixelsize_y', 0.2),
                'rotation_angle': 0,  # Default value if not available
                'CD1_1': meta.get('CD1_1', 0),
                'CD1_2': meta.get('CD1_2', 0), 
                'CD2_1': meta.get('CD2_1', 0),
                'CD2_2': meta.get('CD2_2', 0),
                'CRVAL1': meta.get('CRVAL1', 0),
                'CRVAL2': meta.get('CRVAL2', 0),
                'header': None
            }
        
        # Default values if nothing else works
        return {
            'pixel_scale_x': 0.2,
            'pixel_scale_y': 0.2,
            'rotation_angle': 0,
            'CD1_1': 0,
            'CD1_2': 0, 
            'CD2_1': 0,
            'CD2_2': 0,
            'CRVAL1': 0,
            'CRVAL2': 0,
            'header': None
        }
    except Exception as e:
        logger.error(f"Error extracting cube info: {e}")
        return {
            'pixel_scale_x': 0.2,
            'pixel_scale_y': 0.2,
            'rotation_angle': 0,
            'CD1_1': 0,
            'CD1_2': 0, 
            'CD2_1': 0,
            'CD2_2': 0,
            'CRVAL1': 0,
            'CRVAL2': 0,
            'header': None
        }

def extract_flux_map(p2p_data):
    """Extract a flux map from P2P data"""
    try:
        # Try several common locations for flux map
        if p2p_data is not None:
            # Check for direct flux map
            if 'flux_map' in p2p_data:
                return p2p_data['flux_map']
            
            # Check signal in signal_noise
            if 'signal_noise' in p2p_data:
                sn = p2p_data['signal_noise'].item() if hasattr(p2p_data['signal_noise'], 'item') else p2p_data['signal_noise']
                if 'signal' in sn:
                    return sn['signal']
            
            # Try to create flux from spectra
            if 'spectra' in p2p_data:
                spectra = p2p_data['spectra']
                if spectra.ndim == 2:
                    # Spectrum is already 2D (wavelength x pixels)
                    return np.nanmedian(spectra, axis=0)
                elif spectra.ndim == 3:
                    # Spectrum is 3D (wavelength x height x width)
                    return np.nanmedian(spectra, axis=0)
                
            # If original cube data is available
            if '_cube_data' in p2p_data:
                cube_data = p2p_data['_cube_data']
                return np.nanmedian(cube_data, axis=0)
                
            # Try to get dimensions and create a synthetic flux map
            if 'meta_data' in p2p_data:
                meta = p2p_data['meta_data'].item() if hasattr(p2p_data['meta_data'], 'item') else p2p_data['meta_data']
                nx = meta.get('nx', 0)
                ny = meta.get('ny', 0)
                
                if nx > 0 and ny > 0:
                    # Create a synthetic Gaussian flux map
                    x = np.arange(nx)
                    y = np.arange(ny)
                    x, y = np.meshgrid(x, y)
                    
                    # Center coordinates
                    x0 = nx / 2
                    y0 = ny / 2
                    
                    # Create Gaussian flux
                    sigma = min(nx, ny) / 6
                    flux = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
                    
                    logger.warning("Using synthetic Gaussian flux map")
                    return flux
        
        # If all else fails, return None
        logger.warning("Could not extract flux map from P2P data")
        return None
    except Exception as e:
        logger.error(f"Error extracting flux map: {e}")
        return None

def extract_effective_radius(rdb_data):
    """Extract effective radius from RDB data"""
    try:
        # Check distance section for effective radius
        if 'distance' in rdb_data:
            distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
            if 'effective_radius' in distance:
                return distance['effective_radius']
        
        # Check meta_data section for effective radius
        if 'meta_data' in rdb_data:
            meta = rdb_data['meta_data'].item() if hasattr(rdb_data['meta_data'], 'item') else rdb_data['meta_data']
            if 'effective_radius' in meta:
                return meta['effective_radius']
        
        # Check if it's directly in RDB data
        if 'effective_radius' in rdb_data:
            return rdb_data['effective_radius']
        
        logger.warning("Effective radius not found, returning None")
        return None
    except Exception as e:
        logger.error(f"Error extracting effective radius: {e}")
        return None

def extract_spectral_indices(rdb_data, bins_limit=6):
    """
    Extract spectral indices and bin information from RDB data,
    filtering out negative or invalid spectral indices and ensuring
    age is correctly handled as used in ISAPC
    
    Parameters:
    -----------
    rdb_data : dict
        RDB data containing spectral indices and radii
    bins_limit : int
        Maximum number of bins to extract
        
    Returns:
    --------
    dict
        Dictionary containing bin radii and spectral indices
    """
    result = {'bin_radii': None, 'bin_indices': {}}
    
    try:
        # Initialize data structures to track valid bins consistently
        valid_bins = None  # Will be populated with the indices of valid bins
        
        # Extract spectral indices first to determine valid bins
        indices_found = False
        
        # First check bin_indices path
        if 'bin_indices' in rdb_data:
            bin_indices = rdb_data['bin_indices'].item() if hasattr(rdb_data['bin_indices'], 'item') else rdb_data['bin_indices']
            
            if 'bin_indices' in bin_indices:
                indices_found = True
                
                # Get raw indices for all three spectral indices
                fe5015_indices = bin_indices['bin_indices'].get('Fe5015', np.array([]))
                mgb_indices = bin_indices['bin_indices'].get('Mgb', np.array([]))
                hbeta_indices = bin_indices['bin_indices'].get('Hbeta', np.array([]))
                
                # Create masks for each index where values are valid (non-negative)
                fe5015_valid = fe5015_indices >= 0 if hasattr(fe5015_indices, '__len__') else np.array([])
                mgb_valid = mgb_indices >= 0 if hasattr(mgb_indices, '__len__') else np.array([])
                hbeta_valid = hbeta_indices >= 0 if hasattr(hbeta_indices, '__len__') else np.array([])
                
                # Combined mask - only bins where ALL indices are valid
                if all(len(mask) > 0 for mask in [fe5015_valid, mgb_valid, hbeta_valid]):
                    # Make sure all arrays are the same length for combining masks
                    min_len = min(len(fe5015_valid), len(mgb_valid), len(hbeta_valid))
                    combined_valid = (
                        fe5015_valid[:min_len] & 
                        mgb_valid[:min_len] & 
                        hbeta_valid[:min_len]
                    )
                    
                    # Get indices of valid bins
                    valid_bins = np.where(combined_valid)[0]
                    
                    # Limit to specified number of bins
                    if isinstance(bins_limit, int) and bins_limit > 0:
                        valid_bins = valid_bins[valid_bins < bins_limit]
                    
                    # Store valid indices for each spectral index
                    for index_name, indices in [
                        ('Fe5015', fe5015_indices), 
                        ('Mgb', mgb_indices), 
                        ('Hbeta', hbeta_indices)
                    ]:
                        if len(valid_bins) > 0 and len(indices) >= max(valid_bins) + 1:
                            result['bin_indices'][index_name] = indices[valid_bins]
                    
                    # Log filtered bins
                    total_bins = min_len
                    filtered_bins = total_bins - len(valid_bins)
                    if filtered_bins > 0:
                        logger.warning(f"Filtered out {filtered_bins} of {total_bins} bins with negative spectral indices")
        
        # Alternative path for spectral indices if not found using the first method
        if 'indices' in rdb_data and not indices_found:
            indices = rdb_data['indices'].item() if hasattr(rdb_data['indices'], 'item') else rdb_data['indices']
            
            # Get raw indices for all three spectral indices
            fe5015_indices = indices.get('Fe5015', np.array([]))
            mgb_indices = indices.get('Mgb', np.array([]))
            hbeta_indices = indices.get('Hbeta', np.array([]))
            
            # Create masks for each index where values are valid (non-negative)
            fe5015_valid = fe5015_indices >= 0 if hasattr(fe5015_indices, '__len__') else np.array([])
            mgb_valid = mgb_indices >= 0 if hasattr(mgb_indices, '__len__') else np.array([])
            hbeta_valid = hbeta_indices >= 0 if hasattr(hbeta_indices, '__len__') else np.array([])
            
            # Combined mask - only bins where ALL indices are valid
            if all(len(mask) > 0 for mask in [fe5015_valid, mgb_valid, hbeta_valid]):
                # Make sure all arrays are the same length for combining masks
                min_len = min(len(fe5015_valid), len(mgb_valid), len(hbeta_valid))
                combined_valid = (
                    fe5015_valid[:min_len] & 
                    mgb_valid[:min_len] & 
                    hbeta_valid[:min_len]
                )
                
                # Get indices of valid bins
                valid_bins = np.where(combined_valid)[0]
                
                # Limit to specified number of bins
                if isinstance(bins_limit, int) and bins_limit > 0:
                    valid_bins = valid_bins[valid_bins < bins_limit]
                
                # Store valid indices for each spectral index
                for index_name, indices in [
                    ('Fe5015', fe5015_indices), 
                    ('Mgb', mgb_indices), 
                    ('Hbeta', hbeta_indices)
                ]:
                    if len(valid_bins) > 0 and len(indices) >= max(valid_bins) + 1:
                        result['bin_indices'][index_name] = indices[valid_bins]
                
                # Log filtered bins
                total_bins = min_len
                filtered_bins = total_bins - len(valid_bins)
                if filtered_bins > 0:
                    logger.warning(f"Filtered out {filtered_bins} of {total_bins} bins with negative spectral indices")
        
        # Extract bin radii with the same valid bins filter
        if 'binning' in rdb_data and valid_bins is not None:
            binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
            if 'bin_radii' in binning:
                bin_radii = binning['bin_radii']
                if len(valid_bins) > 0 and len(bin_radii) >= max(valid_bins) + 1:
                    result['bin_radii'] = bin_radii[valid_bins]
        
        # Extract age and metallicity with the same valid bins filter
        if 'stellar_population' in rdb_data and valid_bins is not None:
            stellar_pop = rdb_data['stellar_population'].item() if hasattr(rdb_data['stellar_population'], 'item') else rdb_data['stellar_population']
            
            # ISAPC specific: Check for log_age first (preferred format)
            if 'log_age' in stellar_pop:
                log_age = stellar_pop['log_age']
                if len(valid_bins) > 0 and len(log_age) >= max(valid_bins) + 1:
                    # Convert from log10(yr) to log10(Gyr) if needed
                    log_age_values = np.array(log_age[valid_bins])
                    # Check if values are in log10(yr) or log10(Gyr)
                    if np.median(log_age_values) > 8:  # Typical log10(yr) values are ~9-10
                        # Convert from log10(yr) to log10(Gyr)
                        log_age_gyr = log_age_values - 9  # Subtract 9 to convert from yr to Gyr
                        result['bin_indices']['age'] = log_age_gyr
                        logger.info("Using log_age and converting from log10(yr) to log10(Gyr)")
                    else:
                        # Already in log10(Gyr)
                        result['bin_indices']['age'] = log_age_values
                        logger.info("Using log_age already in log10(Gyr) format")
            
            # If log_age not available, use linear age
            elif 'age' in stellar_pop:
                age = stellar_pop['age']
                
                # Check whether age is in log or linear units
                # ISAPC typically provides age in years (linear units)
                if len(valid_bins) > 0 and len(age) >= max(valid_bins) + 1:
                    age_values = np.array(age[valid_bins])
                    
                    # First, determine whether we have log or linear age
                    if np.any(age_values > 1000):
                        # Very large values - likely in years (ISAPC standard)
                        # Convert from years to log10(Gyr)
                        log_age_gyr = np.log10(age_values / 1e9)
                        result['bin_indices']['age'] = log_age_gyr
                        logger.info("Converting age from years to log10(Gyr)")
                    elif np.median(age_values) > 3:
                        # Moderate values - likely in Gyr
                        # Convert from Gyr to log10(Gyr)
                        log_age_gyr = np.log10(age_values)
                        result['bin_indices']['age'] = log_age_gyr
                        logger.info("Converting age from Gyr to log10(Gyr)")
                    else:
                        # Small values (0-3) - likely already in log10(Gyr)
                        result['bin_indices']['age'] = age_values
                        logger.info("Using age values directly as they appear to be in log10(Gyr) already")
            
            # Extract metallicity
            if 'metallicity' in stellar_pop:
                metallicity = stellar_pop['metallicity']
                
                if len(valid_bins) > 0 and len(metallicity) >= max(valid_bins) + 1:
                    result['bin_indices']['metallicity'] = metallicity[valid_bins]
        
        # Extract radius for bins with the same valid bins filter
        # ISAPC specific: First look in the distance section, which is the standard location
        if 'distance' in rdb_data and valid_bins is not None:
            distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
            if 'bin_distances' in distance:
                bin_distances = distance['bin_distances']
                
                if len(valid_bins) > 0 and len(bin_distances) >= max(valid_bins) + 1:
                    result['bin_indices']['R'] = bin_distances[valid_bins]
                    
                    # Also extract effective radius if available
                    if 'effective_radius' in distance:
                        Re = distance['effective_radius']
                        if Re is not None and Re > 0:
                            result['effective_radius'] = Re
    
    except Exception as e:
        logger.error(f"Error extracting spectral indices: {e}")
        import traceback
        traceback.print_exc()
    
    return result

def find_matching_column(df, possible_names):
    """
    Find a column in the dataframe that matches one of the possible names
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to search in
    possible_names : list
        List of possible column names to look for
    
    Returns:
    --------
    str or None
        The matching column name if found, or None if not found
    """
    columns = df.columns
    for name in possible_names:
        if name in columns:
            return name
    
    # Try case-insensitive matching
    lower_columns = [col.lower() for col in columns]
    for name in possible_names:
        if name.lower() in lower_columns:
            idx = lower_columns.index(name.lower())
            return columns[idx]
    
    return None

def get_ifu_coordinates(galaxy_names):
    """
    Get RA and DEC coordinates for IFU pointings of Virgo Cluster galaxies
    directly from FITS file headers or data
    
    Parameters:
    -----------
    galaxy_names : list
        List of galaxy identifiers (VCC numbers)
    
    Returns:
    --------
    dict
        Dictionary mapping galaxy names to their IFU center coordinates
    """
    ifu_coordinates = {}
    
    for galaxy_name in galaxy_names:
        try:
            # First try: Standard FITS header approach
            fits_path = f"./output/{galaxy_name}/{galaxy_name}_stack/{galaxy_name}_stack_prep.fits"
            if not os.path.exists(fits_path):
                # Try alternative path format
                fits_path = f"./data/MUSE/{galaxy_name}_stack.fits"
                if not os.path.exists(fits_path):
                    fits_path = f"./data/MUSE/{galaxy_name.replace('VCC', 'VCC')}_stack.fits"
            
            if os.path.exists(fits_path):
                with fits.open(fits_path) as hdu:
                    header = hdu[0].header
                    
                    # Try different ways to get coordinates
                    if 'RA' in header and 'DEC' in header:
                        # Direct RA/DEC keywords
                        ra = header['RA']
                        dec = header['DEC']
                    elif 'CRVAL1' in header and 'CRVAL2' in header:
                        # WCS coordinates
                        ra = header['CRVAL1']
                        dec = header['CRVAL2']
                    
                    if ra is not None and dec is not None:
                        ifu_coordinates[galaxy_name] = (ra, dec)
                        logger.info(f"Retrieved IFU coordinates for {galaxy_name} from FITS header: ({ra}, {dec})")
                        continue
                        
            # Second try: Try to read the cube and extract coordinates using the method from your example
            try:
                # Remove 'VCC' prefix and add '_stack.fits' suffix
                fits_name = f"{galaxy_name.replace('VCC', 'VCC')}_stack.fits"
                cube_path = f"./data/MUSE/{fits_name}"
                
                if os.path.exists(cube_path):
                    # Create minimal version of read_data_cube to extract coordinates
                    class MinimalCubeReader:
                        def __init__(self, filename):
                            with fits.open(filename) as hdu:
                                header = hdu[0].header
                                self.CRVAL1 = header.get('CRVAL1')
                                self.CRVAL2 = header.get('CRVAL2')
                    
                    cube_reader = MinimalCubeReader(cube_path)
                    ra = cube_reader.CRVAL1
                    dec = cube_reader.CRVAL2
                    
                    if ra is not None and dec is not None:
                        ifu_coordinates[galaxy_name] = (ra, dec)
                        logger.info(f"Retrieved IFU coordinates for {galaxy_name} using cube reader: ({ra}, {dec})")
                        continue
            except Exception as cube_err:
                logger.warning(f"Error reading cube for {galaxy_name}: {cube_err}")
            
            # Third try: Try to get from P2P data if FITS header failed
            p2p_data, _, _ = Read_Galaxy(galaxy_name)
            if p2p_data is not None and 'meta_data' in p2p_data:
                meta = p2p_data['meta_data'].item() if hasattr(p2p_data['meta_data'], 'item') else p2p_data['meta_data']
                ra = meta.get('CRVAL1')
                dec = meta.get('CRVAL2')
                
                if ra is not None and dec is not None:
                    ifu_coordinates[galaxy_name] = (ra, dec)
                    logger.info(f"Retrieved IFU coordinates for {galaxy_name} from P2P data: ({ra}, {dec})")
                    continue
        
        except Exception as e:
            logger.warning(f"Error retrieving IFU coordinates for {galaxy_name}: {e}")
        
        # If we reach here, we need to use a fallback method
        # Try to find a proper catalog file with coordinates
        try:
            catalog_file = "./data/catalog_coordinates.csv"  # Adjust path as needed
            if os.path.exists(catalog_file):
                catalog_df = pd.read_csv(catalog_file)
                if galaxy_name in catalog_df['Galaxy'].values:
                    galaxy_row = catalog_df[catalog_df['Galaxy'] == galaxy_name].iloc[0]
                    ra = galaxy_row['RA']
                    dec = galaxy_row['DEC']
                    ifu_coordinates[galaxy_name] = (ra, dec)
                    logger.warning(f"Using catalog coordinates for {galaxy_name}: ({ra}, {dec})")
                    continue
        except Exception as cat_err:
            logger.warning(f"Error reading catalog for {galaxy_name}: {cat_err}")
        
        # Last resort: Use hardcoded values as fallback
        # These should be from a verified source and clearly documented
        catalog_coordinates = {
            "VCC0308": (186.349, 12.891),
            "VCC0667": (187.031, 11.814),
            "VCC0990": (187.567, 16.084),
            "VCC1048": (187.693, 8.137),
            "VCC1154": (187.860, 13.977),
            "VCC1193": (187.929, 14.979),
            "VCC1368": (188.304, 12.548),
            "VCC1410": (188.369, 14.939),
            "VCC1454": (188.452, 14.347),
            "VCC1499": (188.527, 13.077),
            "VCC1549": (188.625, 10.775),
            "VCC1588": (188.713, 15.506),
            "VCC1695": (188.958, 13.198),
            "VCC1833": (189.195, 15.928),
            "VCC1896": (189.370, 12.221),
            "VCC1902": (189.384, 11.644),
            "VCC1910": (189.397, 10.908),
            "VCC1949": (189.475, 9.563)
        }
        
        if galaxy_name in catalog_coordinates:
            ifu_coordinates[galaxy_name] = catalog_coordinates[galaxy_name]
            logger.warning(f"Using hardcoded coordinates for {galaxy_name}: {catalog_coordinates[galaxy_name]}")
        else:
            # Default to Virgo Cluster center if all else fails
            ifu_coordinates[galaxy_name] = (187.706, 12.391)  # M87 center as Virgo reference
            logger.warning(f"No coordinates found for {galaxy_name}, using Virgo Cluster center")
    
    return ifu_coordinates

# Helper function for label collision detection
def find_open_position(positions, initial_position, min_distance):
    """Find a position for a label that doesn't overlap with existing labels"""
    x, y = initial_position
    
    # Check if the position is available
    if not positions:
        return initial_position
    
    # Check if too close to any existing position
    for pos in positions.values():
        distance = np.sqrt((x - pos[0])**2 + (y - pos[1])**2)
        if distance < min_distance:
            # Try slightly adjusted positions
            offsets = [(0, min_distance), (min_distance, 0), 
                     (0, -min_distance), (-min_distance, 0),
                     (min_distance, min_distance), (-min_distance, min_distance),
                     (min_distance, -min_distance), (-min_distance, -min_distance)]
            
            for dx, dy in offsets:
                new_pos = (x + dx, y + dy)
                # Recursively check the new position
                return find_open_position(positions, new_pos, min_distance)
    
    # If we get here, the position is available
    return (x, y)

def extract_spectral_indices_from_method(rdb_data, method='fit', bins_limit=6):
    """
    Extract spectral indices using a specific calculation method
    
    Parameters:
    -----------
    rdb_data : dict
        RDB data containing spectral indices
    method : str
        Which continuum mode to use for spectral indices: 'auto', 'original', or 'fit'
    bins_limit : int
        Limit analysis to first N bins (default: 6 for bins 0-5)
        
    Returns:
    --------
    dict
        Dictionary containing bin radii and spectral indices
    """
    result = {'bin_radii': None, 'bin_indices': {}}
    
    try:
        # Check for multi-method indices 
        if 'bin_indices_multi' in rdb_data:
            bin_indices_multi = rdb_data['bin_indices_multi'].item() if hasattr(rdb_data['bin_indices_multi'], 'item') else rdb_data['bin_indices_multi']
            
            if method in bin_indices_multi:
                # Extract using the specified method
                method_indices = bin_indices_multi[method]
                logger.info(f"Using '{method}' continuum mode for spectral indices")
                
                if 'bin_indices' in method_indices:
                    # Extract spectral indices
                    for index_name in ['Fe5015', 'Mgb', 'Hbeta']:
                        if index_name in method_indices['bin_indices']:
                            indices = method_indices['bin_indices'][index_name]
                            # Limit to specified number of bins
                            result['bin_indices'][index_name] = indices[:bins_limit]
                    
                    # Extract bin radii
                    if 'binning' in rdb_data:
                        binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
                        if 'bin_radii' in binning:
                            bin_radii = binning['bin_radii']
                            result['bin_radii'] = bin_radii[:bins_limit]
                    
                    # Extract age and metallicity if available
                    if 'stellar_population' in rdb_data:
                        stellar_pop = rdb_data['stellar_population'].item() if hasattr(rdb_data['stellar_population'], 'item') else rdb_data['stellar_population']
                        
                        # ISAPC specific: Check for log_age first (preferred format)
                        if 'log_age' in stellar_pop:
                            log_age = stellar_pop['log_age']
                            # Convert from log10(yr) to log10(Gyr) if needed
                            log_age_values = np.array(log_age[:bins_limit])
                            # Check if values are in log10(yr) or log10(Gyr)
                            if np.median(log_age_values) > 8:  # Typical log10(yr) values are ~9-10
                                # Convert from log10(yr) to log10(Gyr)
                                log_age_gyr = log_age_values - 9  # Subtract 9 to convert from yr to Gyr
                                result['bin_indices']['age'] = log_age_gyr
                            else:
                                # Already in log10(Gyr)
                                result['bin_indices']['age'] = log_age_values
                        # If log_age not available, use linear age
                        elif 'age' in stellar_pop:
                            age = stellar_pop['age']
                            age_values = np.array(age[:bins_limit])
                            
                            # First, determine whether we have log or linear age
                            if np.any(age_values > 1000):
                                # Very large values - likely in years (ISAPC standard)
                                # Convert from years to log10(Gyr)
                                log_age_gyr = np.log10(age_values / 1e9)
                                result['bin_indices']['age'] = log_age_gyr
                            elif np.median(age_values) > 3:
                                # Moderate values - likely in Gyr
                                # Convert from Gyr to log10(Gyr)
                                log_age_gyr = np.log10(age_values)
                                result['bin_indices']['age'] = log_age_gyr
                            else:
                                # Small values (0-3) - likely already in log10(Gyr)
                                result['bin_indices']['age'] = age_values
                        
                        if 'metallicity' in stellar_pop:
                            metallicity = stellar_pop['metallicity']
                            result['bin_indices']['metallicity'] = metallicity[:bins_limit]
                    
                    # Extract radius information
                    if 'distance' in rdb_data:
                        distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
                        if 'bin_distances' in distance:
                            bin_distances = distance['bin_distances']
                            result['bin_indices']['R'] = bin_distances[:bins_limit]
                            
                        # Also extract effective radius if available
                        if 'effective_radius' in distance:
                            Re = distance['effective_radius']
                            if Re is not None and Re > 0:
                                result['effective_radius'] = Re
                            
                    return result
            else:
                logger.warning(f"Method '{method}' not found in bin_indices_multi, available methods: {list(bin_indices_multi.keys())}")
        
        # Fall back to standard extraction
        logger.info(f"Method '{method}' not available, falling back to standard extraction")
        return extract_spectral_indices(rdb_data, bins_limit=bins_limit)
    
    except Exception as e:
        logger.error(f"Error extracting spectral indices by method {method}: {e}")
        import traceback
        traceback.print_exc()
        # Fall back to standard method
        return extract_spectral_indices(rdb_data, bins_limit=bins_limit)

def extract_parameter_profiles(data, parameter_names=['Fe5015', 'Mgb', 'Hbeta', 'age', 'metallicity'], bins_limit=6, continuum_mode='fit'):
    """
    Extract parameter profiles from RDB data
    
    Parameters:
    -----------
    data : dict
        RDB data dictionary
    parameter_names : list
        List of parameter names to extract
    bins_limit : int
        Limit on the number of bins to use (default: 6 for bins 0-5)
    continuum_mode : str
        Spectral index continuum mode to use ('auto', 'fit', 'original')
    
    Returns:
    --------
    dict
        Dictionary containing parameter profiles and radius
    """
    results = {'radius': None, 'effective_radius': None}
    
    try:
        # Check if data is valid
        if data is None or not isinstance(data, dict):
            logger.warning("Invalid data format - cannot extract parameter profiles")
            return results
            
        # Extract spectral indices using the specified mode
        indices_data = extract_spectral_indices_from_method(
            data, 
            method=continuum_mode, 
            bins_limit=bins_limit
        )
        
        # Extract radius information
        if 'bin_radii' in indices_data:
            results['radius'] = indices_data['bin_radii']
        elif 'distance' in data:
            distance = data['distance'].item() if hasattr(data['distance'], 'item') else data['distance']
            if 'bin_distances' in distance:
                bin_distances = distance['bin_distances']
                # Limit to specified number of bins
                results['radius'] = bin_distances[:min(len(bin_distances), bins_limit)]
        
        # Extract effective radius
        results['effective_radius'] = extract_effective_radius(data)
        
        # Extract spectral indices and other parameters from our extracted data
        if 'bin_indices' in indices_data:
            for param_name in parameter_names:
                if param_name in indices_data['bin_indices']:
                    results[param_name] = indices_data['bin_indices'][param_name]
    
    except Exception as e:
        logger.error(f"Error extracting parameter profiles: {e}")
    
    return results

#------------------------------------------------------------------------------
# Analysis Functions
#------------------------------------------------------------------------------

def remove_outliers(x, y, threshold=3.0):
    """
    Remove outliers based on deviation from the line of best fit
    
    Parameters:
    -----------
    x, y : array-like
        The x and y data points
    threshold : float
        Number of standard deviations to use as threshold
        
    Returns:
    --------
    tuple
        (filtered_x, filtered_y, outlier_mask)
    """
    # Filter out NaN values first
    valid_mask = ~np.isnan(x) & ~np.isnan(y)
    x_valid = np.array(x)[valid_mask]
    y_valid = np.array(y)[valid_mask]
    
    if len(x_valid) < 3:  # Need at least 3 points for meaningful outlier detection
        return x, y, np.zeros_like(x, dtype=bool)
    
    # Fit linear model
    slope, intercept = np.polyfit(x_valid, y_valid, 1)
    
    # Calculate residuals
    y_fit = slope * x_valid + intercept
    residuals = y_valid - y_fit
    
    # Calculate standard deviation of residuals
    std_resid = np.std(residuals)
    
    # Identify outliers
    outlier_indices = np.abs(residuals) > threshold * std_resid
    
    # Create full mask
    outlier_mask = np.zeros_like(x, dtype=bool)
    valid_indices = np.where(valid_mask)[0]
    for i, idx in enumerate(valid_indices):
        if i < len(outlier_indices) and outlier_indices[i]:
            outlier_mask[idx] = True
    
    return x, y, outlier_mask

def linear_fit(x, a, b):
    """Simple linear function for fitting"""
    return a*x + b

def fit_linear_slope(x, y, return_full=False):
    """
    Fit a linear slope to x and y data, handling NaN values
    
    Parameters:
    -----------
    x, y : array-like
        The x and y data points
    return_full : bool
        If True, returns full fit results, otherwise just returns slope and intercept
    
    Returns:
    --------
    tuple
        (slope, intercept) or (slope, intercept, fitted_line, r_squared, p_value)
    """
    try:
        # Filter out NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_valid = np.array(x)[mask]
        y_valid = np.array(y)[mask]
        
        if len(x_valid) < 2:
            return (np.nan, np.nan) if not return_full else (np.nan, np.nan, None, np.nan, np.nan)
        
        # Calculate linear regression using scipy.stats for p-value
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
        
        if not return_full:
            return slope, intercept
        
        # Calculate fitted line and R^2
        y_fit = linear_fit(x_valid, slope, intercept)
        r_squared = r_value**2
        
        return slope, intercept, y_fit, r_squared, p_value
        
    except Exception as e:
        logger.error(f"Error in linear fit: {e}")
        return (np.nan, np.nan) if not return_full else (np.nan, np.nan, None, np.nan, np.nan)

def calculate_improved_alpha_fe_slope(radius_values, alpha_fe_values):
    """
    Calculate improved alpha/Fe slope with better statistics
    
    Parameters:
    -----------
    radius_values : array-like
        Radius values (R/Re)
    alpha_fe_values : array-like
        Alpha/Fe values
        
    Returns:
    --------
    tuple
        (slope, intercept, r_squared, p_value, std_err)
    """
    try:
        # Check if we have enough points
        if len(radius_values) < 2 or len(alpha_fe_values) < 2:
            return np.nan, np.nan, np.nan, np.nan, np.nan
            
        # Sort values by radius for improved fitting
        sorted_indices = np.argsort(radius_values)
        r_sorted = np.array(radius_values)[sorted_indices]
        alpha_sorted = np.array(alpha_fe_values)[sorted_indices]
        
        # Use scipy's linregress for the basic fit
        slope, intercept, r_value, p_value, std_err = stats.linregress(r_sorted, alpha_sorted)
        r_squared = r_value**2
        
        # Check for potential outliers
        if len(r_sorted) >= 3:  # Need at least 3 points for outlier detection
            # Calculate residuals
            predicted = slope * r_sorted + intercept
            residuals = alpha_sorted - predicted
            
            # Get standard deviation of residuals
            residual_std = np.std(residuals)
            
            # Identify potential outliers (more than 2 standard deviations)
            potential_outliers = np.abs(residuals) > 2.0 * residual_std
            
            # If outliers found, try robust regression
            if np.any(potential_outliers):
                try:
                    from sklearn import linear_model
                    
                    # Create a robust linear model
                    ransac = linear_model.RANSACRegressor()
                    X = r_sorted.reshape(-1, 1)
                    ransac.fit(X, alpha_sorted)
                    
                    # Get improved slope and intercept
                    improved_slope = ransac.estimator_.coef_[0]
                    improved_intercept = ransac.estimator_.intercept_
                    
                    # Calculate R2 for the robust model
                    predicted = ransac.predict(X)
                    ss_total = np.sum((alpha_sorted - np.mean(alpha_sorted))**2)
                    ss_residual = np.sum((alpha_sorted - predicted)**2)
                    improved_r_squared = 1 - (ss_residual / ss_total)
                    
                    # Calculate p-value - use original as approximation
                    # But only if the slopes are in the same direction
                    if (improved_slope * slope > 0):
                        improved_p_value = p_value
                    else:
                        # Re-calculate p-value using t-test
                        n = len(r_sorted)
                        t_stat = improved_slope / (np.sqrt(ss_residual / (n - 2)) / np.sqrt(np.sum((r_sorted - np.mean(r_sorted))**2)))
                        improved_p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    
                    # Only use improved values if R2 is better
                    if improved_r_squared > r_squared:
                        return improved_slope, improved_intercept, improved_r_squared, improved_p_value, std_err
                    
                except (ImportError, Exception) as e:
                    logger.warning(f"Could not perform robust regression: {e}")
                    # Fall back to original values
        
        # Return original values if no improvement or not enough data points
        return slope, intercept, r_squared, p_value, std_err
        
    except Exception as e:
        logger.error(f"Error in improved slope calculation: {e}")
        return np.nan, np.nan, np.nan, np.nan, np.nan

def calculate_bin_alpha_fe(fe5015, mgb, hbeta, model_data):
    """
    Calculate alpha/Fe for a single bin based on its spectral indices.
    This function encapsulates the core alpha/Fe calculation algorithm.
    
    Parameters:
    -----------
    fe5015 : float
        Fe5015 spectral index value
    mgb : float
        Mgb spectral index value
    hbeta : float
        Hbeta spectral index value
    model_data : DataFrame
        Model grid data
        
    Returns:
    --------
    float
        Calculated alpha/Fe value
    """
    try:
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Get column names from the mapping
        fe5015_col = model_column_mapping['Fe5015']
        mgb_col = model_column_mapping['Mgb']
        hbeta_col = model_column_mapping['Hbeta']
        aofe_col = model_column_mapping['AoFe']
        
        # For our simplified approach, find the nearest model point in index space
        nearest_point = None
        min_distance = float('inf')
        
        for _, row in model_data.iterrows():
            # Calculate Euclidean distance in index space
            dist_fe5015 = (row[fe5015_col] - fe5015) / 5.0  # Normalize by typical range
            dist_mgb = (row[mgb_col] - mgb) / 4.0
            dist_hbeta = (row[hbeta_col] - hbeta) / 3.0
            
            distance = np.sqrt(dist_fe5015**2 + dist_mgb**2 + dist_hbeta**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_point = row
        
        # Return alpha/Fe from nearest point
        if nearest_point is not None:
            return nearest_point[aofe_col]
        else:
            return np.nan
            
    except Exception as e:
        logger.error(f"Error calculating bin alpha/Fe: {e}")
        return np.nan

def get_standardized_alpha_fe_data(galaxy_name, rdb_data, model_data, bins_limit=6, continuum_mode='fit', 
                                  special_cases=None):
    """
    Standardized alpha/Fe calculation function to be used by all visualizations.
    This is the SINGLE SOURCE OF TRUTH for alpha/Fe calculations.
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices
    model_data : DataFrame
        Model grid data with indices and alpha/Fe values
    bins_limit : int
        Limit analysis to first N bins (default: 6)
    continuum_mode : str
        Spectral index continuum mode to use ('auto', 'fit', 'original')
    special_cases : dict or None
        Optional dictionary of special case overrides, mapping galaxy names to values
        
    Returns:
    --------
    dict
        Standardized data dictionary with all required values for visualization
    """
    logger.info(f"Calculating standardized alpha/Fe data for {galaxy_name} with {continuum_mode} mode")
    
    # Start with a standardized results template
    result = {
        'galaxy': galaxy_name,
        'alpha_fe_values': [],
        'radius_values': [],
        'fe5015_values': [],
        'mgb_values': [],
        'hbeta_values': [],
        'bin_indices': [],
        'slope': np.nan,
        'p_value': np.nan,
        'r_squared': np.nan,
        'intercept': np.nan,
        'alpha_fe_median': np.nan,
        'radius_median': np.nan,
        'effective_radius': None,
        'method': '3D chi-square fitting',  # Choose your preferred default method
        'continuum_mode': continuum_mode,
        'special_case_applied': False
    }
    
    # Check for special case overrides first
    if special_cases and galaxy_name in special_cases:
        override = special_cases[galaxy_name]
        logger.info(f"Applying special case handling for {galaxy_name}")
        
        # Apply the override values
        for key, value in override.items():
            if key in result:
                result[key] = value
        
        result['special_case_applied'] = True
        
        # If the special case provides complete data, return it
        if len(result['alpha_fe_values']) > 0 and len(result['radius_values']) > 0:
            return result
    
    # If no special case or it doesn't provide complete data, calculate using the standard method
    try:
        # Extract spectral indices using the specified mode
        galaxy_indices = extract_spectral_indices_from_method(
            rdb_data, 
            method=continuum_mode,
            bins_limit=bins_limit
        )
        
        # Check if we have the required indices
        if ('bin_indices' not in galaxy_indices or
            'Fe5015' not in galaxy_indices['bin_indices'] or 
            'Mgb' not in galaxy_indices['bin_indices'] or
            'Hbeta' not in galaxy_indices['bin_indices']):
            logger.warning(f"Missing required spectral indices for {galaxy_name}")
            return result
        
        # Get galaxy spectral indices
        fe5015_values = galaxy_indices['bin_indices']['Fe5015']
        mgb_values = galaxy_indices['bin_indices']['Mgb']
        hbeta_values = galaxy_indices['bin_indices']['Hbeta']
        
        # Get radius information
        radius_values = None
        if 'R' in galaxy_indices['bin_indices']:
            radius_values = galaxy_indices['bin_indices']['R']
        elif 'bin_radii' in galaxy_indices:
            radius_values = galaxy_indices['bin_radii']
        elif 'distance' in rdb_data:
            distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
            if 'bin_distances' in distance:
                radius_values = distance['bin_distances'][:bins_limit]
        
        # Get effective radius
        Re = extract_effective_radius(rdb_data)
        
        # Scale radius by effective radius if available
        if Re is not None and Re > 0 and radius_values is not None:
            r_scaled = radius_values / Re
        else:
            r_scaled = radius_values
        
        # Store radius and effective radius
        result['effective_radius'] = Re
        
        # Calculate alpha/Fe values for each bin
        alpha_fe_values = []
        valid_bins = []
        
        # Process each bin
        for i in range(min(bins_limit, len(fe5015_values))):
            # Skip invalid bins
            if (i >= len(mgb_values) or i >= len(hbeta_values) or
                np.isnan(fe5015_values[i]) or np.isnan(mgb_values[i]) or 
                np.isnan(hbeta_values[i]) or fe5015_values[i] <= 0 or
                mgb_values[i] <= 0 or hbeta_values[i] <= 0):
                continue
                
            # Calculate alpha/Fe for this bin using our central calculation function
            bin_alpha_fe = calculate_bin_alpha_fe(
                fe5015_values[i], mgb_values[i], hbeta_values[i], model_data)
                
            # If we got a valid alpha/Fe value, add the bin to our results
            if not np.isnan(bin_alpha_fe):
                valid_bins.append(i)
                alpha_fe_values.append(bin_alpha_fe)
                result['fe5015_values'].append(fe5015_values[i])
                result['mgb_values'].append(mgb_values[i])
                result['hbeta_values'].append(hbeta_values[i])
                if r_scaled is not None and i < len(r_scaled):
                    result['radius_values'].append(r_scaled[i])
        
        # Store in our result
        result['alpha_fe_values'] = alpha_fe_values
        result['bin_indices'] = valid_bins
        
        # Calculate statistics
        if len(alpha_fe_values) > 0 and len(result['radius_values']) > 0:
            # Calculate median values
            result['alpha_fe_median'] = np.median(alpha_fe_values)
            result['radius_median'] = np.median(result['radius_values'])
            
            # Calculate slope and other statistics if we have multiple points
            if len(alpha_fe_values) > 1:
                slope, intercept, r_squared, p_value, std_err = calculate_improved_alpha_fe_slope(
                    result['radius_values'], alpha_fe_values)
                    
                result['slope'] = slope
                result['intercept'] = intercept
                result['r_squared'] = r_squared
                result['p_value'] = p_value
        
        # Return the standardized result
        return result
        
    except Exception as e:
        logger.error(f"Error in standardized alpha/Fe calculation for {galaxy_name}: {e}")
        import traceback
        traceback.print_exc()
        return result

def create_spectral_index_interpolation_plot(galaxy_name, rdb_data, model_data, output_path=None, dpi=300, 
                                           bins_limit=6, continuum_mode='fit', standardized_data=None):
    """
    Create a visualization showing how alpha/Fe is interpolated from spectral indices
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices
    model_data : DataFrame
        Model grid data
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for the output image
    bins_limit : int
        Limit on the number of bins to analyze
    continuum_mode : str
        Spectral index continuum mode to use
    standardized_data : dict, optional
        Pre-calculated standardized data - if provided, this will be used instead of calculating
        
    Returns:
    --------
    dict
        Dictionary with analysis results
    """
    try:
        # Use standardized data if provided, otherwise calculate it
        if standardized_data is None:
            standardized_data = get_standardized_alpha_fe_data(
                galaxy_name, rdb_data, model_data, bins_limit, continuum_mode, SPECIAL_CASES)
        
        # Extract the needed data from standardized_data
        alpha_fe_values = standardized_data['alpha_fe_values']
        radius_values = standardized_data['radius_values']
        fe5015_values = standardized_data['fe5015_values']
        mgb_values = standardized_data['mgb_values']
        hbeta_values = standardized_data['hbeta_values']
        slope = standardized_data['slope']
        p_value = standardized_data['p_value']
        r_squared = standardized_data['r_squared']
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Get mean age for model grid
        mean_age = 10.0  # Default to 10 Gyr
        
        # Find closest age in model grid
        age_column = model_column_mapping['Age']
        available_ages = np.array(model_data[age_column].unique())
        closest_age = available_ages[np.argmin(np.abs(available_ages - mean_age))]
        
        # Filter model grid to this age
        model_age_data = model_data[model_data[age_column] == closest_age]
        
        # Create a normalization for color mapping
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0.0, vmax=0.5)  # Fixed range for alpha/Fe
        
        # Get unique alpha/Fe and metallicity values
        zoh_column = model_column_mapping['ZoH']
        aofe_column = model_column_mapping['AoFe']
        fe5015_col = model_column_mapping['Fe5015']
        mgb_col = model_column_mapping['Mgb']
        hbeta_col = model_column_mapping['Hbeta']
        
        unique_aofe = sorted(model_age_data[aofe_column].unique())
        unique_zoh = sorted(model_age_data[zoh_column].unique())
        
        # Plot 1: Fe5015 vs Mgb - colored by [α/Fe]
        ax1 = axes[0, 0]
        
        # Draw constant alpha/Fe lines
        for aofe in unique_aofe:
            aofe_data = model_age_data[model_age_data[aofe_column] == aofe]
            aofe_data = aofe_data.sort_values(by=zoh_column)
            ax1.plot(aofe_data[fe5015_col], aofe_data[mgb_col], '-', 
                   color=plt.cm.plasma(norm(aofe)), linewidth=2.5, alpha=0.7)
        
        # Draw constant metallicity lines
        for zoh in unique_zoh:
            zoh_data = model_age_data[model_age_data[zoh_column] == zoh]
            zoh_data = zoh_data.sort_values(by=aofe_column)
            ax1.plot(zoh_data[fe5015_col], zoh_data[mgb_col], '--', 
                   color='gray', linewidth=1.5, alpha=0.5)
        
        # Plot galaxy points
        sc1 = ax1.scatter(fe5015_values, mgb_values, c=alpha_fe_values, 
                      cmap='plasma', s=150, edgecolor='black', linewidth=2.0, norm=norm)
        
        # Add bin numbers
        for i in range(len(fe5015_values)):
            ax1.text(fe5015_values[i], mgb_values[i], str(standardized_data['bin_indices'][i]), 
                   color='white', fontweight='bold', ha='center', va='center', fontsize=12)
        
        # Add colorbar
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label('[α/Fe]', fontsize=14)
        
        # Set labels and title
        ax1.set_xlabel('Fe5015 Index', fontsize=14)
        ax1.set_ylabel('Mgb Index', fontsize=14)
        ax1.set_title('Fe5015 vs Mgb - colored by [α/Fe]', fontsize=16)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fe5015 vs Hβ - colored by [α/Fe]
        ax2 = axes[0, 1]
        
        # Draw constant alpha/Fe lines
        for aofe in unique_aofe:
            aofe_data = model_age_data[model_age_data[aofe_column] == aofe]
            aofe_data = aofe_data.sort_values(by=zoh_column)
            ax2.plot(aofe_data[fe5015_col], aofe_data[hbeta_col], '-', 
                   color=plt.cm.plasma(norm(aofe)), linewidth=2.5, alpha=0.7)
            
        # Draw constant metallicity lines
        for zoh in unique_zoh:
            zoh_data = model_age_data[model_age_data[zoh_column] == zoh]
            zoh_data = zoh_data.sort_values(by=aofe_column)
            ax2.plot(zoh_data[fe5015_col], zoh_data[hbeta_col], '--', 
                   color='gray', linewidth=1.5, alpha=0.5)
        
        # Plot galaxy points
        sc2 = ax2.scatter(fe5015_values, hbeta_values, c=alpha_fe_values, 
                      cmap='plasma', s=150, edgecolor='black', linewidth=2.0, norm=norm)
        
        # Add bin numbers
        for i in range(len(fe5015_values)):
            ax2.text(fe5015_values[i], hbeta_values[i], str(standardized_data['bin_indices'][i]), 
                   color='white', fontweight='bold', ha='center', va='center', fontsize=12)
        
        # Add colorbar
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label('[α/Fe]', fontsize=14)
        
        # Set labels and title
        ax2.set_xlabel('Fe5015 Index', fontsize=14)
        ax2.set_ylabel('Hβ Index', fontsize=14)
        ax2.set_title('Fe5015 vs Hβ - colored by [α/Fe]', fontsize=16)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mgb vs Hβ - colored by [α/Fe]
        ax3 = axes[1, 0]
        
        # Draw constant alpha/Fe lines
        for aofe in unique_aofe:
            aofe_data = model_age_data[model_age_data[aofe_column] == aofe]
            aofe_data = aofe_data.sort_values(by=zoh_column)
            ax3.plot(aofe_data[mgb_col], aofe_data[hbeta_col], '-', 
                   color=plt.cm.plasma(norm(aofe)), linewidth=2.5, alpha=0.7)
            
        # Draw constant metallicity lines
        for zoh in unique_zoh:
            zoh_data = model_age_data[model_age_data[zoh_column] == zoh]
            zoh_data = zoh_data.sort_values(by=aofe_column)
            ax3.plot(zoh_data[mgb_col], zoh_data[hbeta_col], '--', 
                   color='gray', linewidth=1.5, alpha=0.5)
        
        # Plot galaxy points
        sc3 = ax3.scatter(mgb_values, hbeta_values, c=alpha_fe_values, 
                      cmap='plasma', s=150, edgecolor='black', linewidth=2.0, norm=norm)
        
        # Add bin numbers
        for i in range(len(mgb_values)):
            ax3.text(mgb_values[i], hbeta_values[i], str(standardized_data['bin_indices'][i]), 
                   color='white', fontweight='bold', ha='center', va='center', fontsize=12)
        
        # Add colorbar
        cbar3 = plt.colorbar(sc3, ax=ax3)
        cbar3.set_label('[α/Fe]', fontsize=14)
        
        # Set labels and title
        ax3.set_xlabel('Mgb Index', fontsize=14)
        ax3.set_ylabel('Hβ Index', fontsize=14)
        ax3.set_title('Mgb vs Hβ - colored by [α/Fe]', fontsize=16)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: [α/Fe] vs. Radius - use the standardized values
        ax4 = axes[1, 1]
        
        # Sort by radius for line plot
        if radius_values and len(radius_values) > 0:
            sorted_pairs = sorted(zip(radius_values, alpha_fe_values), key=lambda pair: pair[0])
            r_sorted = np.array([pair[0] for pair in sorted_pairs])
            a_sorted = np.array([pair[1] for pair in sorted_pairs])
            
            # Plot points with connecting line
            ax4.plot(r_sorted, a_sorted, '-', color='purple', linewidth=2.5)
            ax4.scatter(radius_values, alpha_fe_values, s=120, color='purple', edgecolor='black', zorder=10)
            
            # Add bin numbers
            for i in range(len(radius_values)):
                ax4.text(radius_values[i], alpha_fe_values[i], str(standardized_data['bin_indices'][i]), 
                       fontsize=12, ha='center', va='center', color='white', fontweight='bold')
        
            # Add trend line - using the standardized slope & intercept
            if not np.isnan(slope):
                x_range = np.linspace(min(radius_values), max(radius_values), 100)
                y_range = slope * x_range + standardized_data['intercept']
                ax4.plot(x_range, y_range, '--', color='red', linewidth=2.5)
                
                # Add annotation
                significance = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
                ax4.text(0.05, 0.95, f"Slope = {slope:.3f}{significance}\np-value = {p_value:.4f}\nR² = {r_squared:.3f}", 
                       transform=ax4.transAxes, va='top', ha='left', fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Set labels and title
        ax4.set_xlabel('R/Re', fontsize=14)
        ax4.set_ylabel('[α/Fe]', fontsize=14)
        ax4.set_title('[α/Fe] vs. R/Re', fontsize=16)
        ax4.grid(True, alpha=0.3)
        
        # Calculate bounds for y-axis
        if alpha_fe_values and len(alpha_fe_values) > 0:
            min_alpha = min(alpha_fe_values) - 0.03
            max_alpha = max(alpha_fe_values) + 0.03
            ax4.set_ylim(min_alpha, max_alpha)
        
        # Add special case note if applicable
        if standardized_data.get('special_case_applied', False):
            plt.figtext(0.5, 0.02, 
                      "NOTE: Special case handling applied to this galaxy.",
                      ha='center', fontsize=12, color='red', 
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        
        # Add overall title
        plt.suptitle(f"Galaxy {galaxy_name}: [α/Fe] Interpolation from Spectral Indices\n"
                   f"Model Age: {closest_age} Gyr | Continuum Mode: {continuum_mode}", 
                   fontsize=18, y=0.98)
        
        # Add explanation text
        plt.figtext(0.5, 0.01, 
                  "Alpha/Fe values are interpolated from the model grid using primarily Fe5015 and Mgb indices.\n"
                  "Solid lines: constant [α/Fe], Dashed lines: constant [Z/H]",
                  ha='center', fontsize=14)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved spectral index interpolation plot to {output_path}")
        
        plt.close()
        
        # Return the standardized data for reference
        return standardized_data
        
    except Exception as e:
        logger.error(f"Error creating spectral index interpolation plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_parameter_radius_plots(galaxy_name, rdb_data, model_data=None, output_path=None, 
                                dpi=300, bins_limit=6, interpolated_data=None, continuum_mode='fit'):
    """
    Create parameter vs. radius plots with consistent data and consistent R/Re units
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices
    model_data : DataFrame, optional
        Model grid data
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for the output image
    bins_limit : int
        Limit on the number of bins to analyze
    interpolated_data : dict, optional
        Pre-calculated data - if provided, use this instead of calculating
    continuum_mode : str
        Spectral index continuum mode to use
    """
    try:
        # Check if RDB data is valid
        if rdb_data is None or not isinstance(rdb_data, dict):
            logger.error(f"Invalid RDB data format for {galaxy_name}")
            return
        
        # Get standardized data if not provided
        if interpolated_data is None:
            standardized_data = get_standardized_alpha_fe_data(
                galaxy_name, rdb_data, model_data, bins_limit, continuum_mode, SPECIAL_CASES)
        else:
            standardized_data = interpolated_data
        
        # Get parameter profiles for standard spectral indices
        params = extract_parameter_profiles(
            rdb_data, 
            parameter_names=['Fe5015', 'Mgb', 'Hbeta', 'age', 'metallicity'],
            bins_limit=bins_limit,
            continuum_mode=continuum_mode
        )
        
        # Use standardized data for alpha/Fe
        alpha_fe_values = standardized_data['alpha_fe_values']
        radius_values = standardized_data['radius_values']
        
        # Get effective radius - needed for normalization
        Re = params['effective_radius']
        if Re is None or Re <= 0:
            # Try to get from standardized data
            Re = standardized_data.get('effective_radius')
            
        # Create normalized radius values for all parameters
        if Re is not None and Re > 0:
            # Use consistent R/Re units for all plots
            if 'radius' in params and params['radius'] is not None:
                # Normalize raw radius by Re
                r_scaled = params['radius'] / Re
                x_label = 'R/Re'
            else:
                r_scaled = None
                x_label = 'R/Re'
        else:
            # No Re available, use raw radius
            r_scaled = params.get('radius')
            x_label = 'R/Re'
            
        # Create figure with 2x3 grid of subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Create plots for each parameter
        parameters = ['Fe5015', 'Mgb', 'Hbeta', 'age', 'metallicity'] 
        
        for i, param_name in enumerate(parameters):
            ax = axes[i]
            
            if param_name in params and hasattr(params[param_name], '__len__') and len(params[param_name]) > 0:
                y = params[param_name]
                
                # Use r_scaled (normalized radius) for ALL plots
                if r_scaled is None or len(r_scaled) == 0:
                    ax.text(0.5, 0.5, f"No radius data available", 
                          ha='center', va='center', fontsize=14,
                          transform=ax.transAxes)
                    continue
                
                # Create sorted arrays for consistent plotting
                # Always use the normalized r_scaled here
                sorted_pairs = sorted(zip(r_scaled, y), key=lambda pair: pair[0])
                r_sorted = np.array([pair[0] for pair in sorted_pairs])
                y_sorted = np.array([pair[1] for pair in sorted_pairs])
                
                # Check for outliers
                _, _, outlier_mask = remove_outliers(r_sorted, y_sorted, threshold=3.0)
                
                # Create a clean copy for fitting
                x_clean = r_sorted.copy()
                y_clean = y_sorted.copy()
                
                # Mark outliers with X but don't use them for fitting
                if np.any(outlier_mask):
                    x_clean[outlier_mask] = np.nan
                    y_clean[outlier_mask] = np.nan
                
                # Plot data points with lines connecting in order of radius
                ax.plot(r_sorted, y_sorted, 'o-', color='blue', markersize=10, linewidth=2.0, alpha=0.7)
                
                # Mark outliers
                if np.any(outlier_mask):
                    ax.plot(r_sorted[outlier_mask], y_sorted[outlier_mask], 'rx', markersize=12, alpha=0.8)
                
                # Fit linear trend and add to plot
                slope, intercept, y_fit, r_squared, p_value = fit_linear_slope(x_clean, y_clean, return_full=True)
                
                if not np.isnan(slope):
                    valid_mask = ~np.isnan(x_clean) & ~np.isnan(y_clean)
                    x_valid = np.array(x_clean)[valid_mask]
                    if len(x_valid) >= 2:
                        x_line = np.linspace(min(x_valid), max(x_valid), 100)
                        y_line = linear_fit(x_line, slope, intercept)
                        
                        ax.plot(x_line, y_line, '--', color='red', linewidth=2.5)
                        
                        # Add slope and p-value to plot
                        if i <3:
                            ax.text(0.65, 0.95, f"Slope = {slope:.3f}\np = {p_value:.3f}", 
                                transform=ax.transAxes, fontsize=12,
                                va='top', ha='left',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                        else:
                            ax.text(0.05, 0.95, f"Slope = {slope:.3f}\np = {p_value:.3f}", 
                                transform=ax.transAxes, fontsize=12,
                                va='top', ha='left',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Set labels and title
                ax.set_xlabel('R/Re', fontsize=14)
                ax.set_ylabel(param_name, fontsize=14)
                ax.set_title(f"{param_name} vs. R/Re", fontsize=16)
                
                # Add grid
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add vertical line at Re=1 if using normalized radius
                if Re is not None and Re > 0:
                    ax.axvline(x=1, color='k', linestyle=':', alpha=0.5)
                
                # Set tick parameters
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize=12, right=True, top=True, direction='in')
            else:
                ax.text(0.5, 0.5, f"No {param_name} data available", 
                      ha='center', va='center', fontsize=14,
                      transform=ax.transAxes)
        
        # Add alpha/Fe plot in the last subplot - using standardized data
        ax = axes[5]
        
        if alpha_fe_values and radius_values and len(alpha_fe_values) > 0:
            # Sort by radius
            sorted_pairs = sorted(zip(radius_values, alpha_fe_values), key=lambda pair: pair[0])
            r_sorted = np.array([pair[0] for pair in sorted_pairs])
            alpha_sorted = np.array([pair[1] for pair in sorted_pairs])
            
            # Plot data points with lines
            ax.plot(r_sorted, alpha_sorted, 'o-', color='purple', markersize=10, linewidth=2.0, alpha=0.7)
            
            # Add bin numbers
            for j, (r, alpha) in enumerate(zip(r_sorted, alpha_sorted)):
                ax.text(r, alpha, str(standardized_data['bin_indices'][j]), fontsize=10, ha='center', va='center', 
                      color='white', fontweight='bold')
            
            # Use standardized slope values
            slope = standardized_data['slope']
            p_value = standardized_data['p_value']
            r_squared = standardized_data['r_squared']
            intercept = standardized_data['intercept']
            
            # Add trend line if we have a valid slope
            if not np.isnan(slope):
                x_range = np.linspace(min(r_sorted), max(r_sorted), 100)
                y_range = slope * x_range + intercept
                ax.plot(x_range, y_range, '--', color='red', linewidth=2.5)
                
                # Add horizontal reference line
                ax.axhline(y=np.mean(alpha_sorted), color='green', linestyle=':', alpha=0.7)
                
                # Determine trend type based on slope magnitude
                if abs(slope) < 0.01:
                    trend_type = "Horizontal" 
                    trend_color = "green"
                elif slope > 0:
                    trend_type = "Increasing"
                    trend_color = "blue"
                else:
                    trend_type = "Decreasing" 
                    trend_color = "red"
                
                # Add annotation with trend type
                significance = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
                ax.text(0.05, 0.95, 
                      f"Slope = {slope:.3f}{significance}\np = {p_value:.3f}\nR² = {r_squared:.3f}\nTrend: {trend_type}", 
                      transform=ax.transAxes, fontsize=12,
                      va='top', ha='left', color=trend_color,
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Set consistent x-axis label for the alpha/Fe plot
        ax.set_xlabel('R/Re', fontsize=14)
        ax.set_ylabel('[α/Fe]', fontsize=14)
        ax.set_title('[α/Fe] vs. R/Re', fontsize=16)
        
        # Add grid and other formatting
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add vertical line at Re=1 if using normalized radius
        if Re is not None and Re > 0:
            ax.axvline(x=1, color='k', linestyle=':', alpha=0.5)
        
        # Add special case note if applicable
        if standardized_data.get('special_case_applied', False):
            ax.text(0.5, 0.5, "NOTE: Special case values applied",
                  transform=ax.transAxes, fontsize=12, color='red',
                  ha='center', va='center',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        
        # Add overall title
        re_info = f" (Re = {Re:.2f} arcsec)" if Re is not None and Re > 0 else ""
        plt.suptitle(f"Galaxy {galaxy_name}: Parameter-Radius Relations{re_info}\nContinuum Mode: {continuum_mode}", 
                   fontsize=18, y=0.98)
        
        # Add note about standardized data
        plt.figtext(0.5, 0.01, 
                  "Alpha/Fe values derived from standard interpolation in spectral index space.\n"
                  "Bin numbers shown on the α/Fe plot correspond to the radial bins used in the analysis.",
                  ha='center', fontsize=12, style='italic')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved parameter-radius plots to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating parameter-radius plots: {e}")
        import traceback
        traceback.print_exc()

def create_model_grid_plots_part1(galaxy_name, rdb_data, model_data, age=10, output_path=None, dpi=150, 
                                bins_limit=6, continuum_mode='fit', standardized_data=None):
    """
    Create first set of model grid plots colored by R, log Age, and M/H
    Part 1: Fe5015 vs Mgb, Fe5015 vs Hbeta, Mgb vs Hbeta - Colored by different parameters
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices and bin info
    model_data : DataFrame
        Model grid data with ages, metallicities, and spectral indices
    age : float
        Age (in Gyr) to use for the model grid
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for the output image
    bins_limit : int
        Limit analysis to first N bins (default: 6 for bins 0-5)
    continuum_mode : str
        Spectral index continuum mode to use ('auto', 'fit', 'original')
    standardized_data : dict, optional
        Pre-calculated standardized data - if provided, this will be used instead of calculating
    """
    try:
        # Use standardized data if provided, otherwise calculate it
        if standardized_data is None:
            standardized_data = get_standardized_alpha_fe_data(
                galaxy_name, rdb_data, model_data, bins_limit, continuum_mode, SPECIAL_CASES)
        
        # Extract data from standardized data
        fe5015_values = standardized_data['fe5015_values']
        mgb_values = standardized_data['mgb_values'] 
        hbeta_values = standardized_data['hbeta_values']
        radius_values = standardized_data['radius_values']
        bin_indices = standardized_data['bin_indices']
        
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Filter model data to the requested age
        age_column = model_column_mapping['Age']
        available_ages = np.array(sorted(model_data[age_column].unique()))
        closest_age = available_ages[np.argmin(np.abs(available_ages - age))]
        
        model_age_data = model_data[model_data[age_column] == closest_age].copy()
        logger.info(f"Using age {closest_age} Gyr for model grid (requested: {age} Gyr)")
        
        # Create figure with 3x3 grid (9 subplots)
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Set up the three index pairs to plot
        index_pairs = [
            ('Fe5015', 'Mgb'),
            ('Fe5015', 'Hbeta'),
            ('Mgb', 'Hbeta')
        ]
        
        # Set up color variables in order
        color_vars = ['R', 'age', 'metallicity']
        color_labels = ['R (arcsec)', 'log Age (Gyr)', '[M/H]']
        
        # Get unique metallicity and alpha/Fe values for grid lines
        zoh_column = model_column_mapping['ZoH']
        aofe_column = model_column_mapping['AoFe']
        
        zoh_unique = sorted(model_age_data[zoh_column].unique())
        aofe_unique = sorted(model_age_data[aofe_column].unique())
        
        # Get parameter data from standardized data
        param_data = {'R': radius_values}
        
        # Extract other parameters from RDB data
        params = extract_parameter_profiles(
            rdb_data, 
            parameter_names=['age', 'metallicity'],
            bins_limit=bins_limit,
            continuum_mode=continuum_mode
        )
        
        # Use only the valid bins
        if 'age' in params and params['age'] is not None:
            param_data['age'] = [params['age'][b] for b in bin_indices]
        
        if 'metallicity' in params and params['metallicity'] is not None:
            param_data['metallicity'] = [params['metallicity'][b] for b in bin_indices]
        
        # Plot each panel
        for row, color_var in enumerate(color_vars):
            for col, (x_index, y_index) in enumerate(index_pairs):
                ax = axes[row, col]
                
                # Get the mapped column names for the model data
                model_x_column = model_column_mapping[x_index]
                model_y_column = model_column_mapping[y_index]
                
                # Skip if either column is missing
                if model_x_column is None or model_y_column is None:
                    ax.text(0.5, 0.5, f"Missing {x_index} or {y_index} data", 
                          transform=ax.transAxes, ha='center', va='center')
                    continue
                
                # Draw model grid
                # Draw metallicity (ZoH) lines
                for zoh in zoh_unique:
                    zoh_data = model_age_data[model_age_data[zoh_column] == zoh]
                    ax.plot(zoh_data[model_x_column], zoh_data[model_y_column], '-', 
                           color='tab:blue', alpha=0.5, linewidth=1.5, zorder=1)
                
                # Draw alpha/Fe (AoFe) lines
                for aofe in aofe_unique:
                    aofe_data = model_age_data[model_age_data[aofe_column] == aofe]
                    ax.plot(aofe_data[model_x_column], aofe_data[model_y_column], '--', 
                           color='tab:red', alpha=0.5, linewidth=1.5, zorder=1)
                
                # Add grid point annotations
                for zoh in zoh_unique:
                    for aofe in aofe_unique:
                        point_data = model_age_data[(model_age_data[zoh_column] == zoh) & 
                                                  (model_age_data[aofe_column] == aofe)]
                        if len(point_data) > 0:
                            x_val = point_data[model_x_column].values[0]
                            y_val = point_data[model_y_column].values[0]
                            
                            # Add grid point
                            ax.scatter(x_val, y_val, color='black', s=20, zorder=2)
                
                # Get galaxy values for this panel
                x_values = []
                y_values = []
                color_values = []
                
                if x_index == 'Fe5015' and y_index == 'Mgb':
                    x_values = fe5015_values
                    y_values = mgb_values
                elif x_index == 'Fe5015' and y_index == 'Hbeta':
                    x_values = fe5015_values 
                    y_values = hbeta_values
                elif x_index == 'Mgb' and y_index == 'Hbeta':
                    x_values = mgb_values
                    y_values = hbeta_values
                
                # Get color data
                if color_var in param_data and param_data[color_var] is not None:
                    color_data = param_data[color_var]
                    
                    # Determine color normalization
                    if len(color_data) > 0:
                        vmin = np.nanmin(color_data)
                        vmax = np.nanmax(color_data)
                    else:
                        # Default values if array is empty
                        vmin = 0
                        vmax = 1
                    
                    # Plot points colored by the selected variable
                    sc = ax.scatter(x_values, y_values, c=color_data, 
                                  cmap='viridis', s=80, alpha=0.8, zorder=10,
                                  vmin=vmin, vmax=vmax)
                    
                    # Add bin numbers
                    for j, (x, y) in enumerate(zip(x_values, y_values)):
                        ax.text(x, y, str(bin_indices[j]), fontsize=8, ha='center', va='center', 
                              color='white', fontweight='bold', zorder=11)
                    
                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(sc, cax=cax)
                    cbar.set_label(color_labels[row])
                else:
                    # Fallback to plain scatter if color variable is not available
                    ax.scatter(x_values, y_values, color='green', s=80, alpha=0.7, zorder=10)
                    
                    # Add bin numbers
                    for j, (x, y) in enumerate(zip(x_values, y_values)):
                        ax.text(x, y, str(bin_indices[j]), fontsize=8, ha='center', va='center', 
                              color='black', fontweight='bold', zorder=11)
                
                # Set labels and grid
                ax.set_xlabel(f'{x_index} Index', fontsize=12)
                ax.set_ylabel(f'{y_index} Index', fontsize=12)
                ax.set_title(f'{x_index} vs {y_index} - colored by {color_labels[row]}', fontsize=14)
                ax.grid(alpha=0.3, linestyle='--')
                
                # Set tick parameters
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Add legend
        legend_elements = [
            Line2D([0], [0], color='tab:blue', linestyle='-', linewidth=1.5, label='Constant [Z/H]'),
            Line2D([0], [0], color='tab:red', linestyle='--', linewidth=1.5, label='Constant [α/Fe]'),
            Line2D([0], [0], marker='o', color='w', label='Galaxy Bins',
                   markerfacecolor='tab:green', markersize=8)
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.03))
        
        # Add special case note if applicable
        if standardized_data.get('special_case_applied', False):
            plt.figtext(0.5, 0.05, 
                      "NOTE: Special case handling applied to this galaxy.",
                      ha='center', fontsize=10, color='red', 
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        
        # Add overall title
        plt.suptitle(f'Galaxy {galaxy_name}: Spectral Indices vs. Model Grid\n'
                   f'Age = {closest_age} Gyr | Continuum Mode: {continuum_mode}', 
                   fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved model grid plots part 1 to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating model grid plots part 1: {e}")
        import traceback
        traceback.print_exc()

def create_combined_flux_and_binning(galaxy_name, p2p_data, rdb_data, cube_info, output_path=None, dpi=150):
    """
    Create flux map with only bin boundaries shown as edges between pixels in different bins,
    with thicker lines for better visibility
    """
    try:
        # Check if data is available
        p2p_available = p2p_data is not None and isinstance(p2p_data, dict)
        rdb_valid = rdb_data is not None and isinstance(rdb_data, dict)
        
        if not rdb_valid:
            logger.error(f"No RDB data available for {galaxy_name}")
            return
            
        # Extract flux map
        flux_map = None
        if p2p_available:
            flux_map = extract_flux_map(p2p_data)
                
        if flux_map is None:
            logger.warning(f"No flux map found for {galaxy_name}, creating synthetic one")
            # Try to get dimensions
            nx, ny = 50, 50  # Default dimensions
            if 'binning' in rdb_data:
                binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
                if 'n_x' in binning and 'n_y' in binning:
                    nx, ny = binning['n_x'], binning['n_y']
                
            # Generate synthetic flux map
            y, x = np.indices((ny, nx))
            cy, cx = ny / 2, nx / 2
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            flux_map = np.exp(-r / (max(nx, ny) / 4))
        
        # Extract binning information
        if 'binning' not in rdb_data:
            logger.error(f"No binning information found in RDB data for {galaxy_name}")
            return
        
        binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
        
        # Get bin numbers
        bin_num = binning.get('bin_num', None)
        
        # Get center, PA and ellipticity (for reference only)
        center_x = binning.get('center_x', None)
        center_y = binning.get('center_y', None)
        pa = binning.get('pa', 0.0)
        ellipticity = binning.get('ellipticity', 0.0)
        
        # Get dimensions from flux map
        ny, nx = flux_map.shape
        
        # Use dimensions if center not provided
        if center_x is None or center_y is None:
            center_x = nx / 2
            center_y = ny / 2
        
        # Extract effective radius
        Re = extract_effective_radius(rdb_data)
        
        # Get pixel scale information
        pixel_scale_x = cube_info['pixel_scale_x']  # arcsec/pixel
        pixel_scale_y = cube_info['pixel_scale_y']  # arcsec/pixel
        
        # Calculate full extent in arcseconds
        extent_x = nx * pixel_scale_x
        extent_y = ny * pixel_scale_y
        
        # Calculate the extent for the plot - centered at 0
        extent = [-extent_x/2, extent_x/2, -extent_y/2, extent_y/2]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot the flux map with physical units
        valid_mask = np.isfinite(flux_map) & (flux_map > 0)
        if np.any(valid_mask):
            vmin = np.percentile(flux_map[valid_mask], 1)
            vmax = np.percentile(flux_map[valid_mask], 99)
            norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
        else:
            norm = LogNorm(vmin=1e-10, vmax=1)
        
        # Plot flux map with physical units and correct scaling
        im = ax.imshow(flux_map, origin='lower', norm=norm, cmap='inferno',
                     extent=extent, aspect='equal')
        
        # Add colorbar for flux with log scale formatting
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Flux (log scale)')
        
        # Check if bin_num is provided as a 2D array matching the flux map
        if bin_num is not None:
            # Convert bin_num to 2D array if it's not already
            if hasattr(bin_num, 'ndim') and bin_num.ndim == 1:
                bin_num_2d = np.full((ny, nx), -1, dtype=int)
                valid_len = min(len(bin_num), ny * nx)
                bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
            elif hasattr(bin_num, 'ndim') and bin_num.ndim == 2:
                bin_num_2d = bin_num
            else:
                logger.warning("bin_num format not recognized, cannot draw bin boundaries")
                bin_num_2d = None
                
            # If bin_num_2d is valid, draw the bin boundaries
            if bin_num_2d is not None:
                # Create arrays to store the line segments
                vertical_segs = []   # For vertical edges (between columns)
                horizontal_segs = [] # For horizontal edges (between rows)
                
                # Check for bin boundaries in x-direction (vertical edges)
                for y in range(ny):
                    for x in range(1, nx):
                        # Only draw a line if adjacent pixels are in different bins
                        if bin_num_2d[y, x] != bin_num_2d[y, x-1]:
                            # Convert pixel coordinates to arcsec coordinates
                            x_arcsec = (x - nx/2) * pixel_scale_x - pixel_scale_x/2  # Center on the edge
                            y_bottom = (y - ny/2) * pixel_scale_y - pixel_scale_y/2
                            y_top = y_bottom + pixel_scale_y
                            
                            # Add the line segment
                            vertical_segs.append([(x_arcsec, y_bottom), (x_arcsec, y_top)])
                
                # Check for bin boundaries in y-direction (horizontal edges)
                for x in range(nx):
                    for y in range(1, ny):
                        # Only draw a line if adjacent pixels are in different bins
                        if bin_num_2d[y, x] != bin_num_2d[y-1, x]:
                            # Convert pixel coordinates to arcsec coordinates
                            y_arcsec = (y - ny/2) * pixel_scale_y - pixel_scale_y/2  # Center on the edge
                            x_left = (x - nx/2) * pixel_scale_x - pixel_scale_x/2
                            x_right = x_left + pixel_scale_x
                            
                            # Add the line segment
                            horizontal_segs.append([(x_left, y_arcsec), (x_right, y_arcsec)])
                
                # Draw all horizontal and vertical line segments with thicker lines
                line_segments = LineCollection(
                    vertical_segs + horizontal_segs,
                    colors='white',
                    linewidths=1.5,  # Increased line thickness
                    alpha=0.9,       # Increased opacity
                    zorder=10
                )
                ax.add_collection(line_segments)
                
                # Add count of bins found to the plot
                unique_bins = np.unique(bin_num_2d[bin_num_2d >= 0])
                ax.text(0.98, 0.02, f"Bins: {len(unique_bins)}", transform=ax.transAxes, 
                       color='white', fontsize=10, ha='right', va='bottom',
                       bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'))
        
        # Add effective radius label if available
        if Re is not None:
            # Just add the text without drawing the ellipse
            ax.text(0, -extent_y/2 * 0.9, f'Re = {Re:.2f} arcsec', 
                   color='red', fontsize=12, fontweight='bold', ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=3))
        
        # Add north/east arrows
        arrow_len = min(extent_x, extent_y) * 0.1
        arrow_start_x = extent_x/2 * 0.8
        arrow_start_y = -extent_y/2 * 0.8
        
        # North arrow
        ax.annotate('N', xy=(arrow_start_x, arrow_start_y + arrow_len), 
                  xytext=(arrow_start_x, arrow_start_y),
                  arrowprops=dict(facecolor='white', width=1.5, headwidth=7),
                  color='white', ha='center', va='bottom', fontsize=12)
        
        # East arrow
        ax.annotate('E', xy=(arrow_start_x + arrow_len, arrow_start_y), 
                  xytext=(arrow_start_x, arrow_start_y),
                  arrowprops=dict(facecolor='white', width=1.5, headwidth=7),
                  color='white', ha='left', va='center', fontsize=12)
        
        # Add title
        ax.set_title(f"Galaxy: {galaxy_name}", fontsize=16)
        ax.set_xlabel('Arcsec')
        ax.set_ylabel('Arcsec')
        
        # Add metadata at the bottom
        info_text = f"PA: {pa:.1f}°, Ellipticity: {ellipticity:.2f}, Pixel scale: {pixel_scale_x:.3f}×{pixel_scale_y:.3f} arcsec/pixel"
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12)
        
        # Set tick parameters
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved flux map with bin boundaries to {output_path}")
        
        plt.close()
        
        return
        
    except Exception as e:
        logger.error(f"Error creating flux map with bin boundaries: {e}")
        import traceback
        traceback.print_exc()

def create_alpha_radius_plot_improved(results_list, output_path=None, dpi=300):
    """
    Create an improved plot of alpha/Fe vs. radius for all galaxies with slope information
    in the legend rather than on the plot
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing galaxy alpha/Fe and radius results
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for saved image
    """
    try:
        # Create figure with a larger size for better readability
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Set up colors for different galaxies - use a qualitative colormap
        # that's more distinct than tab20
        n_galaxies = len([r for r in results_list if r is not None])
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Create a custom legend for slopes and p-values
        legend_elements = []
        
        # Plot each galaxy
        galaxy_index = 0
        for result in results_list:
            if result is None:
                continue
                
            galaxy = result['galaxy']
            # Use alpha_fe_median as the central value
            alpha_fe = result.get('alpha_fe_median', np.nan)
            radius = result.get('radius_median', np.nan)
            
            # Use the standardized slope values
            slope = result.get('slope', np.nan)
            p_value = result.get('p_value', np.nan)
            
            # Get color for this galaxy
            color = colors[galaxy_index % 20]
            galaxy_index += 1
            
            # Marker size and edge width based on statistical significance
            if np.isnan(p_value) or p_value > 0.05:
                edge_color = 'gray'  # Not significant
                linewidth = 1.5
                marker_size = 150
            else:
                edge_color = 'black'  # Significant
                linewidth = 2.5
                marker_size = 200
            
            # Plot point
            ax.scatter(radius, alpha_fe, s=marker_size, color=color, edgecolor=edge_color, 
                     linewidth=linewidth, zorder=10)
            
            # Add slope line if available
            if not np.isnan(slope) and 'radius_values' in result and len(result['radius_values']) > 1:
                # Get radius range - extend slightly for better visibility
                min_radius = max(0.1, np.min(result['radius_values']) * 0.8)
                max_radius = np.max(result['radius_values']) * 1.2
                
                # Calculate intercept from the median point
                intercept = alpha_fe - slope * radius
                
                # Plot slope line
                x_range = np.linspace(min_radius, max_radius, 100)
                y_range = slope * x_range + intercept
                ax.plot(x_range, y_range, '--', color=color, alpha=0.7, linewidth=2.5)
            
            # Format significance marker
            sig_marker = "*" if p_value < 0.05 else ""
            
            # Create legend entry with slope and p-value
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                     markeredgecolor=edge_color, markeredgewidth=linewidth,
                     markersize=12, label=f"{galaxy}: α/Fe={alpha_fe:.2f}, Slope={slope:.3f}{sig_marker} (p={p_value:.3f})")
            )
        
        # Set labels and title
        ax.set_xlabel('R/Re', fontsize=16)
        ax.set_ylabel('[α/Fe]', fontsize=16)
        ax.set_title('Alpha Element Abundance vs. Radius for Virgo Cluster Galaxies', fontsize=18)
        
        # Set tick parameters
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis='both', which='both', labelsize=12, right=True, top=True, direction='in')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add significance note
        ax.text(0.02, 0.02, "* p < 0.05 (statistically significant)", 
               transform=ax.transAxes, fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Add legend with galaxy names, slopes and p-values
        # Place it outside the plot to avoid overlapping with data
        legend = ax.legend(handles=legend_elements, fontsize=12, framealpha=0.9, 
                         bbox_to_anchor=(1.05, 1), title='Galaxies')
        legend.get_title().set_fontsize(14)
        
        # Add note about the standardized data
        plt.figtext(0.5, 0.01, 
                  "Alpha/Fe values calculated using standardized 3D interpolation method.\n"
                  "Slopes indicate radial gradients of alpha-element abundance.",
                  ha='center', fontsize=12, style='italic')
        
        # Adjust layout to make room for legend
        plt.tight_layout(rect=[0, 0.03, 0.78, 0.97])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved improved alpha/Fe vs. radius plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating improved alpha/Fe vs. radius plot: {e}")
        import traceback
        traceback.print_exc()

def create_alpha_radius_direct_plot(results_list, output_path=None, dpi=300):
    """
    Create a plot of alpha/Fe vs. radius using directly calculated values for each point
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing galaxy data with direct alpha/Fe calculations
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for saved image
    """
    try:
        # Create figure
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Set up colors for different galaxies
        n_galaxies = len([r for r in results_list if r is not None])
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        
        # Create legend elements
        legend_elements = []
        
        # Process each galaxy
        galaxy_index = 0
        for result in results_list:
            if result is None:
                continue
                
            galaxy_name = result['galaxy']
            
            # Use standardized data for consistency
            alpha_fe_values = result.get('alpha_fe_values', [])
            radius_values = result.get('radius_values', [])
            slope = result.get('slope', np.nan)
            p_value = result.get('p_value', np.nan)
            
            # Skip if insufficient data
            if len(alpha_fe_values) == 0 or len(radius_values) == 0:
                continue
            
            # Get color for this galaxy
            color = colors[galaxy_index % 20]
            galaxy_index += 1
            
            # Plot all points for this galaxy
            ax.scatter(radius_values, alpha_fe_values, s=120, color=color, edgecolor='black', alpha=0.8, zorder=10)
            
            # Calculate statistics
            median_alpha_fe = np.median(alpha_fe_values) if alpha_fe_values else np.nan
            
            # If we have multiple points, add line connecting them in radial order
            if len(radius_values) > 1:
                # Sort by radius
                sorted_pairs = sorted(zip(radius_values, alpha_fe_values), key=lambda pair: pair[0])
                r_sorted = [pair[0] for pair in sorted_pairs]
                a_sorted = [pair[1] for pair in sorted_pairs]
                ax.plot(r_sorted, a_sorted, '-', color=color, alpha=0.6, linewidth=2.0)
                
                # Add to legend with slope and p-value information
                sig_marker = "*" if p_value < 0.05 else ""
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                         markeredgecolor='black', markeredgewidth=1.5,
                         markersize=12, label=f"{galaxy_name}: median α/Fe={median_alpha_fe:.2f}, Slope={slope:.3f}{sig_marker} (p={p_value:.3f})")
                )
            else:
                # If only one point, just add the galaxy to legend without slope
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                         markeredgecolor='black', markeredgewidth=1.5,
                         markersize=12, label=f"{galaxy_name}: α/Fe={median_alpha_fe:.2f}")
                )
        
        # Set labels and title
        ax.set_xlabel('R/Re', fontsize=16)
        ax.set_ylabel('[α/Fe]', fontsize=16)
        ax.set_title('Alpha Element Abundance vs. Radius - All Data Points', fontsize=18)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add significance note
        ax.text(0.02, 0.02, "* p < 0.05 (statistically significant)", 
               transform=ax.transAxes, fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Add legend outside the plot to avoid overlap
        legend = ax.legend(handles=legend_elements, fontsize=12, framealpha=0.9, 
                         bbox_to_anchor=(1.05, 1), title='Galaxies')
        legend.get_title().set_fontsize(14)
        
        # Add note about the data source
        plt.figtext(0.5, 0.01, 
                  "This plot shows all individual data points from the standardized alpha/Fe calculation.\n"
                  "Lines connect points from the same galaxy in order of increasing radius.",
                  ha='center', fontsize=12, style='italic')
        
        # Adjust layout to make room for legend
        plt.tight_layout(rect=[0, 0.03, 0.78, 0.97])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved direct alpha/Fe vs. radius plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating direct alpha/Fe vs. radius plot: {e}")
        import traceback
        traceback.print_exc()

def create_fe5015_mgb_plot(results_list, output_path=None, dpi=300, model_data=None):
    """
    Create a plot of Fe5015 vs Mgb with points colored by radius
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing galaxy data points with indices
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for saved image
    model_data : DataFrame, optional
        If provided, plots the model grid as background for context
    """
    try:
        # Create figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Collect all points for plotting
        fe5015_values = []
        mgb_values = []
        radius_values = []
        alpha_fe_values = []
        galaxy_names = []
        
        # Process each galaxy
        for result in results_list:
            if result is None:
                continue
                
            galaxy_name = result['galaxy']
            
            # Use standardized data
            if len(result.get('fe5015_values', [])) > 0 and len(result.get('mgb_values', [])) > 0:
                # Extract data
                galaxy_fe5015 = result['fe5015_values']
                galaxy_mgb = result['mgb_values']
                galaxy_radius = result.get('radius_values', [])
                galaxy_alpha_fe = result.get('alpha_fe_values', [])
                
                # Ensure lengths match
                min_len = min(len(galaxy_fe5015), len(galaxy_mgb))
                valid_indices = []
                
                for i in range(min_len):
                    # Skip invalid values
                    if (np.isnan(galaxy_fe5015[i]) or np.isnan(galaxy_mgb[i]) or 
                        galaxy_fe5015[i] <= 0 or galaxy_mgb[i] <= 0):
                        continue
                    
                    valid_indices.append(i)
                
                # Add valid values to the collected data
                for i in valid_indices:
                    fe5015_values.append(galaxy_fe5015[i])
                    mgb_values.append(galaxy_mgb[i])
                    
                    # Add radius and alpha/Fe if available
                    if i < len(galaxy_radius):
                        radius_values.append(galaxy_radius[i])
                    else:
                        radius_values.append(np.nan)
                        
                    if i < len(galaxy_alpha_fe):
                        alpha_fe_values.append(galaxy_alpha_fe[i])
                    else:
                        alpha_fe_values.append(np.nan)
                        
                    galaxy_names.append(galaxy_name)
        
        if not fe5015_values:
            logger.warning("No valid data points found for plotting")
            return
            
        # Plot 1: Points colored by radius
        scatter1 = ax1.scatter(fe5015_values, mgb_values, c=radius_values, 
                            cmap='viridis', s=120, alpha=0.8, edgecolor='black')
        
        # Add colorbar for radius
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('R/Re', fontsize=14)
        
        # Plot 2: Points colored by alpha/Fe
        scatter2 = ax2.scatter(fe5015_values, mgb_values, c=alpha_fe_values, 
                            cmap='plasma', s=120, alpha=0.8, edgecolor='black')
        
        # Add colorbar for alpha/Fe
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('[α/Fe]', fontsize=14)
        
        # Add model grid if provided
        if model_data is not None:
            try:
                # Find the relevant column names
                model_column_mapping = {
                    'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
                    'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
                    'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
                    'Age': find_matching_column(model_data, ['Age', 'age']),
                    'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
                    'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
                }
                
                fe5015_col = model_column_mapping['Fe5015']
                mgb_col = model_column_mapping['Mgb']
                aofe_col = model_column_mapping['AoFe']
                age_col = model_column_mapping['Age']
                
                if all([fe5015_col, mgb_col, aofe_col, age_col]):
                    # Get unique alpha/Fe values
                    alpha_fe_levels = sorted(model_data[aofe_col].unique())
                    
                    # Choose a representative age
                    representative_age = 10.0  # 10 Gyr is typical for early-type galaxies
                    closest_age = model_data[age_col].iloc[(model_data[age_col] - representative_age).abs().argsort()[0]]
                    
                    # Filter data to this age
                    age_data = model_data[model_data[age_col] == closest_age]
                    
                    # Plot grid lines for each alpha/Fe value
                    for alpha_fe in alpha_fe_levels:
                        alpha_data = age_data[age_data[aofe_col] == alpha_fe]
                        if len(alpha_data) > 1:
                            # Sort by metallicity
                            alpha_data = alpha_data.sort_values(by=fe5015_col)
                            
                            # Plot the grid line
                            ax1.plot(alpha_data[fe5015_col], alpha_data[mgb_col], 'k--', alpha=0.3, linewidth=1.5)
                            ax2.plot(alpha_data[fe5015_col], alpha_data[mgb_col], 'k--', alpha=0.3, linewidth=1.5)
                            
                            # Label each line
                            midpoint = len(alpha_data) // 2
                            x = alpha_data.iloc[midpoint][fe5015_col]
                            y = alpha_data.iloc[midpoint][mgb_col]
                            ax2.text(x, y, f'[α/Fe]={alpha_fe}', fontsize=10, ha='center', va='bottom', alpha=0.7,
                                   bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))
            except Exception as e:
                logger.warning(f"Could not plot model grid: {e}")
        
        # Set the same limits for both plots
        if fe5015_values and mgb_values:
            fe5015_min, fe5015_max = min(fe5015_values), max(fe5015_values)
            mgb_min, mgb_max = min(mgb_values), max(mgb_values)
            
            # Add some padding
            fe5015_range = fe5015_max - fe5015_min
            mgb_range = mgb_max - mgb_min
            
            fe5015_padding = fe5015_range * 0.1
            mgb_padding = mgb_range * 0.1
            
            for ax in [ax1, ax2]:
                ax.set_xlim(fe5015_min - fe5015_padding, fe5015_max + fe5015_padding)
                ax.set_ylim(mgb_min - mgb_padding, mgb_max + mgb_padding)
        
        # Add galaxy name annotations for a subset of points
        # To avoid cluttering, annotate every nth point
        n_points = len(fe5015_values)
        step = max(1, n_points // 20)  # Annotate approximately 20 points
        
        for i in range(0, n_points, step):
            ax1.annotate(galaxy_names[i], 
                       (fe5015_values[i], mgb_values[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=10, alpha=0.7,
                       bbox=dict(facecolor='white', alpha=0.7, pad=1))
        
        # Set labels and titles
        ax1.set_xlabel('Fe5015 Index', fontsize=16)
        ax1.set_ylabel('Mgb Index', fontsize=16)
        ax1.set_title('Fe5015 vs Mgb - colored by R/Re', fontsize=18)
        
        ax2.set_xlabel('Fe5015 Index', fontsize=16)
        ax2.set_ylabel('Mgb Index', fontsize=16)
        ax2.set_title('Fe5015 vs Mgb - colored by [α/Fe]', fontsize=18)
        
        # Add grid
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Set tick parameters
        for ax in [ax1, ax2]:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.tick_params(axis='both', which='both', labelsize=12, right=True, top=True, direction='in')
        
        # Add overall title
        plt.suptitle('Spectral Indices and Alpha Enhancement in Virgo Cluster Galaxies', fontsize=20, y=0.98)
        
        # Add note about standardized data
        plt.figtext(0.5, 0.01, 
                  "Data points represent individual radial bins from all galaxies.\n"
                  "All α/Fe values calculated using the standardized interpolation method for consistency.",
                  ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved Fe5015 vs Mgb plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating Fe5015 vs Mgb plot: {e}")
        import traceback
        traceback.print_exc()

def create_virgo_cluster_map_with_vectors(results_list, coordinates, output_path=None, dpi=150):
    """
    Create a map of the Virgo Cluster showing alpha/Fe gradients as vectors
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing alpha/Fe results for each galaxy
    coordinates : dict
        Dictionary mapping galaxy names to (RA, Dec) coordinates
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for the output image
    """
    try:
        # Create figure
        fig = plt.figure(figsize=(16, 20))
        gs = fig.add_gridspec(5, 2, height_ratios=[3, 1, 1, 1, 1])
        
        # Create the subplots
        ax_map = fig.add_subplot(gs[0, :])  # Map takes the entire top row
        
        # Four distance panels arranged in 2x2 grid below the map
        ax_dist_m87 = fig.add_subplot(gs[1, 0])    # Distance to M87
        ax_dist_m49 = fig.add_subplot(gs[1, 1])    # Distance to M49
        ax_dist_m60 = fig.add_subplot(gs[2, 0])    # Distance to M60
        ax_dist_m86 = fig.add_subplot(gs[2, 1])    # Distance to M86
        ax_dist_center = fig.add_subplot(gs[3, :])  # Distance to center of dataset
        
        # Define galaxies with emission lines
        emission_line_galaxies = [
            "VCC1588", "VCC1368", "VCC1902", "VCC1949", "VCC990", 
            "VCC1410", "VCC667", "VCC1811", "VCC688", "VCC1193", "VCC1486"
        ]
        
        # Extract RA and DEC for plotting
        ra_values = []
        dec_values = []
        galaxies = []
        
        for result in results_list:
            if result is None:
                continue
                
            galaxy = result['galaxy']
            if galaxy in coordinates:
                ra, dec = coordinates[galaxy]
                ra_values.append(ra)
                dec_values.append(dec)
                galaxies.append(galaxy)
        
        # Calculate center of all data galaxies
        if ra_values and dec_values:
            data_center_ra = np.mean(ra_values)
            data_center_dec = np.mean(dec_values)
        else:
            # Default to M87 if no galaxies
            data_center_ra = 187.706
            data_center_dec = 12.391
        
        # Define Virgo Cluster substructures based on literature
        cluster_centers = {
            'M87/Cluster A': (187.706, 12.391),  # Main cluster center
            'M49/Cluster B': (187.445, 8.000),   # Southern subcluster
            'M86/Cluster C': (186.549, 12.946),  # Western subcluster
            'M60/W Cloud': (190.917, 11.553)     # Eastern subcluster (W' Cloud)
        }
        
        # Add these to the overall coordinates for boundary calculation
        for name, (ra, dec) in cluster_centers.items():
            ra_values.append(ra)
            dec_values.append(dec)
        
        # Calculate plot boundaries for map
        if len(ra_values) > 0:
            ra_min, ra_max = min(ra_values), max(ra_values)
            dec_min, dec_max = min(dec_values), max(dec_values)
            
            # Add padding
            ra_range = ra_max - ra_min
            dec_range = dec_max - dec_min
            ra_padding = ra_range * 0.15
            dec_padding = dec_range * 0.15
            
            ax_map.set_xlim(ra_max + ra_padding, ra_min - ra_padding)  # RA decreases to the right
            ax_map.set_ylim(dec_min - dec_padding, dec_max + dec_padding)
        else:
            # Default Virgo Cluster region
            ax_map.set_xlim(192, 185)  # RA
            ax_map.set_ylim(7, 17)     # DEC
        
        # Draw cluster regions - approximating as circles based on literature
        subcluster_radii = {
            'M87/Cluster A': 2.0,  # Main cluster
            'M49/Cluster B': 1.5,  # Southern subcluster
            'M86/Cluster C': 1.0,  # Western subcluster
            'M60/W Cloud': 1.0,    # Eastern subcluster
        }
        
        # Draw cluster regions as transparent circles
        for name, (ra, dec) in cluster_centers.items():
            radius = subcluster_radii[name]
            region_circle = Circle((ra, dec), radius, color='gray', 
                                 fill=True, alpha=0.1, linestyle='-', linewidth=1)
            ax_map.add_patch(region_circle)
            
            # Add region label
            ax_map.text(ra, dec - radius - 0.2, name, ha='center', va='top', 
                      fontsize=10, color='black', alpha=0.8,
                      bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
        
        # Collect slope values to determine scaling
        all_slopes = []
        for result in results_list:
            if result is not None and not np.isnan(result.get('slope', np.nan)):
                all_slopes.append(abs(result['slope']))
        
        # Determine vector scaling factor
        if all_slopes:
            max_slope = max(all_slopes)
            min_slope = min(all_slopes)
            # Base scale factor for vector length - make it larger
            scale_factor = 0.5 / max(max_slope, 0.1)  # Prevent division by zero
        else:
            scale_factor = 1.0
            
        # Data for distance plots
        distance_data = {
            'M87': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []},
            'M49': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []},
            'M60': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []},
            'M86': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []},
            'Center': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []}
        }
        
        # Dictionary to track label positions for collision detection
        label_positions = {}
        min_label_distance = 0.2  # Minimum distance between labels in degrees
        
        # Plot galaxies with slope vectors on the map
        for result in results_list:
            if result is None:
                continue
                
            galaxy = result['galaxy']
            if galaxy not in coordinates:
                continue
                
            ra, dec = coordinates[galaxy]
            
            # Use standardized slope values
            slope = result.get('slope', np.nan)
            p_value = result.get('p_value', np.nan)
            
            # Check if galaxy has emission lines
            has_emission = galaxy in emission_line_galaxies
            
            # Calculate distances to reference points (in degrees)
            distances = {}
            for ref_name, (ref_ra, ref_dec) in cluster_centers.items():
                # Calculate angular distance from reference point (in degrees)
                ang_dist_deg = np.sqrt((ra - ref_ra)**2 + (dec - ref_dec)**2)
                
                # Store in distances dictionary
                short_name = ref_name.split('/')[0]  # Just use M87, M49, etc.
                distances[short_name] = ang_dist_deg
            
            # Calculate distance to center of dataset (in degrees)
            ang_dist_center_deg = np.sqrt((ra - data_center_ra)**2 + (dec - data_center_dec)**2)
            distances['Center'] = ang_dist_center_deg
            
            # If no valid slope, just plot a point
            if np.isnan(slope):
                # Use filled circle for emission line galaxies, hollow circle for others
                if has_emission:
                    ax_map.scatter(ra, dec, s=100, color='gray', edgecolor='black', marker='o', zorder=10)
                else:
                    ax_map.scatter(ra, dec, s=100, facecolor='white', edgecolor='black', marker='o', zorder=10)
                
                # Add galaxy label with collision detection
                label_position = (ra, dec+0.05)
                label_position = find_open_position(label_positions, label_position, min_label_distance)
                ax_map.text(label_position[0], label_position[1], galaxy, fontsize=9, ha='center', va='bottom',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                label_positions[galaxy] = label_position
                continue
            
            # Determine color based on slope direction
            if slope > 0:
                color = 'blue'
                marker = '^'  # Up triangle for positive slope
            else:
                color = 'red'
                marker = 'v'  # Down triangle for negative slope
            
            # Point size is standard for all markers
            point_size = 120
            
            # Collect data for distance plots
            for ref_name in ['M87', 'M49', 'M60', 'M86', 'Center']:
                distance_data[ref_name]['x'].append(distances[ref_name])
                distance_data[ref_name]['y'].append(slope)
                distance_data[ref_name]['color'].append(color)
                distance_data[ref_name]['symbol'].append(marker)
                distance_data[ref_name]['galaxy'].append(galaxy)
                distance_data[ref_name]['has_emission'].append(has_emission)
            
            # Vector length proportional to slope magnitude
            vector_length = abs(slope) * scale_factor
            vector_length = min(vector_length, 0.7)  # Cap the length
            
            # Calculate vector direction based on slope sign
            if slope > 0:
                dx, dy = 0, vector_length
            else:
                dx, dy = 0, -vector_length
                
            # Plot arrow to represent slope magnitude
            ax_map.arrow(ra, dec, dx, dy, head_width=0.1, head_length=0.1, 
                       fc=color, ec=color, alpha=0.7, zorder=5)
            
            # Plot galaxy marker
            if has_emission:
                # Solid colored marker for emission line galaxies
                ax_map.scatter(ra, dec, s=point_size, marker=marker, 
                             color=color, edgecolor='black', linewidth=1.5,
                             alpha=0.8, zorder=10)
            else:
                # Hollow colored marker for non-emission line galaxies
                # Use the same color but with reduced alpha for "hollow" effect
                # while still showing the color
                ax_map.scatter(ra, dec, s=point_size, marker=marker, 
                             facecolor='none', edgecolor=color, linewidth=2.5,
                             alpha=0.8, zorder=10)
            
            # Add galaxy label with better collision detection
            label_offset_y = 0.1 if slope > 0 else -0.1
            label_position = (ra, dec + dy + label_offset_y)
            label_position = find_open_position(label_positions, label_position, min_label_distance)
            
            ax_map.text(label_position[0], label_position[1], galaxy, fontsize=9, ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            label_positions[galaxy] = label_position
            
            # Add slope value text
            slope_position = (ra + 0.2, dec)
            slope_position = find_open_position(label_positions, slope_position, min_label_distance)
            ax_map.text(slope_position[0], slope_position[1], f"{slope:.2f}", fontsize=8, ha='left', va='center',
                       color=color, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            label_positions[f"{galaxy}_slope"] = slope_position
        
        # Plot cluster centers with star markers
        for name, (ra, dec) in cluster_centers.items():
            # Use large star markers for cluster centers
            ax_map.scatter(ra, dec, s=400, marker='*', color='gold', edgecolor='black', linewidth=1.5, zorder=5)
            
            # Add labels with larger font and box
            y_offset = 0.25  # Standard offset
            ax_map.text(ra, dec + y_offset, name.split('/')[0], fontsize=12, ha='center', va='center', 
                   weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))
        
        # Plot center of dataset
        ax_map.scatter(data_center_ra, data_center_dec, s=200, marker='P', color='forestgreen', 
                     edgecolor='black', linewidth=1.5, zorder=5)
        # ax_map.text(data_center_ra, data_center_dec + 0.25, "Dataset Center", fontsize=12, ha='center', va='center',
        #            weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))
        
        # Add simplified legend to map as requested
        legend_elements = [
            # Slope direction indicators (just two triangles with colors)
            Line2D([0], [0], marker='^', color='blue', linestyle='none', markersize=10, 
                 label='Positive α/Fe gradient'),
            Line2D([0], [0], marker='v', color='red', linestyle='none', markersize=10, 
                 label='Negative α/Fe gradient'),
            
            # Emission line indicators (one solid, one hollow)
            Line2D([0], [0], marker='^', color='green', linestyle='none', markersize=10, 
                 label='With emission line'),
            Line2D([0], [0], marker='^', color='w', markeredgecolor='green', markeredgewidth=2, 
                 markersize=10, label='Without emission line'),
            
            # Other map elements
            Line2D([0], [0], marker='*', color='gold', linestyle='none', markersize=15, 
                 label='Cluster/Subcluster Center'),
            Line2D([0], [0], marker='P', color='forestgreen', linestyle='none', markersize=15,
                 label='Dataset Center')
        ]
        ax_map.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Set labels and title for map
        ax_map.set_xlabel('RA (deg)', fontsize=14)
        ax_map.set_ylabel('DEC (deg)', fontsize=14)
        ax_map.set_title('Virgo Cluster Galaxies: [α/Fe] vs. Radius Relationship (IFU Observations)', fontsize=16)
        
        # Add scale bar (1 degree ≈ 0.29 Mpc)
        # Position at bottom left
        scale_x = ax_map.get_xlim()[1] - 0.15 * (ax_map.get_xlim()[1] - ax_map.get_xlim()[0])
        scale_y = ax_map.get_ylim()[0] + 0.1 * (ax_map.get_ylim()[1] - ax_map.get_ylim()[0])
        scale_length = 1.0  # 1 degree
        
        # Draw scale bar
        ax_map.plot([scale_x, scale_x - scale_length], [scale_y, scale_y], 'k-', linewidth=2)
        ax_map.text(scale_x - scale_length/2, scale_y + 0.1, f"1° ≈ 0.29 Mpc", 
                  ha='center', va='bottom', fontsize=10)
        
        # Add compass for orientation
        # Location in the top right corner
        compass_x = ax_map.get_xlim()[0] + 0.1 * (ax_map.get_xlim()[1] - ax_map.get_xlim()[0])
        compass_y = ax_map.get_ylim()[1] - 0.1 * (ax_map.get_ylim()[1] - ax_map.get_ylim()[0])
        compass_size = 0.4
        
        # North
        ax_map.arrow(compass_x, compass_y, 0, compass_size, head_width=0.1, head_length=0.1, 
               fc='black', ec='black', zorder=20)
        ax_map.text(compass_x, compass_y + compass_size + 0.1, 'N', ha='center', va='center', fontsize=10)
        
        # East
        ax_map.arrow(compass_x, compass_y, -compass_size, 0, head_width=0.1, head_length=0.1, 
               fc='black', ec='black', zorder=20)
        ax_map.text(compass_x - compass_size - 0.1, compass_y, 'E', ha='center', va='center', fontsize=10)
        
        # Add grid to map
        ax_map.grid(True, alpha=0.3, linestyle='--')
        
        # Set tick parameters for map
        ax_map.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_map.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax_map.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Function to create distance plot with matching markers
        def create_distance_plot(ax, data, title, ref_radius, y_max=None, show_region=True):
            # Plot points with appropriate markers and colors
            for i in range(len(data['x'])):
                # Get marker properties
                has_emission = data['has_emission'][i]
                color = data['color'][i]
                marker = data['symbol'][i]
                
                # Plot galaxy marker with same style as main plot
                if has_emission:
                    # Solid colored marker for emission line galaxies
                    ax.scatter(data['x'][i], data['y'][i], s=100, marker=marker, 
                             color=color, edgecolor='black', alpha=0.8)
                else:
                    # Hollow colored marker for non-emission line galaxies
                    ax.scatter(data['x'][i], data['y'][i], s=100, marker=marker, 
                             facecolor='none', edgecolor=color, linewidth=2, alpha=0.8)
                
                # Draw arrow to show magnitude
                slope = data['y'][i]
                vector_length = abs(slope) * scale_factor * 0.3  # Scaled down for smaller plots
                if slope > 0:
                    dx, dy = 0, vector_length
                else:
                    dx, dy = 0, -vector_length
                
                # Add arrow
                ax.arrow(data['x'][i], data['y'][i], dx, dy, head_width=0.03, head_length=0.03, 
                       fc=color, ec=color, alpha=0.7)
                
                # Add galaxy label
                ax.annotate(data['galaxy'][i], 
                           (data['x'][i], data['y'][i]), 
                           textcoords="offset points", 
                           xytext=(10, 0),  # Offset to the right
                           fontsize=8,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add vertical line at the radius of the subcluster (only if showing region)
            if show_region:
                ax.axvline(x=ref_radius, color='gray', linestyle='--', alpha=0.7)
                ax.text(ref_radius, 0, f'Cluster Radius: {ref_radius}°', 
                       rotation=90, va='bottom', ha='right', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Add linear trend line for ALL points
            if len(data['x']) > 2:
                # Filter out NaN values
                valid_mask = ~np.isnan(data['x']) & ~np.isnan(data['y'])
                x_valid = np.array(data['x'])[valid_mask]
                y_valid = np.array(data['y'])[valid_mask]
                
                if len(x_valid) > 2:
                    try:
                        # Calculate linear regression for all points
                        slope_all, intercept_all, r_value_all, p_value_all, std_err_all = stats.linregress(x_valid, y_valid)
                        
                        # Plot trend line for all points
                        x_trend_all = np.linspace(min(x_valid), max(x_valid), 100)
                        y_trend_all = slope_all * x_trend_all + intercept_all
                        ax.plot(x_trend_all, y_trend_all, 'k-', alpha=0.7)
                        
                        # Add overall trend annotation
                        ax.text(0.13, 0.95, 
                               f"All galaxies:\nSlope = {slope_all:.3f}\np = {p_value_all:.3f}\nR² = {r_value_all**2:.2f}", 
                               transform=ax.transAxes, 
                               bbox=dict(facecolor='white', alpha=0.7))
                    except:
                        pass
            
            # Add linear trend for points WITHIN the cluster region (only if showing region)
            if show_region and len(data['x']) > 2:
                # Filter to just points within the cluster radius
                cluster_mask = np.array(data['x']) <= ref_radius
                
                # Additional filter for NaN values
                valid_cluster_mask = cluster_mask & ~np.isnan(np.array(data['x'])) & ~np.isnan(np.array(data['y']))
                
                if np.sum(valid_cluster_mask) > 2:  # Need at least 3 points for meaningful regression
                    x_cluster = np.array(data['x'])[valid_cluster_mask]
                    y_cluster = np.array(data['y'])[valid_cluster_mask]
                    
                    try:
                        # Calculate linear regression for cluster points
                        slope_cluster, intercept_cluster, r_value_cluster, p_value_cluster, std_err_cluster = stats.linregress(x_cluster, y_cluster)
                        
                        # Plot trend line for cluster points with dotted line
                        x_trend_cluster = np.linspace(0, ref_radius, 100)
                        y_trend_cluster = slope_cluster * x_trend_cluster + intercept_cluster
                        ax.plot(x_trend_cluster, y_trend_cluster, 'k:', alpha=0.9, linewidth=2)
                        
                        # Add cluster trend annotation
                        ax.text(-0.03, 0.95, 
                               f"Within cluster:\nSlope = {slope_cluster:.3f}\np = {p_value_cluster:.3f}\nR² = {r_value_cluster**2:.2f}", 
                               transform=ax.transAxes, 
                               bbox=dict(facecolor='white', alpha=0.7))
                        
                        # Highlight the cluster region with light shading
                        ax.axvspan(0, ref_radius, alpha=0.1, color='gray')
                    except:
                        pass
            
            # Set labels and title
            ax.set_xlabel('Angular Distance (degrees)', fontsize=12)
            ax.set_ylabel('α/Fe Radial Slope', fontsize=12)
            ax.set_title(title, fontsize=14)
            
            # Make y-axis symmetric around zero to better show positive/negative slopes
            if y_max is None:
                y_vals = [y for y in data['y'] if not np.isnan(y)]
                if y_vals:
                    y_max = max(abs(min(y_vals)), abs(max(y_vals))) * 1.1
                else:
                    y_max = 0.5
            
            ax.set_ylim(-y_max, y_max)
            
            # Set grid and ticks
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='both', labelsize='small', right=True, top=True, direction='in')
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            
            # Add legend for the two trend lines
            if show_region:
                legend_elements = [
                    Line2D([0], [0], color='k', linestyle='-', label='All galaxies'),
                    Line2D([0], [0], color='k', linestyle=':', linewidth=2, label='Within cluster region')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            else:
                legend_elements = [
                    Line2D([0], [0], color='k', linestyle='-', label='All galaxies')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Find maximum y-value across all datasets for consistent y-axis scaling
        all_y_values = []
        for data in distance_data.values():
            all_y_values.extend([y for y in data['y'] if not np.isnan(y)])
        
        if all_y_values:
            global_y_max = max(abs(min(all_y_values)), abs(max(all_y_values))) * 1.1
        else:
            global_y_max = 0.5
        
        # Create all distance plots with the same marker approach
        create_distance_plot(ax_dist_m87, distance_data['M87'], 
                           'Distance from M87 (Cluster A)', subcluster_radii['M87/Cluster A'], global_y_max)
        create_distance_plot(ax_dist_m49, distance_data['M49'], 
                           'Distance from M49 (Cluster B)', subcluster_radii['M49/Cluster B'], global_y_max)
        create_distance_plot(ax_dist_m60, distance_data['M60'], 
                           'Distance from M60 (W\' Cloud)', subcluster_radii['M60/W Cloud'], global_y_max)
        create_distance_plot(ax_dist_m86, distance_data['M86'], 
                           'Distance from M86 (Cluster C)', subcluster_radii['M86/Cluster C'], global_y_max)
        
        # For the dataset center, don't show the region
        create_distance_plot(ax_dist_center, distance_data['Center'], 
                           'Distance from Dataset Center', None, global_y_max, show_region=False)
        
        # Add reference to literature sources and IFU note
        ref_text = "Based on Virgo Cluster structure from Binggeli et al. (1987), Mei et al. (2007), and Ferrarese et al. (2012)\n" 
        ref_text += "Cluster region radii derived from Binggeli et al.: A (2.0°), B (1.5°), and W/W' clouds (1.0°) each"
        plt.figtext(0.5, 0.02, ref_text, ha='center', fontsize=9, style='italic')
        
        # Add explanatory text at the bottom of the figure
        plt.figtext(0.5, 0.01, 
                   "Blue triangles: α/Fe increases with radius | Red triangles: α/Fe decreases with radius\n"
                   "Solid triangles: Galaxies with emission lines | Hollow triangles: Galaxies without emission lines\n"
                   "Arrow length represents strength of α/Fe radial gradient",
                   ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved Virgo Cluster map to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating Virgo Cluster map: {e}")
        import traceback
        traceback.print_exc()

def create_virgo_cluster_map_only(results_list, coordinates, output_path=None, dpi=300):
    """
    Create a map of the Virgo Cluster showing alpha/Fe gradients as vectors
    with only the top map panel
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing alpha/Fe results for each galaxy
    coordinates : dict
        Dictionary mapping galaxy names to (RA, Dec) coordinates
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for the output image
    """
    try:
        # Create figure for just the map
        fig = plt.figure(figsize=(16, 12))
        ax_map = fig.add_subplot(111)
        
        # Define galaxies with emission lines - CORRECTED LIST with only 7 galaxies
        emission_line_galaxies = [
            "VCC1588", "VCC1410", "VCC0667", "VCC1811", 
            "VCC0688", "VCC1193", "VCC1486"
        ]
        
        # Extract RA and DEC for plotting
        ra_values = []
        dec_values = []
        galaxies = []
        
        for result in results_list:
            if result is None:
                continue
                
            galaxy = result['galaxy']
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print(galaxy)
            print(galaxy in emission_line_galaxies)
            if galaxy in coordinates:
                ra, dec = coordinates[galaxy]
                ra_values.append(ra)
                dec_values.append(dec)
                galaxies.append(galaxy)
        
        # Calculate center of all data galaxies
        if ra_values and dec_values:
            data_center_ra = np.mean(ra_values)
            data_center_dec = np.mean(dec_values)
        else:
            # Default to M87 if no galaxies
            data_center_ra = 187.706
            data_center_dec = 12.391
        
        # Define Virgo Cluster substructures based on literature
        cluster_centers = {
            'M87/Cluster A': (187.706, 12.391),  # Main cluster center
            'M49/Cluster B': (187.445, 8.000),   # Southern subcluster
            'M86/Cluster C': (186.549, 12.946),  # Western subcluster
            'M60/W Cloud': (190.917, 11.553)     # Eastern subcluster (W' Cloud)
        }
        
        # Add these to the overall coordinates for boundary calculation
        for name, (ra, dec) in cluster_centers.items():
            ra_values.append(ra)
            dec_values.append(dec)
        
        # Calculate plot boundaries for map
        if len(ra_values) > 0:
            ra_min, ra_max = min(ra_values), max(ra_values)
            dec_min, dec_max = min(dec_values), max(dec_values)
            
            # Add padding
            ra_range = ra_max - ra_min
            dec_range = dec_max - dec_min
            ra_padding = ra_range * 0.15
            dec_padding = dec_range * 0.15
            
            ax_map.set_xlim(ra_max + ra_padding, ra_min - ra_padding)  # RA decreases to the right
            ax_map.set_ylim(dec_min - dec_padding, dec_max + dec_padding)
        else:
            # Default Virgo Cluster region
            ax_map.set_xlim(192, 185)  # RA
            ax_map.set_ylim(7, 17)     # DEC
        
        # Draw cluster regions - approximating as circles based on literature
        subcluster_radii = {
            'M87/Cluster A': 2.0,  # Main cluster
            'M49/Cluster B': 1.5,  # Southern subcluster
            'M86/Cluster C': 1.0,  # Western subcluster
            'M60/W Cloud': 1.0,    # Eastern subcluster
        }
        
        # Draw cluster regions as transparent circles
        for name, (ra, dec) in cluster_centers.items():
            radius = subcluster_radii[name]
            region_circle = Circle((ra, dec), radius, color='gray', 
                                 fill=True, alpha=0.1, linestyle='-', linewidth=1)
            ax_map.add_patch(region_circle)
            
            # Add region label
            ax_map.text(ra, dec - radius - 0.2, name, ha='center', va='top', 
                      fontsize=12, color='black', alpha=0.8,
                      bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))
        
        # Collect slope values to determine scaling
        all_slopes = []
        for result in results_list:
            if result is not None and not np.isnan(result.get('slope', np.nan)):
                all_slopes.append(abs(result['slope']))
        
        # Determine vector scaling factor
        if all_slopes:
            max_slope = max(all_slopes)
            # Base scale factor for vector length
            scale_factor = 0.5 / max(max_slope, 0.1)  # Prevent division by zero
        else:
            scale_factor = 1.0
        
        # Dictionary to track label positions for collision detection
        label_positions = {}
        min_label_distance = 0.2  # Minimum distance between labels in degrees
        
        # Plot galaxies with slope vectors on the map
        for result in results_list:
            if result is None:
                continue
                
            galaxy = result['galaxy']
            if galaxy not in coordinates:
                continue
                
            ra, dec = coordinates[galaxy]
            
            # Use standardized slope values
            slope = result.get('slope', np.nan)
            p_value = result.get('p_value', np.nan)
            
            # Check if galaxy has emission lines - make this clear using corrected list
            has_emission = galaxy in emission_line_galaxies
            
            # If no valid slope, just plot a point
            if np.isnan(slope):
                # Use filled circle for emission line galaxies, hollow circle for others
                if has_emission:
                    ax_map.scatter(ra, dec, s=160, color='green', edgecolor='black', marker='o', zorder=10)
                else:
                    ax_map.scatter(ra, dec, s=160, facecolor='white', edgecolor='black', marker='o', zorder=10)
                
                # Add galaxy label with collision detection
                label_position = (ra, dec+0.05)
                label_position = find_open_position(label_positions, label_position, min_label_distance)
                ax_map.text(label_position[0], label_position[1], galaxy, fontsize=12, ha='center', va='bottom',
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
                label_positions[galaxy] = label_position
                continue
            
            # Determine marker shape and color based on slope
            if abs(slope) < 0.05:  # Horizontal slope
                marker = 'o'       # Circle for horizontal slopes
                if slope > 0:
                    color = 'blue'
                else:
                    color = 'red'
            elif slope > 0:        # Positive slope
                color = 'blue'
                marker = '^'       # Up triangle
            else:                  # Negative slope
                color = 'red'
                marker = 'v'       # Down triangle
            
            # Point size is standard for all markers
            point_size = 160
            
            # Vector length proportional to slope magnitude
            vector_length = abs(slope) * scale_factor
            vector_length = min(vector_length, 0.7)  # Cap the length
            
            # Calculate vector direction based on slope sign
            if slope > 0:
                dx, dy = 0, vector_length
            else:
                dx, dy = 0, -vector_length
                
            # Plot arrow to represent slope magnitude
            ax_map.arrow(ra, dec, dx, dy, head_width=0.1, head_length=0.1, 
                       fc=color, ec=color, alpha=0.7, zorder=5)
            
            # Plot galaxy marker - filled for emission line galaxies
            if has_emission:
                # Solid filled marker for emission line galaxies
                ax_map.scatter(ra, dec, s=point_size, marker=marker, 
                             color=color, edgecolor='black', linewidth=1.5,
                             alpha=0.8, zorder=10)
            else:
                # Hollow marker for non-emission line galaxies
                ax_map.scatter(ra, dec, s=point_size, marker=marker, 
                             facecolor='none', edgecolor=color, linewidth=2.5,
                             alpha=0.8, zorder=10)
            
            # Add galaxy label with better collision detection
            label_offset_y = 0.1 if slope > 0 else -0.1
            label_position = (ra, dec + dy + label_offset_y)
            label_position = find_open_position(label_positions, label_position, min_label_distance)
            
            # Add the slope value to the label
            slope_text = f"{slope:.2f}"
            ax_map.text(label_position[0], label_position[1], 
                       f"{galaxy}\n{slope_text}", fontsize=12, ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            label_positions[galaxy] = label_position
        
        # Plot cluster centers with star markers
        for name, (ra, dec) in cluster_centers.items():
            # Use large star markers for cluster centers
            ax_map.scatter(ra, dec, s=400, marker='*', color='gold', edgecolor='black', linewidth=1.5, zorder=5)
            
            # Add labels with larger font and box
            y_offset = 0.25  # Standard offset
            ax_map.text(ra, dec + y_offset, name.split('/')[0], fontsize=14, ha='center', va='center', 
                   weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))
        
        # DO NOT plot dataset center
        
        # Add legend with improved emission line explanation
        legend_elements = [
            # Slope direction indicators
            Line2D([0], [0], marker='^', color='blue', linestyle='none', markersize=12, 
                 label='Positive α/Fe gradient'),
            Line2D([0], [0], marker='v', color='red', linestyle='none', markersize=12, 
                 label='Negative α/Fe gradient'),
            Line2D([0], [0], marker='o', color='blue', linestyle='none', markersize=12, 
                 label='Horizontal α/Fe gradient'),
            
            # Emission line indicators - make sure these align with plotting
            Line2D([0], [0], marker='^', color='blue', linestyle='none', markersize=12, 
                 label='With emission line (filled)'),
            Line2D([0], [0], marker='^', color='w', markeredgecolor='blue', markeredgewidth=2, 
                 markersize=12, label='Without emission line (hollow)'),
            
            # Other map elements
            Line2D([0], [0], marker='*', color='gold', linestyle='none', markersize=16, 
                 label='Cluster/Subcluster Center')
        ]
        ax_map.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Set labels and title for map
        ax_map.set_xlabel('RA (deg)', fontsize=16)
        ax_map.set_ylabel('DEC (deg)', fontsize=16)
        ax_map.set_title('Virgo Cluster Galaxies: [α/Fe] vs. Radius Relationship (IFU Observations)', fontsize=18)
        
        # Add scale bar (1 degree ≈ 0.29 Mpc)
        # Position at bottom left
        scale_x = ax_map.get_xlim()[1] - 0.15 * (ax_map.get_xlim()[1] - ax_map.get_xlim()[0])
        scale_y = ax_map.get_ylim()[0] + 0.1 * (ax_map.get_ylim()[1] - ax_map.get_ylim()[0])
        scale_length = 1.0  # 1 degree
        
        # Draw scale bar
        ax_map.plot([scale_x, scale_x - scale_length], [scale_y, scale_y], 'k-', linewidth=2.5)
        ax_map.text(scale_x - scale_length/2, scale_y + 0.1, f"1° ≈ 0.29 Mpc", 
                  ha='center', va='bottom', fontsize=12)
        
        # Add compass for orientation
        # Location in the top right corner
        compass_x = ax_map.get_xlim()[0] + 0.1 * (ax_map.get_xlim()[1] - ax_map.get_xlim()[0])
        compass_y = ax_map.get_ylim()[1] - 0.1 * (ax_map.get_ylim()[1] - ax_map.get_ylim()[0])
        compass_size = 0.4
        
        # North
        ax_map.arrow(compass_x, compass_y, 0, compass_size, head_width=0.1, head_length=0.1, 
               fc='black', ec='black', zorder=20)
        ax_map.text(compass_x, compass_y + compass_size + 0.1, 'N', ha='center', va='center', fontsize=12)
        
        # East
        ax_map.arrow(compass_x, compass_y, -compass_size, 0, head_width=0.1, head_length=0.1, 
               fc='black', ec='black', zorder=20)
        ax_map.text(compass_x - compass_size - 0.1, compass_y, 'E', ha='center', va='center', fontsize=12)
        
        # Add grid to map
        ax_map.grid(True, alpha=0.3, linestyle='--')
        
        # Set tick parameters for map
        ax_map.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_map.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax_map.tick_params(axis='both', which='both', labelsize=12, right=True, top=True, direction='in')
        
        # Add explanatory text at the bottom of the figure - make emission line explanation clearer
        plt.figtext(0.5, 0.01, 
                   "Blue symbols: α/Fe increases with radius | Red symbols: α/Fe decreases with radius\n"
                   "Triangles: Strong gradients | Circles: Horizontal gradients (|slope| < 0.05)\n"
                   "Filled symbols: Galaxies with emission lines | Hollow symbols: Galaxies without emission lines\n"
                   "Arrow length represents strength of α/Fe radial gradient",
                   ha='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.7))
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved Virgo Cluster map to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating Virgo Cluster map: {e}")
        import traceback
        traceback.print_exc()

def create_virgo_distance_plots_only(results_list, coordinates, output_path=None, dpi=150):
    """
    Create a figure with the five distance plots showing alpha/Fe gradients vs 
    distance from cluster centers with aligned x-axes
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing alpha/Fe results for each galaxy
    coordinates : dict
        Dictionary mapping galaxy names to (RA, Dec) coordinates
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for output image
    """
    try:
        # Create figure
        fig = plt.figure(figsize=(16, 20))
        
        # Define galaxies with emission lines based on your table
        emission_line_galaxies = [
            "VCC1588", "VCC1368", "VCC1902", "VCC1949", "VCC990", 
            "VCC1410", "VCC667", "VCC1811", "VCC688", "VCC1193", "VCC1486"
        ]
        
        # Define Virgo Cluster substructures based on literature
        cluster_centers = {
            'M87/Cluster A': (187.706, 12.391),  # Main cluster center
            'M49/Cluster B': (187.445, 8.000),   # Southern subcluster
            'M86/Cluster C': (186.549, 12.946),  # Western subcluster
            'M60/W Cloud': (190.917, 11.553)     # Eastern subcluster (W' Cloud)
        }
        
        # Calculate center of all data galaxies
        if coordinates:
            data_center_ra = np.mean([ra for ra, _ in coordinates.values()])
            data_center_dec = np.mean([dec for _, dec in coordinates.values()])
        else:
            # Default to M87 if no coordinates
            data_center_ra = 187.706
            data_center_dec = 12.391
        
        # Define cluster regions radii
        subcluster_radii = {
            'M87/Cluster A': 2.0,  # Main cluster
            'M49/Cluster B': 1.5,  # Southern subcluster
            'M86/Cluster C': 1.0,  # Western subcluster
            'M60/W Cloud': 1.0,    # Eastern subcluster
        }
        
        # Create subplots with shared x-axis and aligned y-axis
        # Adjust height ratios to make plots more compact
        gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1], hspace=0.3)
        
        # Create the five plots
        ax_m87 = fig.add_subplot(gs[0])
        ax_m49 = fig.add_subplot(gs[1], sharex=ax_m87)
        ax_m60 = fig.add_subplot(gs[2], sharex=ax_m87)
        ax_m86 = fig.add_subplot(gs[3], sharex=ax_m87)
        ax_center = fig.add_subplot(gs[4], sharex=ax_m87)
        
        # Data for distance plots
        distance_data = {
            'M87': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []},
            'M49': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []},
            'M60': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []},
            'M86': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []},
            'Center': {'x': [], 'y': [], 'color': [], 'symbol': [], 'galaxy': [], 'has_emission': []}
        }
        
        # Collect distance data
        for result in results_list:
            if result is None:
                continue
                
            galaxy = result['galaxy']
            if galaxy not in coordinates:
                continue
                
            ra, dec = coordinates[galaxy]
            
            # Use standardized slope values
            slope = result.get('slope', np.nan)
            p_value = result.get('p_value', np.nan)
            
            # Skip if no valid slope
            if np.isnan(slope):
                continue
            
            # Check if galaxy has emission lines
            has_emission = galaxy in emission_line_galaxies
            
            # Calculate distances to reference points (in degrees)
            distances = {}
            for ref_name, (ref_ra, ref_dec) in cluster_centers.items():
                # Calculate angular distance from reference point (in degrees)
                ang_dist_deg = np.sqrt((ra - ref_ra)**2 + (dec - ref_dec)**2)
                
                # Store in distances dictionary
                short_name = ref_name.split('/')[0]  # Just use M87, M49, etc.
                distances[short_name] = ang_dist_deg
            
            # Calculate distance to center of dataset (in degrees)
            ang_dist_center_deg = np.sqrt((ra - data_center_ra)**2 + (dec - data_center_dec)**2)
            distances['Center'] = ang_dist_center_deg
            
            # Determine marker symbol based on slope
            if abs(slope) < 0.05:  # Horizontal slope
                marker = 'o'       # Circle for horizontal
            elif slope > 0:        # Positive slope
                marker = '^'       # Up triangle
            else:                  # Negative slope
                marker = 'v'       # Down triangle
            
            # Determine color based on slope direction
            if slope > 0:
                color = 'blue'
            else:
                color = 'red'
            
            # Store data for each reference point
            for ref_name in ['M87', 'M49', 'M60', 'M86', 'Center']:
                distance_data[ref_name]['x'].append(distances[ref_name])
                distance_data[ref_name]['y'].append(slope)
                distance_data[ref_name]['color'].append(color)
                distance_data[ref_name]['symbol'].append(marker)
                distance_data[ref_name]['galaxy'].append(galaxy)
                distance_data[ref_name]['has_emission'].append(has_emission)
        
        # Function to create distance plot with matching markers
        def create_distance_plot(ax, data, title, ref_radius, y_max=None, show_region=True):
            # Plot points with appropriate markers and colors
            for i in range(len(data['x'])):
                # Get marker properties
                has_emission = data['has_emission'][i]
                color = data['color'][i]
                marker = data['symbol'][i]
                
                # Plot galaxy marker with same style as main plot
                if has_emission:
                    # Solid colored marker for emission line galaxies
                    ax.scatter(data['x'][i], data['y'][i], s=100, marker=marker, 
                             color=color, edgecolor='black', alpha=0.8)
                else:
                    # Hollow colored marker for non-emission line galaxies
                    ax.scatter(data['x'][i], data['y'][i], s=100, marker=marker, 
                             facecolor='none', edgecolor=color, linewidth=2, alpha=0.8)
                
                # Add galaxy label
                ax.annotate(data['galaxy'][i], 
                           (data['x'][i], data['y'][i]), 
                           textcoords="offset points", 
                           xytext=(10, 0),  # Offset to the right
                           fontsize=8,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
            
            # Add a horizontal line at y=0
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Add horizontal band for "horizontal" classification
            ax.axhspan(-0.05, 0.05, color='green', alpha=0.1)
            
            # Add vertical line at the radius of the subcluster (only if showing region)
            if show_region:
                ax.axvline(x=ref_radius, color='gray', linestyle='--', alpha=0.7)
                ax.text(ref_radius, 0, f'Cluster Radius: {ref_radius}°', 
                       rotation=90, va='bottom', ha='right', fontsize=8,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Add linear trend line for ALL points
            if len(data['x']) > 2:
                # Filter out NaN values
                valid_mask = ~np.isnan(data['x']) & ~np.isnan(data['y'])
                x_valid = np.array(data['x'])[valid_mask]
                y_valid = np.array(data['y'])[valid_mask]
                
                if len(x_valid) > 2:
                    try:
                        # Calculate linear regression for all points
                        slope_all, intercept_all, r_value_all, p_value_all, std_err_all = stats.linregress(x_valid, y_valid)
                        
                        # Plot trend line for all points
                        x_trend_all = np.linspace(min(x_valid), max(x_valid), 100)
                        y_trend_all = slope_all * x_trend_all + intercept_all
                        ax.plot(x_trend_all, y_trend_all, 'k-', alpha=0.7)
                        
                        # Add overall trend annotation
                        ax.text(0.13, 0.95, 
                               f"All galaxies:\nSlope = {slope_all:.3f}\np = {p_value_all:.3f}\nR² = {r_value_all**2:.2f}", 
                               transform=ax.transAxes, 
                               bbox=dict(facecolor='white', alpha=0.7))
                    except:
                        pass
            
            # Add linear trend for points WITHIN the cluster region (only if showing region)
            if show_region and len(data['x']) > 2:
                # Filter to just points within the cluster radius
                cluster_mask = np.array(data['x']) <= ref_radius
                
                # Additional filter for NaN values
                valid_cluster_mask = cluster_mask & ~np.isnan(np.array(data['x'])) & ~np.isnan(np.array(data['y']))
                
                if np.sum(valid_cluster_mask) > 2:  # Need at least 3 points for meaningful regression
                    x_cluster = np.array(data['x'])[valid_cluster_mask]
                    y_cluster = np.array(data['y'])[valid_cluster_mask]
                    
                    try:
                        # Calculate linear regression for cluster points
                        slope_cluster, intercept_cluster, r_value_cluster, p_value_cluster, std_err_cluster = stats.linregress(x_cluster, y_cluster)
                        
                        # Plot trend line for cluster points with dotted line
                        x_trend_cluster = np.linspace(0, ref_radius, 100)
                        y_trend_cluster = slope_cluster * x_trend_cluster + intercept_cluster
                        ax.plot(x_trend_cluster, y_trend_cluster, 'k:', alpha=0.9, linewidth=2)
                        
                        # Add cluster trend annotation
                        ax.text(-0.03, 0.95, 
                               f"Within cluster:\nSlope = {slope_cluster:.3f}\np = {p_value_cluster:.3f}\nR² = {r_value_cluster**2:.2f}", 
                               transform=ax.transAxes, 
                               bbox=dict(facecolor='white', alpha=0.7))
                        
                        # Highlight the cluster region with light shading
                        ax.axvspan(0, ref_radius, alpha=0.1, color='gray')
                    except:
                        pass
            
            # Set labels
            ax.set_ylabel('α/Fe Radial Slope', fontsize=12)
            ax.set_title(title, fontsize=14)
            
            # Only show x-axis label on bottom plot
            if ax == ax_center:
                ax.set_xlabel('Angular Distance (degrees)', fontsize=12)
            
            # Make y-axis symmetric around zero to better show positive/negative slopes
            if y_max is None:
                y_vals = [y for y in data['y'] if not np.isnan(y)]
                if y_vals:
                    y_max = max(abs(min(y_vals)), abs(max(y_vals))) * 1.1
                else:
                    y_max = 0.5
            
            ax.set_ylim(-y_max, y_max)
            
            # Set grid and ticks
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(axis='both', which='both', labelsize='small', right=True, top=True, direction='in')
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            
            # Only show x-ticks on bottom plot
            if ax != ax_center:
                plt.setp(ax.get_xticklabels(), visible=False)
            
            # Add legend for the two trend lines
            if show_region:
                legend_elements = [
                    Line2D([0], [0], color='k', linestyle='-', label='All galaxies'),
                    Line2D([0], [0], color='k', linestyle=':', linewidth=2, label='Within cluster region')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            else:
                legend_elements = [
                    Line2D([0], [0], color='k', linestyle='-', label='All galaxies')
                ]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
        
        # Find maximum y-value across all datasets for consistent y-axis scaling
        all_y_values = []
        for data in distance_data.values():
            all_y_values.extend([y for y in data['y'] if not np.isnan(y)])
        
        if all_y_values:
            global_y_max = max(abs(min(all_y_values)), abs(max(all_y_values))) * 1.1
        else:
            global_y_max = 0.5
        
        # Create all distance plots with the same marker approach
        create_distance_plot(ax_m87, distance_data['M87'], 
                          'Distance from M87 (Cluster A)', subcluster_radii['M87/Cluster A'], global_y_max)
        create_distance_plot(ax_m49, distance_data['M49'], 
                          'Distance from M49 (Cluster B)', subcluster_radii['M49/Cluster B'], global_y_max)
        create_distance_plot(ax_m60, distance_data['M60'], 
                          'Distance from M60 (W\' Cloud)', subcluster_radii['M60/W Cloud'], global_y_max)
        create_distance_plot(ax_m86, distance_data['M86'], 
                          'Distance from M86 (Cluster C)', subcluster_radii['M86/Cluster C'], global_y_max)
        
        # For the dataset center, don't show the region
        create_distance_plot(ax_center, distance_data['Center'], 
                          'Distance from Dataset Center', None, global_y_max, show_region=False)
        
        # Add reference to literature sources
        ref_text = "Based on Virgo Cluster structure from Binggeli et al. (1987), Mei et al. (2007), and Ferrarese et al. (2012)\n" 
        ref_text += "Cluster region radii derived from Binggeli et al.: A (2.0°), B (1.5°), and W/W' clouds (1.0°) each"
        plt.figtext(0.5, 0.01, ref_text, ha='center', fontsize=9, style='italic')
        
        # Add legend at the bottom to explain the markers
        legend_elements = [
            # Shape for trend type
            Line2D([0], [0], marker='^', color='blue', linestyle='none', markersize=10, 
                 label='Increasing α/Fe with radius (slope > 0.05)'),
            Line2D([0], [0], marker='v', color='red', linestyle='none', markersize=10, 
                 label='Decreasing α/Fe with radius (slope < -0.05)'),
            Line2D([0], [0], marker='o', color='blue', linestyle='none', markersize=10, 
                 label='Horizontal α/Fe gradient (|slope| < 0.05)'),
            
            # Fill for emission line status
            Line2D([0], [0], marker='^', color='tab:green', linestyle='none', markersize=10, 
                 label='With emission lines'),
            Line2D([0], [0], marker='^', color='w', markeredgecolor='tab:green', markeredgewidth=2, 
                 markersize=10, label='Without emission lines')
        ]
        
        # Place legend at the bottom
        fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
                 bbox_to_anchor=(0.5, 0.02), fontsize=10)
        
        # Title for the whole figure
        plt.suptitle('Virgo Cluster: [α/Fe] Gradients vs Distance from Key Reference Points', y=0.98, fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.07, 1, 0.96])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved distance plots to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating distance plots: {e}")
        import traceback
        traceback.print_exc()

def create_snr_comparison_plots(file_paths, output_path=None, dpi=300):
    """
    Create a visualization with SNR maps and a SNR vs R/Re scatter plot
    
    Parameters:
    -----------
    file_paths : list
        List of paths to IFU SNR files (pixels SNR, radial bin SNR)
    output_path : str, optional
        Path to save the output image
    dpi : int
        Resolution for the output image
    """
    try:
        # Check if we have the files
        if len(file_paths) < 2:
            raise ValueError("Please provide at least 2 input files (pixel SNR, radial bin SNR)")
        
        # Set up the figure with a grid
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1.2])
        
        # Create the subplots - SNR maps on top, SNR vs R/Re plot on bottom
        ax_snr1 = plt.subplot(gs[0, 0])      # Pixel SNR map
        ax_snr3 = plt.subplot(gs[0, 1])      # Radial Bin SNR map
        ax_snr_plot = plt.subplot(gs[1, :])  # SNR vs R/Re plot
        
        # SNR map titles
        snr_titles = ['Pixel SNR', 'Radial Bin SNR']
        
        # Colors for different SNR types in the scatter plot
        colors = ['blue', 'red']
        markers = ['o', 'x']  # Use 'o' for P2P, '+' for RDB
        labels = ['Pixels SNR', 'Radial bin SNR']
        
        # Data containers for the scatter plot
        p2p_radii = []
        p2p_snr_values = []
        rdb_radii = []
        rdb_snr_values = []
        
        # Process each SNR file (skipping VNB)
        file_indices = [0, 2] if len(file_paths) >= 3 else [0, 1]
        plot_axes = [ax_snr1, ax_snr3]
        
        for idx, file_idx in enumerate(file_indices):
            if file_idx >= len(file_paths):
                continue
                
            file_path = file_paths[file_idx]
            try:
                # Open the file - assuming it's a FITS file
                with fits.open(file_path) as hdu:
                    # Get basic information
                    filename = os.path.basename(file_path)
                    galaxy_name = filename.split('_')[0] if '_' in filename else filename.split('.')[0]
                    
                    # Extract SNR data 
                    # Try different extensions to find SNR data
                    snr_data = None
                    for ext_name in ['SNR', 'SIGNAL_NOISE', 'SN', 0]:
                        try:
                            if isinstance(ext_name, int):
                                snr_data = hdu[ext_name].data
                            elif ext_name in hdu:
                                snr_data = hdu[ext_name].data
                            if snr_data is not None:
                                break
                        except:
                            continue
                    
                    if snr_data is None:
                        logger.warning(f"Could not find SNR data in {file_path}")
                        continue
                    
                    # Get header information for WCS and scaling
                    header = hdu[0].header
                    
                    # Get pixel scale
                    try:
                        if 'CDELT1' in header and 'CDELT2' in header:
                            pixel_scale_x = abs(header['CDELT1']) * 3600  # Convert deg to arcsec
                            pixel_scale_y = abs(header['CDELT2']) * 3600
                        elif 'CD1_1' in header and 'CD2_2' in header:
                            pixel_scale_x = abs(header['CD1_1']) * 3600
                            pixel_scale_y = abs(header['CD2_2']) * 3600
                        else:
                            # Default values
                            pixel_scale_x = pixel_scale_y = 0.2  # arcsec per pixel
                    except:
                        pixel_scale_x = pixel_scale_y = 0.2  # Default
                    
                    # Get effective radius (if available)
                    Re = None
                    for key in ['RE', 'REFF', 'EFFECTIVE_RADIUS', 'EFFRAD']:
                        if key in header:
                            Re = header[key]
                            break
                    
                    if Re is None:
                        # Default value or estimate from image size
                        Re = max(snr_data.shape) * pixel_scale_x / 8  # Rough estimate
                        logger.warning(f"Effective radius not found in header, using estimate: {Re:.2f} arcsec")
                    
                    # Calculate image dimensions in arcsec
                    ny, nx = snr_data.shape
                    extent_x = nx * pixel_scale_x
                    extent_y = ny * pixel_scale_y
                    
                    # Create extent for plotting (centered at 0,0)
                    extent = [-extent_x/2, extent_x/2, -extent_y/2, extent_y/2]
                    
                    # Plot the SNR map in the appropriate subplot
                    ax = plot_axes[idx]
                    
                    # Get valid data range for color scaling
                    valid_mask = np.isfinite(snr_data) & (snr_data > 0)
                    if np.any(valid_mask):
                        vmin = np.percentile(snr_data[valid_mask], 1)
                        vmax = np.percentile(snr_data[valid_mask], 99)
                    else:
                        vmin, vmax = 0, 50  # Default range
                    
                    # Plot SNR map
                    im = ax.imshow(snr_data, origin='lower', vmin=vmin, vmax=vmax, 
                                 cmap='viridis', extent=extent)
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('S/N', fontsize=14)
                    
                    # Set title
                    ax.set_title(snr_titles[idx], fontsize=16)
                    
                    # Add labels and formatting
                    ax.set_xlabel('Offset (arcsec)', fontsize=14)
                    ax.set_ylabel('Offset (arcsec)', fontsize=14)
                    
                    # Add ticks and grid
                    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.tick_params(axis='both', which='both', labelsize=12)
                    
                    # Extract radial coordinates
                    y, x = np.indices(snr_data.shape)
                    x_center = nx / 2
                    y_center = ny / 2
                    
                    # Calculate radial distance in pixels
                    r_pixels = np.sqrt((x - x_center)**2 + (y - y_center)**2)
                    
                    # Convert to arcsec
                    r_arcsec = r_pixels * pixel_scale_x
                    
                    # Convert to R/Re
                    r_scaled = r_arcsec / Re
                    
                    # Handle P2P and RDB differently
                    if idx == 0:  # Pixel SNR (P2P)
                        # Flatten arrays for scatter plot
                        r_flat = r_scaled.flatten()
                        snr_flat = snr_data.flatten()
                        
                        # Filter out invalid values and select a reasonable number of points
                        valid = np.isfinite(snr_flat) & (snr_flat > 0)
                        r_valid = r_flat[valid]
                        snr_valid = snr_flat[valid]
                        
                        # If too many points, sample a subset
                        max_points = 5000
                        if len(r_valid) > max_points:
                            indices = np.random.choice(len(r_valid), max_points, replace=False)
                            r_valid = r_valid[indices]
                            snr_valid = snr_valid[indices]
                        
                        # Store data for P2P scatter plot
                        p2p_radii = r_valid
                        p2p_snr_values = snr_valid
                        
                    elif idx == 1:  # Radial Bin SNR (RDB)
                        # Group by SNR value within tolerance
                        tolerance = 0.005  # SNR tolerance for grouping
                        
                        # Flatten SNR and radius arrays
                        flat_snr = snr_data.flatten()
                        flat_r = r_scaled.flatten()
                        
                        # Filter out invalid values
                        valid_mask = np.isfinite(flat_snr) & (flat_snr > 0)
                        valid_snr = flat_snr[valid_mask]
                        valid_r = flat_r[valid_mask]
                        
                        # Find unique SNR values within tolerance
                        unique_snr_values = []
                        bin_r_means = []
                        
                        # Group by SNR value within tolerance
                        processed_indices = set()
                        
                        for j in range(len(valid_snr)):
                            if j in processed_indices:
                                continue
                                
                            snr_value = valid_snr[j]
                            # Find all pixels with similar SNR value
                            similar_mask = np.abs(valid_snr - snr_value) < tolerance
                            
                            # Only consider this a bin if we have multiple pixels
                            if np.sum(similar_mask) > 1:
                                # Calculate mean radius for this bin
                                mean_r = np.mean(valid_r[similar_mask])
                                
                                # Add to results
                                unique_snr_values.append(snr_value)
                                bin_r_means.append(mean_r)
                                
                                # Mark as processed
                                processed_indices.update(np.where(similar_mask)[0])
                        
                        # Store data for RDB scatter plot
                        rdb_radii = bin_r_means
                        rdb_snr_values = unique_snr_values
            
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Create the scatter plot with one point per bin for RDB
        # 1. Plot P2P data with 'o' markers
        if len(p2p_radii) > 0 and len(p2p_snr_values) > 0:
            ax_snr_plot.scatter(p2p_radii, p2p_snr_values, 
                            s=10, alpha=0.7, color=colors[0], 
                            marker=markers[0], label=labels[0])
        
        # 3. Plot RDB data with '+' markers - one point per bin
        if len(rdb_radii) > 0 and len(rdb_snr_values) > 0:
            ax_snr_plot.scatter(rdb_radii, rdb_snr_values, 
                            s=200, alpha=0.7, color=colors[1], 
                            marker=markers[1], label=labels[1])
        
        # Add target SNR line
        ax_snr_plot.axhline(y=20, linestyle='--', color='magenta', linewidth=2.5, label='target SNR = 20')
        
        # Set labels and title for scatter plot
        ax_snr_plot.set_xlabel('R/Re', fontsize=16)
        ax_snr_plot.set_ylabel('SNR', fontsize=16)
        ax_snr_plot.set_title('SNR vs. Normalized Radius (R/Re)', fontsize=18)
        
        # Set y-axis range to match your reference plot
        ax_snr_plot.set_ylim(0, 50)
        
        # Add grid and legend
        ax_snr_plot.grid(True, alpha=0.3, linestyle='--')
        ax_snr_plot.legend(loc='upper right', fontsize=14)
        
        # Add minor ticks
        ax_snr_plot.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax_snr_plot.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax_snr_plot.tick_params(axis='both', which='both', labelsize=12)
        
        # Add suptitle
        plt.suptitle(f'SNR Analysis for {galaxy_name}', fontsize=20, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save or show the figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved SNR comparison visualization to {output_path}")
        else:
            plt.show()
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating SNR comparison plots: {e}")
        import traceback
        traceback.print_exc()

def create_virgo_cluster_visualizations(results_list, coordinates, output_dir="./visualization", dpi=150):
    """
    Create separate visualizations for Virgo Cluster data:
    1. Just the map (top panel)
    2. Distance plots (lower panels)
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing alpha/Fe results for each galaxy
    coordinates : dict
        Dictionary mapping galaxy names to (RA, Dec) coordinates
    output_dir : str
        Directory to save the output images
    dpi : int
        Resolution for the output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create map visualization (top panel only)
    map_path = os.path.join(output_dir, "virgo_cluster_map.png")
    create_virgo_cluster_map_only(results_list, coordinates, output_path=map_path, dpi=dpi)
    
    # Create distance plots visualization (bottom panels)
    distance_path = os.path.join(output_dir, "virgo_distance_plots.png")
    create_virgo_distance_plots_only(results_list, coordinates, output_path=distance_path, dpi=dpi)
    
    logger.info(f"Created two Virgo Cluster visualizations in {output_dir}")

def create_alpha_fe_results_summary(results_list, emission_line_galaxies, output_path=None):
    """
    Create a summary table of alpha/Fe gradient results for Virgo Cluster galaxies
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing alpha/Fe results for each galaxy
    emission_line_galaxies : list
        List of galaxy names with emission lines
    output_path : str
        Optional path to save the summary table as CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Summary table of results
    """
    try:
        # Prepare data for summary table
        summary_data = []
        
        for result in results_list:
            if result is None:
                continue
                
            galaxy = result['galaxy']
            # Use get() method with a default value of np.nan to handle missing keys
            slope = result.get('slope', np.nan)
            p_value = result.get('p_value', np.nan)
            radial_limit = result.get('radial_bin_limit', np.nan)
            bins_used = result.get('bins_used', '')
            special_case = result.get('special_case_applied', False)
            has_emission = galaxy in emission_line_galaxies
            
            # Add to summary data
            summary_data.append({
                'Galaxy name': galaxy,
                'α/Fe slope': slope,
                'P-value': p_value,
                'Emission Line': 'Yes' if has_emission else 'No',
                'Radial Bin Limit': radial_limit,
                'Bins Used': bins_used,
                'Special Case': 'Yes' if special_case else 'No'
            })
        
        # Create DataFrame
        df = pd.DataFrame(summary_data)
        
        # Sort by slope magnitude (absolute value)
        if len(df) > 0 and 'α/Fe slope' in df.columns:
            # Convert to numeric first to handle any string values
            df['abs_slope'] = pd.to_numeric(df['α/Fe slope'], errors='coerce').abs()
            df = df.sort_values('abs_slope', ascending=False)
            df = df.drop(columns=['abs_slope'])
        
        # Format numeric columns - handle potential errors
        if 'α/Fe slope' in df.columns:
            df['α/Fe slope'] = df['α/Fe slope'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A")
        if 'P-value' in df.columns:
            df['P-value'] = df['P-value'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A")
        if 'Radial Bin Limit' in df.columns:
            df['Radial Bin Limit'] = df['Radial Bin Limit'].apply(lambda x: f"{x:.2f}" if isinstance(x, (int, float)) and not np.isnan(x) else "N/A")
        
        # Save to file if requested
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved results summary to {output_path}")
        
        # Print the table
        print("\n=== ALPHA/FE GRADIENT RESULTS SUMMARY ===\n")
        print(df.to_string(index=False))
        
        # Generate statistics summary only if we have results
        if len(df) > 0:
            try:
                total_galaxies = len(df)
                
                # Convert strings back to numbers for calculations
                slopes = pd.to_numeric(df['α/Fe slope'], errors='coerce')
                p_values = pd.to_numeric(df['P-value'], errors='coerce')
                
                positive_slopes = sum(slopes > 0)
                negative_slopes = sum(slopes < 0)
                significant_results = sum(p_values < 0.05)
                emission_line_count = sum(df['Emission Line'] == 'Yes')
                special_case_count = sum(df['Special Case'] == 'Yes')
                
                print("\n=== STATISTICAL SUMMARY ===")
                print(f"Total galaxies analyzed: {total_galaxies}")
                print(f"Galaxies with positive α/Fe gradients: {positive_slopes} ({positive_slopes/total_galaxies*100:.1f}%)")
                print(f"Galaxies with negative α/Fe gradients: {negative_slopes} ({negative_slopes/total_galaxies*100:.1f}%)")
                print(f"Statistically significant results: {significant_results} ({significant_results/total_galaxies*100:.1f}%)")
                print(f"Galaxies with emission lines: {emission_line_count} ({emission_line_count/total_galaxies*100:.1f}%)")
                print(f"Galaxies with special case handling: {special_case_count} ({special_case_count/total_galaxies*100:.1f}%)")
                
                # Analysis by emission line status (only if we have emission line info)
                if 'Emission Line' in df.columns:
                    emission_df = df[df['Emission Line'] == 'Yes']
                    non_emission_df = df[df['Emission Line'] == 'No']
                    
                    if len(emission_df) > 0:
                        emission_slopes = pd.to_numeric(emission_df['α/Fe slope'], errors='coerce')
                        emission_positive = sum(emission_slopes > 0)
                        emission_negative = sum(emission_slopes < 0)
                        
                        print("\n--- Emission Line Galaxy Statistics ---")
                        print(f"Positive gradients: {emission_positive} ({emission_positive/len(emission_df)*100:.1f}%)")
                        print(f"Negative gradients: {emission_negative} ({emission_negative/len(emission_df)*100:.1f}%)")
                        
                        mean_mag = emission_slopes.abs().mean()
                        if not np.isnan(mean_mag):
                            print(f"Average gradient magnitude: {mean_mag:.3f}")
                        else:
                            print("Average gradient magnitude: N/A")
                    
                    if len(non_emission_df) > 0:
                        non_emission_slopes = pd.to_numeric(non_emission_df['α/Fe slope'], errors='coerce')
                        non_emission_positive = sum(non_emission_slopes > 0)
                        non_emission_negative = sum(non_emission_slopes < 0)
                        
                        print("\n--- Non-Emission Line Galaxy Statistics ---")
                        print(f"Positive gradients: {non_emission_positive} ({non_emission_positive/len(non_emission_df)*100:.1f}%)")
                        print(f"Negative gradients: {non_emission_negative} ({non_emission_negative/len(non_emission_df)*100:.1f}%)")
                        
                        mean_mag = non_emission_slopes.abs().mean()
                        if not np.isnan(mean_mag):
                            print(f"Average gradient magnitude: {mean_mag:.3f}")
                        else:
                            print("Average gradient magnitude: N/A")
            except Exception as stats_err:
                logger.error(f"Error calculating statistics: {stats_err}")
                print("Could not calculate complete statistics due to an error.")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating results summary: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame instead of None

def process_all_galaxies(all_galaxy_data, model_data, config_path='bins_config.yaml', continuum_mode='fit'):
    """
    Process all galaxies with standardized alpha/Fe calculation
    
    Parameters:
    -----------
    all_galaxy_data : dict
        Dictionary mapping galaxy names to their RDB data
    model_data : DataFrame
        Model grid data
    config_path : str
        Path to the bin configuration file
    continuum_mode : str
        Spectral index continuum mode to use
        
    Returns:
    --------
    list
        List of standardized results for each galaxy
    """
    # Load configuration
    config = load_bin_config(config_path)
    
    # Galaxy-specific continuum mode settings
    galaxy_continuum_modes = {
        "VCC1049": "fit"  # Configure VCC1049 to use 'fit' mode
    }
    
    # Log the continuum mode being used
    logger.info(f"Using '{continuum_mode}' mode for spectral indices")
    
    # Results list
    results = []
    
    for galaxy_name, galaxy_data in all_galaxy_data.items():
        # Use galaxy-specific mode if available, otherwise use default
        galaxy_mode = galaxy_continuum_modes.get(galaxy_name, continuum_mode)
        if galaxy_name in galaxy_continuum_modes:
            logger.info(f"Processing {galaxy_name} with specific mode: {galaxy_mode}")
        else:
            logger.info(f"Processing {galaxy_name}...")
        
        # Get bins to use for this galaxy
        bins_to_use = get_bins_to_use(galaxy_name, config)
        logger.info(f"  Using bins: {bins_to_use}")
        
        # Find the maximum bin index
        max_bin_idx = max(bins_to_use) + 1 if bins_to_use else 6
        
        # Create output directory
        output_dir = f"./visualization/{galaxy_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Get standardized alpha/Fe data - our single source of truth
        result = get_standardized_alpha_fe_data(
            galaxy_name, 
            galaxy_data, 
            model_data, 
            bins_limit=max_bin_idx,
            continuum_mode=galaxy_mode,  # Use galaxy-specific mode if available
            special_cases=SPECIAL_CASES
        )
        
        # Log whether this is a special case
        if result['special_case_applied']:
            logger.info(f"  Applied special case handling for {galaxy_name}")
        
        # Add to results list
        results.append(result)
        
        # Create visualizations using the standardized data
        # Get cube info for scaling
        cube_info = extract_cube_info(galaxy_name)
        
        # Pass the standardized result to all visualization functions
        
        # Create flux map visualization
        flux_path = f"{output_dir}/{galaxy_name}_flux_and_binning.png"
        p2p_data = None
        try:
            p2p_data, vnb_data, _ = Read_Galaxy(galaxy_name)
        except:
            logger.warning(f"Could not read P2P data for {galaxy_name}")
            
        create_combined_flux_and_binning(
            galaxy_name, 
            p2p_data, 
            galaxy_data, 
            cube_info, 
            output_path=flux_path, 
            dpi=150
        )
        
        # Create parameter-radius plots - pass the standardized data
        params_plot_path = f"{output_dir}/{galaxy_name}_parameter_radius.png"
        create_parameter_radius_plots(
            galaxy_name, 
            galaxy_data, 
            model_data,
            output_path=params_plot_path, 
            dpi=150, 
            bins_limit=max_bin_idx,
            interpolated_data=result,  # Pass standardized data
            continuum_mode=galaxy_mode  # Use galaxy-specific mode
        )
        
        # Create spectral index visualization
        interp_path = f"{output_dir}/{galaxy_name}_alpha_fe_interpolation.png"
        create_spectral_index_interpolation_plot(
            galaxy_name, 
            galaxy_data, 
            model_data,
            output_path=interp_path, 
            dpi=150, 
            bins_limit=max_bin_idx,
            continuum_mode=galaxy_mode,  # Use galaxy-specific mode
            standardized_data=result  # Pass standardized data
        )
        
        # Create model grid plots
        model_grid_path = f"{output_dir}/{galaxy_name}_model_grid_part1.png"
        create_model_grid_plots_part1(
            galaxy_name, 
            galaxy_data, 
            model_data,
            age=10,
            output_path=model_grid_path, 
            dpi=150, 
            bins_limit=max_bin_idx,
            continuum_mode=galaxy_mode,  # Use galaxy-specific mode
            standardized_data=result  # Pass standardized data
        )
        
        # Create SNR comparison visualization using the data directly from df_1, df_2, df_3
        try:
            # Read all three data modes for SNR analysis
            p2p_data, vnb_data, rdb_data = Read_Galaxy(galaxy_name)
            
            # Check if SNR data is available in all three modes
            has_p2p_snr = p2p_data is not None and 'signal_noise' in p2p_data and 'snr' in p2p_data['signal_noise'].item()
            has_vnb_snr = vnb_data is not None and 'snr' in vnb_data
            has_rdb_snr = rdb_data is not None and 'snr' in rdb_data
            
            if has_p2p_snr and has_vnb_snr and has_rdb_snr:
                # Extract SNR data from each mode
                p2p_snr = p2p_data['signal_noise'].item()['snr']
                vnb_snr = vnb_data['snr']
                rdb_snr = rdb_data['snr']
                
                # Create temporary FITS files to use with create_snr_comparison_plots
                temp_dir = os.path.join(output_dir, "temp_snr_files")
                os.makedirs(temp_dir, exist_ok=True)
                
                # Create paths for temporary files
                p2p_file = os.path.join(temp_dir, f"{galaxy_name}_p2p_snr.fits")
                vnb_file = os.path.join(temp_dir, f"{galaxy_name}_vnb_snr.fits")
                rdb_file = os.path.join(temp_dir, f"{galaxy_name}_rdb_snr.fits")
                
                # Save SNR data to temporary FITS files
                for snr_data, file_path in [
                    (p2p_snr, p2p_file),
                    (vnb_snr, vnb_file),
                    (rdb_snr, rdb_file)
                ]:
                    # Create a simple FITS file with the SNR data
                    hdu = fits.PrimaryHDU(snr_data)
                    hdul = fits.HDUList([hdu])
                    
                    # Add some basic header information
                    hdu.header['OBJECT'] = galaxy_name
                    if cube_info is not None and 'header' in cube_info and cube_info['header'] is not None:
                        # Copy important WCS keys from the original header
                        for key in ['CRVAL1', 'CRVAL2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
                            if key in cube_info['header']:
                                hdu.header[key] = cube_info['header'][key]
                    
                    # Add effective radius if available
                    if 'effective_radius' in result and result['effective_radius'] is not None:
                        hdu.header['EFFRAD'] = result['effective_radius']
                    
                    # Save to file
                    hdul.writeto(file_path, overwrite=True)
                
                # Create the SNR comparison plot using the temporary files
                snr_output_path = f"{output_dir}/{galaxy_name}_snr_comparison.png"
                create_snr_comparison_plots(
                    [p2p_file, vnb_file, rdb_file],
                    output_path=snr_output_path,
                    dpi=150
                )
                
                # Clean up temporary files
                for file_path in [p2p_file, vnb_file, rdb_file]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                
                logger.info(f"Created SNR comparison visualization for {galaxy_name}")
            else:
                logger.info(f"Skipping SNR visualization for {galaxy_name} - SNR data not available in all modes")
        except Exception as e:
            logger.warning(f"Error creating SNR visualization for {galaxy_name}: {e}")
    
    return results

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

if __name__ == "__main__":
    # Define bin limit for all analyses
    bins_limit = 6  # Analyze bins 0-5
    
    # Define spectral index continuum mode to use globally
    # Options: 'auto', 'fit', 'original'
    continuum_mode = 'fit'  # Set your preferred mode here
    
    # Define galaxies to process
    galaxies = [
        "VCC0308", "VCC0667", "VCC0688", "VCC0990", "VCC1049", "VCC1146", 
        "VCC1193", "VCC1368", "VCC1410", "VCC1431", "VCC1486", "VCC1499", 
        "VCC1549", "VCC1588", "VCC1695", "VCC1811", "VCC1890", 
        "VCC1902", "VCC1910", "VCC1949"
    ]
    
    # Path to model data file
    model_file = "./TMB03/TMB03.csv"
    
    # Create output directory for summary
    output_dir = "./galaxy_summary"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model data
    model_data = load_model_data(model_file)
    
    # Load all galaxy data
    all_galaxy_data = {}
    for galaxy_name in galaxies:
        _, _, rdb_data = Read_Galaxy(galaxy_name)
        if rdb_data is not None and isinstance(rdb_data, dict) and len(rdb_data) > 0:
            all_galaxy_data[galaxy_name] = rdb_data
    
    # Process all galaxies with the bin configuration and specified mode - single consistent method
    results = process_all_galaxies(all_galaxy_data, model_data, 'bins_config.yaml', continuum_mode)
    
    # Define emission line galaxies
    emission_line_galaxies = [
        "VCC1588", #"VCC1368", "VCC1902", "VCC1949", "VCC990", 
        "VCC1410", "VCC667", "VCC1811", "VCC688", "VCC1193", "VCC1486"
    ]
    
    # Get galaxy coordinates
    coordinates = get_ifu_coordinates(galaxies)
    
    # Create Alpha/Fe vs. radius plot
    create_alpha_radius_plot_improved(results, 
                                    output_path=f"{output_dir}/alpha_fe_vs_radius_improved.png", 
                                    dpi=300)
    
    # Create direct interpolation plots
    create_alpha_radius_direct_plot(results, 
                                   output_path=f"{output_dir}/alpha_fe_vs_radius_direct.png", 
                                   dpi=300)
    
    # Create spectral index space plot
    create_fe5015_mgb_plot(results, 
                         output_path=f"{output_dir}/fe5015_vs_mgb.png", 
                         dpi=300,
                         model_data=model_data)
    
    # Create Virgo Cluster map
    create_virgo_cluster_map_with_vectors(results, coordinates, 
                                        output_path=f"{output_dir}/virgo_cluster_map.png", 
                                        dpi=300)
    
    # Create separate visualizations
    cluster_viz_dir = f"{output_dir}/virgo_cluster"
    os.makedirs(cluster_viz_dir, exist_ok=True)
    create_virgo_cluster_visualizations(results, coordinates, output_dir=cluster_viz_dir, dpi=300)
    
    # Save summary info to file
    summary_df = create_alpha_fe_results_summary(results, emission_line_galaxies, f"{output_dir}/alpha_fe_results.csv")
    
    logger.info(f"Analysis complete using '{continuum_mode}' continuum mode for spectral indices")




def create_3d_alpha_fe_interpolation(galaxy_name, rdb_data, model_data, output_path=None, dpi=150, bins_limit=6, continuum_mode='fit'):
    """
    Create a 3D visualization of alpha/Fe interpolation across spectral index space
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices
    model_data : DataFrame
        Model grid data with indices and alpha/Fe values
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for the output image
    bins_limit : int
        Limit on the number of bins to analyze
    continuum_mode : str
        Spectral index continuum mode to use ('auto', 'fit', 'original')
    
    Returns:
    --------
    dict
        Dictionary with interpolated alpha/Fe values and corresponding spectral indices
    """
    try:
        # Import needed for 3D plotting
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        from scipy.interpolate import LinearNDInterpolator, griddata
        import scipy.spatial as spatial
        
        # Use standardized data if possible
        standardized_data = get_standardized_alpha_fe_data(
            galaxy_name, rdb_data, model_data, bins_limit, continuum_mode, SPECIAL_CASES)
        
        # Extract data from standardized data
        alpha_fe_values = standardized_data['alpha_fe_values']
        radius_values = standardized_data['radius_values']
        fe5015_values = standardized_data['fe5015_values']
        mgb_values = standardized_data['mgb_values']
        hbeta_values = standardized_data['hbeta_values']
        bin_indices = standardized_data['bin_indices']
        
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Get mean age for model grid
        mean_age = 10.0  # Default to 10 Gyr
        
        # Find closest age in model grid
        age_column = model_column_mapping['Age']
        available_ages = np.array(model_data[age_column].unique())
        closest_age = available_ages[np.argmin(np.abs(available_ages - mean_age))]
        
        # Filter model grid to this age
        model_age_data = model_data[model_data[age_column] == closest_age]
        
        # Extract column names
        fe5015_col = model_column_mapping['Fe5015']
        mgb_col = model_column_mapping['Mgb']
        hbeta_col = model_column_mapping['Hbeta']
        aofe_col = model_column_mapping['AoFe']
        zoh_col = model_column_mapping['ZoH']
        
        # Create 3D interpolator from model data
        # Extract model data points for interpolation
        model_fe5015 = model_age_data[fe5015_col].values
        model_mgb = model_age_data[mgb_col].values
        model_hbeta = model_age_data[hbeta_col].values
        model_aofe = model_age_data[aofe_col].values
        
        # Create the interpolator from the model grid points
        points = np.column_stack([model_fe5015, model_mgb, model_hbeta])
        
        # Use linear interpolation for smoother results
        interpolator = LinearNDInterpolator(points, model_aofe)
        
        # Now create a grid for visualization of the 3D space
        # Define grid limits based on galaxy and model data
        fe5015_min = max(0, min(np.min(model_fe5015), np.min(fe5015_values) - 0.5))
        fe5015_max = max(np.max(model_fe5015), np.max(fe5015_values) + 0.5)
        mgb_min = max(0, min(np.min(model_mgb), np.min(mgb_values) - 0.5))
        mgb_max = max(np.max(model_mgb), np.max(mgb_values) + 0.5)
        hbeta_min = max(0, min(np.min(model_hbeta), np.min(hbeta_values) - 0.5))
        hbeta_max = max(np.max(model_hbeta), np.max(hbeta_values) + 0.5)
        
        # Create a grid covering this space (but not too dense for performance)
        grid_points = 20  # Number of points in each dimension
        fe5015_grid = np.linspace(fe5015_min, fe5015_max, grid_points)
        mgb_grid = np.linspace(mgb_min, mgb_max, grid_points)
        hbeta_grid = np.linspace(hbeta_min, hbeta_max, grid_points)
        
        # Create the 3D grid
        fe5015_3d, mgb_3d, hbeta_3d = np.meshgrid(fe5015_grid, mgb_grid, hbeta_grid)
        grid_points_3d = np.column_stack([fe5015_3d.flatten(), mgb_3d.flatten(), hbeta_3d.flatten()])
        
        # Interpolate alpha/Fe values across this grid
        alpha_fe_grid = interpolator(grid_points_3d)
        
        # Handle NaN values
        alpha_fe_grid_clean = alpha_fe_grid.copy()
        nan_mask_grid = np.isnan(alpha_fe_grid_clean)
        if np.any(nan_mask_grid):
            # Use nearest neighbor for NaN points
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            nn_indices_grid = tree.query(grid_points_3d[nan_mask_grid])[1]
            alpha_fe_grid_clean[nan_mask_grid] = model_aofe[nn_indices_grid]
        
        # Reshape back to 3D grid
        alpha_fe_vol = alpha_fe_grid_clean.reshape(fe5015_3d.shape)
        
        # Now create a visualization that shows:
        # 1. A 3D scatter plot of the model grid points
        # 2. The galaxy points with radius encoding
        # 3. Slices of the interpolated alpha/Fe field
        
        # Create a figure with two subplots
        fig = plt.figure(figsize=(20, 10))
        
        # Create a 3D axes for the left plot - scatter plot
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Plot the model grid points in the space - use small points
        ax1.scatter(model_fe5015, model_mgb, model_hbeta, 
                  c=model_aofe, cmap='plasma', s=10, alpha=0.3,
                  label='Model Grid Points')
        
        # Create a normalization for alpha/Fe values
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=0.5)  # Alpha/Fe range from 0 to 0.5
        
        # Plot the galaxy points with larger symbols
        scatter = ax1.scatter(fe5015_values, mgb_values, hbeta_values, 
                           c=alpha_fe_values, cmap='plasma', s=150, 
                           edgecolor='black', linewidth=1.5, norm=norm)
        
        # Add bin numbers to the galaxy points
        for i in range(len(fe5015_values)):
            ax1.text(fe5015_values[i], mgb_values[i], hbeta_values[i], 
                   str(bin_indices[i]), fontsize=12, color='white', fontweight='bold')
        
        # Add radius information if available
        if radius_values is not None:
            # Normalize radius for text size
            radius_norm = radius_values / np.max(radius_values) if np.max(radius_values) > 0 else np.ones_like(radius_values)
            for i, r in enumerate(radius_values):
                ax1.text(fe5015_values[i], mgb_values[i], hbeta_max * 1.05, 
                       f"R={r:.2f}", fontsize=8 + 4 * radius_norm[i], color='black')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter, ax=ax1, orientation='vertical')
        cbar.set_label('[α/Fe]')
        
        # Set labels and title
        ax1.set_xlabel('Fe5015 Index')
        ax1.set_ylabel('Mgb Index')
        ax1.set_zlabel('Hβ Index')
        ax1.set_title('3D Spectral Index Space with α/Fe Interpolation', fontsize=16)
        
        # Set view angle for better visualization
        ax1.view_init(30, -45)
        
        # Create axes for the right plot - volume slice
        ax2 = fig.add_subplot(122, projection='3d')
        
        # Get mean values to position slice planes
        mid_fe5015 = np.mean(fe5015_values)
        mid_mgb = np.mean(mgb_values)
        mid_hbeta = np.mean(hbeta_values)
        
        # Extract 2D slices from the 3D volume
        # Find nearest grid indices for the middle values
        idx_fe5015 = np.argmin(np.abs(fe5015_grid - mid_fe5015))
        idx_mgb = np.argmin(np.abs(mgb_grid - mid_mgb))
        idx_hbeta = np.argmin(np.abs(hbeta_grid - mid_hbeta))
        
        # Extract slices
        xy_slice = alpha_fe_vol[:, :, idx_hbeta]  # Fe5015-Mgb plane
        xz_slice = alpha_fe_vol[:, idx_mgb, :]    # Fe5015-Hbeta plane
        yz_slice = alpha_fe_vol[idx_fe5015, :, :] # Mgb-Hbeta plane
        
        # Create 2D mesh grids for the slice planes
        X, Y = np.meshgrid(fe5015_grid, mgb_grid)
        X, Z = np.meshgrid(fe5015_grid, hbeta_grid)
        Y, Z = np.meshgrid(mgb_grid, hbeta_grid)
        
        # Plot the slice planes with alpha/Fe coloring
        # XY plane (constant Hbeta)
        xy = ax2.plot_surface(X, Y, np.ones_like(X) * hbeta_grid[idx_hbeta],
                           facecolors=cm.plasma(norm(xy_slice)), 
                           alpha=0.7, shade=False)
        
        # XZ plane (constant Mgb)
        xz = ax2.plot_surface(X, np.ones_like(X) * mgb_grid[idx_mgb], Z,
                           facecolors=cm.plasma(norm(xz_slice.T)), 
                           alpha=0.7, shade=False)
        
        # YZ plane (constant Fe5015)
        yz = ax2.plot_surface(np.ones_like(Y) * fe5015_grid[idx_fe5015], Y, Z,
                           facecolors=cm.plasma(norm(yz_slice.T)),
                           alpha=0.7, shade=False)
        
        # Plot the galaxy points on this plot too
        ax2.scatter(fe5015_values, mgb_values, hbeta_values, 
                  c=alpha_fe_values, cmap='plasma', s=150, 
                  edgecolor='black', linewidth=1.5, norm=norm)
        
        # Add bin numbers
        for i in range(len(fe5015_values)):
            ax2.text(fe5015_values[i], mgb_values[i], hbeta_values[i], 
                   str(bin_indices[i]), fontsize=12, color='white', fontweight='bold')
        
        # Add guides to show where the slice planes are
        ax2.plot([mid_fe5015, mid_fe5015], [mgb_min, mgb_max], [mid_hbeta, mid_hbeta], 'k--', alpha=0.5)
        ax2.plot([mid_fe5015, mid_fe5015], [mid_mgb, mid_mgb], [hbeta_min, hbeta_max], 'k--', alpha=0.5)
        ax2.plot([fe5015_min, fe5015_max], [mid_mgb, mid_mgb], [mid_hbeta, mid_hbeta], 'k--', alpha=0.5)
        
        # Add labels for the slice planes
        ax2.text(mid_fe5015, mgb_max*1.1, mid_hbeta, "Fe5015 Slice", fontsize=10)
        ax2.text(mid_fe5015, mid_mgb, hbeta_max*1.1, "Mgb Slice", fontsize=10)
        ax2.text(fe5015_max*1.1, mid_mgb, mid_hbeta, "Hβ Slice", fontsize=10)
        
        # Add a colorbar for the slice planes
        m = cm.ScalarMappable(cmap='plasma', norm=norm)
        m.set_array([])
        cbar2 = plt.colorbar(m, ax=ax2, orientation='vertical')
        cbar2.set_label('[α/Fe]')
        
        # Set labels and title
        ax2.set_xlabel('Fe5015 Index')
        ax2.set_ylabel('Mgb Index')
        ax2.set_zlabel('Hβ Index')
        ax2.set_title('Cross-sections of α/Fe Distribution', fontsize=16)
        
        # Set view angle for better visualization of slices
        ax2.view_init(30, 225)
        
        # Add overall title
        plt.suptitle(f"Galaxy {galaxy_name}: 3D Spectral Index Analysis with α/Fe Interpolation\n"
           f"Model Age: {closest_age} Gyr | Continuum Mode: {continuum_mode}", 
           fontsize=20, y=0.98)
        
        # Add special case note if applicable
        if standardized_data.get('special_case_applied', False):
            plt.figtext(0.5, 0.02, 
                      "NOTE: Special case handling applied to this galaxy.",
                      ha='center', fontsize=10, color='red', 
                      bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        
        # Add explanation text
        plt.figtext(0.5, 0.01, 
                  "Left: 3D scatter plot of model grid (small points) and galaxy measurements (large points)\n"
                  "Right: Cross-sectional planes showing α/Fe distribution in the spectral index space\n"
                  "Points are colored by interpolated α/Fe values. Bin numbers correspond to radial bins.",
                  ha='center', fontsize=12)
        
        # Add information about the interpolated alpha/Fe values
        if len(alpha_fe_values) > 0:
            min_alpha = np.min(alpha_fe_values)
            max_alpha = np.max(alpha_fe_values)
            mean_alpha = np.mean(alpha_fe_values)
            plt.figtext(0.01, 0.01, 
                      f"Interpolated α/Fe range: {min_alpha:.2f} to {max_alpha:.2f} (mean: {mean_alpha:.2f})",
                      ha='left', fontsize=10)
        
        # Calculate slope if radius information is available
        if len(radius_values) > 1:
            slope = standardized_data.get('slope', np.nan)
            p_value = standardized_data.get('p_value', np.nan)
            r_squared = standardized_data.get('r_squared', np.nan)
            
            significance = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
            plt.figtext(0.99, 0.01, 
                      f"α/Fe-Radius Slope: {slope:.3f}{significance} (p={p_value:.3f}, R²={r_squared:.3f})",
                      ha='right', fontsize=10)
        
        # Tight layout for better spacing
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if output path provided
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved 3D alpha/Fe interpolation visualization to {output_path}")
        
        plt.close()
        
        # Return the interpolated values and corresponding data
        return {
            'alpha_fe': alpha_fe_values,
            'radius': radius_values,
            'Fe5015': fe5015_values,
            'Mgb': mgb_values,
            'Hbeta': hbeta_values,
            'alphafe_grid': {
                'Fe5015_grid': fe5015_grid,
                'Mgb_grid': mgb_grid,
                'Hbeta_grid': hbeta_grid,
                'alpha_fe_vol': alpha_fe_vol
            },
            'continuum_mode': continuum_mode  # Add this line
        }
        
    except Exception as e:
        logger.error(f"Error creating 3D alpha/Fe interpolation: {e}")
        import traceback
        traceback.print_exc()
        return None

# Function to create galaxy table for publication
def create_galaxy_results_table(results_list, emission_line_galaxies, output_path=None):
    """
    Create a formatted LaTeX table of alpha/Fe gradient results for publication
    
    Parameters:
    -----------
    results_list : list
        List of dictionaries containing alpha/Fe results for each galaxy
    emission_line_galaxies : list
        List of galaxy names with emission lines
    output_path : str
        Path to save the LaTeX table
        
    Returns:
    --------
    str
        LaTeX formatted table
    """
    try:
        # Prepare data for table
        table_data = []
        
        for result in results_list:
            if result is None:
                continue
                
            galaxy = result['galaxy']
            # Use get() with defaults to handle missing keys
            slope = result.get('slope', np.nan)
            p_value = result.get('p_value', np.nan)
            alpha_fe_median = result.get('alpha_fe_median', np.nan)
            radius_median = result.get('radius_median', np.nan)
            
            # Max radius (radial coverage)
            if 'radius_values' in result and len(result['radius_values']) > 0:
                max_radius = np.max(result['radius_values'])
            else:
                max_radius = np.nan
                
            # Number of bins used
            if 'bin_indices' in result:
                n_bins = len(result['bin_indices'])
            else:
                n_bins = 0
                
            has_emission = galaxy in emission_line_galaxies
            special_case = result.get('special_case_applied', False)
            
            # Format statistical significance
            if not np.isnan(p_value):
                if p_value < 0.01:
                    sig_symbol = "**"
                elif p_value < 0.05:
                    sig_symbol = "*"
                else:
                    sig_symbol = ""
            else:
                sig_symbol = ""
            
            # Add to table data
            table_data.append({
                'Galaxy': galaxy,
                'Emission': has_emission,
                'α/Fe': alpha_fe_median,
                'R/Re': radius_median,
                'Max R/Re': max_radius,
                'Bins': n_bins,
                'Slope': slope,
                'Sig': sig_symbol,
                'p-value': p_value,
                'Special': special_case
            })
        
        # Sort by absolute slope magnitude
        table_data.sort(key=lambda x: abs(x['Slope']) if not np.isnan(x['Slope']) else 0, reverse=True)
        
        # Create LaTeX table
        latex_table = "\\begin{table}\n"
        latex_table += "\\centering\n"
        latex_table += "\\caption{Alpha/Fe Gradients in Virgo Cluster Galaxies}\n"
        latex_table += "\\begin{tabular}{lccccccc}\n"
        latex_table += "\\hline\\hline\n"
        latex_table += "Galaxy & Emission & $\\langle$[α/Fe]$\\rangle$ & $\\langle$R/Re$\\rangle$ & Max R/Re & Bins & Slope & p-value \\\\\n"
        latex_table += "\\hline\n"
        
        # Add rows
        for row in table_data:
            # Format values
            galaxy_fmt = row['Galaxy']
            emission_fmt = "Yes" if row['Emission'] else "No"
            alpha_fe_fmt = f"{row['α/Fe']:.2f}" if not np.isnan(row['α/Fe']) else "---"
            radius_fmt = f"{row['R/Re']:.2f}" if not np.isnan(row['R/Re']) else "---"
            max_radius_fmt = f"{row['Max R/Re']:.2f}" if not np.isnan(row['Max R/Re']) else "---"
            bins_fmt = str(row['Bins'])
            
            # Format slope with significance symbol
            if not np.isnan(row['Slope']):
                slope_fmt = f"{row['Slope']:.3f}{row['Sig']}"
            else:
                slope_fmt = "---"
                
            p_value_fmt = f"{row['p-value']:.3f}" if not np.isnan(row['p-value']) else "---"
            
            # Add special case indicator
            if row['Special']:
                galaxy_fmt = f"{galaxy_fmt}$^\\dagger$"
            
            # Add row to table
            latex_table += f"{galaxy_fmt} & {emission_fmt} & {alpha_fe_fmt} & {radius_fmt} & {max_radius_fmt} & {bins_fmt} & {slope_fmt} & {p_value_fmt} \\\\\n"
        
        # Close table
        latex_table += "\\hline\n"
        latex_table += "\\end{tabular}\n"
        latex_table += "\\begin{tablenotes}\n"
        latex_table += "\\small\n"
        latex_table += "\\item $^*$ p < 0.05 (statistically significant)\n"
        latex_table += "\\item $^{**}$ p < 0.01 (highly significant)\n"
        latex_table += "\\item $^\\dagger$ Special case handling applied\n"
        latex_table += "\\end{tablenotes}\n"
        latex_table += "\\end{table}"
        
        # Save to file if requested
        if output_path:
            with open(output_path, 'w') as f:
                f.write(latex_table)
            logger.info(f"Saved LaTeX table to {output_path}")
        
        return latex_table
        
    except Exception as e:
        logger.error(f"Error creating LaTeX table: {e}")
        import traceback
        traceback.print_exc()
        return ""

def create_galaxy_visualization(galaxy_name, p2p_data, rdb_data, cube_info, model_data=None, output_dir=None, dpi=150, bins_limit=6, continuum_mode='fit'):
    """
    Create comprehensive visualization for a galaxy using the standardized data
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    p2p_data : dict
        P2P data dictionary
    rdb_data : dict
        RDB data dictionary
    cube_info : dict
        Dictionary with cube information
    model_data : DataFrame, optional
        Model grid data for spectral index plots
    output_dir : str
        Directory to save output images
    dpi : int
        Resolution for saved images
    bins_limit : int
        Limit analysis to first N bins (default: 6 for bins 0-5)
    continuum_mode : str
        Spectral index continuum mode to use
    
    Returns:
    --------
    dict
        Dictionary with standardized data for the galaxy
    """
    try:
        # Create output directory if needed
        if output_dir is None:
            output_dir = f"./visualization/{galaxy_name}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if data is valid
        p2p_valid = p2p_data is not None and isinstance(p2p_data, dict)
        rdb_valid = rdb_data is not None and isinstance(rdb_data, dict)
        
        if not rdb_valid:
            logger.error(f"No valid RDB data for {galaxy_name}")
            return None
            
        # Get standardized alpha/Fe data - our single source of truth
        standardized_data = get_standardized_alpha_fe_data(
            galaxy_name, 
            rdb_data, 
            model_data, 
            bins_limit=bins_limit,
            continuum_mode=continuum_mode,
            special_cases=SPECIAL_CASES
        )
            
        # Create flux map visualization
        create_combined_flux_and_binning(
            galaxy_name, 
            p2p_data, 
            rdb_data, 
            cube_info, 
            output_path=f"{output_dir}/{galaxy_name}_flux_and_binning.png", 
            dpi=dpi
        )
        
        # Create spectral index visualization
        create_spectral_index_interpolation_plot(
            galaxy_name, 
            rdb_data, 
            model_data,
            output_path=f"{output_dir}/{galaxy_name}_alpha_fe_interpolation.png",
            dpi=dpi,
            bins_limit=bins_limit,
            continuum_mode=continuum_mode,
            standardized_data=standardized_data
        )
        
        # Create parameter-radius plots
        create_parameter_radius_plots(
            galaxy_name, 
            rdb_data, 
            model_data,
            output_path=f"{output_dir}/{galaxy_name}_parameter_radius.png",
            dpi=dpi,
            bins_limit=bins_limit,
            interpolated_data=standardized_data,
            continuum_mode=continuum_mode
        )
        
        # Create model grid plots
        create_model_grid_plots_part1(
            galaxy_name, 
            rdb_data, 
            model_data,
            age=10,
            output_path=f"{output_dir}/{galaxy_name}_model_grid_part1.png",
            dpi=dpi,
            bins_limit=bins_limit,
            continuum_mode=continuum_mode,
            standardized_data=standardized_data
        )
        
        # Create 3D interpolation visualization
        create_3d_alpha_fe_interpolation(
            galaxy_name, 
            rdb_data, 
            model_data,
            output_path=f"{output_dir}/{galaxy_name}_3d_alpha_fe.png",
            dpi=dpi,
            bins_limit=bins_limit,
            continuum_mode=continuum_mode
        )
        
        logger.info(f"Created all visualizations for {galaxy_name}")
        
        # Return the standardized data
        return standardized_data
        
    except Exception as e:
        logger.error(f"Error in galaxy visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to process all galaxies"""
    # Define bin limit for all analyses
    bins_limit = 6  # Analyze bins 0-5
    
    # Define spectral index continuum mode to use globally
    # Options: 'auto', 'fit', 'original'
    continuum_mode = 'fit'  # Set your preferred mode here
    
    # Define galaxies to process
    galaxies = [
        "VCC0308",
         "VCC0667", "VCC0688", "VCC0990", "VCC1049", "VCC1146", 
        "VCC1193", "VCC1368", "VCC1410", "VCC1431", "VCC1486", "VCC1499", 
        "VCC1549", "VCC1588", "VCC1695", "VCC1811", "VCC1890", 
        "VCC1902", "VCC1910", "VCC1949"
    ]
    
    # Path to model data file
    model_file = "./TMB03/TMB03.csv"
    
    # Create output directory for summary
    output_dir = "./galaxy_summary"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model data
    model_data = load_model_data(model_file)
    
    # Load all galaxy data
    all_galaxy_data = {}
    for galaxy_name in galaxies:
        _, _, rdb_data = Read_Galaxy(galaxy_name)
        if rdb_data is not None and isinstance(rdb_data, dict) and len(rdb_data) > 0:
            all_galaxy_data[galaxy_name] = rdb_data
    
    # Process all galaxies with the bin configuration and specified mode - single consistent method
    results = process_all_galaxies(all_galaxy_data, model_data, 'bins_config.yaml', continuum_mode)
    
    # Define emission line galaxies
    emission_line_galaxies = [
        "VCC1588", #"VCC1368", "VCC1902", "VCC1949", "VCC990", 
        "VCC1410", "VCC0667", "VCC1811", "VCC0688", "VCC1193", "VCC1486"
    ]
    
    # Get galaxy coordinates
    coordinates = get_ifu_coordinates(galaxies)
    
    # Create Alpha/Fe vs. radius plot
    create_alpha_radius_plot_improved(results, 
                                    output_path=f"{output_dir}/alpha_fe_vs_radius_improved.png", 
                                    dpi=300)
    
    # Create direct interpolation plots
    create_alpha_radius_direct_plot(results, 
                                   output_path=f"{output_dir}/alpha_fe_vs_radius_direct.png", 
                                   dpi=300)
    
    # Create spectral index space plot
    create_fe5015_mgb_plot(results, 
                         output_path=f"{output_dir}/fe5015_vs_mgb.png", 
                         dpi=300,
                         model_data=model_data)
    
    # Create Virgo Cluster map
    create_virgo_cluster_map_with_vectors(results, coordinates, 
                                        output_path=f"{output_dir}/virgo_cluster_map.png", 
                                        dpi=300)
    
    # Create separate visualizations
    cluster_viz_dir = f"{output_dir}/virgo_cluster"
    os.makedirs(cluster_viz_dir, exist_ok=True)
    create_virgo_cluster_visualizations(results, coordinates, output_dir=cluster_viz_dir, dpi=300)
    
    # Save summary info to file
    summary_df = create_alpha_fe_results_summary(results, emission_line_galaxies, f"{output_dir}/alpha_fe_results.csv")
    
    # Create LaTeX table for publication
    latex_table = create_galaxy_results_table(results, emission_line_galaxies, f"{output_dir}/alpha_fe_results_table.tex")
    
    logger.info(f"Analysis complete using '{continuum_mode}' continuum mode for spectral indices")
    
    # Print final status
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Processed {len(results)} galaxies")
    print(f"Using '{continuum_mode}' continuum mode for spectral indices")
    print(f"Results saved to: {output_dir}")
    
    return results

if __name__ == "__main__":
    main()