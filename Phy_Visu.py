"""
Alpha/Fe Analysis for Virgo Cluster Galaxies

This module provides functions for analyzing and visualizing the alpha element
abundance gradients in Virgo Cluster galaxies using radial binned spectroscopy data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arrow, Circle
from matplotlib.colors import LogNorm, Normalize, ListedColormap
from matplotlib.ticker import AutoMinorLocator
from matplotlib.lines import Line2D
from scipy import stats
import logging
import os
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#------------------------------------------------------------------------------
# Data Loading Functions
#------------------------------------------------------------------------------

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
    filtering out negative or invalid spectral indices
    
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
            
            if 'age' in stellar_pop:
                age = stellar_pop['age']
                # If age is in years, convert to Gyr
                if np.any(age > 100):  # Assuming age > 100 means it's in years
                    age = age / 1e9
                    
                if len(valid_bins) > 0 and len(age) >= max(valid_bins) + 1:
                    result['bin_indices']['age'] = age[valid_bins]
            
            if 'metallicity' in stellar_pop:
                metallicity = stellar_pop['metallicity']
                
                if len(valid_bins) > 0 and len(metallicity) >= max(valid_bins) + 1:
                    result['bin_indices']['metallicity'] = metallicity[valid_bins]
        
        # Extract radius for bins with the same valid bins filter
        if 'distance' in rdb_data and valid_bins is not None:
            distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
            if 'bin_distances' in distance:
                bin_distances = distance['bin_distances']
                
                if len(valid_bins) > 0 and len(bin_distances) >= max(valid_bins) + 1:
                    result['bin_indices']['R'] = bin_distances[valid_bins]
    
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
                        
            # Second try: Try to read the cube and extract coordinates
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

def extract_parameter_profiles(data, parameter_names=['Fe5015', 'Mgb', 'Hbeta', 'age', 'metallicity'], bins_limit=6):
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
            
        # Extract radius information
        if 'distance' in data:
            distance = data['distance'].item() if hasattr(data['distance'], 'item') else data['distance']
            if 'bin_distances' in distance:
                bin_distances = distance['bin_distances']
                # Limit to specified number of bins
                results['radius'] = bin_distances[:min(len(bin_distances), bins_limit)]
            elif 'binning' in data and 'bin_radii' in data['binning']:
                binning = data['binning'].item() if hasattr(data['binning'], 'item') else data['binning']
                bin_radii = binning['bin_radii']
                # Limit to specified number of bins
                results['radius'] = bin_radii[:min(len(bin_radii), bins_limit)]
        
        # Extract effective radius
        results['effective_radius'] = extract_effective_radius(data)
        
        # Extract spectral indices
        if 'bin_indices' in data and results['radius'] is not None:
            bin_indices = data['bin_indices'].item() if hasattr(data['bin_indices'], 'item') else data['bin_indices']
            if 'bin_indices' in bin_indices:
                for param in parameter_names[:3]:  # First three are spectral indices
                    if param in bin_indices['bin_indices']:
                        indices = bin_indices['bin_indices'][param]
                        # Limit to specified number of bins
                        results[param] = indices[:min(len(indices), bins_limit)]
        
        # Try alternative indices path
        if 'indices' in data and results['radius'] is not None:
            indices = data['indices'].item() if hasattr(data['indices'], 'item') else data['indices']
            for param in parameter_names[:3]:  # First three are spectral indices
                if param in indices and param not in results:
                    idx_values = indices[param]
                    # Limit to specified number of bins
                    results[param] = idx_values[:min(len(idx_values), bins_limit)]
        
        # Extract stellar population parameters
        if 'stellar_population' in data and results['radius'] is not None:
            stellar_pop = data['stellar_population'].item() if hasattr(data['stellar_population'], 'item') else data['stellar_population']
            # Get age (convert to log if in linear units)
            if 'age' in stellar_pop and 'age' in parameter_names:
                age = np.array(stellar_pop['age'])
                # Limit to specified number of bins
                age = age[:min(len(age), bins_limit)]
                # Check if age needs to be converted to log10
                if np.any(age > 0) and np.median(age[age > 0]) > 1e8:  # Age in years
                    results['age'] = np.log10(age)
                else:
                    results['age'] = age  # Already in log10 or in Gyr
                    
            # Get metallicity
            if 'metallicity' in stellar_pop and 'metallicity' in parameter_names:
                metallicity = stellar_pop['metallicity']
                # Limit to specified number of bins
                results['metallicity'] = metallicity[:min(len(metallicity), bins_limit)]
    
    except Exception as e:
        logger.error(f"Error extracting parameter profiles: {e}")
    
    return results

def extract_spectral_indices_by_method(rdb_data, method='auto', bins_limit=6):
    """
    Extract spectral indices using the specified calculation method
    
    Parameters:
    -----------
    rdb_data : dict
        RDB data containing spectral indices
    method : str
        Which method to use for spectral indices: 'auto', 'original', 'fit', or 'template'
    bins_limit : int
        Limit analysis to first N bins (default: 6 for bins 0-5)
        
    Returns:
    --------
    dict
        Dictionary containing bin radii and spectral indices
    """
    result = {'bin_radii': None, 'bin_indices': {}}
    
    try:
        # Check for multi-method indices first (bin_indices_multi)
        if method != 'template' and 'bin_indices_multi' in rdb_data:
            bin_indices_multi = rdb_data['bin_indices_multi'].item() if hasattr(rdb_data['bin_indices_multi'], 'item') else rdb_data['bin_indices_multi']
            
            if method in bin_indices_multi:
                # Extract using the specified method
                method_indices = bin_indices_multi[method]
                if 'bin_indices' in method_indices:
                    # Filter valid indices
                    fe5015_indices = method_indices['bin_indices'].get('Fe5015', np.array([]))
                    mgb_indices = method_indices['bin_indices'].get('Mgb', np.array([]))
                    hbeta_indices = method_indices['bin_indices'].get('Hbeta', np.array([]))
                    
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
                        
                        # Extract other parameters
                        _extract_additional_parameters(rdb_data, valid_bins, result)
                        
                        # Log success
                        logger.info(f"Using {method} method for spectral indices")
                        return result
# If template method specifically requested or multi-method not found, use standard extraction
        if method == 'template' or method == 'default':
            return extract_spectral_indices(rdb_data, bins_limit=bins_limit)
            
        # If method not found but we have another method in multi-method, use the first available
        if 'bin_indices_multi' in rdb_data:
            bin_indices_multi = rdb_data['bin_indices_multi'].item() if hasattr(rdb_data['bin_indices_multi'], 'item') else rdb_data['bin_indices_multi']
            
            # Try methods in order of preference
            for fallback_method in ['auto', 'original', 'fit']:
                if fallback_method in bin_indices_multi:
                    logger.warning(f"Method {method} not found, falling back to {fallback_method}")
                    return extract_spectral_indices_by_method(rdb_data, method=fallback_method, bins_limit=bins_limit)
        
        # If all else fails, use standard extraction
        logger.warning(f"Method {method} not available, using standard template-based indices")
        return extract_spectral_indices(rdb_data, bins_limit=bins_limit)
    
    except Exception as e:
        logger.error(f"Error extracting spectral indices by method {method}: {e}")
        import traceback
        traceback.print_exc()
        
        # Fall back to standard extraction
        logger.warning(f"Falling back to standard method due to error")
        return extract_spectral_indices(rdb_data, bins_limit=bins_limit)

def _extract_additional_parameters(rdb_data, valid_bins, result):
    """Helper function to extract additional parameters like radius, age, etc."""
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
        
        if 'age' in stellar_pop:
            age = stellar_pop['age']
            # If age is in years, convert to Gyr
            if np.any(age > 100):  # Assuming age > 100 means it's in years
                age = age / 1e9
                
            if len(valid_bins) > 0 and len(age) >= max(valid_bins) + 1:
                result['bin_indices']['age'] = age[valid_bins]
        
        if 'metallicity' in stellar_pop:
            metallicity = stellar_pop['metallicity']
            
            if len(valid_bins) > 0 and len(metallicity) >= max(valid_bins) + 1:
                result['bin_indices']['metallicity'] = metallicity[valid_bins]
    
    # Extract radius for bins with the same valid bins filter
    if 'distance' in rdb_data and valid_bins is not None:
        distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
        if 'bin_distances' in distance:
            bin_distances = distance['bin_distances']
            
            if len(valid_bins) > 0 and len(bin_distances) >= max(valid_bins) + 1:
                result['bin_indices']['R'] = bin_distances[valid_bins]


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

def extract_alpha_fe_radius(galaxy_name, rdb_data, model_data, config=None):
    """
    Extract the alpha/Fe ratio as a function of radius for a galaxy using specified bins
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices and radii
    model_data : DataFrame
        Model grid data with ages, metallicities, and spectral indices
    config : dict, optional
        Bin configuration dictionary
        
    Returns:
    --------
    dict
        Dictionary containing alpha/Fe values and radii information
    """
    try:
        # Check if RDB data is available and valid
        if rdb_data is None or not isinstance(rdb_data, dict):
            logger.warning(f"Invalid RDB data format for {galaxy_name}")
            return None
            
        # Load bin configuration if not provided
        if config is None:
            config = load_bin_config()
            
        # Get bins to use for this galaxy
        bins_to_use = get_bins_to_use(galaxy_name, config)
        
        # Find the maximum bin index to extract
        max_bin_idx = max(bins_to_use) + 1 if bins_to_use else 6
            
        # Extract spectral indices from galaxy data with necessary bin limit
        galaxy_indices = extract_spectral_indices(rdb_data, bins_limit=max_bin_idx)
        
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Check for required data
        if ('bin_indices' not in galaxy_indices or
            'Fe5015' not in galaxy_indices['bin_indices'] or 
            'Mgb' not in galaxy_indices['bin_indices'] or 
            'Hbeta' not in galaxy_indices['bin_indices']):
            logger.warning(f"Missing required spectral indices for {galaxy_name}")
            return None
        
        # Get galaxy spectral indices for each bin
        galaxy_fe5015 = galaxy_indices['bin_indices']['Fe5015']
        galaxy_mgb = galaxy_indices['bin_indices']['Mgb']
        galaxy_hbeta = galaxy_indices['bin_indices']['Hbeta']
        
        # Check if arrays are valid
        if not hasattr(galaxy_fe5015, '__len__') or not hasattr(galaxy_mgb, '__len__') or not hasattr(galaxy_hbeta, '__len__'):
            logger.warning(f"Invalid index arrays for {galaxy_name}")
            return None
            
        # Make sure they all have the same length
        min_len = min(len(galaxy_fe5015), len(galaxy_mgb), len(galaxy_hbeta))
        galaxy_fe5015 = galaxy_fe5015[:min_len]
        galaxy_mgb = galaxy_mgb[:min_len]
        galaxy_hbeta = galaxy_hbeta[:min_len]
        
        # Get galaxy age if available (for better matching in model grid)
        if 'age' in galaxy_indices['bin_indices'] and hasattr(galaxy_indices['bin_indices']['age'], '__len__'):
            galaxy_age = galaxy_indices['bin_indices']['age'][:min_len]
        else:
            # Default age values
            galaxy_age = np.ones(min_len) * 5.0  # Use 5 Gyr as default age
        
        # Get galaxy radius information
        galaxy_radius = None
        if 'R' in galaxy_indices['bin_indices'] and hasattr(galaxy_indices['bin_indices']['R'], '__len__'):
            galaxy_radius = galaxy_indices['bin_indices']['R'][:min_len]
        elif 'distance' in rdb_data:
            distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
            if 'bin_distances' in distance:
                galaxy_radius = distance['bin_distances'][:min_len]
            elif 'binning' in rdb_data:
                binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
                if 'bin_radii' in binning:
                    galaxy_radius = binning['bin_radii'][:min_len]
        
        if galaxy_radius is None:
            logger.warning(f"No radius information found for {galaxy_name}")
            return None
        
        # Get effective radius if available
        Re = extract_effective_radius(rdb_data)
        
        # Scale radius by effective radius if available
        if Re is not None and Re > 0:
            r_scaled = galaxy_radius / Re
        else:
            r_scaled = galaxy_radius
        
        # Extract model column names
        fe5015_col = model_column_mapping['Fe5015']
        mgb_col = model_column_mapping['Mgb']
        hbeta_col = model_column_mapping['Hbeta']
        age_col = model_column_mapping['Age']
        aofe_col = model_column_mapping['AoFe']
        zoh_col = model_column_mapping['ZoH']
        
        # Get unique values of age, metallicity, and alpha/Fe in the model grid
        model_ages = sorted(model_data[age_col].unique())
        model_zoh = sorted(model_data[zoh_col].unique())
        model_aofe = sorted(model_data[aofe_col].unique())
        
        # Interpolate alpha/Fe for each bin - only using the specified bins
        alpha_fe_values = []
        interpolated_zoh = []
        valid_radii = []
        used_bins = []
        
        for i in range(min_len):
            # Skip bins that are not in the specified list
            if i not in bins_to_use:
                continue
                
            fe5015 = galaxy_fe5015[i]
            mgb = galaxy_mgb[i]
            hbeta = galaxy_hbeta[i]
            age = galaxy_age[i]
            radius = r_scaled[i]
            
            # Skip if any values are NaN
            if np.isnan(fe5015) or np.isnan(mgb) or np.isnan(hbeta) or np.isnan(radius):
                continue
                
            # Skip bins with negative spectral indices (physically invalid)
            if fe5015 < 0 or mgb < 0 or hbeta < 0:
                logger.warning(f"Skipping bin {i} for {galaxy_name} - negative spectral indices: "
                              f"Fe5015={fe5015:.2f}, Mgb={mgb:.2f}, Hbeta={hbeta:.2f}")
                continue
            
            # Find closest model age
            closest_age_idx = np.argmin(np.abs(np.array(model_ages) - age))
            closest_age = model_ages[closest_age_idx]
            
            # Filter model grid to points near this age
            age_filtered = model_data[model_data[age_col] == closest_age]
            
            # Find best match for this bin's indices
            best_alpha = None
            best_zoh = None
            min_distance = float('inf')
            
            for _, row in age_filtered.iterrows():
                # Calculate distance in index space
                fe5015_diff = (row[fe5015_col] - fe5015) / 5.0  # Normalize by typical range
                mgb_diff = (row[mgb_col] - mgb) / 4.0
                hbeta_diff = (row[hbeta_col] - hbeta) / 3.0
                
                distance = np.sqrt(fe5015_diff**2 + mgb_diff**2 + hbeta_diff**2)
                
                if distance < min_distance:
                    min_distance = distance
                    best_alpha = row[aofe_col]
                    best_zoh = row[zoh_col]
            
            if best_alpha is not None:
                alpha_fe_values.append(best_alpha)
                interpolated_zoh.append(best_zoh)
                valid_radii.append(radius)
                used_bins.append(i)  # Keep track of which bins were actually used
        
        # Calculate median values
        if len(alpha_fe_values) > 0:
            median_alpha_fe = np.median(alpha_fe_values)
            median_radius = np.median(valid_radii)
            median_zoh = np.median(interpolated_zoh)
            
            # Calculate slope of alpha/Fe vs. radius
            if len(alpha_fe_values) > 1:
                # Calculate linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(valid_radii, alpha_fe_values)
            else:
                slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
            
            # Get maximum radius for reporting
            max_radius = np.max(valid_radii) if len(valid_radii) > 0 else np.nan
            
            # Return results
            return {
                'galaxy': galaxy_name,
                'alpha_fe_median': median_alpha_fe,
                'alpha_fe_values': alpha_fe_values,
                'radius_median': median_radius,
                'radius_values': valid_radii,
                'effective_radius': Re,
                'metallicity_median': median_zoh,
                'slope': slope,  # Slope of alpha/Fe vs. radius
                'p_value': p_value,
                'r_squared': r_value**2 if not np.isnan(r_value) else np.nan,
                'std_err': std_err,
                'bins_used': ','.join(map(str, used_bins)) if used_bins else "",  # Ensure it's always a string
                'radial_bin_limit': max_radius  # Add max radius to the results
            }
        else:
            logger.warning(f"Could not calculate alpha/Fe for any bins in {galaxy_name}")
            return None
    
    except Exception as e:
        logger.error(f"Error extracting alpha/Fe for {galaxy_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_alpha_fe_direct(galaxy_name, rdb_data, model_data, bins_limit=6):
    """
    Calculate alpha/Fe for each data point using interpolation from the model grid
    Simplified version using only Fe5015 and Mgb indices
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices
    model_data : DataFrame
        Model grid data with indices and alpha/Fe values
    bins_limit : int
        Limit analysis to first N bins (default: 6 for bins 0-5)
    
    Returns:
    --------
    dict
        Dictionary containing points with their indices, radii, and interpolated alpha/Fe values
    """
    try:
        # Extract spectral indices from galaxy data with specified bin limit
        galaxy_indices = extract_spectral_indices(rdb_data, bins_limit=bins_limit)
        
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Check for required data
        if ('bin_indices' not in galaxy_indices or
            'Fe5015' not in galaxy_indices['bin_indices'] or 
            'Mgb' not in galaxy_indices['bin_indices']):
            logger.warning(f"Missing required spectral indices for {galaxy_name}")
            return None
        
        # Get galaxy spectral indices for each bin
        galaxy_fe5015 = galaxy_indices['bin_indices']['Fe5015']
        galaxy_mgb = galaxy_indices['bin_indices']['Mgb']
        
        # Get Hbeta if available (for display only, not used in interpolation)
        if 'Hbeta' in galaxy_indices['bin_indices']:
            galaxy_hbeta = galaxy_indices['bin_indices']['Hbeta']
        else:
            galaxy_hbeta = np.ones_like(galaxy_fe5015) * np.nan
        
        # Check if arrays are valid
        if not hasattr(galaxy_fe5015, '__len__') or not hasattr(galaxy_mgb, '__len__'):
            logger.warning(f"Invalid index arrays for {galaxy_name}")
            return None
            
        # Make sure they all have the same length
        min_len = min(len(galaxy_fe5015), len(galaxy_mgb))
        galaxy_fe5015 = galaxy_fe5015[:min_len]
        galaxy_mgb = galaxy_mgb[:min_len]
        galaxy_hbeta = galaxy_hbeta[:min_len] if len(galaxy_hbeta) >= min_len else np.ones(min_len) * np.nan
        
        # Get galaxy age if available
        if 'age' in galaxy_indices['bin_indices'] and hasattr(galaxy_indices['bin_indices']['age'], '__len__'):
            galaxy_age = galaxy_indices['bin_indices']['age'][:min_len]
        else:
            # Default age values
            galaxy_age = np.ones(min_len) * 5.0  # Use 5 Gyr as default age
        
        # Get radii
        galaxy_radius = None
        if 'R' in galaxy_indices['bin_indices'] and hasattr(galaxy_indices['bin_indices']['R'], '__len__'):
            galaxy_radius = galaxy_indices['bin_indices']['R'][:min_len]
        elif 'distance' in rdb_data:
            distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
            if 'bin_distances' in distance:
                galaxy_radius = distance['bin_distances'][:min_len]
            elif 'binning' in rdb_data:
                binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
                if 'bin_radii' in binning:
                    galaxy_radius = binning['bin_radii'][:min_len]
        
        if galaxy_radius is None:
            logger.warning(f"No radius information found for {galaxy_name}")
            return None
        
        # Get effective radius if available
        Re = extract_effective_radius(rdb_data)
        
        # Scale radius by effective radius if available
        if Re is not None and Re > 0:
            r_scaled = galaxy_radius / Re
        else:
            r_scaled = galaxy_radius
            
        # Extract model column names
        fe5015_col = model_column_mapping['Fe5015']
        mgb_col = model_column_mapping['Mgb']
        hbeta_col = model_column_mapping['Hbeta']
        age_col = model_column_mapping['Age']
        aofe_col = model_column_mapping['AoFe']
        zoh_col = model_column_mapping['ZoH']
        
        # Get unique values from the model grid
        model_ages = sorted(model_data[age_col].unique())
        
        # Store results for each individual point
        points = []
        
        # Process each data point
        for i in range(min_len):
            fe5015 = galaxy_fe5015[i]
            mgb = galaxy_mgb[i]
            hbeta = galaxy_hbeta[i]
            age = galaxy_age[i]
            radius = r_scaled[i]
            
            # Skip if Fe5015 or Mgb are NaN
            if np.isnan(fe5015) or np.isnan(mgb) or np.isnan(radius):
                continue
            
            # Skip negative values (invalid measurements)
            if fe5015 <= 0 or mgb <= 0:
                continue
            
            # Find closest model age
            closest_age_idx = np.argmin(np.abs(np.array(model_ages) - age))
            closest_age = model_ages[closest_age_idx]
            
            # Filter model grid to points near this age
            age_filtered = model_data[model_data[age_col] == closest_age]
            
            # Perform weighted interpolation using only Fe5015 and Mgb
            distances = []
            for _, row in age_filtered.iterrows():
                fe5015_diff = (row[fe5015_col] - fe5015) / 5.0  # Normalize by typical range
                mgb_diff = (row[mgb_col] - mgb) / 4.0
                
                # Use only Fe5015 and Mgb for distance calculation
                distance = np.sqrt(fe5015_diff**2 + mgb_diff**2)
                distances.append((distance, row[aofe_col], row[zoh_col]))
            
            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            k = min(5, len(distances))
            nearest_neighbors = distances[:k]
            
            # Apply inverse distance weighting for interpolation
            total_weight = 0
            weighted_alpha_sum = 0
            weighted_zoh_sum = 0
            
            for dist, alpha, zoh in nearest_neighbors:
                # Avoid division by zero
                weight = 1.0 / max(dist, 1e-6)
                total_weight += weight
                weighted_alpha_sum += alpha * weight
                weighted_zoh_sum += zoh * weight
            
            # Calculate weighted average
            interpolated_alpha = weighted_alpha_sum / total_weight
            interpolated_zoh = weighted_zoh_sum / total_weight
            distance_metric = nearest_neighbors[0][0]  # Distance to closest point
            
            # Store the interpolated values
            points.append({
                'radius': radius,
                'Fe5015': fe5015,
                'Mgb': mgb,
                'Hbeta': hbeta,
                'alpha_fe': interpolated_alpha,
                'metallicity': interpolated_zoh,
                'age': age,
                'distance_metric': distance_metric,
                'bin_index': i  # Store which bin this came from
            })
        
        # Check if we found any valid points
        if not points:
            logger.warning(f"No valid points found for {galaxy_name}")
            return None
            
        # Return results
        return {
            'galaxy': galaxy_name,
            'effective_radius': Re,
            'points': points,
            'bins_used': ','.join(map(str, [p['bin_index'] for p in points]))
        }
        
    except Exception as e:
        logger.error(f"Error calculating interpolated alpha/Fe for {galaxy_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_all_galaxies(all_galaxy_data, model_data, config_path='bins_config.yaml'):
    """
    Process all galaxies with the bin configuration
    
    Parameters:
    -----------
    all_galaxy_data : dict
        Dictionary mapping galaxy names to their RDB data
    model_data : DataFrame
        Model grid data with indices and alpha/Fe values
    config_path : str
        Path to the bin configuration file
        
    Returns:
    --------
    list
        List of results for each galaxy
    """
    # Load configuration
    config = load_bin_config(config_path)
    
    # Process each galaxy
    results = []
    
    for galaxy_name, galaxy_data in all_galaxy_data.items():
        logger.info(f"Processing {galaxy_name}...")
        
        # Get bins to use for this galaxy
        bins_to_use = get_bins_to_use(galaxy_name, config)
        logger.info(f"  Using bins: {bins_to_use}")
        
        # Extract alpha/Fe values 
        result = extract_alpha_fe_radius(galaxy_name, galaxy_data, model_data, config)
        
        # Add to results list even if None - we'll filter later
        results.append(result)
        
        # Print basic result - safely check if result exists and has required keys
        if result is None:
            logger.info(f"  No valid results obtained")
        elif 'slope' not in result or np.isnan(result['slope']):
            logger.info(f"  No valid gradient (insufficient bins)")
        else:
            logger.info(f"  Slope: {result['slope']:.3f}, p-value: {result.get('p_value', np.nan):.3f}")
            
            # Safely access bins_used key
            if 'bins_used' in result:
                logger.info(f"  Used bins: {result['bins_used']}")
            else:
                logger.info(f"  Used bins information not available")
    
    return results

#------------------------------------------------------------------------------
# Visualization Functions
#------------------------------------------------------------------------------

def create_alpha_radius_plot_improved(results_list, output_path=None, dpi=150):
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
            alpha_fe = result['alpha_fe_median']
            radius = result['radius_median']
            slope = result['slope']
            p_value = result['p_value']
            
            # Get color for this galaxy
            color = colors[galaxy_index % 20]
            galaxy_index += 1
            
            # Marker size and edge width based on statistical significance
            if np.isnan(p_value) or p_value > 0.05:
                edge_color = 'gray'  # Not significant
                linewidth = 1
                marker_size = 100
            else:
                edge_color = 'black'  # Significant
                linewidth = 2
                marker_size = 150
            
            # Plot point
            ax.scatter(radius, alpha_fe, s=marker_size, color=color, edgecolor=edge_color, 
                     linewidth=linewidth, zorder=10)
            
            # Add slope line if available
            if not np.isnan(slope) and len(result['alpha_fe_values']) > 1:
                # Get radius range - extend slightly for better visibility
                min_radius = max(0.1, np.min(result['radius_values']) * 0.8)
                max_radius = np.max(result['radius_values']) * 1.2
                
                # Calculate intercept from the median point
                intercept = alpha_fe - slope * radius
                
                # Plot slope line
                x_range = np.linspace(min_radius, max_radius, 100)
                y_range = slope * x_range + intercept
                ax.plot(x_range, y_range, '--', color=color, alpha=0.7, linewidth=2)
            
            # Format significance marker
            sig_marker = "*" if p_value < 0.05 else ""
            
            # Create legend entry with slope and p-value
            legend_elements.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                     markeredgecolor=edge_color, markeredgewidth=linewidth,
                     markersize=10, label=f"{galaxy}: α/Fe={alpha_fe:.2f}, Slope={slope:.3f}{sig_marker} (p={p_value:.3f})")
            )
        
        # Set labels and title
        ax.set_xlabel('R/Re', fontsize=14)
        ax.set_ylabel('[α/Fe]', fontsize=14)
        ax.set_title('Alpha Element Abundance vs. Radius for Virgo Cluster Galaxies', fontsize=16)
        
        # Set tick parameters
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add significance note
        ax.text(0.02, 0.02, "* p < 0.05 (statistically significant)", 
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Add legend with galaxy names, slopes and p-values
        # Place it outside the plot to avoid overlapping with data
        legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9, 
                         bbox_to_anchor=(1.02, 1), title='Galaxies')
        legend.get_title().set_fontsize(12)
        
        # Adjust layout to make room for legend
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved improved alpha/Fe vs. radius plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating improved alpha/Fe vs. radius plot: {e}")
        import traceback
        traceback.print_exc()

def create_fe5015_mgb_plot(results_list, output_path=None, dpi=150, model_data=None):
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
            
            # Check if we have the necessary data
            if 'points' in result and result['points']:
                # Only use points from explicitly selected bins
                bins_used = result.get('bins_used', '')
                if bins_used:
                    bins_list = [int(b.strip()) for b in bins_used.split(',') if b.strip().isdigit()]
                    
                    # Extract points only from the bins that were actually used
                    for point in result['points']:
                        # Skip if any values are invalid or if bin index not in the used bins
                        bin_index = point.get('bin_index', -1)
                        if bin_index not in bins_list:
                            continue
                            
                        fe5015 = point.get('Fe5015')
                        mgb = point.get('Mgb')
                        
                        # Skip if any spectral indices are invalid or negative
                        if (fe5015 is None or mgb is None or 
                            np.isnan(fe5015) or np.isnan(mgb) or 
                            fe5015 < 0 or mgb < 0):
                            continue
                        
                        fe5015_values.append(fe5015)
                        mgb_values.append(mgb)
                        radius_values.append(point.get('radius', 0))
                        alpha_fe_values.append(point.get('alpha_fe', 0))
                        galaxy_names.append(galaxy_name)
                else:
                    # If bins_used is not available, include all points but filter out negative values
                    for point in result['points']:
                        fe5015 = point.get('Fe5015')
                        mgb = point.get('Mgb')
                        
                        # Skip if any spectral indices are invalid or negative
                        if (fe5015 is None or mgb is None or 
                            np.isnan(fe5015) or np.isnan(mgb) or 
                            fe5015 < 0 or mgb < 0):
                            continue
                        
                        fe5015_values.append(fe5015)
                        mgb_values.append(mgb)
                        radius_values.append(point.get('radius', 0))
                        alpha_fe_values.append(point.get('alpha_fe', 0))
                        galaxy_names.append(galaxy_name)
        
        if not fe5015_values:
            logger.warning("No valid data points found for plotting")
            return
            
        # Plot 1: Points colored by radius
        scatter1 = ax1.scatter(fe5015_values, mgb_values, c=radius_values, 
                            cmap='viridis', s=80, alpha=0.8, edgecolor='black')
        
        # Add colorbar for radius
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('R/Re', fontsize=12)
        
        # Plot 2: Points colored by alpha/Fe
        scatter2 = ax2.scatter(fe5015_values, mgb_values, c=alpha_fe_values, 
                            cmap='plasma', s=80, alpha=0.8, edgecolor='black')
        
        # Add colorbar for alpha/Fe
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('[α/Fe]', fontsize=12)
        
        # Add model grid if provided
        if model_data is not None:
            try:
                # Find the relevant column names
                fe5015_col = find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index'])
                mgb_col = find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index'])
                aofe_col = find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
                age_col = find_matching_column(model_data, ['Age', 'age'])
                
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
                            ax1.plot(alpha_data[fe5015_col], alpha_data[mgb_col], 'k--', alpha=0.3, linewidth=1)
                            ax2.plot(alpha_data[fe5015_col], alpha_data[mgb_col], 'k--', alpha=0.3, linewidth=1)
                            
                            # Label each line
                            midpoint = len(alpha_data) // 2
                            x = alpha_data.iloc[midpoint][fe5015_col]
                            y = alpha_data.iloc[midpoint][mgb_col]
                            ax2.text(x, y, f'[α/Fe]={alpha_fe}', fontsize=8, ha='center', va='bottom', alpha=0.7,
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
                       fontsize=8, alpha=0.7)
        
        # Set labels and titles
        ax1.set_xlabel('Fe5015 Index', fontsize=14)
        ax1.set_ylabel('Mgb Index', fontsize=14)
        ax1.set_title('Fe5015 vs Mgb - colored by R/Re', fontsize=16)
        
        ax2.set_xlabel('Fe5015 Index', fontsize=14)
        ax2.set_ylabel('Mgb Index', fontsize=14)
        ax2.set_title('Fe5015 vs Mgb - colored by [α/Fe]', fontsize=16)
        
        # Add grid
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Set tick parameters
        for ax in [ax1, ax2]:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Add overall title
        plt.suptitle('Spectral Indices and Alpha Enhancement in Virgo Cluster Galaxies', fontsize=18, y=0.98)
        
        # Add note about interpolation method
        plt.figtext(0.5, 0.01, 
                  "α/Fe values interpolated from TMB03 model grid using Fe5015, Mgb and Hβ indices.\n"
                  "Model grid lines (dashed) show constant α/Fe paths for a 10 Gyr stellar population.",
                  ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
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

def create_alpha_radius_direct_plot(results_list, output_path=None, dpi=150):
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
            if result is None or 'points' not in result or not result['points']:
                continue
                
            galaxy_name = result['galaxy']
            
            # Get color for this galaxy
            color = colors[galaxy_index % 20]
            galaxy_index += 1
            
            # Extract radii and alpha/Fe values
            radii = [point['radius'] for point in result['points']]
            alpha_fe = [point['alpha_fe'] for point in result['points']]
            
            # Plot each point
            ax.scatter(radii, alpha_fe, s=80, color=color, edgecolor='black', alpha=0.8, zorder=10)
            
            # Calculate statistics
            median_alpha_fe = np.median(alpha_fe) if alpha_fe else np.nan
            
            # Fit linear slope
            if len(radii) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(radii, alpha_fe)
                
                # Add to legend with slope and p-value information
                sig_marker = "*" if p_value < 0.05 else ""
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                         markeredgecolor='black', markeredgewidth=1,
                         markersize=10, label=f"{galaxy_name}: median α/Fe={median_alpha_fe:.2f}, Slope={slope:.3f}{sig_marker} (p={p_value:.3f})")
                )
            else:
                # If only one point, just add the galaxy to legend without slope
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                         markeredgecolor='black', markeredgewidth=1,
                         markersize=10, label=f"{galaxy_name}: α/Fe={median_alpha_fe:.2f}")
                )
        
        # Set labels and title
        ax.set_xlabel('R/Re', fontsize=14)
        ax.set_ylabel('[α/Fe]', fontsize=14)
        ax.set_title('Alpha Element Abundance vs. Radius - Direct Grid Interpolation', fontsize=16)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add significance note
        ax.text(0.02, 0.02, "* p < 0.05 (statistically significant)", 
               transform=ax.transAxes, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Add legend
        legend = ax.legend(handles=legend_elements, loc='upper left', fontsize=10, framealpha=0.9, 
                         bbox_to_anchor=(1.02, 1), title='Galaxies')
        legend.get_title().set_fontsize(12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved direct alpha/Fe vs. radius plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating direct alpha/Fe vs. radius plot: {e}")
        import traceback
        traceback.print_exc()

def create_combined_flux_and_binning(galaxy_name, p2p_data, rdb_data, cube_info, output_path=None, dpi=150):
    """Create combined flux map and display bin boundaries directly on the map"""
    try:
        # Check if data is available
        p2p_available = p2p_data is not None and isinstance(p2p_data, dict)
        rdb_available = rdb_data is not None and isinstance(rdb_data, dict)
        
        if not rdb_available:
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
            cy, cx = ny // 2, nx // 2
            r = np.sqrt((x - cx)**2 + (y - cy)**2)
            flux_map = np.exp(-r / (max(nx, ny) / 4))
        
        # Extract binning information
        if 'binning' not in rdb_data:
            logger.error(f"No binning information found in RDB data for {galaxy_name}")
            return
        
        binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
        
        # Get bin numbers and bin radii
        bin_num = binning.get('bin_num', None)
        bin_radii = binning.get('bin_radii', None)
        n_rings = binning.get('n_rings', 0)
        if bin_radii is not None and hasattr(bin_radii, '__len__'):
            n_rings = max(n_rings, len(bin_radii))
        
        # Get center, PA and ellipticity
        center_x = binning.get('center_x', None)
        center_y = binning.get('center_y', None)
        pa = binning.get('pa', 0.0)
        ellipticity = binning.get('ellipticity', 0.0)
        
        # Get dimensions from flux map
        ny, nx = flux_map.shape
        
        # Create a figure with just the flux map
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get dimensions
        if center_x is None or center_y is None:
            center_x = nx / 2
            center_y = ny / 2
        
        # Extract effective radius
        Re = None
        if 'distance' in rdb_data:
            distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
            if 'effective_radius' in distance:
                Re = distance['effective_radius']
        elif 'meta_data' in rdb_data:
            meta = rdb_data['meta_data'].item() if hasattr(rdb_data['meta_data'], 'item') else rdb_data['meta_data']
            if 'effective_radius' in meta:
                Re = meta['effective_radius']
        elif 'effective_radius' in rdb_data:
            Re = rdb_data['effective_radius']
        
        # Get pixel scale information
        pixel_scale_x = cube_info['pixel_scale_x']  # arcsec/pixel
        pixel_scale_y = cube_info['pixel_scale_y']  # arcsec/pixel
        
        # Calculate full extent in arcseconds
        extent_x = nx * pixel_scale_x
        extent_y = ny * pixel_scale_y
        
        # Calculate the extent for the plot - centered at 0
        extent = [-extent_x/2, extent_x/2, -extent_y/2, extent_y/2]
        
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
        
        # Convert center from pixels to arcseconds for ellipse drawing
        center_x_arcsec = (center_x - nx/2) * pixel_scale_x
        center_y_arcsec = (center_y - ny/2) * pixel_scale_y
        
        # Draw elliptical bin boundaries on the flux map
        if bin_radii is not None and hasattr(bin_radii, '__len__') and len(bin_radii) > 0:
            for i, radius in enumerate(sorted(bin_radii)):  # Sort for proper drawing order
                # Create ellipse in arcseconds
                ellipse = Ellipse(
                    (center_x_arcsec, center_y_arcsec),
                    2 * radius,  # Major axis diameter in arcsec
                    2 * radius * (1 - ellipticity),  # Minor axis diameter in arcsec
                    angle=pa,
                    fill=False,
                    edgecolor='white',
                    linestyle='-',
                    linewidth=1.0,
                    alpha=0.7
                )
                ax.add_patch(ellipse)
                
                # Add bin number label to the first few bins
                if i < 6:  # Label bins 0-5
                    # Calculate position for bin label - slightly offset from ellipse edge
                    theta = np.radians(45)  # Place at 45 degrees
                    label_x = center_x_arcsec + (radius * 0.8) * np.cos(theta)
                    label_y = center_y_arcsec + (radius * 0.8 * (1 - ellipticity)) * np.sin(theta)
                    
                    # Add bin number
                    ax.text(label_x, label_y, str(i), 
                          color='white', fontsize=12, fontweight='bold', ha='center', va='center',
                          bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=2))
        
        # Add effective radius as a dashed red ellipse if available
        if Re is not None:
            ell_Re = Ellipse(
                (center_x_arcsec, center_y_arcsec),
                2 * Re,  # Major axis diameter in arcsec
                2 * Re * (1 - ellipticity),  # Minor axis diameter in arcsec
                angle=pa,
                fill=False,
                edgecolor='red',
                linestyle='--',
                linewidth=2.0,
                alpha=0.8
            )
            ax.add_patch(ell_Re)
            
            # Add label for Re
            ax.text(0, -extent_y/2 * 0.9, f'Re = {Re:.2f} arcsec', 
                   color='red', fontsize=12, fontweight='bold', ha='center', va='center',
                   bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=3))
        
        # Add North/East arrows
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
        
        # Add title and axis labels
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
            logger.info(f"Saved combined visualization to {output_path}")
        
        plt.close()
        
        return
        
    except Exception as e:
        logger.error(f"Error creating combined visualization: {e}")
        import traceback
        traceback.print_exc()

def create_interp_verification_plot(galaxy_name, rdb_data, model_data, output_path=None, dpi=150, bins_limit=6):
    """
    Create a visualization showing how alpha/Fe is interpolated across the entire spectral index plane
    
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
        Resolution for the image
    bins_limit : int
        Limit on the number of bins to analyze
    """
    try:
        # Extract spectral indices from galaxy data
        galaxy_indices = extract_spectral_indices(rdb_data, bins_limit=bins_limit)
        
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Check if necessary data is available
        if ('bin_indices' not in galaxy_indices or
            'Fe5015' not in galaxy_indices['bin_indices'] or 
            'Mgb' not in galaxy_indices['bin_indices'] or 
            'Hbeta' not in galaxy_indices['bin_indices']):
            logger.warning(f"Missing required spectral indices for {galaxy_name}")
            return
        
        # Get galaxy spectral indices
        galaxy_fe5015 = galaxy_indices['bin_indices']['Fe5015']
        galaxy_mgb = galaxy_indices['bin_indices']['Mgb']
        galaxy_hbeta = galaxy_indices['bin_indices']['Hbeta']
        
        # Get age if available
        if 'age' in galaxy_indices['bin_indices']:
            galaxy_age = galaxy_indices['bin_indices']['age']
        else:
            # Default age
            galaxy_age = np.ones_like(galaxy_fe5015) * 5.0  # 5 Gyr default
        
        # Calculate mean age for model grid selection
        if hasattr(galaxy_age, '__len__') and len(galaxy_age) > 0:
            mean_age = np.nanmean(galaxy_age)
        else:
            mean_age = 5.0  # Default
            
        # Get column references
        age_column = model_column_mapping['Age']
        zoh_column = model_column_mapping['ZoH']
        aofe_column = model_column_mapping['AoFe']
        fe5015_col = model_column_mapping['Fe5015']
        mgb_col = model_column_mapping['Mgb']
        hbeta_col = model_column_mapping['Hbeta']
        
        # Find closest age in model grid
        available_ages = np.array(model_data[age_column].unique())
        closest_age = available_ages[np.argmin(np.abs(available_ages - mean_age))]
        
        # Filter model grid to this age
        model_age_data = model_data[model_data[age_column] == closest_age]
        
        # Get unique alpha/Fe and metallicity values
        unique_aofe = sorted(model_age_data[aofe_column].unique())
        unique_zoh = sorted(model_age_data[zoh_column].unique())
        
        # Create normalization for colormap
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=min(unique_aofe), vmax=max(unique_aofe))
        
        # Create figure with 2x2 grid of plots
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(2, 2)
        
        # Create the four plots
        ax1 = fig.add_subplot(gs[0, 0])  # Fe5015 vs Mgb - Model Grid
        ax2 = fig.add_subplot(gs[0, 1])  # Fe5015 vs Mgb - Interpolated Surface
        ax3 = fig.add_subplot(gs[1, 0])  # Mgb vs Hbeta - Model Grid
        ax4 = fig.add_subplot(gs[1, 1])  # Mgb vs Hbeta - Interpolated Surface
        
        # Get galaxy data points
        direct_result = calculate_alpha_fe_direct(galaxy_name, rdb_data, model_data, bins_limit=bins_limit)
        
        if direct_result is None or 'points' not in direct_result or not direct_result['points']:
            logger.warning(f"No alpha/Fe interpolation results for {galaxy_name}")
            return
            
        # Extract data points
        points = direct_result['points']
        
        # Collect data for plotting
        fe5015_values = [point['Fe5015'] for point in points]
        mgb_values = [point['Mgb'] for point in points]
        hbeta_values = [point['Hbeta'] for point in points]
        original_alpha_fe_values = [point['alpha_fe'] for point in points]
        radius_values = [point['radius'] for point in points]
        bin_indices = [i for i in range(len(fe5015_values))]
        
        # Calculate median values for fixed dimensions
        fixed_hbeta = np.median(hbeta_values) if len(hbeta_values) > 0 else 2.0
        fixed_fe5015 = np.median(fe5015_values) if len(fe5015_values) > 0 else 5.0
        
        # Create 2D interpolation for Fe5015 vs Mgb (fixing Hbeta)
        # Import necessary functions
        from scipy.interpolate import griddata, Rbf, SmoothBivariateSpline
        from scipy.ndimage import gaussian_filter
        
        # Define grid for Fe5015 vs Mgb with high resolution for smoother contours
        if len(fe5015_values) > 0 and len(mgb_values) > 0:
            fe5015_min = max(0, min(fe5015_values) - 1.5)
            fe5015_max = max(fe5015_values) + 1.5
            mgb_min = max(0, min(mgb_values) - 1.5)
            mgb_max = max(mgb_values) + 1.5
        else:
            # Default ranges if no galaxy data
            fe5015_min, fe5015_max = 0, 10
            mgb_min, mgb_max = 0, 5
        
        # Create higher resolution grid for smoother interpolation
        fe5015_grid = np.linspace(fe5015_min, fe5015_max, 200)  # Increased from 100 to 200
        mgb_grid = np.linspace(mgb_min, mgb_max, 200)  # Increased from 100 to 200
        
        # Create 2D grid
        fe5015_mesh, mgb_mesh = np.meshgrid(fe5015_grid, mgb_grid)
        
        # Extract model data for interpolation
        model_fe5015 = model_age_data[fe5015_col].values
        model_mgb = model_age_data[mgb_col].values
        model_alpha_fe = model_age_data[aofe_column].values
        
        # Combine all model points regardless of Hbeta value (2D interpolation simplification)
        model_points = np.column_stack([model_fe5015, model_mgb])
        
        # Use griddata for initial interpolation
        alpha_fe_grid_fe_mgb = griddata(model_points, model_alpha_fe, 
                                     (fe5015_mesh.flatten(), mgb_mesh.flatten()), 
                                     method='linear')
        # Fill in missing values with nearest interpolation
        alpha_fe_grid_fe_mgb_nearest = griddata(model_points, model_alpha_fe, 
                                           (fe5015_mesh.flatten(), mgb_mesh.flatten()), 
                                           method='nearest')
        alpha_fe_grid_fe_mgb = np.where(np.isnan(alpha_fe_grid_fe_mgb), 
                                     alpha_fe_grid_fe_mgb_nearest, 
                                     alpha_fe_grid_fe_mgb)
        
        # Reshape to 2D grid
        alpha_fe_grid_fe_mgb = alpha_fe_grid_fe_mgb.reshape(fe5015_mesh.shape)
        
        # Apply gaussian filter to smooth the grid
        alpha_fe_grid_fe_mgb = gaussian_filter(alpha_fe_grid_fe_mgb, sigma=2.0)
        
        # Also calculate interpolated values for the actual data points
        interpolated_alpha_fe_fe_mgb = griddata(model_points, model_alpha_fe, 
                                           (np.array(fe5015_values), np.array(mgb_values)), 
                                           method='linear')
        # Fill in any missing values
        interpolated_alpha_fe_fe_mgb_nearest = griddata(model_points, model_alpha_fe, 
                                                  (np.array(fe5015_values), np.array(mgb_values)), 
                                                  method='nearest')
        interpolated_alpha_fe_fe_mgb = np.where(np.isnan(interpolated_alpha_fe_fe_mgb), 
                                           interpolated_alpha_fe_fe_mgb_nearest, 
                                           interpolated_alpha_fe_fe_mgb)
        
        # Create 2D interpolation for Mgb vs Hbeta (fixing Fe5015)
        # Define grid for Mgb vs Hbeta with higher resolution
        if len(mgb_values) > 0 and len(hbeta_values) > 0:
            mgb_min = max(0, min(mgb_values) - 1.5)
            mgb_max = max(mgb_values) + 1.5
            hbeta_min = max(0, min(hbeta_values) - 1.5)
            hbeta_max = max(hbeta_values) + 1.5
        else:
            # Default ranges if no galaxy data
            mgb_min, mgb_max = 0, 5
            hbeta_min, hbeta_max = 0, 5
        
        # Create finer grid for interpolation
        mgb_grid = np.linspace(mgb_min, mgb_max, 200)  # Increased from 100 to 200
        hbeta_grid = np.linspace(hbeta_min, hbeta_max, 200)  # Increased from 100 to 200
        
        # Create 2D grid
        mgb_mesh, hbeta_mesh = np.meshgrid(mgb_grid, hbeta_grid)
        
        # Extract model data for interpolation
        model_mgb = model_age_data[mgb_col].values
        model_hbeta = model_age_data[hbeta_col].values
        
        # Combine model points for Mgb vs Hbeta
        model_points = np.column_stack([model_mgb, model_hbeta])
        
        # Use griddata for interpolation
        alpha_fe_grid_mgb_hb = griddata(model_points, model_alpha_fe, 
                                     (mgb_mesh.flatten(), hbeta_mesh.flatten()), 
                                     method='linear')
        # Fill in missing values
        alpha_fe_grid_mgb_hb_nearest = griddata(model_points, model_alpha_fe, 
                                           (mgb_mesh.flatten(), hbeta_mesh.flatten()), 
                                           method='nearest')
        alpha_fe_grid_mgb_hb = np.where(np.isnan(alpha_fe_grid_mgb_hb), 
                                     alpha_fe_grid_mgb_hb_nearest, 
                                     alpha_fe_grid_mgb_hb)
        
        # Reshape to 2D grid
        alpha_fe_grid_mgb_hb = alpha_fe_grid_mgb_hb.reshape(mgb_mesh.shape)
        
        # Apply gaussian filter to smooth the grid
        alpha_fe_grid_mgb_hb = gaussian_filter(alpha_fe_grid_mgb_hb, sigma=2.0)
        
        # Also calculate interpolated values for the actual data points
        interpolated_alpha_fe_mgb_hb = griddata(model_points, model_alpha_fe, 
                                           (np.array(mgb_values), np.array(hbeta_values)), 
                                           method='linear')
        # Fill in any missing values
        interpolated_alpha_fe_mgb_hb_nearest = griddata(model_points, model_alpha_fe, 
                                                  (np.array(mgb_values), np.array(hbeta_values)), 
                                                  method='nearest')
        interpolated_alpha_fe_mgb_hb = np.where(np.isnan(interpolated_alpha_fe_mgb_hb), 
                                           interpolated_alpha_fe_mgb_hb_nearest, 
                                           interpolated_alpha_fe_mgb_hb)
        
        # Create combined array of interpolated values for later use
        # Average the two interpolation methods
        interpolated_alpha_fe = (interpolated_alpha_fe_fe_mgb + interpolated_alpha_fe_mgb_hb) / 2.0
        
        # Store the interpolated alpha/Fe values in a dictionary along with radius and bin indices
        interpolated_results = {
            'alpha_fe': interpolated_alpha_fe,
            'alpha_fe_fe_mgb': interpolated_alpha_fe_fe_mgb,
            'alpha_fe_mgb_hb': interpolated_alpha_fe_mgb_hb,
            'radius': np.array(radius_values),
            'bin_indices': np.array(bin_indices)
        }
        
        # Save this for use in other functions (could be returned or passed)
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        np.savez(os.path.splitext(output_path)[0] + '_interpolated_data.npz', **interpolated_results)
        
        # Plot 1: Fe5015 vs Mgb - Model Grid with galaxy points
        # Draw the model grid
        for aofe in unique_aofe:
            # Get points for this alpha/Fe value
            aofe_data = model_age_data[model_age_data[aofe_column] == aofe]
            
            # Sort by metallicity
            aofe_data = aofe_data.sort_values(zoh_column)
            
            # Draw the line on the plot
            ax1.plot(aofe_data[fe5015_col], aofe_data[mgb_col], '-', 
                   color=plt.cm.plasma(norm(aofe)), linewidth=2, alpha=0.7,
                   label=f'[α/Fe] = {aofe:.1f}')
                   
        # Mark the contours of constant metallicity
        for zoh in unique_zoh:
            # Get points for this metallicity
            zoh_data = model_age_data[model_age_data[zoh_column] == zoh]
            
            # Sort by alpha/Fe
            zoh_data = zoh_data.sort_values(aofe_column)
            
            # Draw the line on the plot
            ax1.plot(zoh_data[fe5015_col], zoh_data[mgb_col], '--', 
                   color='gray', linewidth=1, alpha=0.5)
            
            # Add label at the end of the line
            if len(zoh_data) > 0:
                x = zoh_data[fe5015_col].iloc[-1]
                y = zoh_data[mgb_col].iloc[-1]
                ax1.text(x, y, f'[Z/H]={zoh:.1f}', fontsize=8, 
                       ha='left', va='bottom', color='gray',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Draw galaxy data points with bin numbers - use the interpolated alpha/Fe values for the plot
        sc1 = ax1.scatter(fe5015_values, mgb_values, c=interpolated_alpha_fe_fe_mgb, 
                        cmap='plasma', s=100, zorder=10, edgecolor='black', norm=norm)
        
        # Add bin numbers
        for i, (x, y) in enumerate(zip(fe5015_values, mgb_values)):
            ax1.text(x, y, str(i), fontsize=9, ha='center', va='center',
                   color='white', fontweight='bold', zorder=11)
        
        # Add colorbar
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label('[α/Fe]')
        
        # Set labels and title
        ax1.set_xlabel('Fe5015 Index', fontsize=12)
        ax1.set_ylabel('Mgb Index', fontsize=12)
        ax1.set_title('Fe5015 vs Mgb - Model Grid with Galaxy Data', fontsize=14)
        
        # Add grid
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Fe5015 vs Mgb - Interpolated Surface
        
        # Create filled contour plot with more levels for smoother color transitions
        contour = ax2.contourf(fe5015_mesh, mgb_mesh, alpha_fe_grid_fe_mgb, 
                             levels=50, cmap='plasma', alpha=0.8, norm=norm)
        
        # Add colorbar
        cbar2 = plt.colorbar(contour, ax=ax2)
        cbar2.set_label('[α/Fe] (interpolated)')
        
        # Add contour lines with labels
        contour_lines = ax2.contour(fe5015_mesh, mgb_mesh, alpha_fe_grid_fe_mgb, 
                                  levels=unique_aofe, colors='black', linewidths=1)
        ax2.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        # Draw galaxy data points with bin numbers - use the same interpolated values
        ax2.scatter(fe5015_values, mgb_values, c=interpolated_alpha_fe_fe_mgb, 
                  cmap='plasma', s=100, zorder=10, edgecolor='black', norm=norm)
        
        # Add bin numbers
        for i, (x, y) in enumerate(zip(fe5015_values, mgb_values)):
            ax2.text(x, y, str(i), fontsize=9, ha='center', va='center',
                   color='white', fontweight='bold', zorder=11)
        
        # Set labels and title
        ax2.set_xlabel('Fe5015 Index', fontsize=12)
        ax2.set_ylabel('Mgb Index', fontsize=12)
        ax2.set_title('Fe5015 vs Mgb - Interpolated [α/Fe] Surface (fixed Hβ)', fontsize=14)
        
        # Add fixed Hbeta value annotation
        ax2.text(0.05, 0.95, f"Fixed Hβ = {fixed_hbeta:.2f}", 
               transform=ax2.transAxes, fontsize=10,
               va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add grid
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 3: Mgb vs Hbeta - Model Grid
        # Draw the model grid
        for aofe in unique_aofe:
            # Get points for this alpha/Fe value
            aofe_data = model_age_data[model_age_data[aofe_column] == aofe]
            
            # Sort by metallicity
            aofe_data = aofe_data.sort_values(zoh_column)
            
            # Draw the line on the plot
            ax3.plot(aofe_data[mgb_col], aofe_data[hbeta_col], '-', 
                   color=plt.cm.plasma(norm(aofe)), linewidth=2, alpha=0.7)
        
        # Mark the contours of constant metallicity
        for zoh in unique_zoh:
            # Get points for this metallicity
            zoh_data = model_age_data[model_age_data[zoh_column] == zoh]
            
            # Sort by alpha/Fe
            zoh_data = zoh_data.sort_values(aofe_column)
            
            # Draw the line on the plot
            ax3.plot(zoh_data[mgb_col], zoh_data[hbeta_col], '--', 
                   color='gray', linewidth=1, alpha=0.5)
            
            # Add label at the end of the line
            if len(zoh_data) > 0:
                x = zoh_data[mgb_col].iloc[-1]
                y = zoh_data[hbeta_col].iloc[-1]
                ax3.text(x, y, f'[Z/H]={zoh:.1f}', fontsize=8, 
                       ha='left', va='bottom', color='gray',
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Draw galaxy data points with bin numbers - use the interpolated values
        sc3 = ax3.scatter(mgb_values, hbeta_values, c=interpolated_alpha_fe_mgb_hb, 
                        cmap='plasma', s=100, zorder=10, edgecolor='black', norm=norm)
        
        # Add bin numbers
        for i, (x, y) in enumerate(zip(mgb_values, hbeta_values)):
            ax3.text(x, y, str(i), fontsize=9, ha='center', va='center',
                   color='white', fontweight='bold', zorder=11)
        
        # Add colorbar
        cbar3 = plt.colorbar(sc3, ax=ax3)
        cbar3.set_label('[α/Fe]')
        
        # Set labels and title
        ax3.set_xlabel('Mgb Index', fontsize=12)
        ax3.set_ylabel('Hβ Index', fontsize=12)
        ax3.set_title('Mgb vs Hβ - Model Grid with Galaxy Data', fontsize=14)
        
        # Add grid
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 4: Mgb vs Hbeta - Interpolated Surface
        
        # Create filled contour plot with more levels for smoother color transitions
        contour = ax4.contourf(mgb_mesh, hbeta_mesh, alpha_fe_grid_mgb_hb, 
                             levels=50, cmap='plasma', alpha=0.8, norm=norm)
        
        # Add colorbar
        cbar4 = plt.colorbar(contour, ax=ax4)
        cbar4.set_label('[α/Fe] (interpolated)')
        
        # Add contour lines with labels
        contour_lines = ax4.contour(mgb_mesh, hbeta_mesh, alpha_fe_grid_mgb_hb, 
                                  levels=unique_aofe, colors='black', linewidths=1)
        ax4.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1f')
        
        # Draw galaxy data points with bin numbers - use the interpolated values
        ax4.scatter(mgb_values, hbeta_values, c=interpolated_alpha_fe_mgb_hb, 
                  cmap='plasma', s=100, zorder=10, edgecolor='black', norm=norm)
        
        # Add bin numbers
        for i, (x, y) in enumerate(zip(mgb_values, hbeta_values)):
            ax4.text(x, y, str(i), fontsize=9, ha='center', va='center',
                   color='white', fontweight='bold', zorder=11)
        
        # Set labels and title
        ax4.set_xlabel('Mgb Index', fontsize=12)
        ax4.set_ylabel('Hβ Index', fontsize=12)
        ax4.set_title('Mgb vs Hβ - Interpolated [α/Fe] Surface (fixed Fe5015)', fontsize=14)
        
        # Add fixed Fe5015 value annotation
        ax4.text(0.05, 0.95, f"Fixed Fe5015 = {fixed_fe5015:.2f}", 
               transform=ax4.transAxes, fontsize=10,
               va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add grid
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # Add overall title
        plt.suptitle(f"Galaxy {galaxy_name}: [α/Fe] Interpolation Verification\nModel Age: {closest_age} Gyr", 
                   fontsize=16, y=0.98)
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                  "Left: Model grid lines with galaxy data points. Right: Interpolated [α/Fe] surface (2D projections) with galaxy data.\n"
                  "The color of each galaxy point indicates its interpolated [α/Fe] value. Bin numbers shown for each point.",
                  ha='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved interpolation verification plot to {output_path}")
        
        plt.close()
        
        return interpolated_results
        
    except Exception as e:
        logger.error(f"Error creating interpolation verification plot: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_model_grid_plots_part1(galaxy_name, rdb_data, model_data, age=1, output_path=None, dpi=150, bins_limit=6):
    """
    Create first set of model grid plots colored by R, log Age, and M/H
    Part 1: Fe5015 vs Mgb, Fe5015 vs Hbeta, Mgb vs Hbeta - Colored by R
    """
    try:
        # Extract spectral indices from galaxy data with bin limit
        galaxy_indices = extract_spectral_indices(rdb_data, bins_limit=bins_limit)
        
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
        
        # Get unique values for grid lines
        zoh_column = model_column_mapping['ZoH']
        aofe_column = model_column_mapping['AoFe']
        
        zoh_unique = sorted(model_age_data[zoh_column].unique())
        aofe_unique = sorted(model_age_data[aofe_column].unique())
        
        # Generate a complete model grid with all combinations
        for row, color_var in enumerate(color_vars):
            for col, (x_index, y_index) in enumerate(index_pairs):
                ax = axes[row, col]
                
                # Get column names for model data
                model_x_col = model_column_mapping[x_index]
                model_y_col = model_column_mapping[y_index]
                
                # Skip if data not available
                if (x_index not in galaxy_indices['bin_indices'] or 
                    y_index not in galaxy_indices['bin_indices']):
                    ax.text(0.5, 0.5, f"Missing {x_index} or {y_index} data", 
                          transform=ax.transAxes, ha='center', va='center')
                    continue
                
                # First plot the model grid points for reference (very small markers)
                ax.scatter(model_age_data[model_x_col], model_age_data[model_y_col], 
                         color='black', s=5, alpha=0.2, zorder=1)
                
                # Plot Z/H grid lines (connect points with same alpha/Fe but different Z/H)
                for aofe in aofe_unique:
                    aofe_data = model_age_data[model_age_data[aofe_column] == aofe]
                    if len(aofe_data) > 1:
                        # Sort by metallicity for proper line drawing
                        aofe_data = aofe_data.sort_values(by=zoh_column)
                        
                        # Draw the grid line
                        ax.plot(aofe_data[model_x_col], aofe_data[model_y_col], '-', 
                               color='black', alpha=0.3, linewidth=1, zorder=2)
                        
                        # Add label for the line in the middle point
                        if len(aofe_data) > 2:
                            mid_idx = len(aofe_data) // 2
                            mid_point = aofe_data.iloc[mid_idx]
                            ax.text(mid_point[model_x_col], mid_point[model_y_col], 
                                  f'[α/Fe]={aofe:.1f}', fontsize=8, ha='center', va='center',
                                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                                  alpha=0.7, zorder=3)
                
                # Plot α/Fe grid lines (connect points with same Z/H but different alpha/Fe)
                for zoh in zoh_unique:
                    zoh_data = model_age_data[np.isclose(model_age_data[zoh_column], zoh, atol=0.05)]
                    if len(zoh_data) > 1:
                        # Sort by alpha/Fe for proper line drawing
                        zoh_data = zoh_data.sort_values(by=aofe_column)
                        
                        # Draw the grid line with dashed style
                        ax.plot(zoh_data[model_x_col], zoh_data[model_y_col], '--', 
                               color='red', alpha=0.3, linewidth=1, zorder=2)
                        
                        # Add label for the line
                        if len(zoh_data) > 2:
                            mid_idx = len(zoh_data) // 2
                            mid_point = zoh_data.iloc[mid_idx]
                            ax.text(mid_point[model_x_col], mid_point[model_y_col], 
                                  f'[Z/H]={zoh:.1f}', fontsize=8, ha='center', va='center',
                                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                                  alpha=0.7, zorder=3)
                
                # Get galaxy data
                galaxy_x = galaxy_indices['bin_indices'][x_index]
                galaxy_y = galaxy_indices['bin_indices'][y_index]
                
                # Color by the row's variable if available
                if color_var in galaxy_indices['bin_indices']:
                    color_data = galaxy_indices['bin_indices'][color_var]
                    
                    # Plot galaxy data with coloring
                    sc = ax.scatter(galaxy_x, galaxy_y, c=color_data, cmap='viridis', 
                                  s=120, edgecolor='black', linewidth=1.5, zorder=10)
                    
                    # Add bin numbers with white text
                    for i in range(len(galaxy_x)):
                        ax.text(galaxy_x[i], galaxy_y[i], str(i), 
                               color='white', fontweight='bold', ha='center', va='center',
                               fontsize=10, zorder=11)
                    
                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(sc, cax=cax)
                    cbar.set_label(color_labels[row])
                else:
                    # Plot without color coding
                    ax.scatter(galaxy_x, galaxy_y, color='blue', s=120, edgecolor='black', zorder=10)
                    for i in range(len(galaxy_x)):
                        ax.text(galaxy_x[i], galaxy_y[i], str(i), 
                               color='white', fontweight='bold', ha='center', va='center',
                               fontsize=10, zorder=11)
                
                # Add labels, title and grid
                ax.set_xlabel(f'{x_index} Index', fontsize=12)
                ax.set_ylabel(f'{y_index} Index', fontsize=12)
                ax.set_title(f'{x_index} vs {y_index} - colored by {color_labels[row]}', fontsize=14)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Set tick parameters
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Add overall title
        plt.suptitle(f"Galaxy {galaxy_name}: Spectral Indices vs. Model Grid (Age = {closest_age} Gyr)", 
                   fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved model grid plots part 1 to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating model grid plots part 1: {e}")
        import traceback
        traceback.print_exc()

def create_model_grid_plots_part2(galaxy_name, rdb_data, model_data, ages=[1, 3, 10], output_path=None, dpi=150, bins_limit=6):
    """
    Create second set of model grid plots for multiple ages with age-colored data points
    """
    try:
        # Extract spectral indices from galaxy data with bin limit
        galaxy_indices = extract_spectral_indices(rdb_data, bins_limit=bins_limit)
        
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Create figure with 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Get column references
        age_column = model_column_mapping['Age']
        zoh_column = model_column_mapping['ZoH']
        aofe_column = model_column_mapping['AoFe']
        
        # Choose model ages to use - find closest available ages
        available_ages = np.array(sorted(model_data[age_column].unique()))
        model_ages = []
        for age in ages:
            closest_idx = np.argmin(np.abs(available_ages - age))
            closest_age = available_ages[closest_idx]
            if closest_age not in model_ages:  # Avoid duplicates
                model_ages.append(closest_age)
        
        # Use distinct colors for ages
        age_colors = ['purple', 'teal', 'gold']  # More distinct colors
        
        # Set up different metallicity values for each row
        metallicity_values = [-1.0, 0.0, 0.5]  # Different [Z/H] for each row
        
        # Set up index pairs to plot
        index_pairs = [
            ('Fe5015', 'Mgb'),
            ('Fe5015', 'Hbeta'),
            ('Mgb', 'Hbeta')
        ]
        
        # Create a more detailed version of each subplot
        for row, zoh_value in enumerate(metallicity_values):
            # Find closest metallicity in the model
            available_zoh = np.array(sorted(model_data[zoh_column].unique()))
            closest_zoh_idx = np.argmin(np.abs(available_zoh - zoh_value))
            closest_zoh = available_zoh[closest_zoh_idx]
            
            for col, (x_index, y_index) in enumerate(index_pairs):
                ax = axes[row, col]
                
                # Model column references
                model_x_col = model_column_mapping[x_index]
                model_y_col = model_column_mapping[y_index]
                
                # Skip if indices not available
                if (x_index not in galaxy_indices['bin_indices'] or 
                    y_index not in galaxy_indices['bin_indices']):
                    ax.text(0.5, 0.5, f"Missing {x_index} or {y_index} data", 
                          transform=ax.transAxes, ha='center', va='center')
                    continue
                
                # Draw grid for each age with enhanced visibility
                for i, age in enumerate(model_ages):
                    # Get data for this age
                    age_data = model_data[model_data[age_column] == age]
                    
                    # Filter data close to the target metallicity
                    zoh_tolerance = 0.15  # Broaden this for a more complete grid
                    zoh_mask = np.abs(age_data[zoh_column] - closest_zoh) <= zoh_tolerance
                    zoh_data = age_data[zoh_mask]
                    
                    if len(zoh_data) > 1:
                        # For each alpha/Fe, plot a line 
                        for aofe in sorted(age_data[aofe_column].unique()):
                            aofe_data = zoh_data[zoh_data[aofe_column] == aofe]
                            if len(aofe_data) > 1:
                                # Plot line
                                ax.plot(aofe_data[model_x_col], aofe_data[model_y_col], '-', 
                                       color=age_colors[i], linewidth=2.5, alpha=0.7)
                        
                        # Add grid markers
                        ax.scatter(zoh_data[model_x_col], zoh_data[model_y_col], 
                                 color=age_colors[i], alpha=0.5, s=30)
                
                # Get galaxy data
                galaxy_x = galaxy_indices['bin_indices'][x_index]
                galaxy_y = galaxy_indices['bin_indices'][y_index]
                
                # Get galaxy age if available for color mapping
                if 'age' in galaxy_indices['bin_indices']:
                    galaxy_age = galaxy_indices['bin_indices']['age']
                    
                    # Plot with age-based coloring
                    sc = ax.scatter(galaxy_x, galaxy_y, c=galaxy_age, cmap='plasma', 
                                  s=120, edgecolor='black', linewidth=1.5, zorder=10)
                    
                    # Add bin numbers
                    for j in range(len(galaxy_x)):
                        ax.text(galaxy_x[j], galaxy_y[j], str(j), 
                               color='white', fontweight='bold', ha='center', va='center',
                               fontsize=10, zorder=11)
                    
                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(sc, cax=cax)
                    cbar.set_label('log Age (Gyr)')
                else:
                    # Plot without age coloring
                    ax.scatter(galaxy_x, galaxy_y, color='red', s=120, edgecolor='black', zorder=10)
                    for j in range(len(galaxy_x)):
                        ax.text(galaxy_x[j], galaxy_y[j], str(j), 
                               color='white', fontweight='bold', ha='center', va='center',
                               fontsize=10, zorder=11)
                
                # Labels and grid
                ax.set_xlabel(f'{x_index} Index', fontsize=12)
                ax.set_ylabel(f'{y_index} Index', fontsize=12)
                ax.set_title(f'{x_index} vs {y_index} - [Z/H]={closest_zoh:.1f}', fontsize=14)
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Set tick parameters
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Add legend for model lines
        legend_elements = []
        for i, age in enumerate(model_ages):
            legend_elements.append(
                Line2D([0], [0], color=age_colors[i], linewidth=2.5, label=f'Age = {age} Gyr')
            )
            
        # Add galaxy marker to legend
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red' if 'age' not in galaxy_indices['bin_indices'] else 'blue',
                 markersize=8, markeredgecolor='black', label='Galaxy Bins')
        )
        
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements),
                 bbox_to_anchor=(0.5, 0.05), fontsize=12)
        
        # Add overall title
        plt.suptitle(f'Galaxy {galaxy_name}: Spectral Indices vs. Model Grid at Multiple Ages', 
                   fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.07, 1, 0.95])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved model grid plots part 2 to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating model grid plots part 2: {e}")
        import traceback
        traceback.print_exc()

def create_spectral_index_interpolation_plot(galaxy_name, rdb_data, model_data, output_path=None, dpi=150, bins_limit=6):
    """
    Create a visualization showing how alpha/Fe is interpolated from spectral indices
    Using only Fe5015 vs Mgb for interpolation (simplified version)
    """
    try:
        # Extract spectral indices from galaxy data
        galaxy_indices = extract_spectral_indices(rdb_data, bins_limit=bins_limit)
        
        # Get alpha/Fe calculations
        direct_result = calculate_alpha_fe_direct(galaxy_name, rdb_data, model_data, bins_limit=bins_limit)
        
        if direct_result is None or 'points' not in direct_result or not direct_result['points']:
            logger.warning(f"No alpha/Fe interpolation results for {galaxy_name}")
            return
            
        # Extract data points
        points = direct_result['points']
        
        # Create a figure with 2x2 grid to show the interpolation process
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Plot 1: Fe5015 vs Mgb (colored by alpha/Fe)
        ax1 = axes[0, 0]
        
        # Extract values
        fe5015_values = [point['Fe5015'] for point in points]
        mgb_values = [point['Mgb'] for point in points]
        hbeta_values = [point['Hbeta'] for point in points]
        alpha_fe_values = [point['alpha_fe'] for point in points]
        
        # Plot points
        sc1 = ax1.scatter(fe5015_values, mgb_values, c=alpha_fe_values, cmap='plasma', 
                      s=120, edgecolor='black', linewidth=1.5)
        
        # Add bin numbers
        for i in range(len(fe5015_values)):
            ax1.text(fe5015_values[i], mgb_values[i], str(i), 
                   color='white', fontweight='bold', ha='center', va='center', fontsize=10)
        
        # Add colorbar
        cbar1 = plt.colorbar(sc1, ax=ax1)
        cbar1.set_label('[α/Fe]')
        
        # Labels and title
        ax1.set_xlabel('Fe5015 Index', fontsize=12)
        ax1.set_ylabel('Mgb Index', fontsize=12)
        ax1.set_title('Fe5015 vs Mgb - colored by [α/Fe]', fontsize=14)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 2: Fe5015 vs Hbeta (colored by alpha/Fe)
        ax2 = axes[0, 1]
        
        # Plot points
        sc2 = ax2.scatter(fe5015_values, hbeta_values, c=alpha_fe_values, cmap='plasma', 
                      s=120, edgecolor='black', linewidth=1.5)
        
        # Add bin numbers
        for i in range(len(fe5015_values)):
            ax2.text(fe5015_values[i], hbeta_values[i], str(i), 
                   color='white', fontweight='bold', ha='center', va='center', fontsize=10)
        
        # Add colorbar
        cbar2 = plt.colorbar(sc2, ax=ax2)
        cbar2.set_label('[α/Fe]')
        
        # Labels and title
        ax2.set_xlabel('Fe5015 Index', fontsize=12)
        ax2.set_ylabel('Hβ Index', fontsize=12)
        ax2.set_title('Fe5015 vs Hβ - colored by [α/Fe]', fontsize=14)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 3: Mgb vs Hbeta (colored by alpha/Fe)
        ax3 = axes[1, 0]
        
        # Plot points
        sc3 = ax3.scatter(mgb_values, hbeta_values, c=alpha_fe_values, cmap='plasma', 
                      s=120, edgecolor='black', linewidth=1.5)
        
        # Add bin numbers
        for i in range(len(mgb_values)):
            ax3.text(mgb_values[i], hbeta_values[i], str(i), 
                   color='white', fontweight='bold', ha='center', va='center', fontsize=10)
        
        # Add colorbar
        cbar3 = plt.colorbar(sc3, ax=ax3)
        cbar3.set_label('[α/Fe]')
        
        # Labels and title
        ax3.set_xlabel('Mgb Index', fontsize=12)
        ax3.set_ylabel('Hβ Index', fontsize=12)
        ax3.set_title('Mgb vs Hβ - colored by [α/Fe]', fontsize=14)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Plot 4: Alpha/Fe vs Radius
        ax4 = axes[1, 1]
        
        # Get radius values
        radius_values = [point['radius'] for point in points]
        
        # Sort by radius
        sorted_indices = np.argsort(radius_values)
        radius_sorted = np.array(radius_values)[sorted_indices]
        alpha_fe_sorted = np.array(alpha_fe_values)[sorted_indices]
        
        # Plot points with line
        ax4.plot(radius_sorted, alpha_fe_sorted, 'o-', color='purple', 
               markersize=10, linewidth=2)
        
        # Add bin numbers with black outline for better visibility
        for i in range(len(radius_values)):
            # Plot text with white background for visibility
            ax4.text(radius_values[i], alpha_fe_values[i], str(i),
                   fontsize=10, ha='center', va='center', color='black', fontweight='bold',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1, boxstyle='circle'))
        
        # Calculate gradient line
        if len(radius_values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                radius_values, alpha_fe_values)
            
            # Add trend line
            x_range = np.linspace(min(radius_values), max(radius_values), 100)
            y_range = slope * x_range + intercept
            
            ax4.plot(x_range, y_range, '--', color='red', linewidth=2)
            
            # Add annotation
            significance = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
            ax4.text(0.05, 0.95, f"Slope = {slope:.3f}{significance}\np-value = {p_value:.3f}\nR² = {r_value**2:.3f}", 
                   transform=ax4.transAxes, fontsize=12, va='top',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        # Labels and title
        ax4.set_xlabel('R/Re', fontsize=12)
        ax4.set_ylabel('[α/Fe]', fontsize=12)
        ax4.set_title('[α/Fe] vs. Radius', fontsize=14)
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # Set tick parameters for all subplots
        for ax in axes.flat:
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Add overall title
        plt.suptitle(f"Galaxy {galaxy_name}: Spectral Index Interpolation for [α/Fe]", 
                   fontsize=16, y=0.98)
        
        # Add explanation text
        plt.figtext(0.5, 0.01, 
                  "Alpha/Fe values are interpolated from the model grid using Fe5015, Mgb, and Hβ indices.\n"
                  "Points are colored by their interpolated α/Fe values to show how they map in spectral index space.",
                  ha='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved spectral index interpolation plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating spectral index interpolation plot: {e}")
        import traceback
        traceback.print_exc()

def create_parameter_radius_plots(galaxy_name, rdb_data, model_data=None, output_path=None, dpi=150, bins_limit=6, interpolated_data=None):
    """Create parameter vs. radius plots with linear fits, using Re and including interpolated alpha/Fe"""
    try:
        # Check if RDB data is valid
        if rdb_data is None or not isinstance(rdb_data, dict):
            logger.error(f"Invalid RDB data format for {galaxy_name}")
            return
            
        # Extract parameters with bin limit
        params = extract_parameter_profiles(rdb_data, 
                                          parameter_names=['Fe5015', 'Mgb', 'Hbeta', 'age', 'metallicity'],
                                          bins_limit=bins_limit)
        
        if params['radius'] is None:
            logger.error(f"No radius information found for {galaxy_name}")
            return
        
        # Get effective radius
        Re = params['effective_radius']
        if Re is None:
            logger.warning(f"No effective radius found for {galaxy_name}, using raw radius")
            r_scaled = params['radius']
            x_label = 'Radius (arcsec)'
        else:
            # Normalize radius by Re
            r_scaled = params['radius'] / Re
            x_label = 'R/Re'
        
        # Set up parameter labels
        param_labels = {
            'Fe5015': 'Fe5015 Index',
            'Mgb': 'Mgb Index',
            'Hbeta': 'Hβ Index',
            'age': 'log Age (Gyr)',
            'metallicity': '[M/H]',
            'alpha_fe': '[α/Fe]'
        }
        
        # Create figure with 6 subplots in a 2x3 grid to include alpha/Fe
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        # Create plots for each parameter
        parameters = ['Fe5015', 'Mgb', 'Hbeta', 'age', 'metallicity'] 
        
        for i, param_name in enumerate(parameters):
            ax = axes[i]
            
            if param_name in params and hasattr(params[param_name], '__len__') and len(params[param_name]) > 0:
                y = params[param_name]
                
                # Create sorted arrays for consistent plotting
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
                ax.plot(r_sorted, y_sorted, 'o-', color='blue', markersize=8, alpha=0.7)
                
                # Mark outliers
                if np.any(outlier_mask):
                    ax.plot(r_sorted[outlier_mask], y_sorted[outlier_mask], 'rx', markersize=10, alpha=0.8)
                
                # Fit linear trend and add to plot
                slope, intercept, y_fit, r_squared, p_value = fit_linear_slope(x_clean, y_clean, return_full=True)
                
                if not np.isnan(slope):
                    valid_mask = ~np.isnan(x_clean) & ~np.isnan(y_clean)
                    x_valid = np.array(x_clean)[valid_mask]
                    if len(x_valid) >= 2:
                        x_line = np.linspace(min(x_valid), max(x_valid), 100)
                        y_line = linear_fit(x_line, slope, intercept)
                        
                        ax.plot(x_line, y_line, '--', color='red', linewidth=2)
                        
                        # Add slope and p-value to plot
                        ax.text(0.05, 0.95, f"Slope = {slope:.3f}\np = {p_value:.3f}", 
                              transform=ax.transAxes, fontsize=10,
                              va='top', ha='left',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Set labels and title
                ax.set_xlabel(x_label, fontsize=12)
                ax.set_ylabel(param_labels[param_name], fontsize=12)
                ax.set_title(f"{param_labels[param_name]} vs. {x_label}", fontsize=14)
                
                # Add grid
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add vertical line at Re=1 if using normalized radius
                if Re is not None:
                    ax.axvline(x=1, color='k', linestyle=':', alpha=0.5)
                
                # Set tick parameters
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
            else:
                ax.text(0.5, 0.5, f"No {param_name} data available", 
                      ha='center', va='center', fontsize=12,
                      transform=ax.transAxes)
        
        # Add alpha/Fe plot in the last subplot - using INTERPOLATED values if available
        ax = axes[5]
        
        # Try to load interpolated data from file if not provided directly
        if interpolated_data is None:
            interp_data_path = os.path.join(os.path.dirname(output_path), f"{galaxy_name}_interp_verification_interpolated_data.npz")
            if os.path.exists(interp_data_path):
                try:
                    interp_data = np.load(interp_data_path)
                    interpolated_data = {
                        'alpha_fe': interp_data['alpha_fe'],
                        'radius': interp_data['radius']
                    }
                except Exception as e:
                    logger.warning(f"Error loading interpolated data: {e}")
                    interpolated_data = None
        
        # Calculate directly interpolated alpha/Fe values if not available and model_data is provided
        if interpolated_data is None and model_data is not None:
            interp_verification_output = os.path.join(os.path.dirname(output_path), f"{galaxy_name}_interp_verification.png")
            interpolated_data = create_interp_verification_plot(galaxy_name, rdb_data, model_data, 
                                                            output_path=interp_verification_output,
                                                            bins_limit=bins_limit, dpi=dpi)
        
        # Use interpolated alpha/Fe if available, otherwise try calculating it
        if interpolated_data is not None and 'alpha_fe' in interpolated_data and 'radius' in interpolated_data:
            # Get interpolated alpha/Fe values and corresponding radii
            alpha_values = interpolated_data['alpha_fe']
            alpha_radii = interpolated_data['radius']
            
            # Sort by radius
            if len(alpha_values) > 0 and len(alpha_radii) > 0:
                sorted_pairs = sorted(zip(alpha_radii, alpha_values), key=lambda pair: pair[0])
                r_sorted = np.array([pair[0] for pair in sorted_pairs])
                alpha_sorted = np.array([pair[1] for pair in sorted_pairs])
                
                # Plot data points with lines
                ax.plot(r_sorted, alpha_sorted, 'o-', color='purple', markersize=8, alpha=0.7)
                
                # Add bin numbers to the points
                for j, (r, alpha) in enumerate(zip(r_sorted, alpha_sorted)):
                    ax.text(r, alpha, str(j), fontsize=8, ha='center', va='center', 
                          color='white', fontweight='bold')
                
                # Fit linear trend
                slope, intercept, y_fit, r_squared, p_value = fit_linear_slope(r_sorted, alpha_sorted, return_full=True)
                
                if not np.isnan(slope):
                    # Create line using the pre-computed slope
                    x_range = np.linspace(min(r_sorted), max(r_sorted), 100)
                    y_range = slope * x_range + intercept
                    
                    ax.plot(x_range, y_range, '--', color='red', linewidth=2)
                    
                    # Add slope and p-value annotation
                    significance = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
                    ax.text(0.05, 0.95, f"Slope = {slope:.3f}{significance}\np = {p_value:.3f}\nR² = {r_squared:.3f}", 
                          transform=ax.transAxes, fontsize=10,
                          va='top', ha='left',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                    
                    # Add note about significance symbols
                    if significance:
                        ax.text(0.05, 0.05, f"* p < 0.05\n** p < 0.01", 
                              transform=ax.transAxes, fontsize=8,
                              va='bottom', ha='left',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Set labels and title
                ax.set_xlabel(x_label, fontsize=12)
                ax.set_ylabel('[α/Fe] (interpolated)', fontsize=12)
                ax.set_title('[α/Fe] vs. ' + x_label, fontsize=14)
                
                # Add grid
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add vertical line at Re=1 if using normalized radius
                if Re is not None:
                    ax.axvline(x=1, color='k', linestyle=':', alpha=0.5)
                
                # Set tick parameters
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
            else:
                ax.text(0.5, 0.5, "No interpolated [α/Fe] data available", 
                      ha='center', va='center', fontsize=12,
                      transform=ax.transAxes)
        else:
            # If no interpolated data, try to use direct calculation method
            if model_data is not None:
                direct_result = calculate_alpha_fe_direct(galaxy_name, rdb_data, model_data, bins_limit=bins_limit)
                if direct_result is not None and 'points' in direct_result:
                    points = direct_result['points']
                    
                    # Extract values
                    alpha_fe = [point['alpha_fe'] for point in points]
                    radii = [point['radius'] for point in points]
                    
                    # Sort by radius
                    if len(alpha_fe) > 0 and len(radii) > 0:
                        sorted_pairs = sorted(zip(radii, alpha_fe), key=lambda pair: pair[0])
                        r_sorted = np.array([pair[0] for pair in sorted_pairs])
                        alpha_sorted = np.array([pair[1] for pair in sorted_pairs])
                        
                        # Plot points with lines
                        ax.plot(r_sorted, alpha_sorted, 'o-', color='purple', markersize=8, alpha=0.7)
                        
                        # Add bin numbers
                        for j, (r, alpha) in enumerate(zip(r_sorted, alpha_sorted)):
                            ax.text(r, alpha, str(j), fontsize=8, ha='center', va='center', 
                                  color='white', fontweight='bold')
                        
                        # Fit linear trend
                        slope, intercept, y_fit, r_squared, p_value = fit_linear_slope(r_sorted, alpha_sorted, return_full=True)
                        
                        if not np.isnan(slope):
                            # Create line
                            x_range = np.linspace(min(r_sorted), max(r_sorted), 100)
                            y_range = slope * x_range + intercept
                            
                            ax.plot(x_range, y_range, '--', color='red', linewidth=2)
                            
                            # Add annotations
                            significance = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
                            ax.text(0.05, 0.95, f"Slope = {slope:.3f}{significance}\np = {p_value:.3f}\nR² = {r_squared:.3f}", 
                                  transform=ax.transAxes, fontsize=10,
                                  va='top', ha='left',
                                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                        
                        # Set labels and title
                        ax.set_xlabel(x_label, fontsize=12)
                        ax.set_ylabel('[α/Fe]', fontsize=12)
                        ax.set_title('[α/Fe] vs. ' + x_label, fontsize=14)
                        
                        # Add grid and other formatting
                        ax.grid(True, alpha=0.3, linestyle='--')
                        if Re is not None:
                            ax.axvline(x=1, color='k', linestyle=':', alpha=0.5)
                        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                        ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
                    else:
                        ax.text(0.5, 0.5, "No [α/Fe] data available", 
                              ha='center', va='center', fontsize=12,
                              transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "No [α/Fe] data available", 
                          ha='center', va='center', fontsize=12,
                          transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "Model data required for [α/Fe] interpolation", 
                      ha='center', va='center', fontsize=12,
                      transform=ax.transAxes)
        
        # Add overall title
        re_info = f" (Re = {Re:.2f} arcsec)" if Re is not None else ""
        plt.suptitle(f"Galaxy {galaxy_name}: Parameter-Radius Relations{re_info}", fontsize=16, y=0.98)
        
        # Add note about alpha/Fe interpolation
        if model_data is not None:
            plt.figtext(0.5, 0.01, 
                      "Alpha/Fe values derived from TMB03 models using interpolation in Fe5015-Mgb-Hβ space.\n"
                      "Bin numbers shown on the α/Fe plot correspond to the radial bins used in the analysis.",
                      ha='center', fontsize=10, style='italic')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved parameter-radius plots to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating parameter-radius plots: {e}")
        import traceback
        traceback.print_exc()

def calculate_alpha_fe_direct_multi(galaxy_name, rdb_data, model_data, index_method='auto', 
                                 index_combinations=['all', 'fe_mgb', 'fe_hbeta', 'mgb_hbeta'], 
                                 bins_limit=6):
    """
    Calculate alpha/Fe for each data point using interpolation from the model grid
    with support for multiple spectral index calculation methods and index combinations
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices
    model_data : DataFrame
        Model grid data with indices and alpha/Fe values
    index_method : str
        Which method to use for spectral indices: 'auto', 'original', 'fit', or 'template'
    index_combinations : list
        List of index combinations to use: 'all' (all 3 indices), 'fe_mgb', 'fe_hbeta', 'mgb_hbeta'
    bins_limit : int
        Limit analysis to first N bins (default: 6 for bins 0-5)
    
    Returns:
    --------
    dict
        Dictionary containing points with their indices, radii, and interpolated alpha/Fe values
        for each index combination method
    """
    try:
        # Extract spectral indices based on the selected method
        galaxy_indices = extract_spectral_indices_by_method(rdb_data, method=index_method, bins_limit=bins_limit)
        
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Check for required data
        if ('bin_indices' not in galaxy_indices or
            'Fe5015' not in galaxy_indices['bin_indices'] or 
            'Mgb' not in galaxy_indices['bin_indices'] or 
            'Hbeta' not in galaxy_indices['bin_indices']):
            logger.warning(f"Missing required spectral indices for {galaxy_name} with method {index_method}")
            return None
        
        # Get galaxy spectral indices for each bin
        galaxy_fe5015 = galaxy_indices['bin_indices']['Fe5015']
        galaxy_mgb = galaxy_indices['bin_indices']['Mgb']
        galaxy_hbeta = galaxy_indices['bin_indices']['Hbeta']
        
        # Check if arrays are valid
        if not hasattr(galaxy_fe5015, '__len__') or not hasattr(galaxy_mgb, '__len__') or not hasattr(galaxy_hbeta, '__len__'):
            logger.warning(f"Invalid index arrays for {galaxy_name}")
            return None
            
        # Make sure they all have the same length
        min_len = min(len(galaxy_fe5015), len(galaxy_mgb), len(galaxy_hbeta))
        galaxy_fe5015 = galaxy_fe5015[:min_len]
        galaxy_mgb = galaxy_mgb[:min_len]
        galaxy_hbeta = galaxy_hbeta[:min_len]
        
        # Get galaxy age if available
        if 'age' in galaxy_indices['bin_indices'] and hasattr(galaxy_indices['bin_indices']['age'], '__len__'):
            galaxy_age = galaxy_indices['bin_indices']['age'][:min_len]
        else:
            # Default age values
            galaxy_age = np.ones(min_len) * 5.0  # Use 5 Gyr as default age
        
        # Get radii
        galaxy_radius = None
        if 'R' in galaxy_indices['bin_indices'] and hasattr(galaxy_indices['bin_indices']['R'], '__len__'):
            galaxy_radius = galaxy_indices['bin_indices']['R'][:min_len]
        elif 'distance' in rdb_data:
            distance = rdb_data['distance'].item() if hasattr(rdb_data['distance'], 'item') else rdb_data['distance']
            if 'bin_distances' in distance:
                galaxy_radius = distance['bin_distances'][:min_len]
            elif 'binning' in rdb_data:
                binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
                if 'bin_radii' in binning:
                    galaxy_radius = binning['bin_radii'][:min_len]
        
        if galaxy_radius is None:
            logger.warning(f"No radius information found for {galaxy_name}")
            return None
        
        # Get effective radius if available
        Re = extract_effective_radius(rdb_data)
        
        # Scale radius by effective radius if available
        if Re is not None and Re > 0:
            r_scaled = galaxy_radius / Re
        else:
            r_scaled = galaxy_radius
            
        # Extract model column names
        fe5015_col = model_column_mapping['Fe5015']
        mgb_col = model_column_mapping['Mgb']
        hbeta_col = model_column_mapping['Hbeta']
        age_col = model_column_mapping['Age']
        aofe_col = model_column_mapping['AoFe']
        zoh_col = model_column_mapping['ZoH']
        
        # Get unique values from the model grid
        model_ages = sorted(model_data[age_col].unique())
        model_zoh = sorted(model_data[zoh_col].unique())
        model_aofe = sorted(model_data[aofe_col].unique())
        
        # Create a dictionary to store results for different index combinations
        results_by_combination = {}
        
        # Process each combination of indices
        for combo in index_combinations:
            # Store results for each individual point with this combination
            points = []
            
            # Process each data point
            for i in range(min_len):
                fe5015 = galaxy_fe5015[i]
                mgb = galaxy_mgb[i]
                hbeta = galaxy_hbeta[i]
                age = galaxy_age[i]
                radius = r_scaled[i]
                
                # Skip if any values are NaN
                if np.isnan(fe5015) or np.isnan(mgb) or np.isnan(hbeta) or np.isnan(radius):
                    continue
                
                # Find closest model age
                closest_age_idx = np.argmin(np.abs(np.array(model_ages) - age))
                closest_age = model_ages[closest_age_idx]
                
                # Filter model grid to points near this age
                age_filtered = model_data[model_data[age_col] == closest_age]
                
                # Perform interpolation based on the selected index combination
                if combo == 'all':
                    # Use all three indices with appropriate weights
                    distances = []
                    for _, row in age_filtered.iterrows():
                        fe5015_diff = (row[fe5015_col] - fe5015) / 5.0  # Normalize by typical range
                        mgb_diff = (row[mgb_col] - mgb) / 4.0
                        hbeta_diff = (row[hbeta_col] - hbeta) / 3.0
                        
                        distance = np.sqrt(fe5015_diff**2 + mgb_diff**2 + hbeta_diff**2)
                        distances.append((distance, row[aofe_col], row[zoh_col]))
                
                elif combo == 'fe_mgb':
                    # Use only Fe5015 and Mgb
                    distances = []
                    for _, row in age_filtered.iterrows():
                        fe5015_diff = (row[fe5015_col] - fe5015) / 5.0
                        mgb_diff = (row[mgb_col] - mgb) / 4.0
                        
                        distance = np.sqrt(fe5015_diff**2 + mgb_diff**2)
                        distances.append((distance, row[aofe_col], row[zoh_col]))
                
                elif combo == 'fe_hbeta':
                    # Use only Fe5015 and Hbeta
                    distances = []
                    for _, row in age_filtered.iterrows():
                        fe5015_diff = (row[fe5015_col] - fe5015) / 5.0
                        hbeta_diff = (row[hbeta_col] - hbeta) / 3.0
                        
                        distance = np.sqrt(fe5015_diff**2 + hbeta_diff**2)
                        distances.append((distance, row[aofe_col], row[zoh_col]))
                
                elif combo == 'mgb_hbeta':
                    # Use only Mgb and Hbeta
                    distances = []
                    for _, row in age_filtered.iterrows():
                        mgb_diff = (row[mgb_col] - mgb) / 4.0
                        hbeta_diff = (row[hbeta_col] - hbeta) / 3.0
                        
                        distance = np.sqrt(mgb_diff**2 + hbeta_diff**2)
                        distances.append((distance, row[aofe_col], row[zoh_col]))
                
                # Sort by distance and apply inverse distance weighting
                if distances:
                    # Sort by distance
                    distances.sort(key=lambda x: x[0])
                    
                    # Take up to k nearest neighbors for interpolation
                    k = min(5, len(distances))
                    nearest_neighbors = distances[:k]
                    
                    # Apply inverse distance weighting
                    total_weight = 0
                    weighted_alpha_sum = 0
                    weighted_zoh_sum = 0
                    
                    for dist, alpha, zoh in nearest_neighbors:
                        # Avoid division by zero
                        weight = 1.0 / max(dist, 1e-6)
                        total_weight += weight
                        weighted_alpha_sum += alpha * weight
                        weighted_zoh_sum += zoh * weight
                    
                    # Calculate weighted average
                    interpolated_alpha = weighted_alpha_sum / total_weight
                    interpolated_zoh = weighted_zoh_sum / total_weight
                    distance_metric = nearest_neighbors[0][0]  # Distance to closest point
                    
                    # Store the interpolated values
                    points.append({
                        'radius': radius,
                        'Fe5015': fe5015,
                        'Mgb': mgb,
                        'Hbeta': hbeta,
                        'alpha_fe': interpolated_alpha,
                        'metallicity': interpolated_zoh,
                        'age': age,
                        'distance_metric': distance_metric,
                        'bin_index': i  # Store which bin this came from
                    })
            
            # Store points for this combination
            if points:
                results_by_combination[combo] = points
        
        # Check if we found any valid points
        if not results_by_combination:
            logger.warning(f"No valid points found for {galaxy_name} with method {index_method}")
            return None
        
        # Calculate summary statistics for each combination
        summary = {}
        for combo, points in results_by_combination.items():
            if points:
                # Extract values
                alpha_values = [point['alpha_fe'] for point in points]
                radii = [point['radius'] for point in points]
                
                # Calculate median values
                median_alpha = np.median(alpha_values)
                median_radius = np.median(radii)
                
                # Calculate slope of alpha/Fe vs. radius
                if len(alpha_values) > 1:
                    # Calculate linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(radii, alpha_values)
                else:
                    slope = np.nan
                    p_value = np.nan
                    r_value = np.nan
                    std_err = np.nan
                
                # Store summary for this combination
                summary[combo] = {
                    'alpha_fe_median': median_alpha,
                    'radius_median': median_radius,
                    'slope': slope,
                    'p_value': p_value,
                    'r_squared': r_value**2 if not np.isnan(r_value) else np.nan,
                    'std_err': std_err,
                    'n_points': len(points)
                }
        
        # Return combined results
        return {
            'galaxy': galaxy_name,
            'effective_radius': Re,
            'index_method': index_method,
            'combinations': results_by_combination,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"Error calculating interpolated alpha/Fe for {galaxy_name} with method {index_method}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_alpha_fe_method_comparison_plot(galaxy_name, rdb_data, model_data, output_path=None, 
                                        dpi=150, bins_limit=6, 
                                        methods=['auto', 'original', 'fit', 'template']):
    """
    Create a visualization comparing alpha/Fe calculated using different spectral index methods
    and different combinations of indices
    """
    try:
        # Define index combinations to test
        combinations = ['all', 'fe_mgb', 'fe_hbeta', 'mgb_hbeta']
        
        # Calculate alpha/Fe using all methods and combinations
        results = {}
        available_methods = []
        
        # First check which methods actually return data
        for method in methods:
            result = calculate_alpha_fe_direct_multi(
                galaxy_name, rdb_data, model_data,
                index_method=method,
                index_combinations=combinations,
                bins_limit=bins_limit
            )
            
            # Only include methods with valid results
            if result and 'combinations' in result and any(result['combinations'].values()):
                results[method] = result
                available_methods.append(method)
                logger.info(f"Calculated alpha/Fe with {method} method")
            else:
                logger.warning(f"No valid results for {method} method")
        
        # Create figure
        n_methods = len(available_methods)
        if n_methods == 0:
            logger.warning(f"No valid results for any method for {galaxy_name}")
            # Create an empty figure with a message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"No valid alpha/Fe data for {galaxy_name} with any method",
                   ha='center', va='center', fontsize=14, transform=ax.transAxes)
            plt.tight_layout()
            if output_path:
                plt.savefig(output_path, dpi=dpi)
            plt.close()
            return
        
        # Create figure with the right size for the methods we have
        fig = plt.figure(figsize=(14, 4 * n_methods))
        gs = fig.add_gridspec(n_methods, 4)
        
        # Color each combination consistently
        combo_colors = {
            'all': 'blue',
            'fe_mgb': 'red',
            'fe_hbeta': 'green',
            'mgb_hbeta': 'purple'
        }
        
        # Combo labels for legend
        combo_labels = {
            'all': 'All indices (Fe5015 + Mgb + Hβ)',
            'fe_mgb': 'Fe5015 + Mgb',
            'fe_hbeta': 'Fe5015 + Hβ',
            'mgb_hbeta': 'Mgb + Hβ'
        }
        
        # Track ylim for consistent scaling
        y_min, y_max = float('inf'), float('-inf')
        
        # Plot each method
        for i, method in enumerate(available_methods):
            result = results[method]
            # Create a row of subplots for this method
            axes = [fig.add_subplot(gs[i, j]) for j in range(4)]
            
            # Get effective radius
            Re = result['effective_radius']
            
            # FIRST SUBPLOT: Alpha/Fe vs Radius for all combinations
            ax = axes[0]
            
            # Track if any data was successfully plotted
            combinations_with_data = []
            
            # Plot each combination
            for combo, combo_data in result['combinations'].items():
                if combo_data and len(combo_data) > 0:
                    # Extract data
                    radii = [point['radius'] for point in combo_data]
                    alpha_values = [point['alpha_fe'] for point in combo_data]
                    
                    # Only plot if we have valid data
                    if len(radii) > 0 and len(alpha_values) > 0:
                        # Sort by radius
                        sorted_pairs = sorted(zip(radii, alpha_values), key=lambda pair: pair[0])
                        r_sorted = np.array([pair[0] for pair in sorted_pairs])
                        alpha_sorted = np.array([pair[1] for pair in sorted_pairs])
                        
                        # Plot line and points
                        ax.plot(r_sorted, alpha_sorted, 'o-', color=combo_colors[combo], 
                               label=combo_labels[combo], linewidth=2, alpha=0.7)
                        
                        # Track which combinations had data
                        combinations_with_data.append(combo)
                        
                        # Track y limits
                        if len(alpha_sorted) > 0:
                            y_min = min(y_min, np.min(alpha_sorted))
                            y_max = max(y_max, np.max(alpha_sorted))
            
            # Add grid and labels
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlabel('R/Re' if Re else 'Radius (arcsec)', fontsize=12)
            ax.set_ylabel('[α/Fe]', fontsize=12)
            ax.set_title(f'Alpha/Fe vs Radius - {method.capitalize()} Method', fontsize=13)
            
            # Only add legend if we have data
            if combinations_with_data:
                ax.legend(loc='best', fontsize=9)
            else:
                ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
            
            # Add Re line if available
            if Re:
                ax.axvline(x=1, color='k', linestyle=':', alpha=0.5)
            
            # SPECTRAL INDEX PLANES: Fe5015 vs Mgb, Fe5015 vs Hbeta, Mgb vs Hbeta
            # Define index pairs
            index_pairs = [
                ('Fe5015', 'Mgb'),
                ('Fe5015', 'Hbeta'),
                ('Mgb', 'Hbeta')
            ]
            
            # Plot each index pair
            for j, (x_index, y_index) in enumerate(index_pairs):
                ax = axes[j+1]
                
                # Get all points from combinations that had data
                all_points = []
                for combo in combinations_with_data:
                    combo_data = result['combinations'][combo]
                    for point in combo_data:
                        # Only add point if it contains both required indices
                        if (x_index in point and y_index in point and 
                            not np.isnan(point[x_index]) and not np.isnan(point[y_index])):
                            # Check if this bin is already in the list
                            bin_idx = point.get('bin_index', -1)
                            if not any(p.get('bin_index', -99) == bin_idx for p in all_points):
                                all_points.append(point)
                
                # Plot points colored by alpha/Fe if we have data
                if all_points:
                    x_values = [point[x_index] for point in all_points]
                    y_values = [point[y_index] for point in all_points]
                    alpha_values = [point['alpha_fe'] for point in all_points]
                    
                    # Make sure we have valid data to plot
                    if all(not np.isnan(x) for x in x_values) and all(not np.isnan(y) for y in y_values):
                        # Create scatter plot
                        sc = ax.scatter(x_values, y_values, c=alpha_values, 
                                       cmap='plasma', s=100, alpha=0.8, edgecolor='black')
                        
                        # Add bin numbers
                        for k, point in enumerate(all_points):
                            bin_idx = point.get('bin_index', k)
                            ax.text(point[x_index], point[y_index], str(bin_idx), 
                                   fontsize=9, ha='center', va='center', 
                                   color='white', fontweight='bold')
                        
                        # Add colorbar
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes("right", size="5%", pad=0.05)
                        cbar = plt.colorbar(sc, cax=cax)
                        cbar.set_label('[α/Fe]')
                    else:
                        ax.text(0.5, 0.5, "Invalid data points", 
                               ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, "No data points", 
                           ha='center', va='center', transform=ax.transAxes)
                
                # Add grid and labels
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.set_xlabel(f'{x_index} Index', fontsize=12)
                ax.set_ylabel(f'{y_index} Index', fontsize=12)
                ax.set_title(f'{x_index} vs {y_index}', fontsize=13)
        
        # Set consistent y-limits for alpha/Fe plots
        if y_max > y_min:
            padding = (y_max - y_min) * 0.1
            for i in range(n_methods):
                ax = fig.add_subplot(gs[i, 0])
                ax.set_ylim(y_min - padding, y_max + padding)
        
        # Add overall title
        plt.suptitle(f'Galaxy {galaxy_name}: Alpha/Fe Analysis using Different Index Methods and Combinations', 
                   fontsize=16, y=0.98)
        
        # Add explanatory text
        plt.figtext(0.5, 0.01, 
                  "Comparison of alpha/Fe values calculated using different spectral index calculation methods\n"
                  "and different combinations of spectral indices for interpolation.",
                  ha='center', fontsize=12)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved alpha/Fe method comparison plot to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating alpha/Fe method comparison plot: {e}")
        import traceback
        traceback.print_exc()

def create_galaxy_visualization(galaxy_name, p2p_data, rdb_data, cube_info, model_data=None, output_dir=None, dpi=150, bins_limit=6):
    """
    Create comprehensive visualization for a galaxy
    
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
    
    Returns:
    --------
    None
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
            return
            
        # Create figure 1: Combined flux map and radial binning
        create_combined_flux_and_binning(galaxy_name, p2p_data, rdb_data, cube_info, 
                                       output_path=f"{output_dir}/{galaxy_name}_flux_and_binning.png", 
                                       dpi=dpi)
        
        # Create model grid visualizations and interpolated data if model data is provided
        interpolated_data = None
        if model_data is not None:
            # Create figure with interpolation verification
            interp_verification_output = f"{output_dir}/{galaxy_name}_interp_verification.png"
            interpolated_data = create_interp_verification_plot(galaxy_name, rdb_data, model_data, 
                                                            output_path=interp_verification_output,
                                                            bins_limit=bins_limit, dpi=dpi)
            
            # Create figure 3: Model grid plots part 1 (3x3 grid colored by R, log Age, and [M/H])
            # Pass bins_limit parameter
            create_model_grid_plots_part1(galaxy_name, rdb_data, model_data, age=1,
                                        output_path=f"{output_dir}/{galaxy_name}_model_grid_part1.png",
                                        bins_limit=bins_limit,
                                        dpi=dpi)
            
            # Create figure 4: Model grid plots part 2 (3x3 grid with multiple model ages)
            # Pass bins_limit parameter
            create_model_grid_plots_part2(galaxy_name, rdb_data, model_data, ages=[1, 3, 10],
                                        output_path=f"{output_dir}/{galaxy_name}_model_grid_part2.png",
                                        bins_limit=bins_limit,
                                        dpi=dpi)
            
            # Create figure 5: Spectral index interpolation plot
            create_spectral_index_interpolation_plot(galaxy_name, rdb_data, model_data,
                                                  output_path=f"{output_dir}/{galaxy_name}_alpha_fe_interpolation.png",
                                                  bins_limit=bins_limit,
                                                  dpi=dpi)
        
        # Create figure 2: Parameter-radius relations with linear fits, using Re
        # Pass model_data and bins_limit parameters along with interpolated data
        create_parameter_radius_plots(galaxy_name, rdb_data, model_data,
                                     output_path=f"{output_dir}/{galaxy_name}_parameter_radius.png",
                                     bins_limit=bins_limit,
                                     dpi=dpi,
                                     interpolated_data=interpolated_data)
        
        logger.info(f"Visualization complete for {galaxy_name}")
        
    except Exception as e:
        logger.error(f"Error in visualization for {galaxy_name}: {e}")
        import traceback
        traceback.print_exc()

#------------------------------------------------------------------------------
# Main Analysis Function
#------------------------------------------------------------------------------

def calculate_galaxy_alpha_fe_summary(galaxies, model_file="./TMB03/TMB03.csv", 
                                     output_dir=None, dpi=150, bins_limit=6):
    """
    Calculate alpha/Fe values and radial gradients for all galaxies and create summary plots
    
    Parameters:
    -----------
    galaxies : list
        List of galaxy names to process
    model_file : str
        Path to the model grid file
    output_dir : str
        Directory to save output images
    dpi : int
        Resolution for saved images
    bins_limit : int
        Limit analysis to first N bins (default: 6 for bins 0-5)
    """
    try:
        # Create output directory if needed
        if output_dir is None:
            output_dir = "./galaxy_summary"
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model data
        model_data = load_model_data(model_file)
        if model_data is None or model_data.empty:
            logger.error("Failed to load model data")
            return
            
        logger.info(f"Loaded model data with columns: {list(model_data.columns)}")
        
        # Process each galaxy using both methods
        results_list = []         # Traditional binned analysis
        direct_results_list = []  # Direct point-by-point analysis
        valid_results_count = 0
        
        for galaxy_name in galaxies:
            # Load galaxy data
            p2p_data, vnb_data, rdb_data = Read_Galaxy(galaxy_name)
            
            # Check if RDB data is valid before proceeding
            rdb_valid = rdb_data is not None and isinstance(rdb_data, dict) and len(rdb_data) > 0
            if not rdb_valid:
                logger.warning(f"No valid RDB data for {galaxy_name}")
                results_list.append(None)
                direct_results_list.append(None)
                continue
                
            # Calculate alpha/Fe and radius relationship using traditional method
            # Pass bins_limit parameter to extraction functions
            result = extract_alpha_fe_radius(galaxy_name, rdb_data, model_data, bins_limit=bins_limit)
            results_list.append(result)
            
            # Also calculate using direct point-by-point method with interpolation
            # Pass bins_limit parameter here too
            direct_result = calculate_alpha_fe_direct(galaxy_name, rdb_data, model_data, bins_limit=bins_limit)
            direct_results_list.append(direct_result)
            
            # Log results
            if result:
                valid_results_count += 1
                logger.info(f"{galaxy_name}: α/Fe = {result['alpha_fe_median']:.3f}, R/Re = {result['radius_median']:.3f}")
                if not np.isnan(result['slope']):
                    logger.info(f"  Slope = {result['slope']:.3f}, p-value = {result['p_value']:.3f}")
            else:
                logger.warning(f"Could not calculate α/Fe for {galaxy_name}")
        
        logger.info(f"Successfully processed {valid_results_count} out of {len(galaxies)} galaxies")
        
        if valid_results_count == 0:
            logger.error("No valid results obtained. Cannot create plots.")
            return
            
        # Create improved alpha/Fe vs. radius plot with legend
        create_alpha_radius_plot_improved(results_list, 
                                        output_path=f"{output_dir}/alpha_fe_vs_radius_improved.png", 
                                        dpi=dpi)
        
        # Create direct interpolation plots
        create_alpha_radius_direct_plot(direct_results_list, 
                                       output_path=f"{output_dir}/alpha_fe_vs_radius_direct.png", 
                                       dpi=dpi)
        
        # Pass model_data to use as background grid
        create_fe5015_mgb_plot(direct_results_list, 
                             output_path=f"{output_dir}/fe5015_vs_mgb.png", 
                             dpi=dpi,
                             model_data=model_data)
        
        # Create summary table
        summary_data = []
        for result in results_list:
            if result:
                summary_data.append({
                    'Galaxy': result['galaxy'],
                    'Alpha/Fe': result['alpha_fe_median'],
                    'R/Re': result['radius_median'],
                    'Metallicity': result['metallicity_median'],
                    'α/Fe-R Slope': result['slope'],
                    'p-value': result['p_value'],
                    'R-squared': result['r_squared']
                })
        
        # Save as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{output_dir}/galaxy_summary.csv", index=False)
        
        # Also save a pretty markdown summary
        with open(f"{output_dir}/galaxy_summary.md", "w") as f:
            f.write("# Alpha/Fe Radial Gradient Analysis for Virgo Cluster Galaxies\n\n")
            f.write("## Galaxy Properties\n\n")
            
            # Create markdown table
            f.write("| Galaxy | α/Fe | R/Re | [M/H] | α/Fe-Radius Slope | p-value | Significant? |\n")
            f.write("|--------|------|------|-------|-------------------|---------|-------------|\n")
            
            for result in results_list:
                if result:
                    significant = "Yes" if result['p_value'] < 0.05 else "No"
                    f.write(f"| {result['galaxy']} | {result['alpha_fe_median']:.2f} | {result['radius_median']:.2f} | ")
                    f.write(f"{result['metallicity_median']:.2f} | {result['slope']:.3f} | {result['p_value']:.3f} | {significant} |\n")
            
            f.write("\n\n## Interpretation\n\n")
            f.write("- Positive slope indicates α/Fe increases with radius (more α-enhancement in outskirts)\n")
            f.write("- Negative slope indicates α/Fe decreases with radius (more α-enhancement in center)\n")
            f.write("- Statistically significant trends (p < 0.05) are marked as 'Yes' in the Significant column\n\n")
            
            f.write("## Analysis Method\n\n")
            f.write(f"Alpha/Fe values were derived by interpolating between TMB03 stellar population model grid points\n")
            f.write(f"using inverse distance weighting in the space of spectral indices (Fe5015, Mgb, Hbeta).\n")
            f.write(f"Analysis used the first {bins_limit} radial bins (0-{bins_limit-1}) of each galaxy.\n")
            
            f.write("Analysis completed on " + pd.Timestamp.now().strftime("%Y-%m-%d"))
        
        logger.info(f"Completed galaxy summary analysis. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in galaxy summary analysis: {e}")
        import traceback
        traceback.print_exc()

#------------------------------------------------------------------------------
# Main Execution
#------------------------------------------------------------------------------

if __name__ == "__main__":
    # Define bin limit for all analyses
    bins_limit = 6  # Analyze bins 0-5
    
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
    
    # Process all galaxies with the bin configuration
    results = process_all_galaxies(all_galaxy_data, model_data, 'bins_config.yaml')
    
    # Define emission line galaxies
    emission_line_galaxies = [
        "VCC1588", "VCC1368", "VCC1902", "VCC1949", "VCC990", 
        "VCC1410", "VCC667", "VCC1811", "VCC688", "VCC1193", "VCC1486"
    ]
    
    # Get galaxy coordinates
    coordinates = get_ifu_coordinates(galaxies)
    
    logger.info("Galaxy α/Fe radial gradient analysis complete")

    # Create individual galaxy visualizations
    output_base = "./galaxy_visualizations"
    os.makedirs(output_base, exist_ok=True)
    
    for galaxy_name in galaxies:
        try:
            # Load galaxy data
            p2p_data, vnb_data, rdb_data = Read_Galaxy(galaxy_name)
            
            # Check if RDB data is valid before proceeding
            rdb_valid = rdb_data is not None and isinstance(rdb_data, dict) and len(rdb_data) > 0
            if not rdb_valid:
                logger.warning(f"No valid RDB data for {galaxy_name}, skipping visualization")
                continue
            
            # Extract cube information
            cube_info = extract_cube_info(galaxy_name)
            
            # Create output directory for this galaxy
            output_dir = f"{output_base}/{galaxy_name}"
            
            # Create visualizations including model grid plots - pass bins_limit
            create_galaxy_visualization(galaxy_name, p2p_data, rdb_data, cube_info, 
                                       model_data=model_data,
                                       output_dir=output_dir, dpi=300,
                                       bins_limit=bins_limit)  # Pass bins_limit
                                       
            logger.info(f"Completed visualization for {galaxy_name}")
        except Exception as e:
            logger.error(f"Failed to process {galaxy_name}: {e}")
            continue
    
    logger.info("All visualizations complete!")