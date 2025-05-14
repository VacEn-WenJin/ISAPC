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
            elif 'bin_radii' in rdb_data['binning']:
                binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
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

def calculate_alpha_fe_direct(galaxy_name, rdb_data, model_data, config=None):
    """
    Calculate alpha/Fe for each data point using interpolation from the model grid
    with specified bins only
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices
    model_data : DataFrame
        Model grid data with indices and alpha/Fe values
    config : dict, optional
        Bin configuration dictionary
        
    Returns:
    --------
    dict
        Dictionary containing points with their indices, radii, and interpolated alpha/Fe values
    """
    try:
        # Load bin configuration if not provided
        if config is None:
            config = load_bin_config()
            
        # Get bins to use for this galaxy
        bins_to_use = get_bins_to_use(galaxy_name, config)
        
        # Find the maximum bin index to extract
        max_bin_idx = max(bins_to_use) + 1 if bins_to_use else 6
        
        # Extract spectral indices from galaxy data
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
        
        # Store results for each individual point
        points = []
        
        # Process each data point - only using the specified bins
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
            
            # Find closest model age
            closest_age_idx = np.argmin(np.abs(np.array(model_ages) - age))
            closest_age = model_ages[closest_age_idx]
            
            # Filter model grid to points near this age
            age_filtered = model_data[model_data[age_col] == closest_age]
            
            # Perform weighted interpolation using the k-nearest neighbors
            k = min(5, len(age_filtered))  # Use at most 5 neighbors for interpolation
            
            if k < 3:  # Need at least 3 points for reliable interpolation
                # Fall back to closest point if not enough neighbors
                min_distance = float('inf')
                best_alpha = None
                best_zoh = None
                
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
                        
                interpolated_alpha = best_alpha
                interpolated_zoh = best_zoh
                distance_metric = min_distance
            else:
                # Calculate distances to all points in index space
                distances = []
                for _, row in age_filtered.iterrows():
                    fe5015_diff = (row[fe5015_col] - fe5015) / 5.0
                    mgb_diff = (row[mgb_col] - mgb) / 4.0
                    hbeta_diff = (row[hbeta_col] - hbeta) / 3.0
                    
                    distance = np.sqrt(fe5015_diff**2 + mgb_diff**2 + hbeta_diff**2)
                    distances.append((distance, row[aofe_col], row[zoh_col]))
                
                # Sort by distance and get k nearest neighbors
                distances.sort(key=lambda x: x[0])
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
    """Create combined flux map and radial binning visualization with proper physical scaling"""
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
            # Check for direct flux map
            if 'flux_map' in p2p_data:
                flux_map = p2p_data['flux_map']
            # Check signal in signal_noise
            elif 'signal_noise' in p2p_data:
                sn = p2p_data['signal_noise'].item() if hasattr(p2p_data['signal_noise'], 'item') else p2p_data['signal_noise']
                if 'signal' in sn:
                    flux_map = sn['signal']
            # Try to create flux from spectra
            elif 'spectra' in p2p_data:
                spectra = p2p_data['spectra']
                if hasattr(spectra, 'ndim'):
                    if spectra.ndim == 2:
                        # Spectrum is already 2D (wavelength x pixels)
                        flux_map = np.nanmedian(spectra, axis=0)
                    elif spectra.ndim == 3:
                        # Spectrum is 3D (wavelength x height x width)
                        flux_map = np.nanmedian(spectra, axis=0)
            # If original cube data is available
            elif '_cube_data' in p2p_data:
                cube_data = p2p_data['_cube_data']
                flux_map = np.nanmedian(cube_data, axis=0)
                
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
        
        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
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
        
        # Calculate the extent for both plots - centered at 0
        extent = [-extent_x/2, extent_x/2, -extent_y/2, extent_y/2]
        
        # Plot 1: Flux Map
        valid_mask = np.isfinite(flux_map) & (flux_map > 0)
        if np.any(valid_mask):
            vmin = np.percentile(flux_map[valid_mask], 1)
            vmax = np.percentile(flux_map[valid_mask], 99)
            norm = LogNorm(vmin=max(vmin, 1e-10), vmax=vmax)
        else:
            norm = LogNorm(vmin=1e-10, vmax=1)
        
        # Plot flux map with physical units and correct scaling
        im1 = ax1.imshow(flux_map, origin='lower', norm=norm, cmap='inferno',
                       extent=extent, aspect='equal')  # Use 'equal' to maintain physical scaling
        
        # Add colorbar for flux with log scale formatting
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('Flux (log scale)')
        
        # Add north/east arrows - positioned at bottom right
        arrow_len = min(extent_x, extent_y) * 0.1
        arrow_start_x = extent_x/2 * 0.8
        arrow_start_y = -extent_y/2 * 0.8
        
        # North arrow
        ax1.annotate('N', xy=(arrow_start_x, arrow_start_y + arrow_len), 
                  xytext=(arrow_start_x, arrow_start_y),
                  arrowprops=dict(facecolor='white', width=1.5, headwidth=7),
                  color='white', ha='center', va='bottom', fontsize=12)
        
        # East arrow
        ax1.annotate('E', xy=(arrow_start_x + arrow_len, arrow_start_y), 
                  xytext=(arrow_start_x, arrow_start_y),
                  arrowprops=dict(facecolor='white', width=1.5, headwidth=7),
                  color='white', ha='left', va='center', fontsize=12)
        
        ax1.set_title(f"Flux Map", fontsize=14)
        ax1.set_xlabel('Arcsec')
        ax1.set_ylabel('Arcsec')
        
        # Set tick parameters for ax1
        ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax1.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Plot 2: Radial Binning
        # Create 2D bin number array if needed
        if bin_num is not None:
            if hasattr(bin_num, 'ndim') and bin_num.ndim == 1:
                bin_num_2d = np.full((ny, nx), -1, dtype=int)
                valid_len = min(len(bin_num), ny * nx)
                bin_num_2d.flat[:valid_len] = bin_num[:valid_len]
            else:
                bin_num_2d = bin_num
            
            # Count transitions to determine if this is Voronoi-like binning
            voronoi_like = False
            if hasattr(bin_num_2d, 'ndim') and bin_num_2d.ndim == 2:
                try:
                    transitions = 0
                    for i in range(1, bin_num_2d.shape[0]):
                        transitions += np.sum(bin_num_2d[i, :] != bin_num_2d[i-1, :])
                    for j in range(1, bin_num_2d.shape[1]):
                        transitions += np.sum(bin_num_2d[:, j] != bin_num_2d[:, j-1])
                    
                    # Normalize by array size
                    transition_density = transitions / (bin_num_2d.shape[0] * bin_num_2d.shape[1])
                    voronoi_like = transition_density > 0.1  # Threshold determined empirically
                except Exception:
                    pass
            
            # Count unique bins to determine colormap approach
            unique_bins = np.unique(bin_num_2d)
            unique_bins = unique_bins[unique_bins >= 0]  # Exclude negative (invalid) bins
            num_bins = len(unique_bins)
            
            # Choose colormap based on number of bins and galaxy type
            if num_bins <= 20:
                # Use a discrete colormap for fewer bins (should match both VCC1902 and VCC1549)
                base_cmap = plt.cm.tab20
                # If we need more than 20 colors, cycle through with slight variations
                if num_bins <= 20:
                    cmap = plt.cm.get_cmap('tab20', max(10, num_bins))
                else:
                    # Create a new colormap by cycling through tab20 with slight variations
                    tab20_colors = plt.cm.tab20.colors
                    colors = []
                    for i in range(num_bins):
                        # Cycle through tab20 colors with slight variations in brightness
                        base_idx = i % 20
                        cycle = i // 20
                        color = list(tab20_colors[base_idx])
                        # Adjust brightness slightly for each cycle
                        for j in range(3):
                            color[j] = max(0, min(1, color[j] - cycle * 0.05))
                        colors.append(tuple(color))
                    cmap = ListedColormap(colors)
            else:
                # For many bins, use a standard sequential colormap
                cmap = 'viridis'
            
            # Plot binning map
            im2 = ax2.imshow(bin_num_2d, origin='lower', cmap=cmap,
                         interpolation='nearest', extent=extent, aspect='equal')
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2)
            cbar2.set_label('Bin Number')
            
            # Convert center from pixels to arcseconds
            center_x_arcsec = (center_x - nx/2) * pixel_scale_x
            center_y_arcsec = (center_y - ny/2) * pixel_scale_y
            
            # Draw elliptical bin boundaries - limit to bins 0-5
            if bin_radii is not None and hasattr(bin_radii, '__len__') and len(bin_radii) > 0:
                for radius in sorted(bin_radii)[:6]:  # Sort for proper drawing order and limit to bins 0-5
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
                    ax2.add_patch(ellipse)
            
            # Add bin ID labels - carefully position to avoid overlaps - limit to bins 0-5
            if num_bins <= 15:
                # If few bins, label them all
                bins_to_label = unique_bins
            else:
                # If many bins, just label a few representative ones
                bins_to_label = np.unique(np.linspace(min(unique_bins), max(unique_bins), 10, dtype=int))
                
            for bin_id in bins_to_label:
                # Only label bins 0-5
                if bin_id > 5:
                    continue
                    
                # Find coordinates of this bin
                bin_mask = bin_num_2d == bin_id
                if np.sum(bin_mask) > 0:  # Skip empty bins
                    # Calculate centroid of this bin (in pixels)
                    y_coords, x_coords = np.where(bin_mask)
                    x_center_bin_pix = np.mean(x_coords)
                    y_center_bin_pix = np.mean(y_coords)
                    
                    # Convert to arcsec coordinates
                    x_center_bin_arcsec = (x_center_bin_pix - nx/2) * pixel_scale_x
                    y_center_bin_arcsec = (y_center_bin_pix - ny/2) * pixel_scale_y
                    
                    # Add text label - black text on white background for visibility
                    ax2.text(x_center_bin_arcsec, y_center_bin_arcsec, str(int(bin_id)), 
                           color='black', ha='center', va='center', fontsize=9,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        else:
            ax2.text(0.5, 0.5, "No bin number data available",
                  ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        
        # Set titles and labels
        ax2.set_title(f"Radial Binning ({n_rings} rings)", fontsize=14)
        ax2.set_xlabel('Arcsec')
        ax2.set_ylabel('Arcsec')
        
        # Set tick parameters for ax2
        ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax2.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Add effective radius information if available
        if Re is not None:
            # Draw effective radius ellipse in arcsec
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
            ax2.add_patch(ell_Re)
            
            # Add label for Re (positioned near the bottom)
            ax2.text(0, -extent_y/2 * 0.8, f'Re = {Re:.2f} arcsec', 
                   color='red', fontsize=12, ha='center', va='center',
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Add overall title
        plt.suptitle(f"Galaxy: {galaxy_name}", fontsize=16, y=0.98)
        
        # Add metadata at the bottom
        info_text = f"PA: {pa:.1f}°, Ellipticity: {ellipticity:.2f}"
        if Re is not None:
            info_text += f", Re: {Re:.2f} arcsec"
        info_text += f", Pixel scale: {pixel_scale_x:.3f}×{pixel_scale_y:.3f} arcsec/pixel"
        plt.figtext(0.5, 0.01, info_text, ha='center', fontsize=12)
        
        # Ensure both plots have the same axis limits
        ax1.set_xlim(extent[0], extent[1])
        ax1.set_ylim(extent[2], extent[3])
        ax2.set_xlim(extent[0], extent[1])
        ax2.set_ylim(extent[2], extent[3])
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
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

def create_parameter_radius_plots(galaxy_name, rdb_data, output_path=None, dpi=150, bins_limit=6):
    """Create parameter vs. radius plots with linear fits, using Re instead of R"""
    try:
        # Check if RDB data is valid
        if rdb_data is None or not isinstance(rdb_data, dict):
            logger.error(f"Invalid RDB data format for {galaxy_name}")
            return
            
        # Extract parameters with bin limit to specified number of bins
        params = extract_parameter_profiles(rdb_data, 
                                          parameter_names=['Fe5015', 'Mgb', 'Hbeta', 'age', 'metallicity'],
                                          bins_limit=bins_limit)
        
        if params['radius'] is None:
            logger.error(f"No radius information found for {galaxy_name}")
            return
        
        # Check if radius is valid
        if not hasattr(params['radius'], '__len__') or len(params['radius']) == 0:
            logger.error(f"Invalid radius array for {galaxy_name}")
            return
            
        # Get effective radius
        Re = params['effective_radius']
        if Re is None:
            logger.warning(f"No effective radius found for {galaxy_name}, using raw radius")
            # Just use the raw radius values
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
            'metallicity': '[M/H]'
        }
        
        # Create figure with 5 subplots in a single row
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        # Create plots for each parameter
        for i, param_name in enumerate(['Fe5015', 'Mgb', 'Hbeta', 'age', 'metallicity']):
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
        
        # Add overall title
        re_info = f" (Re = {Re:.2f} arcsec)" if Re is not None else ""
        plt.suptitle(f"Galaxy {galaxy_name}: Parameter-Radius Relations{re_info}", fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save figure if output path is provided
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved parameter-radius plots to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating parameter-radius plots: {e}")
        import traceback
        traceback.print_exc()

def create_model_grid_plots_part1(galaxy_name, rdb_data, model_data, age=1, output_path=None, dpi=150, bins_limit=6):
    """
    Create first set of model grid plots colored by R, log Age, and M/H
    Part 1: Fe5015 vs Mgb, Fe5015 vs Hbeta, Mgb vs Hbeta - Colored by R
    
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
    """
    try:
        # Extract spectral indices from galaxy data with bin limit
        galaxy_indices = extract_spectral_indices(rdb_data, bins_limit=bins_limit)  # Limit to specified bins
        
        # Define column name mapping for the model grid
        # Maps our standard names to whatever is in the model data
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Print the actual mappings being used for debugging
        logger.info(f"Model column mappings: {model_column_mapping}")
        
        # Check if we found all required columns
        missing_columns = [k for k, v in model_column_mapping.items() if v is None]
        if missing_columns:
            logger.error(f"Missing required columns in model data: {missing_columns}")
            logger.info(f"Available columns in model data: {list(model_data.columns)}")
            if 'Age' in missing_columns or 'ZoH' in missing_columns or 'AoFe' in missing_columns:
                logger.error("Critical columns missing, cannot create grid plots")
                return
        
        # Filter model data to the requested age
        age_column = model_column_mapping['Age']
        # Convert available ages to numpy array for calculation
        available_ages = np.array(sorted(model_data[age_column].unique()))
        # Find closest available age
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
        
        # Plot each panel
        for row, color_var in enumerate(color_vars):
            for col, (x_index, y_index) in enumerate(index_pairs):
                ax = axes[row, col]
                
                # Get the mapped column names for the model data
                model_x_column = model_column_mapping[x_index]
                model_y_column = model_column_mapping[y_index]
                
                # Skip if either index is missing from galaxy data or model data
                if (x_index not in galaxy_indices['bin_indices'] or 
                    y_index not in galaxy_indices['bin_indices'] or
                    model_x_column is None or model_y_column is None):
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
                            
                            # Label key grid points (optional)
                            if zoh in [-1.0, -0.5, 0.0, 0.5] and aofe in [0.0, 0.3, 0.5]:
                                label = f'[Z/H]={zoh:.1f}\n[α/Fe]={aofe:.1f}'
                                
                                # Adjust position based on values to avoid overlap
                                if aofe == 0.5 and zoh >= 0.0:
                                    xytext = (-20, 10)  # left top
                                elif aofe == 0.5:
                                    xytext = (-20, -15)  # left bottom
                                elif aofe == 0.0 and zoh >= 0.0:
                                    xytext = (5, 10)    # right top
                                elif aofe == 0.0:
                                    xytext = (5, -15)   # right bottom
                                else:
                                    xytext = (5, 5)     # default right top
                                    
                                ax.annotate(label, (x_val, y_val), 
                                          xytext=xytext, textcoords='offset points',
                                          fontsize=7, alpha=0.8, zorder=3,
                                          bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
                
                # Get galaxy data
                galaxy_x = galaxy_indices['bin_indices'][x_index]
                galaxy_y = galaxy_indices['bin_indices'][y_index]
                
                # Get color data
                if color_var in galaxy_indices['bin_indices']:
                    color_data = galaxy_indices['bin_indices'][color_var]
                    
                    # Determine color normalization
                    vmin = np.nanmin(color_data)
                    vmax = np.nanmax(color_data)
                    
                    # Plot points colored by the selected variable
                    sc = ax.scatter(galaxy_x, galaxy_y, c=color_data, 
                                  cmap='viridis', s=80, alpha=0.8, zorder=10,
                                  vmin=vmin, vmax=vmax)
                    
                    # Add bin numbers
                    for j, (x, y) in enumerate(zip(galaxy_x, galaxy_y)):
                        ax.text(x, y, str(j), fontsize=8, ha='center', va='center', 
                              color='white', fontweight='bold', zorder=11)
                    
                    # Add colorbar
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(sc, cax=cax)
                    cbar.set_label(color_labels[row])
                else:
                    # Fallback to plain scatter if color variable is not available
                    ax.scatter(galaxy_x, galaxy_y, color='green', s=80, alpha=0.7, zorder=10)
                    
                    # Add bin numbers
                    for j, (x, y) in enumerate(zip(galaxy_x, galaxy_y)):
                        ax.text(x, y, str(j), fontsize=8, ha='center', va='center', 
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
        
        # Add overall title
        plt.suptitle(f'Galaxy {galaxy_name}: Spectral Indices vs. Model Grid (Age = {closest_age} Gyr)', fontsize=16, y=0.98)
        
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

def create_model_grid_plots_part2(galaxy_name, rdb_data, model_data, ages=[1, 2, 5], output_path=None, dpi=150, bins_limit=6):
    """
    Create second set of model grid plots for multiple ages with age-colored data points
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices and bin info
    model_data : DataFrame
        Model grid data with ages, metallicities, and spectral indices
    ages : list
        List of ages to use for grid models
    output_path : str
        Path to save the output image
    dpi : int
        Resolution for the output image
    bins_limit : int
        Limit analysis to first N bins (default: 6 for bins 0-5)
    """
    try:
        # Extract spectral indices from galaxy data with bin limit
        galaxy_indices = extract_spectral_indices(rdb_data, bins_limit=bins_limit)  # Limit to specified bins
        
        # Define column name mapping for the model grid
        model_column_mapping = {
            'Fe5015': find_matching_column(model_data, ['Fe5015', 'Fe5015_SI', 'Fe5015_Index']),
            'Mgb': find_matching_column(model_data, ['Mgb', 'Mg_b', 'Mg_b_SI', 'Mgb_Index']),
            'Hbeta': find_matching_column(model_data, ['Hbeta', 'Hb', 'Hbeta_SI', 'Hb_Index', 'Hb_si']),
            'Age': find_matching_column(model_data, ['Age', 'age']),
            'ZoH': find_matching_column(model_data, ['ZoH', 'Z/H', '[Z/H]', 'metallicity', 'MOH', '[M/H]']),
            'AoFe': find_matching_column(model_data, ['AoFe', 'alpha/Fe', '[alpha/Fe]', 'A/Fe', '[A/Fe]', 'alpha'])
        }
        
        # Check if we found all required columns
        missing_columns = [k for k, v in model_column_mapping.items() if v is None]
        if missing_columns:
            logger.error(f"Missing required columns in model data: {missing_columns}")
            logger.info(f"Available columns in model data: {list(model_data.columns)}")
            if 'Age' in missing_columns or 'ZoH' in missing_columns or 'AoFe' in missing_columns:
                logger.error("Critical columns missing, cannot create grid plots")
                return
        
        # Create figure with 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # Set up the three index pairs to plot
        index_pairs = [
            ('Fe5015', 'Mgb'),
            ('Fe5015', 'Hbeta'),
            ('Mgb', 'Hbeta')
        ]
        
        # Get column references for age, metallicity and alpha/Fe
        age_column = model_column_mapping['Age']
        zoh_column = model_column_mapping['ZoH']
        aofe_column = model_column_mapping['AoFe']
        
        # Find available ages in the model - convert to numpy array for calculations
        available_ages = np.array(sorted(model_data[age_column].unique()))
        
        # Choose model ages to plot - try to use requested ages if available
        model_ages = []
        for age in ages:
            # Find closest available age
            closest_age = available_ages[np.argmin(np.abs(available_ages - age))]
            if closest_age not in model_ages:  # Avoid duplicates
                model_ages.append(closest_age)
        
        # If we have fewer than 3 unique ages, add more
        while len(model_ages) < 3 and len(available_ages) > len(model_ages):
            for age in available_ages:
                if age not in model_ages:
                    model_ages.append(age)
                    break
        
        # Ensure we have no more than 3 ages
        model_ages = model_ages[:3]
        
        # Set up color map for model grid age lines
        cmap = plt.cm.viridis
        age_colors = cmap(np.linspace(0, 1, len(model_ages)))
        
        # For each row, set a different metallicity value to highlight
        zoh_values = [-1.0, 0.0, 0.5]  # Different [Z/H] for each row
        
        # Plot each panel
        for row, zoh in enumerate(zoh_values):
            # Find closest available [Z/H] value - convert to numpy array for calculations
            available_zoh = np.array(sorted(model_data[zoh_column].unique()))
            closest_zoh = available_zoh[np.argmin(np.abs(available_zoh - zoh))]
            
            for col, (x_index, y_index) in enumerate(index_pairs):
                ax = axes[row, col]
                
                # Get the mapped column names for the model data
                model_x_column = model_column_mapping[x_index]
                model_y_column = model_column_mapping[y_index]
                
                # Skip if either index is missing from galaxy data or model data
                if (x_index not in galaxy_indices['bin_indices'] or 
                    y_index not in galaxy_indices['bin_indices'] or
                    model_x_column is None or model_y_column is None):
                    ax.text(0.5, 0.5, f"Missing {x_index} or {y_index} data", 
                          transform=ax.transAxes, ha='center', va='center')
                    continue
                
                # Draw model grids for different ages
                for i, age in enumerate(model_ages):
                    age_data = model_data[model_data[age_column] == age]
                    
                    # Draw [Z/H] lines with fixed alpha/Fe
                    for aofe in sorted(age_data[aofe_column].unique()):
                        zoh_aofe_data = age_data[age_data[aofe_column] == aofe]
                        ax.plot(zoh_aofe_data[model_x_column], zoh_aofe_data[model_y_column], '-', 
                               color=age_colors[i], alpha=0.3, linewidth=1.0, zorder=1)
                    
                    # Highlight the line for the specific [Z/H] value for this row
                    zoh_data = age_data[age_data[zoh_column] == closest_zoh]
                    ax.plot(zoh_data[model_x_column], zoh_data[model_y_column], '-', 
                           color=age_colors[i], alpha=0.8, linewidth=2.0, zorder=2)
                
                # Get galaxy data
                galaxy_x = galaxy_indices['bin_indices'][x_index]
                galaxy_y = galaxy_indices['bin_indices'][y_index]
                
                # Check if age data is available
                if 'age' in galaxy_indices['bin_indices']:
                    galaxy_age = galaxy_indices['bin_indices']['age']
                    
                    # Determine color normalization for galaxy ages
                    vmin = np.nanmin(galaxy_age)
                    vmax = np.nanmax(galaxy_age)
                    
                    # Create age colormap that matches the model grid colors
                    # Use the same colormap (viridis) for both model grid and data points
                    galaxy_cmap = plt.cm.viridis
                    
                    # Plot points colored by age
                    sc = ax.scatter(galaxy_x, galaxy_y, c=galaxy_age, 
                                  cmap=galaxy_cmap, s=80, alpha=0.8, zorder=10,
                                  vmin=vmin, vmax=vmax)
                    
                    # Add bin numbers
                    for j, (x, y) in enumerate(zip(galaxy_x, galaxy_y)):
                        ax.text(x, y, str(j), fontsize=8, ha='center', va='center', 
                              color='white', fontweight='bold', zorder=11)
                    
                    # Add colorbar with model age markers
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = plt.colorbar(sc, cax=cax)
                    cbar.set_label('Galaxy Age (Gyr)')
                    
                    # Add model ages to the colorbar
                    for i, age in enumerate(model_ages):
                        # Normalize the age to the colorbar range
                        if vmax > vmin:
                            age_normalized = (age - vmin) / (vmax - vmin)
                            # Only add marker if within range
                            if 0 <= age_normalized <= 1:
                                cbar.ax.axhline(age_normalized, color=age_colors[i], 
                                              linewidth=3, linestyle='-')
                                # Add text label for model age - use different approach to avoid transform issues
                                cbar.ax.text(1.5, age_normalized, f"{age} Gyr", 
                                           va='center', ha='left', fontsize=8,
                                           color=age_colors[i])
                else:
                    # Fallback to plain scatter if age data is not available
                    ax.scatter(galaxy_x, galaxy_y, color='green', s=80, alpha=0.7, zorder=10)
                    
                    # Add bin numbers
                    for j, (x, y) in enumerate(zip(galaxy_x, galaxy_y)):
                        ax.text(x, y, str(j), fontsize=8, ha='center', va='center', 
                              color='black', fontweight='bold', zorder=11)
                
                # Set labels and grid
                ax.set_xlabel(f'{x_index} Index', fontsize=12)
                ax.set_ylabel(f'{y_index} Index', fontsize=12)
                ax.set_title(f'{x_index} vs {y_index} - [Z/H]={closest_zoh:.1f}', fontsize=14)
                ax.grid(alpha=0.3, linestyle='--')
                
                # Set tick parameters
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.tick_params(axis='both', which='both', labelsize='x-small', right=True, top=True, direction='in')
        
        # Add legend for age models
        legend_elements = []
        for i, age in enumerate(model_ages):
            legend_elements.append(
                Line2D([0], [0], color=age_colors[i], linestyle='-', linewidth=2.0, label=f'Age = {age} Gyr')
            )
        
        # Add galaxy data marker to legend
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label='Galaxy Bins',
                   markerfacecolor='tab:green', markersize=8)
        )
        
        fig.legend(handles=legend_elements, loc='upper center', ncol=len(legend_elements), 
                 bbox_to_anchor=(0.5, 0.03), fontsize=10)
        
        # Add overall title
        plt.suptitle(f'Galaxy {galaxy_name}: Spectral Indices vs. Model Grid', fontsize=16, y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save figure
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved model grid plots part 2 to {output_path}")
        
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating model grid plots part 2: {e}")
        import traceback
        traceback.print_exc()

def extract_alpha_fe_radius(galaxy_name, rdb_data, model_data, bins_limit=6):
    """
    Extract the alpha/Fe ratio as a function of radius for a galaxy
    
    Parameters:
    -----------
    galaxy_name : str
        Galaxy name
    rdb_data : dict
        RDB data containing spectral indices and radii
    model_data : DataFrame
        Model grid data with ages, metallicities, and spectral indices
    bins_limit : int
        Limit analysis to first N bins (default: 6 for bins 0-5)
    
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
            elif 'bin_radii' in rdb_data['binning']:
                binning = rdb_data['binning'].item() if hasattr(rdb_data['binning'], 'item') else rdb_data['binning']
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
        
        # Interpolate alpha/Fe for each bin
        alpha_fe_values = []
        interpolated_zoh = []
        valid_radii = []
        
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
                'std_err': std_err
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
        
        # Store results for each individual point
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
            
            # Perform weighted interpolation using the k-nearest neighbors
            k = min(5, len(age_filtered))  # Use at most 5 neighbors for interpolation
            
            if k < 3:  # Need at least 3 points for reliable interpolation
                # Fall back to closest point if not enough neighbors
                min_distance = float('inf')
                best_alpha = None
                best_zoh = None
                
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
                        
                interpolated_alpha = best_alpha
                interpolated_zoh = best_zoh
                distance_metric = min_distance
            else:
                # Calculate distances to all points in index space
                distances = []
                for _, row in age_filtered.iterrows():
                    fe5015_diff = (row[fe5015_col] - fe5015) / 5.0
                    mgb_diff = (row[mgb_col] - mgb) / 4.0
                    hbeta_diff = (row[hbeta_col] - hbeta) / 3.0
                    
                    distance = np.sqrt(fe5015_diff**2 + mgb_diff**2 + hbeta_diff**2)
                    distances.append((distance, row[aofe_col], row[zoh_col]))
                
                # Sort by distance and get k nearest neighbors
                distances.sort(key=lambda x: x[0])
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
                'distance_metric': distance_metric
            })
        
        # Check if we found any valid points
        if not points:
            logger.warning(f"No valid points found for {galaxy_name}")
            return None
            
        # Return results
        return {
            'galaxy': galaxy_name,
            'effective_radius': Re,
            'points': points
        }
        
    except Exception as e:
        logger.error(f"Error calculating interpolated alpha/Fe for {galaxy_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        
        # Define galaxies with emission lines based on your table
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
        ax_map.text(data_center_ra, data_center_dec + 0.25, "Dataset Center", fontsize=12, ha='center', va='center',
                   weight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))
        
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
        
        # Create figure 2: Parameter-radius relations with linear fits, using Re
        # Pass bins_limit parameter
        create_parameter_radius_plots(galaxy_name, rdb_data,
                                     output_path=f"{output_dir}/{galaxy_name}_parameter_radius.png",
                                     bins_limit=bins_limit,
                                     dpi=dpi)
        
        # Create model grid visualizations if model data is provided
        if model_data is not None:
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
        
        # Get galaxy coordinates from IFU pointings
        coordinates = get_ifu_coordinates(galaxies)
        
        # Create Virgo Cluster map with vectors
        create_virgo_cluster_map_with_vectors(results_list, coordinates, 
                                            output_path=f"{output_dir}/virgo_cluster_map.png", 
                                            dpi=dpi)
        
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
            f.write(f"Spatial analysis uses precise IFU pointing centers for accurate representation of observed regions.\n\n")
            
            f.write("Analysis completed on " + pd.Timestamp.now().strftime("%Y-%m-%d"))
        
        logger.info(f"Completed galaxy summary analysis. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in galaxy summary analysis: {e}")
        import traceback
        traceback.print_exc()

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
            has_emission = galaxy in emission_line_galaxies
            
            # Add to summary data
            summary_data.append({
                'Galaxy name': galaxy,
                'α/Fe slope': slope,
                'P-value': p_value,
                'Emission Line': 'Yes' if has_emission else 'No',
                'Radial Bin Limit': radial_limit,
                'Bins Used': bins_used
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
                
                print("\n=== STATISTICAL SUMMARY ===")
                print(f"Total galaxies analyzed: {total_galaxies}")
                print(f"Galaxies with positive α/Fe gradients: {positive_slopes} ({positive_slopes/total_galaxies*100:.1f}%)")
                print(f"Galaxies with negative α/Fe gradients: {negative_slopes} ({negative_slopes/total_galaxies*100:.1f}%)")
                print(f"Statistically significant results: {significant_results} ({significant_results/total_galaxies*100:.1f}%)")
                print(f"Galaxies with emission lines: {emission_line_count} ({emission_line_count/total_galaxies*100:.1f}%)")
                
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
    
    # Create summary table
    summary_df = create_alpha_fe_results_summary(results, emission_line_galaxies, f"{output_dir}/alpha_fe_results.csv")
    
    # Get galaxy coordinates
    coordinates = get_ifu_coordinates(galaxies)
    
    # Create Virgo Cluster map visualization
    create_virgo_cluster_map_with_vectors(results, coordinates, f"{output_dir}/virgo_cluster_map.png", dpi=300)
    
    logger.info("Galaxy α/Fe radial gradient analysis complete")

    # Create individual galaxy visualizations
    output_base = "./galaxy_visualizations"
    os.makedirs(output_base, exist_ok=True)
    
    # for galaxy_name in galaxies:
    #     try:
    #         # Load galaxy data
    #         p2p_data, vnb_data, rdb_data = Read_Galaxy(galaxy_name)
            
    #         # Check if RDB data is valid before proceeding
    #         rdb_valid = rdb_data is not None and isinstance(rdb_data, dict) and len(rdb_data) > 0
    #         if not rdb_valid:
    #             logger.warning(f"No valid RDB data for {galaxy_name}, skipping visualization")
    #             continue
            
    #         # Extract cube information
    #         cube_info = extract_cube_info(galaxy_name)
            
    #         # Create output directory for this galaxy
    #         output_dir = f"{output_base}/{galaxy_name}"
            
    #         # Create visualizations including model grid plots - pass bins_limit
    #         create_galaxy_visualization(galaxy_name, p2p_data, rdb_data, cube_info, 
    #                                    model_data=model_data,
    #                                    output_dir=output_dir, dpi=300,
    #                                    bins_limit=bins_limit)  # Pass bins_limit
                                       
    #         logger.info(f"Completed visualization for {galaxy_name}")
    #     except Exception as e:
    #         logger.error(f"Failed to process {galaxy_name}: {e}")
    #         continue
    
    logger.info("All visualizations complete!")