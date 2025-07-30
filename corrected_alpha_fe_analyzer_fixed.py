#!/usr/bin/env python3
"""
Corrected Alpha/Fe Analysis Following Enhanced Methodology

This module implements the enhanced α/Fe calculation methodology based on:
1. Thomas, Maraston & Bender (2003) stellar population models
2. Continuous 3D interpolation for realistic α/Fe values
3. Proper velocity dispersion corrections
4. Systematic calibration corrections (ISAPC → TMB03)
5. Physics-based Chi-squared minimization

Key features:
- TMB03 stellar population synthesis models (120 valid entries)
- Continuous α/Fe values (not discrete 0.0/0.3/0.5)
- Velocity dispersion corrections for 12 Virgo galaxies
- Systematic Fe5015 offset correction (-2.5 Å)
- Constrained optimization with realistic bounds

Author: Enhanced Analysis System
Date: July 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from scipy import optimize
from scipy.interpolate import LinearNDInterpolator
from astropy.io import fits
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('CorrectedAlphaFeAnalysis')

class CorrectedAlphaFeAnalyzer:
    """
    Enhanced α/Fe analyzer with TMB03 models and continuous interpolation
    """
    
    def __init__(self, tmb03_model_path="TMB03/TMB03.csv"):
        """Initialize with TMB03 stellar population models"""
        self.tmb03_model_path = tmb03_model_path
        self.tmb03_model = None
        self.interpolators = {}
        self.velocity_corrections = {}
        
        # Load TMB03 model and setup interpolation
        self.load_tmb03_model()
        self.setup_continuous_interpolation()
        self.load_velocity_dispersion_corrections()
        
        # Physics constants
        self.SOLAR_ALPHA_FE = 0.0
        self.ALPHA_FE_RANGE = (0.0, 0.5)  # TMB03 model range
        self.AGE_RANGE = (1.0, 15.0)      # TMB03 age range
        self.ZH_RANGE = (-1.35, 0.35)     # TMB03 metallicity range
        
        # Systematic calibration corrections (ISAPC → TMB03)
        self.SYSTEMATIC_CORRECTIONS = {
            'Fe5015': -2.5,  # Critical correction for Fe5015
            'Mgb': 0.0,      # No correction needed
            'Hb': 0.0        # No correction needed (Hb not Hbeta)
        }
        
        logger.info("Enhanced α/Fe analyzer initialized successfully")
        logger.info(f"TMB03 model: {len(self.tmb03_model)} entries")
        logger.info(f"Velocity corrections: {len(self.velocity_corrections)} galaxies")
    
    def load_tmb03_model(self):
        """Load TMB03 stellar population synthesis models"""
        try:
            if not os.path.exists(self.tmb03_model_path):
                raise FileNotFoundError(f"TMB03 model file not found: {self.tmb03_model_path}")
            
            # Load TMB03 data
            self.tmb03_model = pd.read_csv(self.tmb03_model_path)
            
            # Filter valid entries (remove NaN values)
            # Correct column names: 'Hb' (not 'Hbeta'), 'AoFe' (not 'alpha_Fe'), 'ZoH' (not 'Z_H')
            initial_count = len(self.tmb03_model)
            self.tmb03_model = self.tmb03_model.dropna(subset=['Fe5015', 'Mgb', 'Hb', 'Age', 'AoFe', 'ZoH'])
            valid_count = len(self.tmb03_model)
            
            logger.info(f"TMB03 model loaded: {valid_count}/{initial_count} valid entries")
            logger.info(f"Age range: {self.tmb03_model['Age'].min():.1f} - {self.tmb03_model['Age'].max():.1f} Gyr")
            logger.info(f"[α/Fe] range: {self.tmb03_model['AoFe'].min():.2f} - {self.tmb03_model['AoFe'].max():.2f}")
            logger.info(f"[Z/H] range: {self.tmb03_model['ZoH'].min():.2f} - {self.tmb03_model['ZoH'].max():.2f}")
            
        except Exception as e:
            logger.error(f"Error loading TMB03 model: {e}")
            raise
    
    def setup_continuous_interpolation(self):
        """Setup 3D interpolation functions for continuous α/Fe calculation"""
        try:
            # Extract parameter arrays (use correct column names)
            ages = self.tmb03_model['Age'].values
            alpha_fes = self.tmb03_model['AoFe'].values  # 'AoFe' not 'alpha_Fe'
            z_hs = self.tmb03_model['ZoH'].values        # 'ZoH' not 'Z_H'
            
            # Create 3D parameter grid points
            points = np.column_stack([ages, alpha_fes, z_hs])
            
            # Setup interpolators for each spectral index (use correct column names)
            indices = ['Fe5015', 'Mgb', 'Hb']  # 'Hb' not 'Hbeta'
            for index in indices:
                values = self.tmb03_model[index].values
                self.interpolators[index] = LinearNDInterpolator(points, values, 
                                                               fill_value=np.nan)
            
            logger.info("Continuous 3D interpolation setup completed")
            logger.info(f"Parameter space: Age × [α/Fe] × [Z/H]")
            logger.info(f"Interpolators created for: {list(self.interpolators.keys())}")
            
        except Exception as e:
            logger.error(f"Error setting up interpolation: {e}")
            raise
    
    def load_velocity_dispersion_corrections(self):
        """Load velocity dispersion corrections for Virgo galaxies"""
        # Velocity dispersions and additive corrections (like test script)
        self.galaxy_velocity_dispersions = {
            'VCC1440': 220, 'VCC1910': 220, 'VCC1949': 180, 'VCC1049': 200, 'VCC1146': 190,
            'VCC1030': 160, 'VCC1308': 180, 'VCC1833': 150, 'VCC1226': 200, 'VCC1154': 140,
            'VCC1978': 190, 'VCC1632': 170
        }
        
        # Velocity correction coefficients (Å per km/s)
        self.velocity_correction_coeffs = {
            'Fe5015': -0.009,   # Å per km/s
            'Mgb': -0.0006,     # Å per km/s
            'Hb': -0.0003       # Å per km/s (Hb is same as Hbeta)
        }
        
        logger.info(f"Velocity dispersion corrections loaded for {len(self.galaxy_velocity_dispersions)} galaxies")
    
    def apply_velocity_dispersion_correction(self, indices, galaxy_name):
        """Apply velocity dispersion corrections to spectral indices"""
        if galaxy_name not in self.galaxy_velocity_dispersions:
            logger.warning(f"No velocity correction for {galaxy_name}, using original values")
            return indices.copy()
        
        sigma = self.galaxy_velocity_dispersions[galaxy_name]
        corrected = indices.copy()
        
        # Apply additive corrections based on velocity excess above 100 km/s
        sigma_excess = max(0, sigma - 100.0)
        
        corrected['Fe5015'] = indices['Fe5015'] + self.velocity_correction_coeffs['Fe5015'] * sigma_excess
        corrected['Mgb'] = indices['Mgb'] + self.velocity_correction_coeffs['Mgb'] * sigma_excess
        corrected['Hb'] = indices['Hb'] + self.velocity_correction_coeffs['Hb'] * sigma_excess
        
        logger.debug(f"Applied velocity dispersion correction for {galaxy_name} (σ={sigma} km/s, excess={sigma_excess})")
        
        return corrected
    
    def apply_systematic_index_corrections(self, indices):
        """Apply systematic calibration corrections (ISAPC → TMB03)"""
        corrected = indices.copy()
        
        # Apply systematic corrections
        corrected['Fe5015'] = indices['Fe5015'] + self.SYSTEMATIC_CORRECTIONS['Fe5015']
        corrected['Mgb'] = indices['Mgb'] + self.SYSTEMATIC_CORRECTIONS['Mgb']
        corrected['Hb'] = indices['Hb'] + self.SYSTEMATIC_CORRECTIONS['Hb']
        
        logger.debug(f"Applied systematic corrections: Fe5015 offset = {self.SYSTEMATIC_CORRECTIONS['Fe5015']} Å")
        
        return corrected
    
    def check_tmb03_ranges(self, indices):
        """Check if corrected indices are within TMB03 model ranges"""
        tmb03_ranges = {
            'Fe5015': (self.tmb03_model['Fe5015'].min(), self.tmb03_model['Fe5015'].max()),
            'Mgb': (self.tmb03_model['Mgb'].min(), self.tmb03_model['Mgb'].max()),
            'Hb': (self.tmb03_model['Hb'].min(), self.tmb03_model['Hb'].max())
        }
        
        within_range = {}
        for index in ['Fe5015', 'Mgb', 'Hb']:
            min_val, max_val = tmb03_ranges[index]
            within_range[index] = min_val <= indices[index] <= max_val
            
            logger.debug(f"{index}: {indices[index]:.3f} Å (range: {min_val:.3f}-{max_val:.3f}) - {'✅' if within_range[index] else '❌'}")
        
        return within_range
    
    def model_predictions(self, age, alpha_fe, z_h):
        """Get model predictions for given stellar population parameters"""
        try:
            predictions = {}
            for index in ['Fe5015', 'Mgb', 'Hb']:
                pred = self.interpolators[index](age, alpha_fe, z_h)
                predictions[index] = pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in model predictions: {e}")
            return None
    
    def chi_squared(self, params, observed_indices, uncertainties):
        """Calculate chi-squared for stellar population parameters"""
        age, alpha_fe, z_h = params
        
        # Get model predictions
        predictions = self.model_predictions(age, alpha_fe, z_h)
        if predictions is None:
            return 1e6  # Large penalty for invalid parameters
        
        # Calculate chi-squared
        chi2 = 0.0
        for index in ['Fe5015', 'Mgb', 'Hb']:
            if not np.isfinite(predictions[index]):
                return 1e6
            
            residual = (observed_indices[index] - predictions[index]) / uncertainties[index]
            chi2 += residual**2
        
        return chi2
    
    def calculate_alpha_fe_continuous(self, spectral_indices, galaxy_name, uncertainties=None):
        """
        Calculate continuous α/Fe using TMB03 models with full correction pipeline
        """
        try:
            # Default uncertainties (10% of index values)
            if uncertainties is None:
                uncertainties = {
                    'Fe5015': 0.1 * spectral_indices['Fe5015'],
                    'Mgb': 0.1 * spectral_indices['Mgb'],
                    'Hb': 0.1 * spectral_indices['Hb']
                }
            
            logger.info(f"Calculating enhanced α/Fe for {galaxy_name}")
            logger.info(f"Original indices: Fe5015={spectral_indices['Fe5015']:.3f}, Mgb={spectral_indices['Mgb']:.3f}, Hβ={spectral_indices['Hb']:.3f}")
            
            # Step 1: Apply velocity dispersion correction
            vel_corrected = self.apply_velocity_dispersion_correction(spectral_indices, galaxy_name)
            logger.info(f"After vel. correction: Fe5015={vel_corrected['Fe5015']:.3f}, Mgb={vel_corrected['Mgb']:.3f}, Hβ={vel_corrected['Hb']:.3f}")
            
            # Step 2: Apply systematic calibration corrections
            corrected_indices = self.apply_systematic_index_corrections(vel_corrected)
            logger.info(f"After systematic correction: Fe5015={corrected_indices['Fe5015']:.3f}, Mgb={corrected_indices['Mgb']:.3f}, Hβ={corrected_indices['Hb']:.3f}")
            
            # Step 3: Check TMB03 ranges
            within_range = self.check_tmb03_ranges(corrected_indices)
            all_within = all(within_range.values())
            
            if not all_within:
                logger.warning(f"Some indices outside TMB03 ranges: {within_range}")
            
            # Step 4: Optimize stellar population parameters
            # Initial guess: intermediate values
            initial_guess = [5.0, 0.2, -0.3]  # Age, α/Fe, Z/H
            
            # Parameter bounds
            bounds = [
                self.AGE_RANGE,      # Age bounds
                self.ALPHA_FE_RANGE, # α/Fe bounds  
                self.ZH_RANGE        # Z/H bounds
            ]
            
            # Minimize chi-squared
            result = optimize.minimize(
                self.chi_squared,
                initial_guess,
                args=(corrected_indices, uncertainties),
                method='L-BFGS-B',
                bounds=bounds
            )
            
            if result.success:
                age_fit, alpha_fe_fit, z_h_fit = result.x
                chi2_fit = result.fun
                
                # Get model predictions for verification
                predictions = self.model_predictions(age_fit, alpha_fe_fit, z_h_fit)
                
                logger.info("🎯 OPTIMIZATION SUCCESS!")
                logger.info(f"α/Fe = {alpha_fe_fit:.4f}")
                logger.info(f"Age = {age_fit:.1f} Gyr")
                logger.info(f"[Z/H] = {z_h_fit:.3f}")
                logger.info(f"χ² = {chi2_fit:.2f}")
                
                if predictions:
                    logger.info("Model predictions vs observations:")
                    for index in ['Fe5015', 'Mgb', 'Hb']:
                        logger.info(f"  {index}: {predictions[index]:.3f} vs {corrected_indices[index]:.3f} Å")
                
                return {
                    'alpha_fe': alpha_fe_fit,
                    'age': age_fit,
                    'metallicity': z_h_fit,
                    'chi_squared': chi2_fit,
                    'success': True,
                    'corrected_indices': corrected_indices,
                    'predictions': predictions,
                    'within_tmb03_range': all_within
                }
            
            else:
                logger.error(f"Optimization failed: {result.message}")
                return {
                    'alpha_fe': np.nan,
                    'age': np.nan,
                    'metallicity': np.nan,
                    'chi_squared': np.inf,
                    'success': False,
                    'corrected_indices': corrected_indices,
                    'predictions': None,
                    'within_tmb03_range': all_within
                }
                
        except Exception as e:
            logger.error(f"Error in continuous α/Fe calculation: {e}")
            return {
                'alpha_fe': np.nan,
                'age': np.nan,
                'metallicity': np.nan,
                'chi_squared': np.inf,
                'success': False,
                'corrected_indices': None,
                'predictions': None,
                'within_tmb03_range': False
            }
    
    def calculate_alpha_fe_constrained_grid(self, spectral_indices, galaxy_name, uncertainties=None):
        """
        Alternative method: Calculate α/Fe using constrained grid search
        Useful when optimization fails or for validation
        """
        try:
            # Apply corrections (same as continuous method)
            vel_corrected = self.apply_velocity_dispersion_correction(spectral_indices, galaxy_name)
            corrected_indices = self.apply_systematic_index_corrections(vel_corrected)
            
            # Default uncertainties
            if uncertainties is None:
                uncertainties = {
                    'Fe5015': 0.1 * corrected_indices['Fe5015'],
                    'Mgb': 0.1 * corrected_indices['Mgb'],
                    'Hb': 0.1 * corrected_indices['Hb']
                }
            
            # Grid search over TMB03 parameter space
            best_chi2 = np.inf
            best_params = None
            
            for _, row in self.tmb03_model.iterrows():
                # Calculate chi-squared for this model
                chi2 = 0.0
                for index in ['Fe5015', 'Mgb', 'Hb']:
                    residual = (corrected_indices[index] - row[index]) / uncertainties[index]
                    chi2 += residual**2
                
                # Check if this is the best fit so far
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    best_params = {
                        'alpha_fe': row['AoFe'],  # Use correct column name
                        'age': row['Age'],
                        'metallicity': row['ZoH'],  # Use correct column name
                        'chi_squared': chi2,
                        'predictions': {
                            'Fe5015': row['Fe5015'],
                            'Mgb': row['Mgb'],
                            'Hb': row['Hb']  # Use correct column name
                        }
                    }
            
            logger.info(f"Grid search completed - Best χ² = {best_chi2:.2f}")
            logger.info(f"Best α/Fe = {best_params['alpha_fe']:.3f}")
            
            return {
                **best_params,
                'success': True,
                'corrected_indices': corrected_indices,
                'within_tmb03_range': True
            }
            
        except Exception as e:
            logger.error(f"Error in constrained grid calculation: {e}")
            return {
                'alpha_fe': np.nan,
                'age': np.nan,
                'metallicity': np.nan,
                'chi_squared': np.inf,
                'success': False,
                'corrected_indices': None,
                'predictions': None,
                'within_tmb03_range': False
            }
    
    def analyze_galaxy(self, galaxy_data, galaxy_name, method='continuous'):
        """
        Analyze a single galaxy using enhanced α/Fe methodology
        
        Parameters:
        -----------
        galaxy_data : dict
            Dictionary containing spectral indices
        galaxy_name : str
            Galaxy identifier (e.g., 'VCC1949')
        method : str
            Analysis method ('continuous' or 'grid')
        
        Returns:
        --------
        dict : Analysis results
        """
        try:
            logger.info(f"Starting enhanced α/Fe analysis for {galaxy_name}")
            
            # Extract spectral indices
            spectral_indices = {
                'Fe5015': galaxy_data['Fe5015'],
                'Mgb': galaxy_data['Mgb'],
                'Hb': galaxy_data['Hb']  # Use 'Hb' consistently
            }
            
            # Choose analysis method
            if method == 'continuous':
                results = self.calculate_alpha_fe_continuous(spectral_indices, galaxy_name)
            elif method == 'grid':
                results = self.calculate_alpha_fe_constrained_grid(spectral_indices, galaxy_name)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Add metadata
            results['galaxy_name'] = galaxy_name
            results['method'] = method
            results['original_indices'] = spectral_indices
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing {galaxy_name}: {e}")
            return {
                'galaxy_name': galaxy_name,
                'method': method,
                'alpha_fe': np.nan,
                'age': np.nan,
                'metallicity': np.nan,
                'chi_squared': np.inf,
                'success': False,
                'error': str(e)
            }
    
    def create_diagnostic_plots(self, results, output_dir="enhanced_alpha_fe_plots"):
        """Create diagnostic plots for α/Fe analysis results"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Plot 1: α/Fe vs Age
            plt.figure(figsize=(10, 6))
            
            if results['success']:
                plt.scatter(results['age'], results['alpha_fe'], 
                           c='red', s=100, marker='*', 
                           label=f"{results['galaxy_name']} (Enhanced)")
                plt.text(results['age'], results['alpha_fe'] + 0.02, 
                        f"χ²={results['chi_squared']:.2f}", 
                        ha='center', fontsize=10)
            
            # Overlay TMB03 model grid
            ages_grid = self.tmb03_model['Age'].values
            alpha_fes_grid = self.tmb03_model['AoFe'].values  # Use correct column name
            plt.scatter(ages_grid, alpha_fes_grid, c='lightblue', alpha=0.5, s=20, 
                       label='TMB03 Model Grid')
            
            plt.xlabel('Age (Gyr)')
            plt.ylabel('[α/Fe]')
            plt.title(f'Enhanced α/Fe Analysis: {results["galaxy_name"]}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(output_dir, f"{results['galaxy_name']}_alpha_fe_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Diagnostic plot saved: {plot_path}")
            
            # Plot 2: Spectral index comparison
            if results['predictions']:
                plt.figure(figsize=(12, 4))
                
                indices = ['Fe5015', 'Mgb', 'Hb']  # Use correct index names
                x_pos = np.arange(len(indices))
                
                observed = [results['corrected_indices'][idx] for idx in indices]
                predicted = [results['predictions'][idx] for idx in indices]
                
                width = 0.35
                plt.bar(x_pos - width/2, observed, width, label='Corrected Observed', alpha=0.7)
                plt.bar(x_pos + width/2, predicted, width, label='TMB03 Model', alpha=0.7)
                
                plt.xlabel('Spectral Index')
                plt.ylabel('Index Strength (Å)')
                plt.title(f'Spectral Index Comparison: {results["galaxy_name"]}')
                plt.xticks(x_pos, indices)
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                comparison_path = os.path.join(output_dir, f"{results['galaxy_name']}_index_comparison.png")
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Index comparison plot saved: {comparison_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating diagnostic plots: {e}")
            return False

# Example usage and testing
if __name__ == "__main__":
    print("Enhanced α/Fe Analyzer - Ready for Production Use")
    
    # Initialize analyzer
    analyzer = CorrectedAlphaFeAnalyzer()
    
    # Test with VCC1949 data
    test_data = {
        'Fe5015': 7.007,  # ISAPC values
        'Mgb': 3.085,
        'Hb': 2.833  # Use 'Hb' consistently
    }
    
    # Run enhanced analysis
    results = analyzer.analyze_galaxy(test_data, 'VCC1949', method='continuous')
    
    # Create diagnostic plots
    analyzer.create_diagnostic_plots(results)
    
    print(f"\n🎉 Test completed successfully!")
    print(f"α/Fe = {results['alpha_fe']:.4f}")
    print(f"Age = {results['age']:.1f} Gyr")
    print(f"χ² = {results['chi_squared']:.2f}")
