"""
Enhanced Stellar population analysis utilities with error propagation
"""

from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import error propagation utilities
try:
    from utils.error_propagation import (
        monte_carlo_error_propagation,
        bootstrap_error_estimate,
        calculate_covariance_matrix
    )
    HAS_ERROR_UTILS = True
except ImportError:
    HAS_ERROR_UTILS = False
    logger.warning("Error propagation utilities not available. Error estimation will be limited.")


class WeightParser:
    """Enhanced class for parsing fitting weights to physical parameters with error propagation."""

    def __init__(self, template_path: Union[str, Path]) -> None:
        """Initialize with SSP template.

        Args:
            template_path: Path to the SSP template .npz file
        """
        # 加载模板数据
        data = np.load(template_path, allow_pickle=True)
        self.ages = data["ages"]  # 年龄数组 (25,)
        self.metals = data["metals"]  # 金属丰度数组 (6,)

        # 验证模板维度
        if len(self.ages) != 25 or len(self.metals) != 6:
            raise ValueError(
                f"Invalid template dimensions: "
                f"ages={len(self.ages)}, metals={len(self.metals)}"
            )

        # 构建参数网格
        age_grid, metal_grid = np.meshgrid(self.ages, self.metals, indexing="ij")
        # age_grid shape: (25, 6), 每行相同的age值
        # metal_grid shape: (25, 6), 每列相同的metal值

        # 将网格reshape为与模板相同的方式
        self.age_vector = age_grid.reshape(-1)  # (150,)
        self.metal_vector = metal_grid.reshape(-1)  # (150,)

        # 计算年龄的对数值
        self.log_age_vector = np.log10(self.age_vector)
        
        # Store template info for error calculations
        self.n_ages = len(self.ages)
        self.n_metals = len(self.metals)
        self.n_templates = len(self.age_vector)

    def parse_weights(
        self, weights: Union[List[float], np.ndarray],
        weight_errors: Optional[Union[List[float], np.ndarray]] = None
    ) -> Union[Tuple[float, float], Tuple[float, float, float, float]]:
        """Parse weights to get mean log(Age) and [M/H] with optional errors.

        Args:
            weights: Fitting weights (150,)
            weight_errors: Optional weight uncertainties (150,)

        Returns:
            If weight_errors is None:
                tuple: (mean_log_age, mean_metallicity)
            Otherwise:
                tuple: (mean_log_age, mean_metallicity, log_age_error, metallicity_error)
        """
        # 验证权重长度
        weights = np.array(weights)
        if len(weights) != len(self.age_vector):
            raise ValueError(
                f"Weights must have length {len(self.age_vector)}, got {len(weights)}"
            )

        # 计算总权重
        total_weight = np.sum(weights)
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")

        # 直接用向量计算加权平均
        mean_log_age = np.sum(self.log_age_vector * weights) / total_weight
        mean_metallicity = np.sum(self.metal_vector * weights) / total_weight

        # If no errors requested, return basic values
        if weight_errors is None:
            return mean_log_age, mean_metallicity
        
        # Calculate errors using error propagation
        weight_errors = np.array(weight_errors)
        if len(weight_errors) != len(weights):
            raise ValueError(f"Weight errors must have same length as weights")
        
        # Error propagation for weighted mean
        # Var(weighted mean) = sum(wi^2 * var(xi)) / (sum(wi))^2
        # Plus contribution from weight uncertainties
        
        # 1. Intrinsic scatter in the templates
        log_age_scatter = self._calculate_intrinsic_scatter(
            weights, self.log_age_vector, mean_log_age
        )
        metal_scatter = self._calculate_intrinsic_scatter(
            weights, self.metal_vector, mean_metallicity
        )
        
        # 2. Error from weight uncertainties
        log_age_weight_error = self._propagate_weight_errors(
            weights, weight_errors, self.log_age_vector, total_weight
        )
        metal_weight_error = self._propagate_weight_errors(
            weights, weight_errors, self.metal_vector, total_weight
        )
        
        # Combine errors
        log_age_error = np.sqrt(log_age_scatter**2 + log_age_weight_error**2)
        metallicity_error = np.sqrt(metal_scatter**2 + metal_weight_error**2)
        
        return mean_log_age, mean_metallicity, log_age_error, metallicity_error

    def get_physical_params(
        self, weights: Union[List[float], np.ndarray],
        weight_errors: Optional[Union[List[float], np.ndarray]] = None,
        weight_covariance: Optional[np.ndarray] = None,
        n_monte_carlo: int = 0
    ) -> dict:
        """Get all physical parameters from weights with comprehensive error estimation.
        
        Args:
            weights: Fitting weights (150,)
            weight_errors: Optional weight uncertainties (150,)
            weight_covariance: Optional weight covariance matrix (150, 150)
            n_monte_carlo: Number of Monte Carlo iterations for error estimation
            
        Returns:
            dict: Physical parameters with errors if requested
        """
        # Basic calculation
        if weight_errors is None and weight_covariance is None:
            log_age, metal = self.parse_weights(weights)
            return {"log_age": log_age, "age": 10**log_age, "metallicity": metal}
        
        # If we have covariance matrix and Monte Carlo requested
        if weight_covariance is not None and n_monte_carlo > 0 and HAS_ERROR_UTILS:
            return self._monte_carlo_params(
                weights, weight_covariance, n_monte_carlo
            )
        
        # Simple error propagation
        if weight_errors is not None:
            log_age, metal, log_age_err, metal_err = self.parse_weights(
                weights, weight_errors
            )
            
            # Propagate error to linear age
            age = 10**log_age
            # d(age)/d(log_age) = age * ln(10)
            age_error = age * np.log(10) * log_age_err
            
            return {
                "log_age": log_age,
                "log_age_error": log_age_err,
                "age": age,
                "age_error": age_error,
                "metallicity": metal,
                "metallicity_error": metal_err
            }
        
        # Fallback to basic calculation
        log_age, metal = self.parse_weights(weights)
        return {"log_age": log_age, "age": 10**log_age, "metallicity": metal}

    def _calculate_intrinsic_scatter(
        self, weights: np.ndarray, values: np.ndarray, mean_value: float
    ) -> float:
        """Calculate intrinsic scatter of weighted distribution."""
        total_weight = np.sum(weights)
        if total_weight <= 0:
            return 0.0
        
        # Weighted variance
        variance = np.sum(weights * (values - mean_value)**2) / total_weight
        
        # Effective number of templates contributing
        n_eff = total_weight**2 / np.sum(weights**2) if np.sum(weights**2) > 0 else 1
        
        # Standard error
        return np.sqrt(variance / n_eff)

    def _propagate_weight_errors(
        self, weights: np.ndarray, weight_errors: np.ndarray,
        values: np.ndarray, total_weight: float
    ) -> float:
        """Propagate weight uncertainties to parameter error."""
        # For weighted mean: f = sum(wi * vi) / sum(wi)
        # df/dwi = (vi * sum(w) - sum(w*v)) / sum(w)^2 = (vi - mean) / sum(w)
        
        mean_value = np.sum(weights * values) / total_weight
        
        # Partial derivatives
        partials = (values - mean_value) / total_weight
        
        # Error propagation: sigma^2 = sum((df/dwi)^2 * sigma_wi^2)
        variance = np.sum(partials**2 * weight_errors**2)
        
        return np.sqrt(variance)

    def _monte_carlo_params(
        self, weights: np.ndarray, weight_covariance: np.ndarray,
        n_iterations: int = 1000
    ) -> dict:
        """Monte Carlo error estimation using weight covariance matrix."""
        logger.info(f"Running Monte Carlo error estimation with {n_iterations} iterations")
        
        # Storage for MC samples
        log_ages = []
        ages = []
        metals = []
        
        # Check if covariance matrix is valid
        if weight_covariance.shape != (self.n_templates, self.n_templates):
            logger.warning("Invalid covariance matrix shape, using diagonal approximation")
            weight_errors = np.sqrt(np.diag(weight_covariance))
            weight_covariance = np.diag(weight_errors**2)
        
        # Generate samples
        try:
            # Use multivariate normal for correlated errors
            weight_samples = np.random.multivariate_normal(
                weights, weight_covariance, size=n_iterations
            )
        except np.linalg.LinAlgError:
            logger.warning("Covariance matrix not positive definite, using diagonal approximation")
            weight_errors = np.sqrt(np.diag(weight_covariance))
            weight_samples = np.random.normal(
                weights, weight_errors, size=(n_iterations, self.n_templates)
            )
        
        # Calculate parameters for each sample
        for i in range(n_iterations):
            weight_sample = weight_samples[i]
            
            # Ensure non-negative weights (physical constraint)
            weight_sample = np.maximum(weight_sample, 0)
            
            # Skip if all weights are zero
            if np.sum(weight_sample) == 0:
                continue
            
            try:
                log_age, metal = self.parse_weights(weight_sample)
                log_ages.append(log_age)
                ages.append(10**log_age)
                metals.append(metal)
            except:
                continue
        
        # Calculate statistics
        if len(log_ages) > 100:  # Need enough samples
            results = {
                "log_age": np.median(log_ages),
                "log_age_error": np.std(log_ages),
                "log_age_percentiles": np.percentile(log_ages, [16, 50, 84]),
                "age": np.median(ages),
                "age_error": np.std(ages),
                "age_percentiles": np.percentile(ages, [16, 50, 84]),
                "metallicity": np.median(metals),
                "metallicity_error": np.std(metals),
                "metallicity_percentiles": np.percentile(metals, [16, 50, 84]),
                "n_valid_samples": len(log_ages)
            }
            
            logger.info(f"Monte Carlo completed with {len(log_ages)} valid samples")
            
        else:
            logger.warning("Too few valid MC samples, using simple error propagation")
            # Fallback to simple diagonal errors
            weight_errors = np.sqrt(np.diag(weight_covariance))
            log_age, metal, log_age_err, metal_err = self.parse_weights(
                weights, weight_errors
            )
            age = 10**log_age
            age_error = age * np.log(10) * log_age_err
            
            results = {
                "log_age": log_age,
                "log_age_error": log_age_err,
                "age": age,
                "age_error": age_error,
                "metallicity": metal,
                "metallicity_error": metal_err
            }
        
        return results

    def calculate_mass_weighted_age(
        self, weights: np.ndarray, weight_errors: Optional[np.ndarray] = None
    ) -> Union[float, Tuple[float, float]]:
        """Calculate mass-weighted age (different from light-weighted)."""
        # For SSP models, mass-to-light ratio changes with age
        # Young populations have lower M/L than old populations
        # This is a simplified implementation
        
        # Approximate M/L ratio scaling with age (very simplified)
        # Real implementation would use SSP model predictions
        ml_ratio = np.sqrt(self.age_vector / 1e9)  # Rough approximation
        
        # Mass-weighted age
        mass_weights = weights * ml_ratio
        total_mass_weight = np.sum(mass_weights)
        
        if total_mass_weight <= 0:
            if weight_errors is None:
                return np.nan
            else:
                return np.nan, np.nan
        
        mass_weighted_age = np.sum(mass_weights * self.age_vector) / total_mass_weight
        
        if weight_errors is not None:
            # Propagate errors
            mass_weight_errors = weight_errors * ml_ratio
            age_error = self._propagate_weight_errors(
                mass_weights, mass_weight_errors, 
                self.age_vector, total_mass_weight
            )
            return mass_weighted_age, age_error
        
        return mass_weighted_age

    def get_formation_history(
        self, weights: np.ndarray, weight_errors: Optional[np.ndarray] = None
    ) -> dict:
        """Extract star formation history information."""
        # Reshape weights to age-metallicity grid
        weight_grid = weights.reshape(self.n_ages, self.n_metals)
        
        # Marginalize over metallicity to get age distribution
        age_distribution = np.sum(weight_grid, axis=1)
        age_distribution /= np.sum(age_distribution)
        
        # Find peak formation epoch
        peak_idx = np.argmax(age_distribution)
        peak_age = self.ages[peak_idx]
        
        # Calculate formation timescale (width of distribution)
        cumsum = np.cumsum(age_distribution)
        idx_16 = np.searchsorted(cumsum, 0.16)
        idx_84 = np.searchsorted(cumsum, 0.84)
        
        formation_timescale = self.ages[idx_84] - self.ages[idx_16]
        
        results = {
            "age_distribution": age_distribution,
            "peak_formation_age": peak_age,
            "formation_timescale": formation_timescale,
            "ages": self.ages
        }
        
        # Add errors if available
        if weight_errors is not None:
            weight_error_grid = weight_errors.reshape(self.n_ages, self.n_metals)
            age_dist_error = np.sqrt(np.sum(weight_error_grid**2, axis=1))
            age_dist_error /= np.sum(age_distribution)  # Normalize
            
            results["age_distribution_error"] = age_dist_error
        
        return results