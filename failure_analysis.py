#!/usr/bin/env python3
"""
Failure Analysis for VNB and RDB Dual Mode Analysis

This script investigates why certain galaxies failed in VNB/RDB analysis
and provides solutions to fix them.
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path

def setup_logging():
    """Setup logging for failure analysis"""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

logger = setup_logging()

def check_galaxy_data_availability(galaxy_name):
    """
    Comprehensive check of what data is available for each galaxy
    """
    result = {
        'galaxy_name': galaxy_name,
        'alpha_fe_data': False,
        'vnb_data': False,
        'rdb_data': False,
        'p2p_data': False,
        'alpha_fe_path': None,
        'vnb_path': None,
        'rdb_path': None,
        'p2p_path': None,
        'issue_summary': [],
        'suggested_fixes': []
    }
    
    # Check alpha/Fe data
    alpha_fe_path = f"alpha_fe_analysis_results/analysis_20250720_091707/{galaxy_name}/{galaxy_name}_alpha_fe_analysis.npz"
    if os.path.exists(alpha_fe_path):
        result['alpha_fe_data'] = True
        result['alpha_fe_path'] = alpha_fe_path
    else:
        result['issue_summary'].append("Alpha/Fe analysis data missing")
        result['suggested_fixes'].append("Run alpha/Fe analysis for this galaxy")
    
    # Check for analysis results in various possible locations
    possible_vnb_paths = [
        f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_VNB_results.npz",
        f"./output/{galaxy_name}/Data/{galaxy_name}_stack_VNB_results.npz",
        f"./output/{galaxy_name}/{galaxy_name}_stack/Data/{galaxy_name}_stack_VNB_results.npz"
    ]
    
    possible_rdb_paths = [
        f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_RDB_results.npz",
        f"./output/{galaxy_name}/Data/{galaxy_name}_stack_RDB_results.npz",
        f"./output/{galaxy_name}/{galaxy_name}_stack/Data/{galaxy_name}_stack_RDB_results.npz"
    ]
    
    possible_p2p_paths = [
        f"./output/{galaxy_name}_stack/Data/{galaxy_name}_stack_P2P_results.npz",
        f"./output/{galaxy_name}/Data/{galaxy_name}_stack_P2P_results.npz",
        f"./output/{galaxy_name}/{galaxy_name}_stack/Data/{galaxy_name}_stack_P2P_results.npz"
    ]
    
    # Check VNB data
    for path in possible_vnb_paths:
        if os.path.exists(path):
            result['vnb_data'] = True
            result['vnb_path'] = path
            break
    
    # Check RDB data  
    for path in possible_rdb_paths:
        if os.path.exists(path):
            result['rdb_data'] = True
            result['rdb_path'] = path
            break
    
    # Check P2P data
    for path in possible_p2p_paths:
        if os.path.exists(path):
            result['p2p_data'] = True
            result['p2p_path'] = path
            break
    
    # Analysis of issues
    if not result['vnb_data']:
        result['issue_summary'].append("VNB analysis results missing")
        result['suggested_fixes'].append("Run VNB analysis for this galaxy")
    
    if not result['rdb_data']:
        result['issue_summary'].append("RDB analysis results missing")
        result['suggested_fixes'].append("Run RDB analysis for this galaxy")
    
    if not result['p2p_data']:
        result['issue_summary'].append("P2P analysis results missing")
        result['suggested_fixes'].append("Run P2P analysis for this galaxy")
    
    return result

def check_data_quality(galaxy_name, analysis_path):
    """
    Check the quality of existing analysis data
    """
    quality_report = {
        'file_exists': False,
        'file_size_mb': 0,
        'loadable': False,
        'has_required_keys': False,
        'data_summary': {},
        'issues': [],
        'quality_score': 0
    }
    
    try:
        if os.path.exists(analysis_path):
            quality_report['file_exists'] = True
            
            # Check file size
            file_size = os.path.getsize(analysis_path)
            quality_report['file_size_mb'] = file_size / (1024 * 1024)
            
            if file_size < 1000:  # Less than 1KB
                quality_report['issues'].append("File too small - likely incomplete")
                return quality_report
            
            # Try to load the file
            data = np.load(analysis_path, allow_pickle=True)
            quality_report['loadable'] = True
            
            # Check for required keys
            required_keys = ['distance', 'binning']
            available_keys = list(data.keys())
            quality_report['data_summary']['available_keys'] = available_keys
            
            missing_keys = [key for key in required_keys if key not in available_keys]
            if not missing_keys:
                quality_report['has_required_keys'] = True
            else:
                quality_report['issues'].append(f"Missing required keys: {missing_keys}")
            
            # Check distance information
            if 'distance' in data:
                distance_info = data['distance'].item()
                if 'effective_radius' in distance_info:
                    quality_report['data_summary']['effective_radius'] = distance_info['effective_radius']
                    quality_report['quality_score'] += 25
                else:
                    quality_report['issues'].append("Missing effective radius in distance info")
            
            # Check binning information
            if 'binning' in data:
                binning_info = data['binning'].item()
                quality_report['data_summary']['binning_keys'] = list(binning_info.keys()) if isinstance(binning_info, dict) else "Not a dict"
                quality_report['quality_score'] += 25
            
            # Overall quality assessment
            if quality_report['file_exists']:
                quality_report['quality_score'] += 25
            if quality_report['loadable']:
                quality_report['quality_score'] += 25
            
        else:
            quality_report['issues'].append("File does not exist")
    
    except Exception as e:
        quality_report['issues'].append(f"Error loading file: {str(e)}")
        quality_report['loadable'] = False
    
    return quality_report

def analyze_failed_galaxies():
    """
    Analyze all failed galaxies and provide detailed diagnostics
    """
    logger.info("Starting comprehensive failure analysis")
    
    # Get galaxy list
    analysis_dir = "alpha_fe_analysis_results/analysis_20250720_091707"
    galaxy_dirs = [d for d in os.listdir(analysis_dir) 
                   if os.path.isdir(os.path.join(analysis_dir, d)) and d.startswith('VCC')]
    
    failed_galaxies = []
    success_galaxies = []
    all_diagnostics = []
    
    for galaxy_name in sorted(galaxy_dirs):
        logger.info(f"Analyzing {galaxy_name}")
        
        # Check data availability
        availability = check_galaxy_data_availability(galaxy_name)
        
        # Check data quality for existing files
        quality_reports = {}
        
        if availability['vnb_data']:
            quality_reports['vnb'] = check_data_quality(galaxy_name, availability['vnb_path'])
        
        if availability['rdb_data']:
            quality_reports['rdb'] = check_data_quality(galaxy_name, availability['rdb_path'])
        
        if availability['p2p_data']:
            quality_reports['p2p'] = check_data_quality(galaxy_name, availability['p2p_path'])
        
        # Determine success/failure
        can_do_vnb = availability['alpha_fe_data'] and availability['vnb_data']
        can_do_rdb = availability['alpha_fe_data'] and availability['rdb_data']
        
        diagnostic = {
            'galaxy_name': galaxy_name,
            'availability': availability,
            'quality_reports': quality_reports,
            'can_do_vnb': can_do_vnb,
            'can_do_rdb': can_do_rdb,
            'overall_status': 'success' if (can_do_vnb or can_do_rdb) else 'failed'
        }
        
        all_diagnostics.append(diagnostic)
        
        if diagnostic['overall_status'] == 'failed':
            failed_galaxies.append(galaxy_name)
        else:
            success_galaxies.append(galaxy_name)
    
    return all_diagnostics, failed_galaxies, success_galaxies

def generate_fix_commands(diagnostics):
    """
    Generate specific commands to fix each galaxy
    """
    fix_commands = {
        'vnb_missing': [],
        'rdb_missing': [],
        'p2p_missing': [],
        'alpha_fe_missing': []
    }
    
    for diag in diagnostics:
        galaxy = diag['galaxy_name']
        avail = diag['availability']
        
        if not avail['alpha_fe_data']:
            # Need to run complete alpha/Fe analysis
            fix_commands['alpha_fe_missing'].append(f"# Fix {galaxy} - Alpha/Fe analysis missing")
            fix_commands['alpha_fe_missing'].append(f"python run_complete_physics_analysis.py {galaxy}")
            fix_commands['alpha_fe_missing'].append("")
        
        if not avail['vnb_data']:
            fix_commands['vnb_missing'].append(f"# Fix {galaxy} - VNB analysis missing")
            fix_commands['vnb_missing'].append(f"python main.py {galaxy} VNB")
            fix_commands['vnb_missing'].append("")
        
        if not avail['rdb_data']:
            fix_commands['rdb_missing'].append(f"# Fix {galaxy} - RDB analysis missing")
            fix_commands['rdb_missing'].append(f"python main.py {galaxy} RDB")
            fix_commands['rdb_missing'].append("")
        
        if not avail['p2p_data']:
            fix_commands['p2p_missing'].append(f"# Fix {galaxy} - P2P analysis missing")
            fix_commands['p2p_missing'].append(f"python main.py {galaxy} P2P")
            fix_commands['p2p_missing'].append("")
    
    return fix_commands

def print_comprehensive_report(diagnostics, failed_galaxies, success_galaxies):
    """
    Print a comprehensive failure analysis report
    """
    print("="*80)
    print("COMPREHENSIVE DUAL MODE ANALYSIS FAILURE REPORT")
    print("="*80)
    print(f"Total galaxies: {len(diagnostics)}")
    print(f"Successful galaxies: {len(success_galaxies)} - {success_galaxies}")
    print(f"Failed galaxies: {len(failed_galaxies)} - {failed_galaxies}")
    print()
    
    # Summary statistics
    vnb_available = sum(1 for d in diagnostics if d['availability']['vnb_data'])
    rdb_available = sum(1 for d in diagnostics if d['availability']['rdb_data'])
    p2p_available = sum(1 for d in diagnostics if d['availability']['p2p_data'])
    alpha_fe_available = sum(1 for d in diagnostics if d['availability']['alpha_fe_data'])
    
    print("DATA AVAILABILITY SUMMARY:")
    print(f"Alpha/Fe data: {alpha_fe_available}/{len(diagnostics)} galaxies")
    print(f"VNB data: {vnb_available}/{len(diagnostics)} galaxies")
    print(f"RDB data: {rdb_available}/{len(diagnostics)} galaxies")
    print(f"P2P data: {p2p_available}/{len(diagnostics)} galaxies")
    print()
    
    # Detailed failure analysis
    print("DETAILED FAILURE ANALYSIS:")
    print("-" * 60)
    
    for diag in diagnostics:
        if diag['overall_status'] == 'failed':
            galaxy = diag['galaxy_name']
            avail = diag['availability']
            
            print(f"\n{galaxy}:")
            print(f"  Alpha/Fe data: {'✓' if avail['alpha_fe_data'] else '✗'}")
            print(f"  VNB data: {'✓' if avail['vnb_data'] else '✗'}")
            print(f"  RDB data: {'✓' if avail['rdb_data'] else '✗'}")
            print(f"  P2P data: {'✓' if avail['p2p_data'] else '✗'}")
            
            if avail['issue_summary']:
                print(f"  Issues: {', '.join(avail['issue_summary'])}")
            
            if avail['suggested_fixes']:
                print(f"  Fixes: {', '.join(avail['suggested_fixes'])}")
    
    print("\n" + "="*80)

def save_fix_scripts(fix_commands):
    """
    Save shell scripts to fix the issues
    """
    os.makedirs("fix_scripts", exist_ok=True)
    
    # VNB fixes
    if fix_commands['vnb_missing']:
        with open("fix_scripts/fix_vnb_missing.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Script to run missing VNB analyses\n")
            f.write("# Generated automatically by failure analysis\n\n")
            f.write("\n".join(fix_commands['vnb_missing']))
        logger.info("Created fix_scripts/fix_vnb_missing.sh")
    
    # RDB fixes
    if fix_commands['rdb_missing']:
        with open("fix_scripts/fix_rdb_missing.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Script to run missing RDB analyses\n")
            f.write("# Generated automatically by failure analysis\n\n")
            f.write("\n".join(fix_commands['rdb_missing']))
        logger.info("Created fix_scripts/fix_rdb_missing.sh")
    
    # Alpha/Fe fixes
    if fix_commands['alpha_fe_missing']:
        with open("fix_scripts/fix_alpha_fe_missing.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("# Script to run missing Alpha/Fe analyses\n")
            f.write("# Generated automatically by failure analysis\n\n")
            f.write("\n".join(fix_commands['alpha_fe_missing']))
        logger.info("Created fix_scripts/fix_alpha_fe_missing.sh")

def main():
    """
    Main function for failure analysis
    """
    logger.info("Starting comprehensive failure analysis for dual mode analysis")
    
    # Analyze all galaxies
    diagnostics, failed_galaxies, success_galaxies = analyze_failed_galaxies()
    
    # Generate fix commands
    fix_commands = generate_fix_commands(diagnostics)
    
    # Print comprehensive report
    print_comprehensive_report(diagnostics, failed_galaxies, success_galaxies)
    
    # Save fix scripts
    save_fix_scripts(fix_commands)
    
    # Summary of what needs to be done
    print("\nNEXT STEPS:")
    print("-" * 40)
    
    total_vnb_missing = len([d for d in diagnostics if not d['availability']['vnb_data']])
    total_rdb_missing = len([d for d in diagnostics if not d['availability']['rdb_data']])
    
    if total_vnb_missing > 0:
        print(f"1. Run VNB analysis for {total_vnb_missing} galaxies:")
        print("   bash fix_scripts/fix_vnb_missing.sh")
    
    if total_rdb_missing > 0:
        print(f"2. Run RDB analysis for {total_rdb_missing} galaxies:")
        print("   bash fix_scripts/fix_rdb_missing.sh")
    
    print("3. Re-run dual mode analysis:")
    print("   python alpha_gradient_dual.py")
    
    logger.info("Failure analysis complete!")

if __name__ == "__main__":
    main()
