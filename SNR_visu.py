import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from astropy.io import fits
import os
import argparse
from matplotlib.ticker import AutoMinorLocator
import warnings
warnings.filterwarnings("ignore")

def plot_ifu_snr_comparison(file_paths, output_path=None, dpi=150):
    """
    Create a visualization with 3 SNR maps and a SNR vs R/Re scatter plot
    
    Parameters:
    -----------
    file_paths : list
        List of paths to 3 IFU SNR files (pixels SNR, Voronoi bin SNR, Voronoi bin feedback SNR)
    output_path : str, optional
        Path to save the output image
    dpi : int
        Resolution for the output image
    """
    # Check if we have 3 files
    if len(file_paths) != 3:
        raise ValueError("Please provide exactly 3 input files (pixel SNR, Voronoi bin SNR, Voronoi bin feedback SNR)")
    
    # Set up the figure with a 2x3 grid
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1.2])
    
    # Create the subplots - 3 SNR maps on top, SNR vs R/Re plot on bottom
    ax_snr1 = plt.subplot(gs[0, 0])      # Pixel SNR map
    ax_snr2 = plt.subplot(gs[0, 1])      # Voronoi bin SNR map
    ax_snr3 = plt.subplot(gs[0, 2])      # Voronoi bin feedback SNR map
    ax_snr_plot = plt.subplot(gs[1, :])  # SNR vs R/Re plot
    
    # SNR map titles
    snr_titles = ['Pixel SNR', 'Voronoi Bin SNR', 'Voronoi Bin Feedback SNR']
    
    # Colors for different SNR types in the scatter plot
    colors = ['blue', 'green', 'red']
    markers = ['o', 'x', '+']
    labels = ['Pixels SNR', 'Voronoi bin SNR', 'Voronoi Bin feedback SNR']
    
    # Data containers for the scatter plot
    all_radii = []
    all_snr_values = []
    all_types = []
    
    # Process each SNR file
    for i, file_path in enumerate(file_paths):
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
                    print(f"Could not find SNR data in {file_path}")
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
                    print(f"Effective radius not found in header, using estimate: {Re:.2f} arcsec")
                
                # Calculate image dimensions in arcsec
                ny, nx = snr_data.shape
                extent_x = nx * pixel_scale_x
                extent_y = ny * pixel_scale_y
                
                # Create extent for plotting (centered at 0,0)
                extent = [-extent_x/2, extent_x/2, -extent_y/2, extent_y/2]
                
                # Plot the SNR map in the appropriate subplot
                ax = [ax_snr1, ax_snr2, ax_snr3][i]
                
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
                cbar.set_label('S/N/R')
                
                # Set title
                ax.set_title(snr_titles[i])
                
                # Add labels and formatting
                ax.set_xlabel('RA [deg]')
                ax.set_ylabel('DEC [deg]')
                
                # Add ticks and grid
                ax.xaxis.set_minor_locator(AutoMinorLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator(5))
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Extract radial coordinates and SNR values for scatter plot
                y, x = np.indices(snr_data.shape)
                x_center = nx / 2
                y_center = ny / 2
                
                # Calculate radial distance in pixels
                r_pixels = np.sqrt((x - x_center)**2 + (y - y_center)**2)
                
                # Convert to arcsec
                r_arcsec = r_pixels * pixel_scale_x
                
                # Convert to R/Re
                r_scaled = r_arcsec / Re
                
                # Flatten arrays for scatter plot
                r_flat = r_scaled.flatten()
                snr_flat = snr_data.flatten()
                
                # Filter out invalid values and select a reasonable number of points
                # to prevent plot from being too crowded
                valid = np.isfinite(snr_flat) & (snr_flat > 0)
                r_valid = r_flat[valid]
                snr_valid = snr_flat[valid]
                
                # If too many points, sample a subset
                max_points = 5000
                if len(r_valid) > max_points:
                    indices = np.random.choice(len(r_valid), max_points, replace=False)
                    r_valid = r_valid[indices]
                    snr_valid = snr_valid[indices]
                
                # Store data for scatter plot
                all_radii.append(r_valid)
                all_snr_values.append(snr_valid)
                all_types.append(i)
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Create the scatter plot of SNR vs R/Re
    for i in range(len(all_radii)):
        ax_snr_plot.scatter(all_radii[i], all_snr_values[i], 
                           s=5, alpha=0.7, color=colors[i], 
                           marker=markers[i], label=labels[i])
    
    # Add target SNR line
    ax_snr_plot.axhline(y=20, linestyle='--', color='magenta', label='target SNR = 20')
    
    # Set labels and title for scatter plot
    ax_snr_plot.set_xlabel('R/Re [deg]')
    ax_snr_plot.set_ylabel('SNR')
    ax_snr_plot.set_title('SNR vs. Normalized Radius (R/Re)')
    
    # Set y-axis range to match your reference plot
    ax_snr_plot.set_ylim(0, 50)
    
    # Add grid and legend
    ax_snr_plot.grid(True, alpha=0.3, linestyle='--')
    ax_snr_plot.legend(loc='upper right')
    
    # Add minor ticks
    ax_snr_plot.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax_snr_plot.yaxis.set_minor_locator(AutoMinorLocator(5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize IFU SNR data from 3 files')
    parser.add_argument('files', nargs=3, help='Paths to 3 SNR data files (pixel SNR, Voronoi bin SNR, Voronoi bin feedback SNR)')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--dpi', type=int, default=150, help='DPI for output image')
    
    args = parser.parse_args()
    
    plot_ifu_snr_comparison(args.files, args.output, args.dpi)

if __name__ == "__main__":
    main()