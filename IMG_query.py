import os
import numpy as np
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# VCC IDs and their corresponding coordinates from your image
galaxies = {
    'VCC0308': 'J121850.9+075143',
    # 'VCC0667': 'J122348.49+071107.1',
    # 'VCC1049': 'J122754.83+080525.6',
    # 'VCC1146': 'J122857.53+131430.8',
    # 'VCC1193': 'J122930.56+074147.1',
    # 'VCC1368': 'J123132.53+113729.1',
    # 'VCC1410': 'J123203.35+164115.7',
    # 'VCC1431': 'J123223.35+111547.5',
    # 'VCC1486': 'J123310.05+112050.4',
    # 'VCC1549': 'J123414.81+110417.4',
    # 'VCC1588': 'J123450.87+153306.1',
    # 'VCC1695': 'J123654.85+123112.4',
    # 'VCC1811': 'J123951.91+151751.9',
    # 'VCC1890': 'J124146.04+112919.1',
    # 'VCC1902': 'J124159.34+125634.2',
    # 'VCC1910': 'J124208.66+114515.4',
    # 'VCC1949': 'J124257.77+121713',
    # 'VCC0688': 'J122400.25+074706.6',
    # 'VCC0990': 'J122716.93+160128.1'
}

# Create output directory
output_dir = "vcc_galaxies"
os.makedirs(output_dir, exist_ok=True)

# Also create a directory for the grid
grid_dir = os.path.join(output_dir, "grid")
os.makedirs(grid_dir, exist_ok=True)

# Function to convert J-format coordinates to SkyCoord
def parse_j_coords(j_coords):
    # Extract RA and Dec from J-format string
    coords = j_coords.replace('J', '')
    ra_str, dec_str = coords.split('+') if '+' in coords else coords.split('-')
    
    # Format for hours:minutes:seconds
    ra_h = ra_str[:2]
    ra_m = ra_str[2:4]
    ra_s = ra_str[4:]
    
    # Format for degrees:minutes:seconds
    dec_sign = '+' if '+' in coords else '-'
    dec_d = dec_sign + dec_str[:2]
    dec_m = dec_str[2:4]
    dec_s = dec_str[4:] if len(dec_str) > 4 else '00'
    
    ra_hms = f"{ra_h}h{ra_m}m{ra_s}s"
    dec_dms = f"{dec_d}d{dec_m}m{dec_s}s"
    
    return SkyCoord(ra_hms, dec_dms, frame='icrs')

# Function to get decimal coordinates from J-format
def get_decimal_coords(j_coords):
    sky_coord = parse_j_coords(j_coords)
    return sky_coord.ra.degree, sky_coord.dec.degree

# List to store successful images for grid
successful_images = []

# Download each galaxy image
for vcc_id, j_coords in galaxies.items():
    print(f"Processing {vcc_id}...")
    try:
        # Parse the J-format coordinates
        position = parse_j_coords(j_coords)
        ra_deg, dec_deg = position.ra.degree, position.dec.degree
        
        # Get image from SDSS with 2'×2' field of view
        images = SkyView.get_images(position=position, 
                                   survey=['SDSSg', 'SDSSr', 'SDSSi'], 
                                   pixels=500,  
                                   width=2*u.arcmin, 
                                   height=2*u.arcmin)
        
        if images and len(images) == 3:  # We requested 3 filters
            # Create RGB image
            g_data = images[0][0].data
            r_data = images[1][0].data
            i_data = images[2][0].data
            
            # Normalize the data
            g_norm = np.clip((g_data - np.percentile(g_data, 10)) / (np.percentile(g_data, 99) - np.percentile(g_data, 10)), 0, 1)
            r_norm = np.clip((r_data - np.percentile(r_data, 10)) / (np.percentile(r_data, 99) - np.percentile(r_data, 10)), 0, 1)
            i_norm = np.clip((i_data - np.percentile(i_data, 10)) / (np.percentile(i_data, 99) - np.percentile(i_data, 10)), 0, 1)
            
            # Create RGB image (using g for blue, r for green, i for red - approximating true color)
            rgb = np.dstack([i_norm, r_norm, g_norm])
            
            # Plot with galaxy info overlay
            fig, ax = plt.figure(figsize=(5, 5)), plt.gca()
            ax.imshow(rgb, origin='lower')
            
            # Add galaxy ID and coordinates as text
            plt.text(10, 20, f"{vcc_id}", color='white', fontsize=12, fontweight='bold')
            plt.text(10, 40, f"{j_coords}", color='white', fontsize=10)
            
            # Add a small marker on the central coordinates
            ax.plot(rgb.shape[1]//2, rgb.shape[0]//2, 'gs', ms=8, alpha=0.7)
            plt.text(rgb.shape[1]//2 + 10, rgb.shape[0]//2, f"[{ra_deg:.5f}, {dec_deg:.5f}]", 
                    color='yellow', fontsize=8, fontweight='bold')
            
            # Remove axis ticks and labels
            plt.axis('off')
            plt.tight_layout(pad=0)
            
            # Save as individual image
            filename = os.path.join(output_dir, f"{vcc_id}.png")
            plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=100)
            plt.close()
            print(f"✓ Saved image for {vcc_id}")
            
            # Store for grid
            successful_images.append({
                'vcc_id': vcc_id,
                'j_coords': j_coords,
                'rgb': rgb,
                'ra': ra_deg,
                'dec': dec_deg
            })
        else:
            print(f"× No complete image data returned for {vcc_id}")
    except Exception as e:
        print(f"× Error processing {vcc_id}: {e}")

print(f"\nDownloaded {len(successful_images)} images to '{output_dir}' directory")

# Create a grid of images like your original
if successful_images:
    # Determine grid size based on number of images
    n_images = len(successful_images)
    cols = 5  # 5 columns like your original
    rows = (n_images + cols - 1) // cols  # Ceiling division
    
    plt.figure(figsize=(15, 3*rows))
    
    for i, img_data in enumerate(successful_images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(img_data['rgb'], origin='lower')
        
        # Add galaxy ID and coordinates as text
        plt.text(10, 20, f"{img_data['vcc_id']}", color='white', fontsize=8, fontweight='bold')
        plt.text(10, 40, f"{img_data['j_coords']}", color='white', fontsize=6)
        
        # Remove axis ticks
        plt.axis('off')
    
    plt.tight_layout()
    grid_filename = os.path.join(grid_dir, "vcc_galaxy_grid.png")
    plt.savefig(grid_filename, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"✓ Created galaxy grid image: {grid_filename}")

print("Processing complete!")