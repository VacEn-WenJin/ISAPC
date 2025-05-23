import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.skyview import SkyView
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
import matplotlib.patches as patches
from astropy.nddata import Cutout2D
from astropy.io import fits
from reproject import reproject_interp
import os
from tqdm import tqdm

# Create output directory
output_dir = "virgo_cluster"
os.makedirs(output_dir, exist_ok=True)

# Define the galaxies from the table
# Format: Name, RA (in hours, min, sec), Dec (in deg, min, sec), Type, r (in arcmin)
galaxies_data = [
    ("M60", "12h09m42s", "+14d11m13s", "E2", 9.97),
    ("VCC 1588", "12h19m34s", "+14d52m50s", "Sd", 12.35),
    ("VCC 1890", "12h26m56s", "+12d51m10s", "dE", 14.66),
    ("VCC 1368", "12h17m01s", "+14d19m29s", "SBa", 12.46),
    ("VCC 1902", "12h26m07s", "+12d57m12s", "SBa", 12.62),
    ("VCC 1523", "12h19m24s", "+13d38m07s", "dE(N)", 16.75),
    ("VCC 1949", "12h27m56s", "+12d17m51s", "dS0(N)", 13.59),
    ("VCC 990", "12h13m10s", "+16d07m26s", "dS0(N)", 13.92),
    ("VCC 1410", "12h17m53s", "+11d49m00s", "Sd", 13.92),
    ("VCC 1695", "12h23m01s", "+12d43m32s", "dE", 13.72),
    ("VCC 667", "12h09m43s", "+11d13m58s", "Sd", 13.55),
    ("VCC 1549", "12h20m25s", "+13d04m33s", "dE(N)", 14.40),
    ("VCC 308", "12h04m34s", "+12d58m54s", "dE", 13.54),
    ("VCC 1431", "12h18m15s", "+11d07m23s", "dE", 13.59),
    ("VCC 1811", "12h24m46s", "+13d10m11s", "Sc", 12.55),
    ("VCC 688", "12h10m03s", "+11d22m43s", "Sc", 13.54),
    ("VCC 1146", "12h15m22s", "+12d03m12s", "E", 12.18),
    ("VCC 1049", "12h14m11s", "+13d29m30s", "dE(N)", 14.36),
    ("VCC 1193", "12h15m53s", "+14d13m34s", "Sd", 13.81),
    ("VCC 1910", "12h26m12s", "+12d24m05s", "dE(N)", 13.55),
    ("VCC 1486", "12h19m54s", "+13d44m04s", "Sc", 15.16)
]

# Convert galaxy data to SkyCoord objects
galaxies = []
for name, ra_str, dec_str, gal_type, radius in galaxies_data:
    coords = SkyCoord(ra_str, dec_str, frame='icrs')
    galaxies.append({
        'name': name,
        'coords': coords,
        'type': gal_type,
        'radius': radius  # in arcmin
    })

# Center coordinates of the Virgo Cluster
# M87 (Virgo A) is approximately at the center of the cluster
center_ra = 187.7059  # in degrees (12h 30m 49.4s)
center_dec = 12.3911  # in degrees (+12d 23m 28s)
center = SkyCoord(ra=center_ra*u.degree, dec=center_dec*u.degree, frame='icrs')

# Size of the field we want to capture
# The Virgo Cluster spans about 8 degrees
field_width = 8.0 * u.degree
field_height = 8.0 * u.degree

# Size of each subfield (to avoid exceeding SkyView's limits)
subfield_size = 1.0 * u.degree

# Calculate number of subfields needed
n_width = int(np.ceil(field_width / subfield_size))
n_height = int(np.ceil(field_height / subfield_size))

# Calculate the starting coordinates for the grid
start_ra = center_ra - (field_width/2).to(u.degree).value
start_dec = center_dec - (field_height/2).to(u.degree).value

print(f"Starting download of {n_width}x{n_height} grid of subfields...")
print(f"This will cover a {field_width} x {field_height} field centered at RA={center_ra}, Dec={center_dec}")

# Parameters for the final mosaic
mosaic_size = 4000  # pixels
subfield_pixels = int(mosaic_size / max(n_width, n_height))

# Lists to store the subfields
subfields_g = []
subfields_r = []
subfields_i = []
coords_list = []

# Download each subfield
for i in tqdm(range(n_width)):
    for j in range(n_height):
        # Calculate the center of this subfield
        subfield_ra = start_ra + (i + 0.5) * subfield_size.to(u.degree).value
        subfield_dec = start_dec + (j + 0.5) * subfield_size.to(u.degree).value
        
        # Create a SkyCoord for this position
        position = SkyCoord(ra=subfield_ra*u.degree, dec=subfield_dec*u.degree, frame='icrs')
        
        try:
            # Download the g, r, i band images
            images = SkyView.get_images(position=position, 
                                       survey=['SDSSg', 'SDSSr', 'SDSSi'],
                                       pixels=subfield_pixels,
                                       width=subfield_size, 
                                       height=subfield_size)
            
            if images and len(images) == 3:
                # Store the image data and coordinates
                subfields_g.append(images[0][0])
                subfields_r.append(images[1][0])
                subfields_i.append(images[2][0])
                coords_list.append((subfield_ra, subfield_dec))
                
                # Save individual subfield as a color image
                g_data = images[0][0].data
                r_data = images[1][0].data
                i_data = images[2][0].data
                
                # Create a color composite
                # Normalize the data with asinh scaling (Lupton method)
                rgb_image = make_lupton_rgb(i_data, r_data, g_data, 
                                        stretch=10, Q=10, minimum=0)
                
                # Save this subfield
                plt.figure(figsize=(8, 8))
                plt.imshow(rgb_image, origin='lower')
                plt.title(f"Subfield RA={subfield_ra:.3f}, Dec={subfield_dec:.3f}")
                plt.axis('off')
                plt.tight_layout()
                subfield_filename = os.path.join(output_dir, f"virgo_subfield_ra{subfield_ra:.3f}_dec{subfield_dec:.3f}.png")
                plt.savefig(subfield_filename, dpi=100, bbox_inches='tight')
                plt.close()
        
        except Exception as e:
            print(f"Error downloading subfield at RA={subfield_ra:.3f}, Dec={subfield_dec:.3f}: {e}")

print(f"Downloaded {len(subfields_g)} subfields successfully.")

# Now stitch the subfields together into a mosaic
# We'll use a simple approach that doesn't require reprojection for this example
if len(subfields_g) > 0:
    print("Creating mosaic...")
    
    # Create a reference WCS for the full mosaic
    target_wcs = WCS(naxis=2)
    target_wcs.wcs.crpix = [mosaic_size/2, mosaic_size/2]
    target_wcs.wcs.cdelt = [-field_width.to(u.degree).value/mosaic_size, 
                          field_height.to(u.degree).value/mosaic_size]
    target_wcs.wcs.crval = [center_ra, center_dec]
    target_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    
    # Create empty arrays for the mosaic
    mosaic_g = np.zeros((mosaic_size, mosaic_size))
    mosaic_r = np.zeros((mosaic_size, mosaic_size))
    mosaic_i = np.zeros((mosaic_size, mosaic_size))
    
    # Weight map to track overlapping regions
    weight_map = np.zeros((mosaic_size, mosaic_size))
    
    # Reproject and add each subfield to the mosaic
    for idx, (g_img, r_img, i_img) in enumerate(zip(subfields_g, subfields_r, subfields_i)):
        try:
            # Reproject each band to the target WCS
            array_g, footprint_g = reproject_interp(g_img, target_wcs, shape_out=(mosaic_size, mosaic_size))
            array_r, footprint_r = reproject_interp(r_img, target_wcs, shape_out=(mosaic_size, mosaic_size))
            array_i, footprint_i = reproject_interp(i_img, target_wcs, shape_out=(mosaic_size, mosaic_size))
            
            # Create a combined footprint
            footprint = footprint_g & footprint_r & footprint_i
            
            # Add to the mosaic with weighting
            mosaic_g += array_g * footprint
            mosaic_r += array_r * footprint
            mosaic_i += array_i * footprint
            weight_map += footprint
            
            print(f"Added subfield {idx+1}/{len(subfields_g)} to mosaic")
        
        except Exception as e:
            print(f"Error adding subfield {idx+1} to mosaic: {e}")
    
    # Normalize by the weight map (avoiding division by zero)
    weight_map = np.maximum(weight_map, 1)
    mosaic_g /= weight_map
    mosaic_r /= weight_map
    mosaic_i /= weight_map
    
    # Create a color composite of the final mosaic
    # Replace NaNs with zeros
    mosaic_g = np.nan_to_num(mosaic_g)
    mosaic_r = np.nan_to_num(mosaic_r)
    mosaic_i = np.nan_to_num(mosaic_i)
    
    # Create RGB image with Lupton method
    rgb_mosaic = make_lupton_rgb(mosaic_i, mosaic_r, mosaic_g, 
                               stretch=5, Q=8, minimum=0)
    
    # Save the final mosaic
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_mosaic, origin='lower')
    plt.title("Virgo Cluster Mosaic")
    plt.axis('off')
    plt.tight_layout()
    mosaic_filename = os.path.join(output_dir, "virgo_cluster_mosaic.png")
    plt.savefig(mosaic_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mosaic saved to {mosaic_filename}")
    
    # Also save a version with galaxy information
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_mosaic, origin='lower')
    
    # Define a function to convert sky coordinates to pixel coordinates in the mosaic
    def skycoord_to_pixel(coord):
        x, y = target_wcs.world_to_pixel(coord.ra, coord.dec)
        return x, y
    
    # Add galaxies to the image
    for galaxy in galaxies:
        x, y = skycoord_to_pixel(galaxy['coords'])
        
        # Only annotate galaxies that fall within the image
        if 0 <= x < mosaic_size and 0 <= y < mosaic_size:
            # Convert radius in arcmin to pixel size
            # 1 arcmin is 1/60 degrees
            radius_deg = galaxy['radius'] / 60.0  # convert to degrees
            # Calculate how many pixels this corresponds to in our image
            pix_per_deg = mosaic_size / field_width.to(u.degree).value
            radius_pixels = radius_deg * pix_per_deg
            
            # Draw a circle around the galaxy with radius proportional to r
            circle = patches.Circle((x, y), radius_pixels*0.5, fill=False, 
                                   edgecolor='yellow', linewidth=1.5, alpha=0.7)
            plt.gca().add_patch(circle)
            
            # Add galaxy name and type
            plt.text(x, y + radius_pixels*0.6, f"{galaxy['name']}", 
                    color='white', fontsize=12, ha='center', va='bottom',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2))
            
            plt.text(x, y - radius_pixels*0.6, f"Type: {galaxy['type']}, r: {galaxy['radius']}'", 
                    color='white', fontsize=10, ha='center', va='top',
                    bbox=dict(facecolor='black', alpha=0.7, pad=2))
    
    # Add grid lines
    overlay = plt.gca().get_coords_overlay('fk5')
    overlay.grid(color='white', ls='dotted')
    
    # Set up the labels
    overlay[0].set_axislabel('Right Ascension (J2000)')
    overlay[1].set_axislabel('Declination (J2000)')
    
    plt.title("Virgo Cluster Galaxies with Type and Radius Information")
    plt.tight_layout()
    annotated_filename = os.path.join(output_dir, "virgo_cluster_annotated.png")
    plt.savefig(annotated_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Annotated mosaic saved to {annotated_filename}")

print("Processing complete!")