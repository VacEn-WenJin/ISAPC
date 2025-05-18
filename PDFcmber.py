import os
import glob
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
import argparse

def combine_pdf_files(base_dir, galaxy_name, output_dir=None):
    """
    Combine 4 specific PDF files (RDB_fit0.pdf, RDB_fit1.pdf, RDB_fit2.pdf, RDB_fit3.pdf)
    into a single 2x2 grid image
    
    Parameters:
    -----------
    base_dir : str
        Base directory
    galaxy_name : str
        Name of the galaxy (e.g., 'VCC1588')
    output_dir : str, optional
        Directory where to save the output
        
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    # Define the path to the RDB directory
    rdb_dir = os.path.join(base_dir, "output", galaxy_name, f"{galaxy_name}_stack", "Plots", "RDB")
    
    # Check if directory exists
    if not os.path.exists(rdb_dir):
        print(f"Error: RDB directory not found for {galaxy_name}: {rdb_dir}")
        return False
    
    # Define the specific PDF files to combine
    pdf_files = [
        os.path.join(rdb_dir, "RDB_fit0.pdf"),
        os.path.join(rdb_dir, "RDB_fit1.pdf"),
        os.path.join(rdb_dir, "RDB_fit2.pdf"),
        os.path.join(rdb_dir, "RDB_fit3.pdf")
    ]
    
    # Verify all files exist
    missing_files = [f for f in pdf_files if not os.path.exists(f)]
    if missing_files:
        print(f"Error: Missing PDF files for {galaxy_name}:")
        for f in missing_files:
            print(f"  - {os.path.basename(f)}")
        return False
    
    # Extract first page of each PDF as an image
    images = []
    max_width, max_height = 0, 0
    
    for pdf_path in pdf_files:
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(0)  # First page
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            
            # Track maximum dimensions
            max_width = max(max_width, img.width)
            max_height = max(max_height, img.height)
            
            doc.close()
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            return False
    
    # Create a blank canvas for the 2x2 grid
    grid_img = Image.new('RGB', (max_width * 2, max_height * 2), (255, 255, 255))
    
    # Place each image in the grid
    for i, img in enumerate(images):
        row = i // 2
        col = i % 2
        # Center the image in its cell
        x = col * max_width + (max_width - img.width) // 2
        y = row * max_height + (max_height - img.height) // 2
        grid_img.paste(img, (x, y))
    
    # Define output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{galaxy_name}_combined_plots.png")
    else:
        # Create a 'combined' directory in the base directory
        combined_dir = os.path.join(base_dir, "combined")
        os.makedirs(combined_dir, exist_ok=True)
        output_path = os.path.join(combined_dir, f"{galaxy_name}_combined_plots.png")
    
    # Save the grid image
    grid_img.save(output_path)
    print(f"Saved combined image for {galaxy_name} to {output_path}")
    
    return True

def find_all_galaxies(base_dir):
    """
    Find all galaxy directories in the output directory
    
    Parameters:
    -----------
    base_dir : str
        Base directory
        
    Returns:
    --------
    list
        List of galaxy names found
    """
    output_dir = os.path.join(base_dir, "output")
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return []
    
    # Find all subdirectories in the output directory
    galaxy_dirs = [d for d in os.listdir(output_dir) 
                  if os.path.isdir(os.path.join(output_dir, d))]
    
    # Filter to only include likely galaxy names (starting with VCC)
    galaxy_names = [d for d in galaxy_dirs if d.startswith("VCC")]
    
    return galaxy_names

def process_all_galaxies(base_dir, output_dir=None):
    """
    Process all galaxy directories in the base directory
    
    Parameters:
    -----------
    base_dir : str
        Base directory
    output_dir : str, optional
        Directory where to save all output grids
        
    Returns:
    --------
    tuple
        (success_count, fail_count, galaxy_list)
    """
    # Find all galaxy directories
    galaxy_names = find_all_galaxies(base_dir)
    
    if not galaxy_names:
        print(f"No galaxy directories found in {os.path.join(base_dir, 'output')}")
        return 0, 0, []
    
    success_count = 0
    fail_count = 0
    successful_galaxies = []
    
    for galaxy_name in galaxy_names:
        print(f"Processing {galaxy_name}...")
        
        if combine_pdf_files(base_dir, galaxy_name, output_dir):
            success_count += 1
            successful_galaxies.append(galaxy_name)
        else:
            fail_count += 1
    
    return success_count, fail_count, successful_galaxies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine RDB fit PDF files into grid images")
    parser.add_argument("--base_dir", default=".", help="Base directory containing the 'output' folder")
    parser.add_argument("--galaxy", help="Process only this specific galaxy")
    parser.add_argument("--output_dir", help="Directory to save all combined images")
    args = parser.parse_args()
    
    if args.galaxy:
        # Process a specific galaxy
        if combine_pdf_files(args.base_dir, args.galaxy, args.output_dir):
            print(f"Successfully processed {args.galaxy}")
        else:
            print(f"Failed to process {args.galaxy}")
    else:
        # Process all galaxies
        success, failed, galaxies = process_all_galaxies(args.base_dir, args.output_dir)
        print(f"Processed {success + failed} galaxies: {success} successful, {failed} failed")
        if success > 0:
            print("Successfully processed galaxies:")
            for galaxy in galaxies:
                print(f" - {galaxy}")