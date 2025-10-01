import os
import numpy as np
from PIL import Image
import json
from skimage.measure import label

def get_bbox(mask):
    """Get the smallest bounding box that covers all non-zero regions in the mask"""
    nonzero = np.nonzero(mask)
    if len(nonzero[0]) == 0:
        return [0, 0, 0, 0]
    
    y_min = int(np.min(nonzero[0]))
    y_max = int(np.max(nonzero[0]))
    x_min = int(np.min(nonzero[1]))
    x_max = int(np.max(nonzero[1]))
    
    return [x_min, y_min, x_max, y_max]

def process_folders(png_folder, npy_folder, output_json, npy_threshold=0.5):
    # Collect file lists
    png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])
    npy_files = sorted([f for f in os.listdir(npy_folder) if f.endswith('.npy')])
    
    # Ensure the number of files matches
    assert len(png_files) == len(npy_files), "Number of files does not match"
    
    results = {
        "per_mask_metrics": {}
    }
    
    for png_file, npy_file in zip(png_files, npy_files):
        # Ensure file names correspond (without extension)
        assert os.path.splitext(png_file)[0] == os.path.splitext(npy_file)[0], f"File name mismatch: {png_file} vs {npy_file}"
        
        # Load and process the NPY file (prediction mask only)
        npy_path = os.path.join(npy_folder, npy_file)
        npy_prob = np.load(npy_path)
        npy_mask = (npy_prob > npy_threshold).astype(np.uint8)
        
        # Get bounding box and number of connected components
        bbox = get_bbox(npy_mask)
        num_labels = int(np.max(label(npy_mask)))
        
        # Store results (IoU and Dice removed)
        results["per_mask_metrics"][npy_file] = {
            "num_labels": num_labels,
            "bbox": bbox
        }
    
    # Save results to JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

# Example usage
if __name__ == "__main__":
    png_folder = "/path/to/the_original_image/"    # Path to original PNG images
    npy_folder = "/path/to/the_confidence_map_npy/"  # Path to NPY confidence maps
    output_json = "pseudo_bbox_train.json"
    
    results = process_folders(
        png_folder=png_folder,
        npy_folder=npy_folder,
        output_json=output_json,
        npy_threshold=0.5
    )
    print(f"Results saved to {output_json}")
