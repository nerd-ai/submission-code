import json
from pathlib import Path

# Threshold parameters (adjust as needed)
low_thre = 600        # lower bound for bbox area
high_thre = 15000     # upper bound for bbox area
num_thre = 5          # threshold for num_labels

def load_json(file_path):
    """Load a JSON file and return its content as a dict."""
    with open(file_path, 'r') as f:
        return json.load(f)

def calc_bbox_area(bbox):
    """Compute bbox area. bbox format: [x1, y1, x2, y2]."""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height

def calculate_kl_threshold(data2):
    """
    Compute the KL threshold as the 30th percentile (i.e., top 30% lowest KL).
    Assumes data2 has structure: {'per_mask_metrics': {filename: {'KL': value}, ...}}
    """
    kl_values = []
    for _, metrics in data2.get('per_mask_metrics', {}).items():
        if 'KL' in metrics:
            kl_values.append(metrics['KL'])

    if not kl_values:
        raise ValueError("No 'KL' values found in json2 per_mask_metrics.")

    kl_values.sort()  # ascending
    threshold_index = max(0, int(len(kl_values) * 0.3) - 1)
    thre_kl = kl_values[threshold_index]
    print(f"Calculated KL threshold (30th percentile): {thre_kl:.6f}")
    return thre_kl

def process_and_save(
    json_bbox_path,
    json_kl_path,
    output_txt_path="/path/to/filtered_filenames@top30%.txt",
):
    """
    Filter filenames using:
      - KL < KL_30th_percentile
      - low_thre < bbox_area < high_thre
      - num_labels < num_thre
    Then save the selected filenames (as .png) into a TXT file, one per line.
    """
    # Load JSON data
    data_bbox = load_json(json_bbox_path)  # expects 'bbox' and 'num_labels'
    data_kl = load_json(json_kl_path)      # expects 'KL'

    thre_kl = calculate_kl_threshold(data_kl)

    metrics_bbox = data_bbox.get('per_mask_metrics', {})
    metrics_kl = data_kl.get('per_mask_metrics', {})

    selected_files = []

    # Iterate over filenames present in bbox json; match in KL json
    for filename, m1 in metrics_bbox.items():
        m2 = metrics_kl.get(filename)
        if m2 is None:
            # Skip if there is no matching KL entry
            continue

        if 'KL' not in m2:
            continue
        if 'bbox' not in m1 or 'num_labels' not in m1:
            continue

        kl = m2['KL']
        bbox = m1['bbox']  # NOTE: expects key 'bbox' (not 'bbox1')
        num_labels = m1['num_labels']
        bbox_area = calc_bbox_area(bbox)

        # Apply filters (IoU is not used anymore)
        if (kl < thre_kl and
            low_thre < bbox_area < high_thre and
            num_labels < num_thre):
            # Convert .npy to .png for output list
            png_name = filename.replace('.npy', '.png')
            selected_files.append(png_name)

    # Write selected filenames to TXT
    out_path = Path(output_txt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for name in selected_files:
            f.write(f"{name}\n")

    print(f"Total candidates in bbox JSON: {len(metrics_bbox)}")
    print(f"Selected filenames: {len(selected_files)}")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    # Input JSON paths
    json1_path = "/path/to/pseudo_bbox_annotation.json"  # contains bbox/num_labels
    json2_path = "/path/to/kl_results.json"  # contains KL

    if not (Path(json1_path).exists() and Path(json2_path).exists()):
        print("Error: One or both JSON files not found!")
    else:
        process_and_save(json1_path, json2_path)
