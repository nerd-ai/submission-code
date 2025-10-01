import json
from collections import defaultdict
from pathlib import Path

def process_and_save_final(det_json_path, val_json_path, final_output_path):
    """
    Read detection results and validation (COCO-style) JSONs, select the
    highest-score prediction per image, and directly save the final list in
    the desired format (no intermediate file).
    
    Final format (list of dicts):
    [
        {
            "file_name": <str>,
            "image_id": <int>,
            "category_id": 1,
            "bbox": [x, y, width, height],  # integers
            "score": <float>
        },
        ...
    ]
    """
    # Load detection results (a list of predictions)
    with open(det_json_path, "r") as f:
        predictions = json.load(f)

    # Load validation JSON to build id -> file_name map
    with open(val_json_path, "r") as f:
        val_data = json.load(f)

    # Build mapping from image_id to file_name
    id_to_filename = {img["id"]: img["file_name"] for img in val_data["images"]}

    # Group predictions by image_id
    image_predictions = defaultdict(list)
    for pred in predictions:
        # Expect keys: image_id, bbox=[x,y,w,h], score, category_id
        image_predictions[pred["image_id"]].append(pred)

    # For each image, keep the best (highest-score) detection only
    formatted_results = []
    for img_id, preds in image_predictions.items():
        if not preds:
            continue

        best_pred = max(preds, key=lambda x: x["score"])
        bbox = best_pred["bbox"]  # [x, y, width, height]

        # Some datasets store IDs as strings; ensure integer output
        try:
            image_id_int = int(img_id)
        except (TypeError, ValueError):
            image_id_int = img_id  # fallback if already int-like or not convertible

        formatted_results.append({
            "file_name": id_to_filename[img_id],
            "image_id": image_id_int,
            "category_id": 1,  # fixed category id (adjust if needed)
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "score": float(best_pred["score"])
        })

    # Save the final formatted results (no intermediate file)
    out_path = Path(final_output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(formatted_results, f, indent=4)

    print(f"Saved formatted results to {out_path}")
    print(f"Total images with predictions: {len(formatted_results)}")

if __name__ == "__main__":
    # Example paths (adjust as needed)
    det_json_path = "/path/to/test/coco_instances_results.json"
    val_json_path = "/path/to/instance_test.json"
    final_output_path = "inference_for_medsam.json"

    if not (Path(det_json_path).exists() and Path(val_json_path).exists()):
        print("Error: One or both JSON files not found!")
    else:
        process_and_save_final(det_json_path, val_json_path, final_output_path)
