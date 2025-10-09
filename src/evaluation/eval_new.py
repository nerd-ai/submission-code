import numpy as np
if not hasattr(np, "Inf"): np.Inf = np.inf
import cv2
import os
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import argparse
import multiprocessing as mp
from functools import partial
import time

join = os.path.join
basename = os.path.basename

def process_batch_masks(args):
    """Process a batch of mask pairs for metrics calculation."""
    batch_names, gt_path, seg_path, calculate_nsd, nsd_tolerance = args
    
    results = []
    for name in batch_names:
        result = process_single_mask((name, gt_path, seg_path, calculate_nsd, nsd_tolerance))
        results.append(result)
    
    return results

def process_single_mask(args):
    """Process a single mask pair for metrics calculation."""
    name, gt_path, seg_path, calculate_nsd, nsd_tolerance = args
    
    result = {'Name': name, 'DSC': 0.0, 'NSD': 0.0 if calculate_nsd else None}
    
    try:
        # Load masks
        gt_mask_path = join(gt_path, name)
        seg_mask_path = join(seg_path, name)
        
        if not os.path.exists(seg_mask_path):
            # Missing segmentation mask - return zeros
            return result
            
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        seg_mask = cv2.imread(seg_mask_path, cv2.IMREAD_GRAYSCALE)
        
        if gt_mask is None or seg_mask is None:
            return result
            
        # Resize segmentation mask to match ground truth
        seg_mask = cv2.resize(seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Threshold masks
        gt_mask = cv2.threshold(gt_mask, 30, 255, cv2.THRESH_BINARY)[1]
        seg_mask = cv2.threshold(seg_mask, 30, 255, cv2.THRESH_BINARY)[1]
        
        gt_data = np.uint8(gt_mask)
        seg_data = np.uint8(seg_mask)

        gt_labels = np.unique(gt_data)[1:]
        seg_labels = np.unique(seg_data)[1:]
        labels = np.union1d(gt_labels, seg_labels)

        if len(labels) == 0:
            return result

        DSC_arr = []
        NSD_arr = [] if calculate_nsd else None
        
        for i in labels:
            if np.sum(gt_data==i)==0 and np.sum(seg_data==i)==0:
                DSC_i = 1
                NSD_i = 1 if calculate_nsd else None
            elif np.sum(gt_data==i)==0 and np.sum(seg_data==i)>0:
                DSC_i = 0
                NSD_i = 0 if calculate_nsd else None
            else:
                i_gt, i_seg = gt_data == i, seg_data == i
                
                # Compute Dice coefficient
                DSC_i = compute_dice_coefficient(i_gt, i_seg)

                # Compute NSD if requested
                NSD_i = None
                if calculate_nsd:
                    case_spacing = [1, 1, 1]
                    surface_distances = compute_surface_distances(i_gt[..., None], i_seg[..., None], case_spacing)
                    NSD_i = compute_surface_dice_at_tolerance(surface_distances, nsd_tolerance)

            DSC_arr.append(DSC_i)
            if calculate_nsd:
                NSD_arr.append(NSD_i)

        DSC = np.mean(DSC_arr)
        result['DSC'] = round(DSC, 4)
        
        if calculate_nsd:
            NSD = np.mean(NSD_arr)
            result['NSD'] = round(NSD, 4)
            
    except Exception as e:
        print(f"Error processing {name}: {e}")
        
    return result

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Optimized evaluation with batch processing and optional NSD')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth masks')
    parser.add_argument('--seg_path', type=str, required=True, help='Path to segmentation masks')
    parser.add_argument('--calculate_nsd', action='store_true', help='Calculate NSD metric (slower)')
    parser.add_argument('--nsd_tolerance', type=float, default=2.0, help='NSD tolerance in mm (default: 2.0)')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of parallel workers (default: CPU count)')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of masks to process per batch (default: 1)')
    parser.add_argument('--output_csv', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()
    
    gt_path = args.gt_path
    seg_path = args.seg_path
    calculate_nsd = args.calculate_nsd
    nsd_tolerance = args.nsd_tolerance
    num_workers = args.num_workers or mp.cpu_count()
    batch_size = args.batch_size
    
    print(f"Ground truth path: {gt_path}")
    print(f"Segmentation path: {seg_path}")
    print(f"Calculate NSD: {calculate_nsd}")
    if calculate_nsd:
        print(f"NSD tolerance: {nsd_tolerance} mm")
    print(f"Number of workers: {num_workers}")
    print(f"Batch size: {batch_size}")
    
    # Get list of ground truth filenames (reference)
    gt_filenames = os.listdir(gt_path)
    gt_filenames = [x for x in gt_filenames if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')]
    gt_filenames = [x for x in gt_filenames if os.path.exists(join(gt_path, x))]
    gt_filenames.sort()
    
    # Get list of segmentation filenames
    seg_filenames = os.listdir(seg_path)
    seg_filenames = [x for x in seg_filenames if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')]
    seg_filenames = [x for x in seg_filenames if os.path.exists(join(seg_path, x))]
    seg_filenames = set(seg_filenames)
    
    total_gt_masks = len(gt_filenames)
    found_seg_masks = len([f for f in gt_filenames if f in seg_filenames])
    missing_masks = total_gt_masks - found_seg_masks
    
    print(f"\nDataset Statistics:")
    print(f"Total GT masks: {total_gt_masks}")
    print(f"Found segmentation masks: {found_seg_masks}")
    print(f"Missing segmentation masks: {missing_masks}")
    
    if missing_masks > 0:
        print(f"Warning: {missing_masks} segmentation masks are missing!")
        print("Missing masks will be assigned score 0 and normalized in final results.")
    
    # Prepare arguments for batch processing
    if batch_size == 1:
        # Single mask processing (original behavior)
        process_args = [(name, gt_path, seg_path, calculate_nsd, nsd_tolerance) for name in gt_filenames]
        use_batch_processing = False
    else:
        # Batch processing - group filenames into batches
        batches = []
        for i in range(0, len(gt_filenames), batch_size):
            batch = gt_filenames[i:i + batch_size]
            batches.append((batch, gt_path, seg_path, calculate_nsd, nsd_tolerance))
        process_args = batches
        use_batch_processing = True
    
    # Initialize metrics dictionary
    seg_metrics = OrderedDict(
        Name = list(),
        DSC = list(),
    )
    if calculate_nsd:
        seg_metrics['NSD'] = list()
    
    start_time = time.time()
    
    # Process in parallel
    if use_batch_processing:
        print(f"\nProcessing {total_gt_masks} mask pairs in {len(process_args)} batches of size {batch_size}...")
    else:
        print(f"\nProcessing {total_gt_masks} mask pairs...")
        
    if num_workers == 1:
        # Single-threaded processing
        results = []
        if use_batch_processing:
            for args_tuple in tqdm(process_args, desc="Processing batches"):
                batch_results = process_batch_masks(args_tuple)
                results.extend(batch_results)
        else:
            for args_tuple in tqdm(process_args, desc="Processing masks"):
                results.append(process_single_mask(args_tuple))
    else:
        # Multi-threaded processing
        with mp.Pool(processes=num_workers) as pool:
            if use_batch_processing:
                batch_results = list(tqdm(
                    pool.imap(process_batch_masks, process_args),
                    total=len(process_args),
                    desc="Processing batches"
                ))
                # Flatten batch results
                results = []
                for batch in batch_results:
                    results.extend(batch)
            else:
                results = list(tqdm(
                    pool.imap(process_single_mask, process_args),
                    total=len(process_args),
                    desc="Processing masks"
                ))
    
    # Collect results
    for result in results:
        seg_metrics['Name'].append(result['Name'])
        seg_metrics['DSC'].append(result['DSC'])
        if calculate_nsd:
            seg_metrics['NSD'].append(result['NSD'])
    
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Create dataframe
    dataframe = pd.DataFrame(seg_metrics)
    
    # Calculate averages (including all cases, even DSC = 0)
    avg_DSC = dataframe['DSC'].mean()
    avg_NSD = None
    if calculate_nsd:
        avg_NSD = dataframe['NSD'].mean()
    
    # Print results
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS FOR {basename(seg_path)}")
    print("="*60)
    print(f"Dataset: {total_gt_masks} masks")
    print(f"Found: {found_seg_masks} masks")
    print(f"Missing: {missing_masks} masks")
    print(f"Coverage: {found_seg_masks/total_gt_masks*100:.1f}%")
    print()
    print("EVALUATION SCORES (including all cases):")
    print(f"  Average DSC: {avg_DSC:.4f}")
    if calculate_nsd:
        print(f"  Average NSD: {avg_NSD:.4f}")
    print("="*60)
    
    # Save detailed results to CSV if requested
    if args.output_csv:
        # Add summary row
        summary_row = {
            'Name': 'SUMMARY',
            'DSC': avg_DSC,
        }
        if calculate_nsd:
            summary_row['NSD'] = avg_NSD
            
        dataframe = pd.concat([dataframe, pd.DataFrame([summary_row])], ignore_index=True)
        dataframe.to_csv(args.output_csv, index=False)
        print(f"Detailed results saved to: {args.output_csv}")
    
    # Print per-mask details if verbose
    if args.verbose:
        print("\nPer-mask results:")
        for _, row in dataframe.iterrows():
            if row['Name'] != 'SUMMARY':
                nsd_str = f", NSD: {row['NSD']:.4f}" if calculate_nsd else ""
                print(f"  {row['Name']}: DSC: {row['DSC']:.4f}{nsd_str}")

if __name__ == '__main__':
    main()