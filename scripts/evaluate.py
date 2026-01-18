"""
Tracking ve counting değerlendirme
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.geometry import calculate_iou


def parse_file(path, is_gt=False):
    """MOT format dosya okuma"""
    data = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame_id = int(parts[0])
            
            if is_gt:
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                if conf > 0:  # sadece visible
                    data[frame_id].append({'id': track_id, 'bbox': [x, y, x+w, y+h]})
            else:
                track_id = int(parts[1])
                x, y, w, h = map(float, parts[2:6])
                conf = float(parts[6])
                data[frame_id].append({'id': track_id, 'bbox': [x, y, x+w, y+h], 'conf': conf})
    
    return data


def eval_detection(gt_data, det_data, iou_thresh=0.5):
    """Detection metrikleri"""
    tp = fp = fn = 0
    
    for frame_id in gt_data.keys():
        gt_boxes = [obj['bbox'] for obj in gt_data.get(frame_id, [])]
        det_boxes = [obj['bbox'] for obj in det_data.get(frame_id, [])]
        
        matched_gt = set()
        
        for det_box in det_boxes:
            best_iou = max([calculate_iou(det_box, gt_box) for gt_box in gt_boxes] + [0])
            
            if best_iou >= iou_thresh:
                tp += 1
                for j, gt_box in enumerate(gt_boxes):
                    if j not in matched_gt and calculate_iou(det_box, gt_box) == best_iou:
                        matched_gt.add(j)
                        break
            else:
                fp += 1
        
        fn += len(gt_boxes) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {'precision': precision, 'recall': recall, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn}


def eval_tracking(gt_data, track_data, iou_thresh=0.5):
    """Tracking quality metrikleri"""
    id_switches = fragmentations = 0
    gt_to_pred = defaultdict(list)
    last_seen = {}
    
    for frame_id in sorted(gt_data.keys()):
        gt_objs = gt_data.get(frame_id, [])
        pred_objs = track_data.get(frame_id, [])
        
        matched = set()
        
        for gt_obj in gt_objs:
            best_iou = 0
            best_pred = None
            
            for j, pred_obj in enumerate(pred_objs):
                if j not in matched:
                    iou = calculate_iou(gt_obj['bbox'], pred_obj['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_pred = (j, pred_obj['id'])
            
            if best_iou >= iou_thresh and best_pred:
                gt_id = gt_obj['id']
                pred_id = best_pred[1]
                matched.add(best_pred[0])
                
                if gt_id in gt_to_pred and gt_to_pred[gt_id][-1] != pred_id:
                    id_switches += 1
                
                if gt_id in last_seen and frame_id - last_seen[gt_id] > 1:
                    fragmentations += 1
                
                gt_to_pred[gt_id].append(pred_id)
                last_seen[gt_id] = frame_id
    
    return {'id_switches': id_switches, 'fragmentations': fragmentations}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    args = parser.parse_args()
    
    seq = args.sequence
    gt_path = f'data/MOT17/train/{seq}-SDP/gt/gt.txt'
    det_path = f'data/MOT17/train/{seq}-SDP/det/det.txt'
    track_path = f'outputs/{seq}/tracking.txt'
    results_path = f'outputs/{seq}/results.json'
    
    if not os.path.exists(track_path) or not os.path.exists(results_path):
        print("Önce run.py çalıştır")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    gt_data = parse_file(gt_path, is_gt=True) if os.path.exists(gt_path) else None
    det_data = parse_file(det_path, is_gt=True) if os.path.exists(det_path) else None
    track_data = parse_file(track_path)
    
    print(f"\n{'='*60}")
    print(f"EVALUATION: {seq}")
    print(f"{'='*60}\n")
    
    # Detection
    if gt_data and det_data:
        det_metrics = eval_detection(gt_data, det_data)
        print("Detection:")
        print(f"Precision: {det_metrics['precision']:.1%}")
        print(f"Recall:    {det_metrics['recall']:.1%}")
        print(f"F1-Score:  {det_metrics['f1']:.3f}")
    else:
        det_metrics = None
        print("Detection: GT bulunamadı")
    
    # Tracking
    if gt_data:
        track_metrics = eval_tracking(gt_data, track_data)
        print(f"\nTracking:")
        print(f"  ID Switches:    {track_metrics['id_switches']}")
        print(f"  Fragmentations: {track_metrics['fragmentations']}")
    else:
        track_metrics = None
        print("\nTracking: GT bulunamadı")
    
    # Counting
    counts = results['counts']
    print(f"\nCounting:")
    print(f"  Entry:   {counts['entry']}")
    print(f"  Exit:    {counts['exit']}")
    print(f"  Total:   {counts['total_crossings']}")
    print(f"  Tracks:  {counts['unique_tracks']}")
    
    # Save
    eval_results = {
        'sequence': seq,
        'detection': det_metrics,
        'tracking': track_metrics,
        'counting': counts
    }
    
    eval_path = f'outputs/{seq}/evaluation.json'
    with open(eval_path, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Saved: {eval_path}\n")


if __name__ == '__main__':
    main()
