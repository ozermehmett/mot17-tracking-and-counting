import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import yaml

from src.utils.video_io import VideoReader, VideoWriter
from src.core.detector import PersonDetector
from src.core.tracker import ByteTracker
from src.core.counter import LineCounter
from src.utils.visualization import draw_tracks, draw_counting_line, draw_counts


def main():
    parser = argparse.ArgumentParser(description='MOT17 tracking ve counting pipeline')
    parser.add_argument('--sequence', type=str, required=True,
                        help='Sequence adı (örn: MOT17-09, MOT17-02)')
    args = parser.parse_args()
    
    # paths
    sequence_name = args.sequence
    input_dir = f'data/MOT17/train/{sequence_name}-SDP/img1/'
    output_dir = f'outputs/{sequence_name}'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Sequence: {sequence_name}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    
    # counting line config
    with open('configs/counting_lines.yaml', 'r', encoding='utf-8') as f:
        lines_config = yaml.safe_load(f)
    line_coords = lines_config[sequence_name]['line_1']['coordinates']
    line_start = (line_coords[0], line_coords[1])
    line_end = (line_coords[2], line_coords[3])
    
    # pipeline
    reader = VideoReader(input_dir)
    detector = PersonDetector()
    tracker = ByteTracker()
    counter = LineCounter(sequence_name)
    
    # video writer
    writer = VideoWriter(
        os.path.join(output_dir, 'output.mp4'),
        fps=reader.fps,
        width=reader.width,
        height=reader.height
    )
    
    print(f"Frame: {reader.total_frames}")
    
    # main loop
    frame_idx = 0
    for frame_idx in tqdm(range(reader.total_frames), desc="Processing"):
        ret, frame = reader.read()
        if not ret:
            break
        
        # detection
        detections = detector.detect(frame)
        
        # tracking
        tracks = tracker.update(detections)
        
        # counting
        counter.update(tracks)
        
        # visualization
        frame_vis = frame.copy()
        draw_tracks(frame_vis, tracks)
        draw_counting_line(frame_vis, line_start, line_end)
        counts = counter.get_counts()
        draw_counts(frame_vis, counts)
        
        # save
        writer.write(frame_vis)
    
    reader.release()
    writer.release()
    
    # Save results
    final_counts = counter.get_counts()
    results = {
        'sequence': sequence_name,
        'total_frames': frame_idx + 1,
        'counts': {
            'entry': final_counts['entry'],
            'exit': final_counts['exit'],
            'total_crossings': final_counts['total_crossings'],
            'unique_tracks': final_counts['unique_tracks']
        }
    }
    
    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*50)
    print("Results:")
    print("="*50)
    print(f"Entry: {final_counts['entry']}")
    print(f"Exit: {final_counts['exit']}")
    print(f"Total crossings: {final_counts['total_crossings']}")
    print(f"Unique tracks: {final_counts['unique_tracks']}")
    print("="*50)
    print(f"\nVideo saved: {os.path.join(output_dir, 'output.mp4')}")
    print(f"Results saved: {results_path}")


if __name__ == '__main__':
    main()
