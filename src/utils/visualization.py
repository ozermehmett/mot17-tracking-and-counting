import cv2
import numpy as np


def draw_tracks(frame, tracks, show_id=True, show_conf=False):
    """Trackleri çiz"""
    for track in tracks:
        x1, y1, x2, y2, track_id, conf = track
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Bbox
        color = get_color_by_id(int(track_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # ID ve confidence
        label = f'ID:{int(track_id)}'
        if show_conf:
            label += f' {conf:.2f}'
        
        if show_id:
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Ayak noktası
        cx = (x1 + x2) // 2
        cy = y2
        cv2.circle(frame, (cx, cy), 4, color, -1)
    
    return frame


def draw_counting_line(frame, line_start, line_end, color=(0, 255, 0), thickness=3):
    """Sayım çizgisi"""
    cv2.line(frame, 
             (int(line_start[0]), int(line_start[1])),
             (int(line_end[0]), int(line_end[1])),
             color, thickness)
    return frame


def draw_counts(frame, counts, position=(20, 50)):
    """Sayım bilgileri"""
    x, y = position
    
    # Arka plan kutusu
    cv2.rectangle(frame, (x-10, y-35), (x+400, y+20), (0, 0, 0), -1)
    cv2.rectangle(frame, (x-10, y-35), (x+400, y+20), (255, 255, 255), 2)
    
    # Metin
    text = f"Entry: {counts['entry']} | Exit: {counts['exit']} | Total: {counts['total_crossings']}"
    cv2.putText(frame, text, (x, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame


def draw_frame_info(frame, frame_num, total_frames=None, position=(20, 100)):
    """Frame numarası ve bilgileri"""
    x, y = position
    
    if total_frames:
        text = f"Frame: {frame_num}/{total_frames}"
    else:
        text = f"Frame: {frame_num}"
    
    cv2.putText(frame, text, (x, y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def get_color_by_id(track_id):
    """Track ID ye göre yeni renk"""
    np.random.seed(track_id)
    color = tuple(np.random.randint(50, 255, 3).tolist())
    return color


def create_legend(frame, height=80):
    """Görselleştirme için legend ekle"""
    h, w = frame.shape[:2]
    
    # Alt kısımda legend alanı
    legend = np.zeros((height, w, 3), dtype=np.uint8)
    
    # Bilgiler
    cv2.putText(legend, "Tracking + Counting System", (20, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(legend, "Green Line = Counting Line", (20, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(legend, "Colored Box = Tracked Person", (350, 55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
    
    # Frame'e ekle
    combined = np.vstack([frame, legend])
    return combined
