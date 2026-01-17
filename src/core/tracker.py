import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import linear_sum_assignment


def iou(box1, box2):
    """IoU hesapla"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


class KalmanFilter:
    """Constant velocity Kalman filter"""
    
    def __init__(self, bbox):
        # State: [cx, cy, w, h, vx, vy, vw, vh]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        # State covariance
        self.P = np.eye(8, dtype=np.float32)
        self.P[4:, 4:] *= 1000  # yüksek uncertainty for velocities
        
        # Process noise
        self.Q = np.eye(8, dtype=np.float32)
        self.Q[4:, 4:] *= 0.01
        
        # Measurement noise
        self.R = np.eye(4, dtype=np.float32) * 10
        
        # State transition matrix (constant velocity)
        self.F = np.eye(8, dtype=np.float32)
        self.F[:4, 4:] = np.eye(4)  # position += velocity
        
        # Measurement matrix
        self.H = np.eye(4, 8, dtype=np.float32)
    
    def predict(self):
        """Bir sonraki state'i tahmin et"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.get_bbox()
    
    def update(self, bbox):
        """Ölçüm ile state'i güncelle"""
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w, h], dtype=np.float32)
        
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
    
    def get_bbox(self):
        """State'ten bbox çıkar"""
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])


class Track:
    def __init__(self, track_id, box, conf):
        self.id = track_id
        self.box = box
        self.conf = conf
        self.age = 1
        self.lost_frames = 0
        self.kf = KalmanFilter(box)
        
    def predict(self):
        """Kalman prediction"""
        self.box = self.kf.predict()
        
    def update(self, box, conf):
        self.kf.update(box)
        self.box = box
        self.conf = conf
        self.age += 1
        self.lost_frames = 0
        
    def mark_lost(self):
        self.lost_frames += 1


class ByteTracker:
    """ByteTrack with Kalman + Hungarian"""
    
    def __init__(self, config_path="configs/tracker.yaml"):
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        self.config = config['tracker']
        self.track_thresh = self.config['track_thresh']
        self.track_buffer = self.config['track_buffer']
        self.match_thresh = self.config['match_thresh']
        self.low_thresh = self.config['low_thresh']
        
        self.tracks = []
        self.next_id = 1
        
    def update(self, detections):
        """Detectionlari tracklerle eşleştir
        
        Args:
            detections: [[x1, y1, x2, y2, conf], ...]
            
        Returns:
            tracked_objects: [[x1, y1, x2, y2, track_id, conf], ...]
        """
        # Kalman prediction
        for track in self.tracks:
            track.predict()
        
        # Yüksek ve düşük confidence detectionlari ayir
        high_dets = [d for d in detections if d[4] >= self.track_thresh]
        low_dets = [d for d in detections if self.low_thresh <= d[4] < self.track_thresh]
        
        # İlk eşleştirme: Hungarian algorithm
        matches, unmatched_tracks, unmatched_dets = self._match(
            self.tracks, high_dets
        )
        
        # Eşleşen trackleri güncelle
        for track_idx, det_idx in matches:
            det = high_dets[det_idx]
            self.tracks[track_idx].update(det[:4], det[4])
        
        # İkinci eşleştirme: düşük confidence
        if len(unmatched_tracks) > 0 and len(low_dets) > 0:
            remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
            matches_low, unmatched_tracks_low, _ = self._match(
                remaining_tracks, low_dets
            )
            
            for i, det_idx in matches_low:
                track_idx = unmatched_tracks[i]
                det = low_dets[det_idx]
                self.tracks[track_idx].update(det[:4], det[4])
                unmatched_tracks.remove(track_idx)
        
        # Eşleşmeyen trackleri lost olarak işaretle
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_lost()
        
        # Yeni trackler oluştur
        for det_idx in unmatched_dets:
            det = high_dets[det_idx]
            new_track = Track(self.next_id, det[:4], det[4])
            self.next_id += 1
            self.tracks.append(new_track)
        
        # Ölü trackleri temizle
        self.tracks = [t for t in self.tracks if t.lost_frames < self.track_buffer]
        
        # Sonuçları döndür
        results = []
        for track in self.tracks:
            if track.lost_frames == 0:
                results.append([
                    track.box[0], track.box[1], track.box[2], track.box[3],
                    track.id, track.conf
                ])
        
        return results
    
    def _match(self, tracks, detections):
        """Hungarian algorithm ile eşleştirme"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Cost matrix (1 - IoU)
        cost_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                cost_matrix[i, j] = 1 - iou(track.box[:4], det[:4])
        
        # Hungarian
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < (1 - self.match_thresh):
                matches.append((i, j))
                unmatched_tracks.remove(i)
                unmatched_dets.remove(j)
        
        return matches, unmatched_tracks, unmatched_dets
