import cv2
from pathlib import Path
import glob


class VideoReader:
    """Video veya frame okuma"""

    def __init__(self, video_path, fps=30):
        self.video_path = Path(video_path)
        self.is_image_sequence = False
        self.current_frame = 0
        
        if self.video_path.is_dir():
            # Frame dizini (MOT17 gibi)
            self.is_image_sequence = True
            self.frame_files = sorted(glob.glob(str(self.video_path / "*.jpg")))
            if not self.frame_files:
                self.frame_files = sorted(glob.glob(str(self.video_path / "*.png")))
            
            if not self.frame_files:
                raise ValueError(f"Frame bulunamadı: {video_path}")
            
            first_frame = cv2.imread(self.frame_files[0])
            self.height, self.width = first_frame.shape[:2]
            self.total_frames = len(self.frame_files)
            self.fps = fps
            self.cap = None
        else:
            # Video dosyası
            self.cap = cv2.VideoCapture(str(video_path))
            if not self.cap.isOpened():
                raise ValueError(f"Video açılamadı: {video_path}")
            
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def read(self):
        if self.is_image_sequence:
            if self.current_frame < len(self.frame_files):
                frame = cv2.imread(self.frame_files[self.current_frame])
                self.current_frame += 1
                return True, frame
            return False, None
        else:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
            return ret, frame
    
    def release(self):
        if self.cap:
            self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.release()
    
    def get_progress(self):
        if self.total_frames > 0:
            return (self.current_frame / self.total_frames) * 100
        return 0


class VideoWriter:
    """Video yazma için basic class"""
    
    def __init__(self, output_path, fps, width, height, codec='mp4v'):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.writer = cv2.VideoWriter(
            str(output_path), 
            cv2.VideoWriter_fourcc(*codec),
            fps, (width, height)
        )
        
        if not self.writer.isOpened():
            raise ValueError(f"Video writer açılamadı: {output_path}")
    
    def write(self, frame):
        self.writer.write(frame)
    
    def release(self):
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.release()
