import yaml
from pathlib import Path


def line_intersection(p1, p2, line_start, line_end):
    """İki nokta arasındaki çizgi, sanal çizgiyi kesiyor mu"""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = line_start
    x4, y4 = line_end
    
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return False
    
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
    
    return 0 <= t <= 1 and 0 <= u <= 1


def get_bbox_bottom_center(bbox):
    """Bboxın alt ortasını döndür (ayak noktası)"""
    x1, y1, x2, y2 = bbox[:4]
    cx = (x1 + x2) / 2
    return (cx, y2)


class LineCounter:
    """Sanal çizgi üzerinden giriş/çıkış sayımı"""
    
    def __init__(self, sequence_name, config_path="configs/counting_lines.yaml"):
        config_path = Path(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        if sequence_name not in config:
            raise ValueError(f"Sequence '{sequence_name}' configs bulunamadı")
        
        self.config = config[sequence_name]
        self.line_config = self.config['line_1']
        
        # Çizgi koordinatları
        coords = self.line_config['coordinates']
        self.line_start = (coords[0], coords[1])
        self.line_end = (coords[2], coords[3])
        
        # Yön bilgisi
        self.entry_direction = self.line_config['direction']['entry']
        self.exit_direction = self.line_config['direction']['exit']
        
        # Track'lerin önceki pozisyonları
        self.track_positions = {}
        
        # Sayaçlar
        self.entry_count = 0
        self.exit_count = 0
        self.crossed_tracks = {}  # track_id: son geçiş yönü
        
    def update(self, tracks):
        """Trackleri kontrol et, çizgi geçişini say
        
        Args:
            tracks: [[x1, y1, x2, y2, track_id, conf], ...]
        """
        current_track_ids = set()
        
        for track in tracks:
            track_id = int(track[4])
            current_track_ids.add(track_id)
            
            # Mevcut pozisyon (alt orta nokta)
            current_pos = get_bbox_bottom_center(track[:4])
            
            if track_id in self.track_positions:
                prev_pos = self.track_positions[track_id]
                
                # Çizgiyi kesiyor mu?
                if line_intersection(prev_pos, current_pos, self.line_start, self.line_end):
                    # Hangi yönde geçti?
                    direction = self._get_crossing_direction(prev_pos, current_pos)
                    
                    # Aynı yönde ard arda geçiş yapmasın
                    last_direction = self.crossed_tracks.get(track_id)
                    
                    if direction != last_direction:
                        # Sayımı yap
                        if direction == self.entry_direction:
                            self.entry_count += 1
                        elif direction == self.exit_direction:
                            self.exit_count += 1
                        
                        self.crossed_tracks[track_id] = direction
            
            # Pozisyonu güncelle
            self.track_positions[track_id] = current_pos
        
        # Kaybolmuş track'leri temizle
        lost_ids = set(self.track_positions.keys()) - current_track_ids
        for tid in lost_ids:
            del self.track_positions[tid]
            if tid in self.crossed_tracks:
                del self.crossed_tracks[tid]
    
    def _get_crossing_direction(self, prev_pos, current_pos):
        """Geçiş yönünü belirle"""
        prev_x = prev_pos[0]
        curr_x = current_pos[0]
        prev_y = prev_pos[1]
        curr_y = current_pos[1]
        
        # X hareketi daha fazlaysa (sağ/sol), Y hareketi daha fazlaysa (yukarı/aşağı)
        dx = abs(curr_x - prev_x)
        dy = abs(curr_y - prev_y)
        
        if dx > dy:
            # Horizontal hareket
            if curr_x > prev_x:
                return "right"  # soldan sağa
            else:
                return "left"   # sağdan sola
        else:
            # Vertical hareket
            if curr_y > prev_y:
                return "down"   # yukarıdan aşağı
            else:
                return "up"     # aşağıdan yukarı
    
    def get_counts(self):
        """Mevcut sayımları döndür"""
        return {
            'entry': self.entry_count,
            'exit': self.exit_count,
            'total_crossings': self.entry_count + self.exit_count,
            'unique_tracks': len(self.crossed_tracks)
        }
    
    def get_line_coords(self):
        """Çizgi koordinatlarını döndür (görselleştirme için)"""
        return self.line_start, self.line_end
    
    def get_line_color(self):
        """Çizgi rengini döndür"""
        return tuple(self.line_config['color'])
    
    def get_line_thickness(self):
        """Çizgi kalınlığını döndür"""
        return self.line_config['thickness']
