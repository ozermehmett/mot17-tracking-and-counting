import numpy as np


def calculate_iou(box1, box2):
    """İki bbox arasında IoU hesapla
    
    Args:
        box1, box2: [x1, y1, x2, y2]
    
    Returns:
        IoU değeri (0-1)
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0


def bbox_area(bbox):
    """Bbox alanını hesapla"""
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def bbox_center(bbox):
    """Bbox merkez noktasını hesapla"""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return (cx, cy)


def get_bbox_bottom_center(bbox):
    """Bboxın alt noktasını döndür ayak noktası"""
    x1, y1, x2, y2 = bbox[:4]
    cx = (x1 + x2) / 2
    return (cx, y2)


def line_intersection(p1, p2, line_start, line_end):
    """İki nokta arasındaki çizgi, sanal çizgiyi kesiyor mu
    
    Args:
        p1, p2: Hareket eden noktanın önceki ve şimdiki pozisyonu
        line_start, line_end: Sanal çizginin başlangıç ve bitiş noktası
    
    Returns:
        True/False - kesişme var mı
    """
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


def euclidean_distance(p1, p2):
    """Euclidean distance"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
