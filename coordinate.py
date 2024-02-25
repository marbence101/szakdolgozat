import numpy as np

class CoordinateConverter:
    @staticmethod
    def to_center_coordinates(x1, y1, w, h):
        cx = x1 + w / 2
        cy = y1 + h / 2
        return cx, cy, w, h
    
    @staticmethod
    def to_corner_coordinates(cx, cy, w, h):
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        return x1, y1, w, h
    
    @staticmethod
    def to_center_coordinates_batch(batch):
        """
        Convert corner coordinates to center coordinates in a batch.
        Input shape: N x 4 (x1, y1, w, h)
        Output shape: N x 4 (cx, cy, w, h)
        """
        x1, y1, w, h = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]
        cx = x1 + w / 2
        cy = y1 + h / 2
        return np.stack([cx, cy, w, h], axis=1)
    
    @staticmethod
    def to_corner_coordinates_batch(batch):
        """
        Convert center coordinates to corner coordinates in a batch.
        Input shape: N x 4 (cx, cy, w, h)
        Output shape: N x 4 (x1, y1, w, h)
        """
        cx, cy, w, h = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]
        x1 = np.round(cx - w / 2).astype(int)
        y1 = np.round(cy - h / 2).astype(int)
        return np.stack([x1, y1, w+x1, h+y1], axis=1)