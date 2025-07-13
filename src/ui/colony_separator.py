import numpy as np
import cv2
from skimage import measure, morphology
from skimage.segmentation import clear_border
from typing import List, Dict, Tuple
import polars as pl

class ColonySeparator:
    """BiofilmQ-style colony separation tool for automatic biofilm region detection"""
    
    def __init__(self):
        self.intensity_threshold = 0.5
        self.min_colony_size = 100
        self.max_colony_size = 50000
        self.detected_colonies = []
        self.manual_additions = []
        self.manual_removals = []
        
    def detect_colonies_from_segmentation(self, segmented_image: np.ndarray) -> List[Dict]:
        """
        Detect colony regions from a segmented binary image
        
        Args:
            segmented_image: Binary segmented image (0 = background, >0 = cells)
            
        Returns:
            List of colony dictionaries with bounding boxes and properties
        """
        # Convert to binary if needed
        if segmented_image.max() > 1:
            binary_image = (segmented_image > 0).astype(np.uint8)
        else:
            binary_image = segmented_image.astype(np.uint8)
        
        # Apply morphological operations to connect nearby cells into colonies
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # Close small gaps between cells
        closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Fill small holes within colonies
        filled = morphology.remove_small_holes(closed.astype(bool), area_threshold=50).astype(np.uint8)
        
        # Remove small objects (noise)
        cleaned = morphology.remove_small_objects(filled.astype(bool), min_size=self.min_colony_size).astype(np.uint8)
        
        # Clear border objects (incomplete colonies at image edges)
        cleared = clear_border(cleaned).astype(np.uint8)
        
        # Label connected components (each becomes a potential colony)
        labeled_image = measure.label(cleared)
        
        # Extract properties for each detected region
        regions = measure.regionprops(labeled_image)
        
        colonies = []
        colony_id = 1
        
        for region in regions:
            # Filter by size
            if self.min_colony_size <= region.area <= self.max_colony_size:
                # Calculate bounding box
                min_row, min_col, max_row, max_col = region.bbox
                
                # Calculate colony properties
                colony_data = {
                    'colony_id': colony_id,
                    'label': region.label,
                    'area': region.area,
                    'centroid': region.centroid,
                    'bbox': (min_col, min_row, max_col, max_row),  # (x1, y1, x2, y2)
                    'bbox_width': max_col - min_col,
                    'bbox_height': max_row - min_row,
                    'eccentricity': region.eccentricity,
                    'solidity': region.solidity,
                    'extent': region.extent,
                    'perimeter': region.perimeter,
                    'major_axis_length': region.major_axis_length,
                    'minor_axis_length': region.minor_axis_length,
                    'orientation': region.orientation,
                    'source': 'automatic'
                }
                
                colonies.append(colony_data)
                colony_id += 1
        
        self.detected_colonies = colonies
        return colonies
    
    def add_manual_colony(self, polygon_points: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> Dict:
        """
        Add a manually drawn colony region
        
        Args:
            polygon_points: List of (x, y) coordinates defining the polygon
            image_shape: Shape of the image (height, width)
            
        Returns:
            Dictionary containing the new colony data
        """
        if len(polygon_points) < 3:
            return None
        
        # Create mask from polygon
        mask = np.zeros(image_shape, dtype=np.uint8)
        points_array = np.array(polygon_points, dtype=np.int32)
        cv2.fillPoly(mask, [points_array], 1)
        
        # Calculate properties
        region = measure.regionprops(mask)[0]
        
        # Calculate bounding box
        min_row, min_col, max_row, max_col = region.bbox
        
        colony_id = len(self.detected_colonies) + len(self.manual_additions) + 1
        
        colony_data = {
            'colony_id': colony_id,
            'label': colony_id,
            'area': region.area,
            'centroid': region.centroid,
            'bbox': (min_col, min_row, max_col, max_row),
            'bbox_width': max_col - min_col,
            'bbox_height': max_row - min_row,
            'eccentricity': region.eccentricity,
            'solidity': region.solidity,
            'extent': region.extent,
            'perimeter': region.perimeter,
            'major_axis_length': region.major_axis_length,
            'minor_axis_length': region.minor_axis_length,
            'orientation': region.orientation,
            'source': 'manual',
            'polygon_points': polygon_points
        }
        
        self.manual_additions.append(colony_data)
        return colony_data
    
    def remove_colony(self, colony_id: int):
        """Remove a colony by ID"""
        # Remove from detected colonies
        self.detected_colonies = [c for c in self.detected_colonies if c['colony_id'] != colony_id]
        
        # Remove from manual additions
        self.manual_additions = [c for c in self.manual_additions if c['colony_id'] != colony_id]
    
    def get_all_colonies(self) -> List[Dict]:
        """Get all detected colonies (automatic + manual)"""
        return self.detected_colonies + self.manual_additions
    
    def update_parameters(self, intensity_threshold: float = None, 
                         min_colony_size: int = None, 
                         max_colony_size: int = None):
        """Update detection parameters"""
        if intensity_threshold is not None:
            self.intensity_threshold = intensity_threshold
        if min_colony_size is not None:
            self.min_colony_size = min_colony_size
        if max_colony_size is not None:
            self.max_colony_size = max_colony_size
    
    def create_colony_overlay(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create an overlay image showing detected colonies with bounding boxes
        
        Args:
            image_shape: Shape of the original image (height, width)
            
        Returns:
            RGB overlay image with colored bounding boxes
        """
        overlay = np.zeros((*image_shape, 3), dtype=np.uint8)
        
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 192, 203), # Pink
            (0, 128, 128),  # Teal
        ]
        
        all_colonies = self.get_all_colonies()
        
        for i, colony in enumerate(all_colonies):
            color = colors[i % len(colors)]
            x1, y1, x2, y2 = colony['bbox']
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Draw colony ID
            cv2.putText(overlay, f"C{colony['colony_id']}", 
                       (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, 2)
            
            # Draw centroid
            centroid_x, centroid_y = int(colony['centroid'][1]), int(colony['centroid'][0])
            cv2.circle(overlay, (centroid_x, centroid_y), 3, color, -1)
        
        return overlay