"""
Multi-Camera Football Field Homography Analysis System
Handles two cameras with overlapping views for complete field coverage
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import math

class FootballFieldHomography:
    """Football field homography analysis for multi-camera setup"""
    
    def __init__(self):
        # Standard football field dimensions (in meters)
        self.field_length = 105.0  # FIFA standard
        self.field_width = 68.0
        self.penalty_area_length = 16.5
        self.penalty_area_width = 40.3
        self.goal_area_length = 5.5
        self.goal_area_width = 18.3
        self.center_circle_radius = 9.15
        
        # Field coordinate system (origin at center)
        self.field_corners = np.array([
            [-self.field_length/2, -self.field_width/2],  # A (bottom-left)
            [-self.field_length/2, self.field_width/2],   # D (top-left)
            [self.field_length/2, self.field_width/2],    # C (top-right)
            [self.field_length/2, -self.field_width/2]    # B (bottom-right)
        ], dtype=np.float32)
        
        # Camera configurations
        self.camera1_coverage = None  # Left half + overlap
        self.camera2_coverage = None  # Right half + overlap
        self.overlap_region = None    # Center overlap area
        
        # Homography matrices
        self.H1 = None  # Camera 1 to field
        self.H2 = None  # Camera 2 to field
        self.H1_inv = None  # Field to camera 1
        self.H2_inv = None  # Field to camera 2
        
        # Field key points for homography calculation
        self.field_keypoints = self._define_field_keypoints()
        
    def _define_field_keypoints(self):
        """Define key field points for homography calculation"""
        keypoints = {
            # Corner points
            'corner_A': [-self.field_length/2, -self.field_width/2],
            'corner_B': [self.field_length/2, -self.field_width/2],
            'corner_C': [self.field_length/2, self.field_width/2],
            'corner_D': [-self.field_length/2, self.field_width/2],
            
            # Center line points
            'center_top': [0, self.field_width/2],
            'center_bottom': [0, -self.field_width/2],
            'center_circle': [0, 0],
            
            # Penalty area corners (left side)
            'penalty_left_top': [-self.field_length/2 + self.penalty_area_length, 
                                 self.penalty_area_width/2],
            'penalty_left_bottom': [-self.field_length/2 + self.penalty_area_length, 
                                   -self.penalty_area_width/2],
            
            # Penalty area corners (right side)
            'penalty_right_top': [self.field_length/2 - self.penalty_area_length, 
                                 self.penalty_area_width/2],
            'penalty_right_bottom': [self.field_length/2 - self.penalty_area_length, 
                                    -self.penalty_area_width/2],
            
            # Goal area corners (left side)
            'goal_left_top': [-self.field_length/2 + self.goal_area_length, 
                             self.goal_area_width/2],
            'goal_left_bottom': [-self.field_length/2 + self.goal_area_length, 
                                -self.goal_area_width/2],
            
            # Goal area corners (right side)
            'goal_right_top': [self.field_length/2 - self.goal_area_length, 
                              self.goal_area_width/2],
            'goal_right_bottom': [self.field_length/2 - self.goal_area_length, 
                                 -self.goal_area_width/2],
        }
        
        return keypoints
    
    def setup_camera_coverage(self):
        """Define camera coverage areas with overlap"""
        overlap_width = 20.0  # 20 meters overlap in center
        
        # Camera 1: Left half + overlap (A-D side + center)
        self.camera1_coverage = {
            'min_x': -self.field_length/2,
            'max_x': overlap_width/2,
            'min_y': -self.field_width/2,
            'max_y': self.field_width/2,
            'primary_region': (-self.field_length/2, 0),  # Left half
            'overlap_region': (-overlap_width/2, overlap_width/2)  # Center overlap
        }
        
        # Camera 2: Right half + overlap (B-C side + center)
        self.camera2_coverage = {
            'min_x': -overlap_width/2,
            'max_x': self.field_length/2,
            'min_y': -self.field_width/2,
            'max_y': self.field_width/2,
            'primary_region': (0, self.field_length/2),  # Right half
            'overlap_region': (-overlap_width/2, overlap_width/2)  # Center overlap
        }
        
        # Overlap region for cross-camera tracking
        self.overlap_region = {
            'min_x': -overlap_width/2,
            'max_x': overlap_width/2,
            'min_y': -self.field_width/2,
            'max_y': self.field_width/2
        }
    
    def calculate_homography(self, camera_id: int, image_points: np.ndarray, 
                           field_point_names: List[str]) -> bool:
        """
        Calculate homography matrix for a camera
        
        Args:
            camera_id: 1 or 2 for camera identification
            image_points: Corresponding points in image coordinates
            field_point_names: Names of field points corresponding to image points
            
        Returns:
            Success status
        """
        try:
            # Get corresponding field points
            field_points = []
            for name in field_point_names:
                if name in self.field_keypoints:
                    field_points.append(self.field_keypoints[name])
                else:
                    print(f"Warning: Field point '{name}' not found")
                    return False
            
            field_points = np.array(field_points, dtype=np.float32)
            
            if len(image_points) != len(field_points) or len(image_points) < 4:
                print("Error: Need at least 4 corresponding points")
                return False
            
            # Calculate homography
            H, mask = cv2.findHomography(image_points, field_points, 
                                       cv2.RANSAC, 5.0)
            
            if H is None:
                print("Error: Could not calculate homography")
                return False
            
            # Store homography matrices
            if camera_id == 1:
                self.H1 = H
                self.H1_inv = np.linalg.inv(H)
            elif camera_id == 2:
                self.H2 = H
                self.H2_inv = np.linalg.inv(H)
            else:
                print("Error: Invalid camera ID")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error calculating homography: {e}")
            return False
    
    def image_to_field(self, camera_id: int, image_points: np.ndarray) -> np.ndarray:
        """Convert image coordinates to field coordinates"""
        try:
            if camera_id == 1 and self.H1 is not None:
                H = self.H1
            elif camera_id == 2 and self.H2 is not None:
                H = self.H2
            else:
                return None
            
            # Ensure points are in correct format
            if len(image_points.shape) == 1:
                image_points = image_points.reshape(1, -1)
            
            # Add homogeneous coordinate if needed
            if image_points.shape[1] == 2:
                ones = np.ones((image_points.shape[0], 1))
                image_points_h = np.hstack([image_points, ones])
            else:
                image_points_h = image_points
            
            # Transform points
            field_points_h = (H @ image_points_h.T).T
            
            # Convert back to Cartesian coordinates
            field_points = field_points_h[:, :2] / field_points_h[:, 2:3]
            
            return field_points
            
        except Exception as e:
            print(f"Error transforming points: {e}")
            return None
    
    def field_to_image(self, camera_id: int, field_points: np.ndarray) -> np.ndarray:
        """Convert field coordinates to image coordinates"""
        try:
            if camera_id == 1 and self.H1_inv is not None:
                H_inv = self.H1_inv
            elif camera_id == 2 and self.H2_inv is not None:
                H_inv = self.H2_inv
            else:
                return None
            
            # Ensure points are in correct format
            if len(field_points.shape) == 1:
                field_points = field_points.reshape(1, -1)
            
            # Add homogeneous coordinate if needed
            if field_points.shape[1] == 2:
                ones = np.ones((field_points.shape[0], 1))
                field_points_h = np.hstack([field_points, ones])
            else:
                field_points_h = field_points
            
            # Transform points
            image_points_h = (H_inv @ field_points_h.T).T
            
            # Convert back to Cartesian coordinates
            image_points = image_points_h[:, :2] / image_points_h[:, 2:3]
            
            return image_points
            
        except Exception as e:
            print(f"Error transforming points: {e}")
            return None
    
    def is_in_camera_view(self, camera_id: int, field_point: np.ndarray) -> bool:
        """Check if a field point is visible in camera view"""
        if camera_id == 1:
            coverage = self.camera1_coverage
        elif camera_id == 2:
            coverage = self.camera2_coverage
        else:
            return False
        
        x, y = field_point[0], field_point[1]
        return (coverage['min_x'] <= x <= coverage['max_x'] and 
                coverage['min_y'] <= y <= coverage['max_y'])
    
    def is_in_overlap_region(self, field_point: np.ndarray) -> bool:
        """Check if a field point is in the overlap region"""
        if self.overlap_region is None:
            return False
        
        x, y = field_point[0], field_point[1]
        return (self.overlap_region['min_x'] <= x <= self.overlap_region['max_x'] and 
                self.overlap_region['min_y'] <= y <= self.overlap_region['max_y'])
    
    def get_primary_camera(self, field_point: np.ndarray) -> int:
        """Determine which camera has primary view of a field point"""
        x, y = field_point[0], field_point[1]
        
        # If in overlap region, choose based on which side is closer to center
        if self.is_in_overlap_region(field_point):
            return 1 if x < 0 else 2
        
        # Otherwise, choose based on field half
        return 1 if x < 0 else 2
    
    def calculate_field_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """Calculate real-world distance between two field points (in meters)"""
        return np.linalg.norm(point2 - point1)
    
    def calculate_player_speed(self, field_positions: List[np.ndarray], 
                             timestamps: List[float]) -> float:
        """Calculate player speed in m/s from field positions"""
        if len(field_positions) < 2 or len(timestamps) < 2:
            return 0.0
        
        total_distance = 0.0
        total_time = 0.0
        
        for i in range(1, len(field_positions)):
            distance = self.calculate_field_distance(field_positions[i-1], field_positions[i])
            time_diff = timestamps[i] - timestamps[i-1]
            
            if time_diff > 0:
                total_distance += distance
                total_time += time_diff
        
        return total_distance / total_time if total_time > 0 else 0.0
    
    def get_field_zone(self, field_point: np.ndarray) -> str:
        """Determine which zone of the field a point is in"""
        x, y = field_point[0], field_point[1]
        
        # Defensive thirds
        if x < -self.field_length/6:
            if x < -self.field_length/2 + self.penalty_area_length:
                return "left_penalty_area"
            else:
                return "left_defensive_third"
        # Middle third
        elif -self.field_length/6 <= x <= self.field_length/6:
            return "middle_third"
        # Attacking thirds
        else:
            if x > self.field_length/2 - self.penalty_area_length:
                return "right_penalty_area"
            else:
                return "right_attacking_third"
    
    def create_field_visualization(self, width: int = 800, height: int = 600) -> np.ndarray:
        """Create a top-down field visualization"""
        # Create blank field image
        field_img = np.zeros((height, width, 3), dtype=np.uint8)
        field_img[:] = (0, 100, 0)  # Green field
        
        # Calculate scaling factors
        scale_x = width / self.field_length
        scale_y = height / self.field_width
        
        # Draw field lines
        self._draw_field_lines(field_img, scale_x, scale_y, width, height)
        
        return field_img
    
    def _draw_field_lines(self, img: np.ndarray, scale_x: float, scale_y: float, 
                         width: int, height: int):
        """Draw field lines on visualization"""
        white = (255, 255, 255)
        thickness = 2
        
        # Field boundary
        cv2.rectangle(img, (0, 0), (width-1, height-1), white, thickness)
        
        # Center line
        center_x = width // 2
        cv2.line(img, (center_x, 0), (center_x, height-1), white, thickness)
        
        # Center circle
        center_y = height // 2
        radius = int(self.center_circle_radius * scale_x)
        cv2.circle(img, (center_x, center_y), radius, white, thickness)
        
        # Penalty areas
        penalty_width = int(self.penalty_area_length * scale_x)
        penalty_height = int(self.penalty_area_width * scale_y)
        
        # Left penalty area
        penalty_y = (height - penalty_height) // 2
        cv2.rectangle(img, (0, penalty_y), (penalty_width, penalty_y + penalty_height), 
                     white, thickness)
        
        # Right penalty area
        cv2.rectangle(img, (width - penalty_width, penalty_y), 
                     (width-1, penalty_y + penalty_height), white, thickness)
        
        # Goal areas
        goal_width = int(self.goal_area_length * scale_x)
        goal_height = int(self.goal_area_width * scale_y)
        goal_y = (height - goal_height) // 2
        
        # Left goal area
        cv2.rectangle(img, (0, goal_y), (goal_width, goal_y + goal_height), 
                     white, thickness)
        
        # Right goal area
        cv2.rectangle(img, (width - goal_width, goal_y), 
                     (width-1, goal_y + goal_height), white, thickness)
