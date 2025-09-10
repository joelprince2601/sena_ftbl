"""
Hawk-Eye Style Single Video Homography Analysis
Based on the approach from https://github.com/nihal111/hawk_eye
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pickle
import os
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import math

class HawkEyeAnalyzer:
    """Hawk-Eye style homography analysis for single camera football videos"""
    
    def __init__(self):
        # Field template for edge map generation
        self.field_template = None
        self.edge_map_dict = {}  # Dictionary of edge maps and homographies
        self.hog_features = []   # HOG features for matching
        self.homographies = []   # Corresponding homographies
        self.knn_model = None    # KNN model for matching
        
        # Field dimensions (FIFA standard)
        self.field_length = 105.0  # meters
        self.field_width = 68.0    # meters
        
        # Player detection parameters
        self.green_lower = np.array([40, 40, 40])
        self.green_upper = np.array([80, 255, 255])
        
        # Initialize field template
        self._create_field_template()
        
    def _create_field_template(self):
        """Create a standard football field template for edge map generation"""
        # Create field template (top-down view)
        template_width = 800
        template_height = 600
        
        self.field_template = np.zeros((template_height, template_width, 3), dtype=np.uint8)
        self.field_template[:] = (0, 100, 0)  # Green field
        
        # Draw field lines
        white = (255, 255, 255)
        thickness = 2
        
        # Field boundary
        cv2.rectangle(self.field_template, (50, 50), (template_width-50, template_height-50), white, thickness)
        
        # Center line
        center_x = template_width // 2
        cv2.line(self.field_template, (center_x, 50), (center_x, template_height-50), white, thickness)
        
        # Center circle
        center_y = template_height // 2
        cv2.circle(self.field_template, (center_x, center_y), 80, white, thickness)
        
        # Penalty areas
        penalty_width = 120
        penalty_height = 200
        
        # Left penalty area
        penalty_y = (template_height - penalty_height) // 2
        cv2.rectangle(self.field_template, (50, penalty_y), (50 + penalty_width, penalty_y + penalty_height), white, thickness)
        
        # Right penalty area
        cv2.rectangle(self.field_template, (template_width - 50 - penalty_width, penalty_y), 
                     (template_width - 50, penalty_y + penalty_height), white, thickness)
        
        # Goal areas
        goal_width = 60
        goal_height = 120
        goal_y = (template_height - goal_height) // 2
        
        # Left goal area
        cv2.rectangle(self.field_template, (50, goal_y), (50 + goal_width, goal_y + goal_height), white, thickness)
        
        # Right goal area
        cv2.rectangle(self.field_template, (template_width - 50 - goal_width, goal_y), 
                     (template_width - 50, goal_y + goal_height), white, thickness)
    
    def generate_edge_map(self, image: np.ndarray) -> np.ndarray:
        """Generate edge map from input image using Canny edge detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert back to 3-channel for consistency
        edge_map = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return edge_map
    
    def extract_hog_features(self, image: np.ndarray) -> np.ndarray:
        """Extract HOG features from edge map for matching"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        resized = cv2.resize(gray, (128, 128))
        
        # Calculate HOG features
        hog = cv2.HOGDescriptor()
        features = hog.compute(resized)
        
        return features.flatten()
    
    def create_augmented_dictionary(self, training_images: List[np.ndarray], 
                                  training_homographies: List[np.ndarray]):
        """Create augmented dictionary with camera perturbations (pan, tilt, zoom)"""
        self.edge_map_dict = {}
        self.hog_features = []
        self.homographies = []
        
        # Perturbation parameters
        pan_values = [-0.1, 0.1, 0.15, -0.15, -0.18, 0.18]
        tilt_values = [-0.05, 0.05, 0.02, -0.02, -0.035, 0.035]
        zoom_values = [0.95, 1.1, 1.2, 1.15, 0.9, 0.85]
        
        for idx, (image, homography) in enumerate(zip(training_images, training_homographies)):
            # Original image
            edge_map = self.generate_edge_map(image)
            hog_features = self.extract_hog_features(edge_map)
            
            self.edge_map_dict[len(self.hog_features)] = {
                'edge_map': edge_map,
                'homography': homography,
                'source_idx': idx
            }
            self.hog_features.append(hog_features)
            self.homographies.append(homography)
            
            # Generate augmented versions
            for pan in pan_values:
                aug_homography = self._apply_pan_perturbation(homography, pan)
                aug_edge_map = self._generate_augmented_edge_map(image, aug_homography)
                aug_hog = self.extract_hog_features(aug_edge_map)
                
                self.edge_map_dict[len(self.hog_features)] = {
                    'edge_map': aug_edge_map,
                    'homography': aug_homography,
                    'source_idx': idx,
                    'perturbation': f'pan_{pan}'
                }
                self.hog_features.append(aug_hog)
                self.homographies.append(aug_homography)
            
            for tilt in tilt_values:
                aug_homography = self._apply_tilt_perturbation(homography, tilt)
                aug_edge_map = self._generate_augmented_edge_map(image, aug_homography)
                aug_hog = self.extract_hog_features(aug_edge_map)
                
                self.edge_map_dict[len(self.hog_features)] = {
                    'edge_map': aug_edge_map,
                    'homography': aug_homography,
                    'source_idx': idx,
                    'perturbation': f'tilt_{tilt}'
                }
                self.hog_features.append(aug_hog)
                self.homographies.append(aug_homography)
            
            for zoom in zoom_values:
                aug_homography = self._apply_zoom_perturbation(homography, zoom)
                aug_edge_map = self._generate_augmented_edge_map(image, aug_homography)
                aug_hog = self.extract_hog_features(aug_edge_map)
                
                self.edge_map_dict[len(self.hog_features)] = {
                    'edge_map': aug_edge_map,
                    'homography': aug_homography,
                    'source_idx': idx,
                    'perturbation': f'zoom_{zoom}'
                }
                self.hog_features.append(aug_hog)
                self.homographies.append(aug_homography)
        
        # Train KNN model
        self.hog_features = np.array(self.hog_features)
        self.knn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
        self.knn_model.fit(self.hog_features)
    
    def _apply_pan_perturbation(self, homography: np.ndarray, pan_value: float) -> np.ndarray:
        """Apply pan perturbation to homography matrix"""
        # Create rotation matrix for pan
        cos_pan = math.cos(pan_value)
        sin_pan = math.sin(pan_value)
        
        pan_matrix = np.array([
            [cos_pan, -sin_pan, 0],
            [sin_pan, cos_pan, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return pan_matrix @ homography
    
    def _apply_tilt_perturbation(self, homography: np.ndarray, tilt_value: float) -> np.ndarray:
        """Apply tilt perturbation to homography matrix"""
        # Create tilt transformation matrix
        tilt_matrix = np.array([
            [1, 0, 0],
            [tilt_value, 1, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return tilt_matrix @ homography
    
    def _apply_zoom_perturbation(self, homography: np.ndarray, zoom_value: float) -> np.ndarray:
        """Apply zoom perturbation to homography matrix"""
        # Create zoom transformation matrix
        zoom_matrix = np.array([
            [zoom_value, 0, 0],
            [0, zoom_value, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        
        return zoom_matrix @ homography
    
    def _generate_augmented_edge_map(self, image: np.ndarray, homography: np.ndarray) -> np.ndarray:
        """Generate edge map using augmented homography"""
        # Transform field template using inverse homography
        h_inv = np.linalg.inv(homography)
        height, width = image.shape[:2]
        
        # Warp field template to camera view
        warped_template = cv2.warpPerspective(self.field_template, h_inv, (width, height))
        
        # Generate edge map
        edge_map = self.generate_edge_map(warped_template)
        
        return edge_map
    
    def find_best_homography(self, query_image: np.ndarray) -> Tuple[np.ndarray, float, Dict]:
        """Find best matching homography for query image"""
        if self.knn_model is None:
            raise ValueError("Dictionary not created. Call create_augmented_dictionary first.")
        
        # Generate edge map for query
        query_edge_map = self.generate_edge_map(query_image)
        query_hog = self.extract_hog_features(query_edge_map)
        
        # Find nearest neighbor
        distances, indices = self.knn_model.kneighbors([query_hog])
        best_idx = indices[0][0]
        best_distance = distances[0][0]
        
        # Get corresponding homography and metadata
        best_match = self.edge_map_dict[best_idx]
        best_homography = best_match['homography']
        
        match_info = {
            'distance': best_distance,
            'source_idx': best_match['source_idx'],
            'perturbation': best_match.get('perturbation', 'original'),
            'edge_map': best_match['edge_map']
        }
        
        return best_homography, best_distance, match_info
    
    def detect_players(self, image: np.ndarray) -> List[Dict]:
        """Detect players in the image using color-based field segmentation"""
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Create mask for green field
        field_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Invert mask to get non-field areas (players, etc.)
        player_mask = cv2.bitwise_not(field_mask)
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_CLOSE, kernel)
        player_mask = cv2.morphologyEx(player_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(player_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        players = []
        for contour in contours:
            # Filter contours by size and aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Check if contour looks like a player (height > width, reasonable size)
            if area > 200 and h > w * 1.2 and h > 30:
                # Extract player region
                player_region = image[y:y+h, x:x+w]
                
                # Determine team color (simplified)
                team_color = self._determine_team_color(player_region)
                
                players.append({
                    'bbox': (x, y, w, h),
                    'center': (x + w//2, y + h//2),
                    'area': area,
                    'team_color': team_color,
                    'contour': contour
                })
        
        return players
    
    def _determine_team_color(self, player_region: np.ndarray) -> str:
        """Determine team color using K-means clustering"""
        # Reshape image for K-means
        pixels = player_region.reshape(-1, 3)
        
        # Apply K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get dominant colors
        colors = kmeans.cluster_centers_
        
        # Simple heuristic: classify based on color intensity
        color1_intensity = np.sum(colors[0])
        color2_intensity = np.sum(colors[1])
        
        if color1_intensity > color2_intensity:
            return "light_team"
        else:
            return "dark_team"
    
    def project_to_top_view(self, image: np.ndarray, homography: np.ndarray, 
                           players: List[Dict] = None) -> np.ndarray:
        """Project camera view to top view using homography"""
        # Get field template dimensions
        template_height, template_width = self.field_template.shape[:2]
        
        # Warp image to top view
        top_view = cv2.warpPerspective(image, homography, (template_width, template_height))
        
        # Blend with field template for better visualization
        alpha = 0.7
        blended = cv2.addWeighted(top_view, alpha, self.field_template, 1-alpha, 0)
        
        # Project players if provided
        if players:
            for player in players:
                center = np.array([[player['center']]], dtype=np.float32)
                # Transform player position to top view
                transformed_center = cv2.perspectiveTransform(center, homography)
                
                x, y = int(transformed_center[0][0][0]), int(transformed_center[0][0][1])
                
                # Draw player on top view
                color = (0, 0, 255) if player['team_color'] == 'dark_team' else (255, 0, 0)
                cv2.circle(blended, (x, y), 8, color, -1)
                cv2.circle(blended, (x, y), 10, (255, 255, 255), 2)
        
        return blended
    
    def calculate_iou_score(self, predicted_homography: np.ndarray, 
                          ground_truth_homography: np.ndarray, 
                          image_shape: Tuple[int, int]) -> float:
        """Calculate IoU score between predicted and ground truth homographies"""
        height, width = image_shape
        
        # Define image corners
        corners = np.array([
            [[0, 0]], [[width, 0]], [[width, height]], [[0, height]]
        ], dtype=np.float32)
        
        # Transform corners using both homographies
        pred_corners = cv2.perspectiveTransform(corners, predicted_homography)
        gt_corners = cv2.perspectiveTransform(corners, ground_truth_homography)
        
        # Create masks for both projections
        template_shape = self.field_template.shape[:2]
        pred_mask = np.zeros(template_shape, dtype=np.uint8)
        gt_mask = np.zeros(template_shape, dtype=np.uint8)
        
        # Fill polygons
        cv2.fillPoly(pred_mask, [pred_corners.astype(np.int32)], 255)
        cv2.fillPoly(gt_mask, [gt_corners.astype(np.int32)], 255)
        
        # Calculate IoU
        intersection = cv2.bitwise_and(pred_mask, gt_mask)
        union = cv2.bitwise_or(pred_mask, gt_mask)
        
        intersection_area = np.sum(intersection > 0)
        union_area = np.sum(union > 0)
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
