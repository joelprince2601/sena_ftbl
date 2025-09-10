"""
Streamlit Homography Processor for Multi-Camera Football Analysis
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from homography_analyzer import FootballFieldHomography
from tracking_system import EnhancedFootballTracker
from utils import validate_video_file

class StreamlitHomographyProcessor:
    """Streamlit processor for multi-camera homography analysis"""
    
    def __init__(self):
        self.homography = FootballFieldHomography()
        self.tracker1 = None  # Camera 1 tracker
        self.tracker2 = None  # Camera 2 tracker
        self.field_visualization = None
        self.setup_complete = False
        
        # Cross-camera tracking data
        self.cross_camera_tracks = {}
        self.field_positions_history = {}
        self.player_speeds = {}
        
    def setup_processors(self, config: Dict[str, Any]):
        """Setup tracking processors for both cameras"""
        try:
            # Initialize trackers for both cameras
            self.tracker1 = EnhancedFootballTracker(
                model_name=config['model'],
                tracker_type=config['tracker'],
                confidence_threshold=config['confidence_threshold'],
                reid_threshold=config['reid_threshold']
            )
            
            self.tracker2 = EnhancedFootballTracker(
                model_name=config['model'],
                tracker_type=config['tracker'],
                confidence_threshold=config['confidence_threshold'],
                reid_threshold=config['reid_threshold']
            )
            
            # Setup camera coverage
            self.homography.setup_camera_coverage()
            
            # Create field visualization
            self.field_visualization = self.homography.create_field_visualization()
            
            self.setup_complete = True
            return True
            
        except Exception as e:
            st.error(f"Error setting up processors: {e}")
            return False
    
    def calibrate_cameras(self, camera1_points: np.ndarray, camera2_points: np.ndarray,
                         field_point_names: List[str]) -> bool:
        """Calibrate both cameras with field reference points"""
        try:
            # Calibrate camera 1
            success1 = self.homography.calculate_homography(1, camera1_points, field_point_names)
            if not success1:
                st.error("Failed to calibrate Camera 1")
                return False
            
            # Calibrate camera 2
            success2 = self.homography.calculate_homography(2, camera2_points, field_point_names)
            if not success2:
                st.error("Failed to calibrate Camera 2")
                return False
            
            st.success("✅ Both cameras calibrated successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error during calibration: {e}")
            return False
    
    def process_dual_camera_videos(self, video1_file, video2_file, config: Dict[str, Any]) -> List[Dict]:
        """Process videos from both cameras simultaneously"""
        if not self.setup_complete:
            if not self.setup_processors(config):
                return []
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp1:
            tmp1.write(video1_file.read())
            temp_video1_path = tmp1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp2:
            tmp2.write(video2_file.read())
            temp_video2_path = tmp2.name
        
        try:
            return self._process_synchronized_videos(temp_video1_path, temp_video2_path, config)
        finally:
            # Clean up temporary files
            if os.path.exists(temp_video1_path):
                os.unlink(temp_video1_path)
            if os.path.exists(temp_video2_path):
                os.unlink(temp_video2_path)
    
    def _process_synchronized_videos(self, video1_path: str, video2_path: str, 
                                   config: Dict[str, Any]) -> List[Dict]:
        """Process both videos in synchronized manner"""
        # Open video captures
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        if not cap1.isOpened() or not cap2.isOpened():
            st.error("Could not open one or both video files")
            return []
        
        # Get video properties
        fps1 = cap1.get(cv2.CAP_PROP_FPS)
        fps2 = cap2.get(cv2.CAP_PROP_FPS)
        total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Use minimum frame count for synchronization
        total_frames = min(total_frames1, total_frames2)
        target_fps = min(fps1, fps2)
        
        # Calculate timing
        playback_speed = config.get('playback_speed', 1.0)
        max_display_fps = config.get('max_display_fps', 15)
        display_fps = min(target_fps * playback_speed, max_display_fps)
        frame_delay = 1.0 / display_fps if display_fps > 0 else 0.067
        
        # Create UI layout
        st.markdown("## 🎬 Dual Camera Processing")
        
        # Video display containers
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📹 Camera 1 (Left Half)")
            video1_placeholder = st.empty()
        with col2:
            st.markdown("### 📹 Camera 2 (Right Half)")
            video2_placeholder = st.empty()
        
        # Field visualization
        st.markdown("### 🏟️ Field View")
        field_placeholder = st.empty()
        
        # Statistics containers
        stats_container = st.container()
        analytics_container = st.container()
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Processing data
        results_data = []
        frame_count = 0
        
        try:
            while cap1.isOpened() and cap2.isOpened():
                # Read frames from both cameras
                ret1, frame1 = cap1.read()
                ret2, frame2 = cap2.read()
                
                if not ret1 or not ret2:
                    break
                
                frame_count += 1
                timestamp = frame_count / target_fps
                
                start_time = time.time()
                
                # Process both cameras
                tracks1, tracks2, field_tracks = self._process_dual_frame(frame1, frame2, timestamp)
                
                # Update visualizations
                self._update_dual_display(frame1, frame2, tracks1, tracks2, 
                                        video1_placeholder, video2_placeholder)
                
                # Update field visualization
                field_img = self._update_field_visualization(field_tracks)
                field_placeholder.image(field_img, channels="RGB", use_container_width=True)
                
                # Collect statistics
                frame_stats = self._collect_dual_statistics(tracks1, tracks2, field_tracks, 
                                                          frame_count, timestamp)
                results_data.append(frame_stats)
                
                # Update UI
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"Processing frame {frame_count}/{total_frames} "
                               f"({progress*100:.1f}%) - {len(field_tracks)} players tracked")
                
                # Update statistics display
                with stats_container:
                    self._update_statistics_display(frame_stats)
                
                # Frame timing
                processing_time = time.time() - start_time
                sleep_time = max(0, frame_delay - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
            return results_data
            
        finally:
            cap1.release()
            cap2.release()
    
    def _process_dual_frame(self, frame1: np.ndarray, frame2: np.ndarray, 
                           timestamp: float) -> Tuple[List, List, List]:
        """Process frames from both cameras and merge tracking data"""
        # Detect and track players in both cameras
        detections1 = self.tracker1.detect_players(frame1)
        detections2 = self.tracker2.detect_players(frame2)
        
        tracks1 = self.tracker1.update_tracks(frame1, detections1)
        tracks2 = self.tracker2.update_tracks(frame2, detections2)
        
        # Convert to field coordinates
        field_tracks = []
        
        # Process camera 1 tracks
        for track in tracks1:
            if hasattr(track, 'to_tlbr'):
                bbox = track.to_tlbr()
                center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
                
                # Convert to field coordinates
                field_pos = self.homography.image_to_field(1, center.reshape(1, -1))
                if field_pos is not None:
                    field_track = {
                        'track_id': track.track_id,
                        'camera': 1,
                        'image_pos': center,
                        'field_pos': field_pos[0],
                        'bbox': bbox,
                        'timestamp': timestamp,
                        'confidence': getattr(track, 'confidence', 0.5)
                    }
                    field_tracks.append(field_track)
        
        # Process camera 2 tracks
        for track in tracks2:
            if hasattr(track, 'to_tlbr'):
                bbox = track.to_tlbr()
                center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
                
                # Convert to field coordinates
                field_pos = self.homography.image_to_field(2, center.reshape(1, -1))
                if field_pos is not None:
                    field_track = {
                        'track_id': track.track_id,
                        'camera': 2,
                        'image_pos': center,
                        'field_pos': field_pos[0],
                        'bbox': bbox,
                        'timestamp': timestamp,
                        'confidence': getattr(track, 'confidence', 0.5)
                    }
                    field_tracks.append(field_track)
        
        # Merge overlapping tracks
        merged_tracks = self._merge_overlapping_tracks(field_tracks)
        
        return tracks1, tracks2, merged_tracks
    
    def _merge_overlapping_tracks(self, field_tracks: List[Dict]) -> List[Dict]:
        """Merge tracks from overlapping region between cameras"""
        merged = []
        overlap_threshold = 2.0  # 2 meters
        
        # Separate tracks by camera
        cam1_tracks = [t for t in field_tracks if t['camera'] == 1]
        cam2_tracks = [t for t in field_tracks if t['camera'] == 2]
        
        # Find matches in overlap region
        matched_cam2 = set()
        
        for track1 in cam1_tracks:
            if self.homography.is_in_overlap_region(track1['field_pos']):
                best_match = None
                best_distance = float('inf')
                
                for i, track2 in enumerate(cam2_tracks):
                    if i in matched_cam2:
                        continue
                    
                    if self.homography.is_in_overlap_region(track2['field_pos']):
                        distance = np.linalg.norm(track1['field_pos'] - track2['field_pos'])
                        if distance < overlap_threshold and distance < best_distance:
                            best_match = i
                            best_distance = distance
                
                if best_match is not None:
                    # Merge tracks (use higher confidence)
                    track2 = cam2_tracks[best_match]
                    if track1['confidence'] >= track2['confidence']:
                        merged.append(track1)
                    else:
                        merged.append(track2)
                    matched_cam2.add(best_match)
                else:
                    merged.append(track1)
            else:
                merged.append(track1)
        
        # Add unmatched camera 2 tracks
        for i, track2 in enumerate(cam2_tracks):
            if i not in matched_cam2:
                merged.append(track2)
        
        return merged
    
    def _update_dual_display(self, frame1: np.ndarray, frame2: np.ndarray,
                           tracks1: List, tracks2: List,
                           placeholder1, placeholder2):
        """Update display for both camera views"""
        # Draw tracks on frames
        annotated1 = self.tracker1.draw_tracks(frame1, tracks1)
        annotated2 = self.tracker2.draw_tracks(frame2, tracks2)
        
        # Convert to RGB
        rgb1 = cv2.cvtColor(annotated1, cv2.COLOR_BGR2RGB)
        rgb2 = cv2.cvtColor(annotated2, cv2.COLOR_BGR2RGB)
        
        # Update displays
        placeholder1.image(rgb1, channels="RGB", use_container_width=True)
        placeholder2.image(rgb2, channels="RGB", use_container_width=True)
    
    def _update_field_visualization(self, field_tracks: List[Dict]) -> np.ndarray:
        """Update field visualization with player positions"""
        field_img = self.field_visualization.copy()
        
        # Draw player positions on field
        for track in field_tracks:
            field_pos = track['field_pos']
            
            # Convert field coordinates to image coordinates
            img_height, img_width = field_img.shape[:2]
            
            # Scale field position to image coordinates
            x = int((field_pos[0] + self.homography.field_length/2) / 
                   self.homography.field_length * img_width)
            y = int((field_pos[1] + self.homography.field_width/2) / 
                   self.homography.field_width * img_height)
            
            # Ensure coordinates are within image bounds
            x = max(0, min(img_width-1, x))
            y = max(0, min(img_height-1, y))
            
            # Draw player
            color = (255, 0, 0) if track['camera'] == 1 else (0, 0, 255)
            cv2.circle(field_img, (x, y), 8, color, -1)
            
            # Draw player ID
            cv2.putText(field_img, str(track['track_id']), (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)
    
    def _collect_dual_statistics(self, tracks1: List, tracks2: List, 
                               field_tracks: List[Dict], frame_count: int, 
                               timestamp: float) -> Dict:
        """Collect statistics from dual camera processing"""
        return {
            'frame': frame_count,
            'timestamp': timestamp,
            'camera1_players': len(tracks1),
            'camera2_players': len(tracks2),
            'total_field_players': len(field_tracks),
            'overlap_players': len([t for t in field_tracks 
                                  if self.homography.is_in_overlap_region(t['field_pos'])]),
            'left_half_players': len([t for t in field_tracks 
                                    if t['field_pos'][0] < 0]),
            'right_half_players': len([t for t in field_tracks 
                                     if t['field_pos'][0] > 0]),
            'avg_confidence': np.mean([t['confidence'] for t in field_tracks]) 
                            if field_tracks else 0.0
        }
    
    def _update_statistics_display(self, stats: Dict):
        """Update real-time statistics display"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Camera 1 Players", stats['camera1_players'])
        with col2:
            st.metric("Camera 2 Players", stats['camera2_players'])
        with col3:
            st.metric("Total Field Players", stats['total_field_players'])
        with col4:
            st.metric("Overlap Players", stats['overlap_players'])

    def get_field_analytics(self, results_data: List[Dict]) -> Dict:
        """Generate comprehensive field analytics"""
        if not results_data:
            return {}

        df = pd.DataFrame(results_data)

        analytics = {
            'total_frames': len(df),
            'duration': df['timestamp'].max() if 'timestamp' in df else 0,
            'avg_players_per_frame': df['total_field_players'].mean(),
            'max_players_detected': df['total_field_players'].max(),
            'field_coverage': {
                'left_half_avg': df['left_half_players'].mean(),
                'right_half_avg': df['right_half_players'].mean(),
                'overlap_avg': df['overlap_players'].mean()
            },
            'camera_performance': {
                'camera1_avg': df['camera1_players'].mean(),
                'camera2_avg': df['camera2_players'].mean(),
                'camera1_max': df['camera1_players'].max(),
                'camera2_max': df['camera2_players'].max()
            },
            'detection_quality': {
                'avg_confidence': df['avg_confidence'].mean() if 'avg_confidence' in df else 0,
                'detection_consistency': df['total_field_players'].std()
            }
        }

        return analytics

    def create_field_heatmap(self, field_tracks_history: List[List[Dict]]) -> np.ndarray:
        """Create a heatmap of player positions on the field"""
        # Create field visualization
        field_img = self.homography.create_field_visualization(width=800, height=600)
        heatmap = np.zeros((600, 800), dtype=np.float32)

        # Accumulate player positions
        for frame_tracks in field_tracks_history:
            for track in frame_tracks:
                field_pos = track['field_pos']

                # Convert to image coordinates
                x = int((field_pos[0] + self.homography.field_length/2) /
                       self.homography.field_length * 800)
                y = int((field_pos[1] + self.homography.field_width/2) /
                       self.homography.field_width * 600)

                # Ensure coordinates are within bounds
                x = max(0, min(799, x))
                y = max(0, min(599, y))

                # Add to heatmap with Gaussian blur effect
                cv2.circle(heatmap, (x, y), 20, 1.0, -1)

        # Normalize and apply color map
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Apply Gaussian blur for smooth heatmap
        heatmap = cv2.GaussianBlur(heatmap, (41, 41), 0)

        # Convert to color heatmap
        heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Blend with field image
        alpha = 0.6
        result = cv2.addWeighted(field_img, 1-alpha, heatmap_colored, alpha, 0)

        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    def analyze_player_movements(self, field_tracks_history: List[List[Dict]]) -> Dict:
        """Analyze player movement patterns across the field"""
        movement_data = {
            'total_movements': 0,
            'avg_speed': 0.0,
            'max_speed': 0.0,
            'zone_transitions': {},
            'cross_camera_handoffs': 0
        }

        # Track player positions over time
        player_histories = {}

        for frame_idx, frame_tracks in enumerate(field_tracks_history):
            for track in frame_tracks:
                player_id = track['track_id']

                if player_id not in player_histories:
                    player_histories[player_id] = []

                player_histories[player_id].append({
                    'frame': frame_idx,
                    'position': track['field_pos'],
                    'timestamp': track['timestamp'],
                    'camera': track['camera']
                })

        # Analyze each player's movement
        total_speeds = []

        for player_id, history in player_histories.items():
            if len(history) < 2:
                continue

            # Calculate speeds
            for i in range(1, len(history)):
                prev_pos = history[i-1]['position']
                curr_pos = history[i]['position']
                time_diff = history[i]['timestamp'] - history[i-1]['timestamp']

                if time_diff > 0:
                    distance = np.linalg.norm(curr_pos - prev_pos)
                    speed = distance / time_diff  # m/s
                    total_speeds.append(speed)

            # Count camera handoffs
            for i in range(1, len(history)):
                if history[i]['camera'] != history[i-1]['camera']:
                    movement_data['cross_camera_handoffs'] += 1

        if total_speeds:
            movement_data['avg_speed'] = np.mean(total_speeds)
            movement_data['max_speed'] = np.max(total_speeds)
            movement_data['total_movements'] = len(total_speeds)

        return movement_data
