"""
Streamlit Football Player Re-ID Application
Modern web interface for football player tracking and re-identification
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from tracking_system import EnhancedFootballTracker
from config import YOLO_MODELS, TRACKER_CONFIGS, DEFAULT_PARAMS, UI_SETTINGS, VIDEO_SETTINGS, HOMOGRAPHY_SETTINGS, FIELD_CALIBRATION_POINTS
from video_processor import StreamlitVideoProcessor
from utils import ModelManager, validate_video_file, PerformanceMonitor
from keypoint_processor import StreamlitKeypointProcessor
from homography_processor import StreamlitHomographyProcessor
from tactical_analyzer import TacticalAnalyzer
from hawkeye_analyzer import HawkEyeAnalyzer

# Page configuration
st.set_page_config(
    page_title=UI_SETTINGS["page_title"],
    page_icon=UI_SETTINGS["page_icon"],
    layout=UI_SETTINGS["layout"],
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'video_processor' not in st.session_state:
        st.session_state.video_processor = StreamlitVideoProcessor()
    if 'keypoint_processor' not in st.session_state:
        st.session_state.keypoint_processor = StreamlitKeypointProcessor()
    if 'homography_processor' not in st.session_state:
        st.session_state.homography_processor = StreamlitHomographyProcessor()
    if 'tactical_analyzer' not in st.session_state:
        st.session_state.tactical_analyzer = TacticalAnalyzer()
    if 'hawkeye_analyzer' not in st.session_state:
        st.session_state.hawkeye_analyzer = HawkEyeAnalyzer()
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'performance_monitor' not in st.session_state:
        st.session_state.performance_monitor = PerformanceMonitor()
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'video_uploaded' not in st.session_state:
        st.session_state.video_uploaded = False
    if 'results_data' not in st.session_state:
        st.session_state.results_data = []
    if 'homography_results_data' not in st.session_state:
        st.session_state.homography_results_data = []
    if 'homography_calibrated' not in st.session_state:
        st.session_state.homography_calibrated = False
    if 'keypoint_results_data' not in st.session_state:
        st.session_state.keypoint_results_data = []
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Player Tracking"

def create_sidebar():
    """Create sidebar with controls"""
    st.sidebar.markdown("## ⚙️ Configuration")
    
    # Model selection
    st.sidebar.markdown("### 🤖 YOLO Model")
    selected_model = st.sidebar.selectbox(
        "Choose YOLO model:",
        options=list(YOLO_MODELS.keys()),
        format_func=lambda x: f"{YOLO_MODELS[x]['name']} - {YOLO_MODELS[x]['description']}",
        index=0
    )
    
    # Tracker selection - all trackers are now always available
    st.sidebar.markdown("### 🎯 Tracker Type")

    selected_tracker = st.sidebar.selectbox(
        "Choose tracking algorithm:",
        options=list(TRACKER_CONFIGS.keys()),
        format_func=lambda x: f"{TRACKER_CONFIGS[x]['name']} - {TRACKER_CONFIGS[x]['description']}",
        index=0
    )
    
    # Parameters
    st.sidebar.markdown("### 📊 Detection Parameters")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_PARAMS["confidence_threshold"],
        step=0.05,
        help="Minimum confidence for detections"
    )
    
    nms_threshold = st.sidebar.slider(
        "NMS Threshold", 
        min_value=0.1,
        max_value=1.0,
        value=DEFAULT_PARAMS["nms_threshold"],
        step=0.05,
        help="Non-maximum suppression threshold"
    )
    
    reid_threshold = st.sidebar.slider(
        "Re-ID Threshold",
        min_value=0.1, 
        max_value=1.0,
        value=DEFAULT_PARAMS["reid_threshold"],
        step=0.05,
        help="Re-identification similarity threshold"
    )
    
    # Advanced parameters
    with st.sidebar.expander("🔧 Advanced Parameters"):
        max_age = st.slider("Max Age", 10, 200, DEFAULT_PARAMS["max_age"])
        min_hits = st.slider("Min Hits", 1, 10, DEFAULT_PARAMS["min_hits"])

    # Video processing options
    with st.sidebar.expander("🎬 Video Processing"):
        playback_speed = st.selectbox("Playback Speed", [0.25, 0.5, 1.0, 1.5, 2.0], index=2)
        max_display_fps = st.slider("Max Display FPS", 5, 30, 15)
        enable_debug_overlay = st.checkbox("Debug Overlay", value=False)

    return {
        'model': selected_model,
        'tracker': selected_tracker,
        'confidence_threshold': confidence_threshold,
        'nms_threshold': nms_threshold,
        'reid_threshold': reid_threshold,
        'max_age': max_age,
        'min_hits': min_hits,
        'playback_speed': playback_speed,
        'max_display_fps': max_display_fps,
        'enable_debug_overlay': enable_debug_overlay
    }

def create_main_interface():
    """Create main interface for player tracking"""
    # File upload
    st.markdown("## 📹 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=VIDEO_SETTINGS["supported_formats"],
        help=f"Supported formats: {', '.join(VIDEO_SETTINGS['supported_formats'])}"
    )

    return uploaded_file

def create_keypoint_interface():
    """Create interface for keypoint detection"""
    # File upload
    st.markdown("## 📹 Upload Video for Keypoint Detection")
    uploaded_file = st.file_uploader(
        "Choose a video file for keypoint analysis",
        type=VIDEO_SETTINGS["supported_formats"],
        help=f"Supported formats: {', '.join(VIDEO_SETTINGS['supported_formats'])}",
        key="keypoint_uploader"
    )

    return uploaded_file

def create_keypoint_sidebar():
    """Create sidebar for keypoint detection configuration"""
    st.sidebar.markdown("## ⚙️ Keypoint Configuration")

    # Detection type selection
    st.sidebar.markdown("### 🎯 Detection Type")
    detection_types = st.session_state.keypoint_processor.get_detection_types()

    selected_detection = st.sidebar.selectbox(
        "Choose detection type:",
        options=list(detection_types.keys()),
        format_func=lambda x: f"{detection_types[x]['name']}",
        index=0
    )

    # Show description
    st.sidebar.info(detection_types[selected_detection]['description'])

    # Parameters
    st.sidebar.markdown("### 📊 Detection Parameters")

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for keypoint detection"
    )

    # Visualization options
    st.sidebar.markdown("### 🎨 Visualization Options")

    show_connections = st.sidebar.checkbox("Show Connections", value=True)
    show_labels = st.sidebar.checkbox("Show Labels", value=True)
    show_3d = st.sidebar.checkbox("Show 3D Visualization", value=False)

    return {
        'detection_type': selected_detection,
        'confidence_threshold': confidence_threshold,
        'show_connections': show_connections,
        'show_labels': show_labels,
        'show_3d': show_3d
    }

def process_video(video_file, config):
    """Process uploaded video using the enhanced video processor"""
    try:
        # Validate video file
        is_valid, message = validate_video_file(video_file)
        if not is_valid:
            st.error(f"❌ {message}")
            return []

        # Check model availability
        model_manager = st.session_state.model_manager
        model_info = model_manager.get_model_info(config['model'])

        if not model_info['available']:
            with st.spinner(f"📥 Downloading {model_info['name']} model..."):
                model_manager.download_model(config['model'])

        # Process video using the enhanced processor
        st.markdown("## 🎬 Video Processing")

        # Setup processor
        video_processor = st.session_state.video_processor
        video_processor.setup_processor(config)

        # Process the video
        results_data = video_processor.process_uploaded_video(video_file, config)

        # Store results
        st.session_state.results_data = results_data

        return results_data

    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        return []

def process_keypoint_video(video_file, config):
    """Process uploaded video for keypoint detection"""
    try:
        # Validate video file
        is_valid, message = validate_video_file(video_file)
        if not is_valid:
            st.error(f"❌ {message}")
            return []

        # Process video using the keypoint processor
        st.markdown("## 🎬 Keypoint Detection & Tracking")

        # Process the video
        keypoint_processor = st.session_state.keypoint_processor
        results_data = keypoint_processor.process_uploaded_video(
            video_file,
            config['detection_type'],
            config['confidence_threshold'],
            config['show_connections'],
            config['show_labels'],
            config['show_3d']
        )

        # Store results
        st.session_state.keypoint_results_data = results_data

        return results_data

    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        return []

def show_analytics():
    """Show enhanced analytics dashboard"""
    if not st.session_state.results_data:
        st.info("📊 Process a video to see analytics")
        return

    st.markdown("## 📈 Enhanced Analytics Dashboard")

    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.results_data)

    # Enhanced metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Max Players", df['active_players'].max())
    with col2:
        st.metric("Avg Players", f"{df['active_players'].mean():.1f}")
    with col3:
        if 'avg_confidence' in df.columns:
            st.metric("Avg Confidence", f"{df['avg_confidence'].mean():.2f}")
        else:
            st.metric("Total Frames", df['current_frame'].max())
    with col4:
        if 'tracking_efficiency' in df.columns:
            st.metric("Track Efficiency", f"{df['tracking_efficiency'].mean():.2f}")
        else:
            st.metric("Video Duration", f"{df['timestamp'].max():.1f}s")
    with col5:
        if 'moving_players' in df.columns:
            st.metric("Avg Moving", f"{df['moving_players'].mean():.1f}")
        else:
            st.metric("Processing FPS", f"{df.get('processing_fps', [0]).mean():.1f}")

    # Create tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Player Tracking", "🎯 Performance", "📈 Advanced", "🔍 Detailed"])

    with tab1:
        # Player count over time
        fig_players = px.line(df, x='timestamp', y='active_players',
                             title='Active Players Over Time',
                             labels={'timestamp': 'Time (seconds)', 'active_players': 'Active Players'})
        fig_players.update_layout(height=400)
        st.plotly_chart(fig_players, use_container_width=True)

        # ID pool usage over time
        if 'id_pool_usage' in df.columns:
            # Extract numeric values from id_pool_usage strings like "8/12"
            df['used_ids'] = df['id_pool_usage'].str.split('/').str[0].astype(int)
            fig_ids = px.line(df, x='timestamp', y='used_ids',
                             title='ID Pool Usage Over Time',
                             labels={'timestamp': 'Time (seconds)', 'used_ids': 'Used IDs'})
            fig_ids.update_layout(height=400)
            st.plotly_chart(fig_ids, use_container_width=True)

    with tab2:
        # Performance metrics
        if 'processing_fps' in df.columns:
            fig_fps = px.line(df, x='timestamp', y='processing_fps',
                             title='Processing Performance (FPS)',
                             labels={'timestamp': 'Time (seconds)', 'processing_fps': 'FPS'})
            fig_fps.update_layout(height=400)
            st.plotly_chart(fig_fps, use_container_width=True)

        # Confidence over time
        if 'avg_confidence' in df.columns:
            fig_conf = px.line(df, x='timestamp', y='avg_confidence',
                              title='Average Detection Confidence',
                              labels={'timestamp': 'Time (seconds)', 'avg_confidence': 'Confidence'})
            fig_conf.update_layout(height=400)
            st.plotly_chart(fig_conf, use_container_width=True)

    with tab3:
        # Advanced analytics
        if 'moving_players' in df.columns and 'avg_velocity' in df.columns:
            # Movement analysis
            fig_movement = go.Figure()
            fig_movement.add_trace(go.Scatter(x=df['timestamp'], y=df['moving_players'],
                                            mode='lines', name='Moving Players', line=dict(color='green')))
            fig_movement.add_trace(go.Scatter(x=df['timestamp'], y=df['avg_velocity']*100,
                                            mode='lines', name='Avg Velocity (×100)', line=dict(color='red')))
            fig_movement.update_layout(title='Player Movement Analysis',
                                     xaxis_title='Time (seconds)',
                                     yaxis_title='Count / Velocity',
                                     height=400)
            st.plotly_chart(fig_movement, use_container_width=True)

        # Tracking efficiency
        if 'tracking_efficiency' in df.columns:
            fig_efficiency = px.line(df, x='timestamp', y='tracking_efficiency',
                                   title='Tracking Efficiency Over Time',
                                   labels={'timestamp': 'Time (seconds)', 'tracking_efficiency': 'Efficiency'})
            fig_efficiency.update_layout(height=400)
            st.plotly_chart(fig_efficiency, use_container_width=True)

    with tab4:
        # Detailed statistics
        st.markdown("### 📋 Detailed Statistics")

        # Summary statistics
        summary_stats = {
            'Total Frames Processed': df['current_frame'].max(),
            'Video Duration': f"{df['timestamp'].max():.1f} seconds",
            'Average Players per Frame': f"{df['active_players'].mean():.2f}",
            'Maximum Players Detected': df['active_players'].max(),
            'Player Detection Rate': f"{(df['active_players'] > 0).mean() * 100:.1f}%"
        }

        if 'avg_confidence' in df.columns:
            summary_stats['Average Confidence'] = f"{df['avg_confidence'].mean():.3f}"
        if 'tracking_efficiency' in df.columns:
            summary_stats['Average Tracking Efficiency'] = f"{df['tracking_efficiency'].mean():.3f}"

        for key, value in summary_stats.items():
            st.text(f"{key}: {value}")

        # Player detection histogram
        fig_hist = px.histogram(df, x='active_players', nbins=20,
                               title='Distribution of Active Players per Frame',
                               labels={'active_players': 'Number of Active Players', 'count': 'Frequency'})
        fig_hist.update_layout(height=400)
        st.plotly_chart(fig_hist, use_container_width=True)

        # Data export option
        if st.button("📥 Download Analytics Data"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="football_analytics.csv",
                mime="text/csv"
            )

        # Tactical Map Integration
        st.markdown("### 🏟️ Tactical Field Map")
        if st.button("🎯 Generate Tactical Map"):
            with st.spinner("Generating tactical field visualization..."):
                # Create tactical field visualization
                tactical_analyzer = st.session_state.tactical_analyzer

                # Mock player data for demonstration
                mock_players = []
                for i in range(min(22, df['active_players'].max())):  # Max 22 players
                    mock_players.append({
                        'track_id': i + 1,
                        'field_pos': np.array([
                            np.random.uniform(-50, 50),  # Random field position
                            np.random.uniform(-30, 30)
                        ])
                    })

                # Create formation visualization
                formation_fig = tactical_analyzer.create_formation_visualization(
                    mock_players, "Current Formation"
                )
                st.plotly_chart(formation_fig, use_container_width=True)

                # Formation analysis
                formation_analysis = tactical_analyzer.analyze_team_formation(mock_players)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Detected Formation", formation_analysis.get('formation', 'Unknown'))
                with col2:
                    st.metric("Formation Confidence", f"{formation_analysis.get('confidence', 0):.2f}")
                with col3:
                    st.metric("Team Compactness", f"{formation_analysis.get('compactness', 0):.1f}m")

def show_keypoint_analytics():
    """Show football pose analytics dashboard"""
    if not st.session_state.keypoint_results_data:
        st.info("📊 Process a video to see football pose analytics")
        return

    st.markdown("## 📈 Football Pose Analytics Dashboard")

    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.keypoint_results_data)

    # Create metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Max Players", df['football_players'].max())
    with col2:
        st.metric("Avg Players", f"{df['football_players'].mean():.1f}")
    with col3:
        st.metric("Avg Pose Quality", f"{df['avg_pose_quality'].mean():.2f}")
    with col4:
        st.metric("Avg Confidence", f"{df['avg_confidence'].mean():.2f}")

    # Players count over time
    fig_players = px.line(df, x='timestamp', y='football_players',
                         title='Football Players Detected Over Time',
                         labels={'timestamp': 'Time (seconds)', 'football_players': 'Number of Players'})
    st.plotly_chart(fig_players, use_container_width=True)

    # Action analysis over time
    if 'action_counts' in df.columns:
        # Extract action data
        action_data = []
        for _, row in df.iterrows():
            action_counts = row.get('action_counts', {})
            for action, count in action_counts.items():
                action_data.append({
                    'timestamp': row['timestamp'],
                    'action': action.title(),
                    'count': count
                })

        if action_data:
            action_df = pd.DataFrame(action_data)
            fig_actions = px.line(action_df, x='timestamp', y='count', color='action',
                                 title='Football Actions Over Time',
                                 labels={'timestamp': 'Time (seconds)', 'count': 'Number of Players'})
            st.plotly_chart(fig_actions, use_container_width=True)

    # Processing performance
    fig_perf = px.line(df, x='timestamp', y='processing_fps',
                      title='Processing Performance (FPS)',
                      labels={'timestamp': 'Time (seconds)', 'processing_fps': 'Processing FPS'})
    st.plotly_chart(fig_perf, use_container_width=True)

    # Pose quality over time
    fig_quality = px.line(df, x='timestamp', y='avg_pose_quality',
                         title='Average Pose Quality Over Time',
                         labels={'timestamp': 'Time (seconds)', 'avg_pose_quality': 'Pose Quality'})
    st.plotly_chart(fig_quality, use_container_width=True)

def show_model_status():
    """Show model availability status"""
    st.sidebar.markdown("### 📦 Model Status")

    model_manager = st.session_state.model_manager

    for model_name, model_config in YOLO_MODELS.items():
        model_info = model_manager.get_model_info(model_name)

        if model_info['available']:
            st.sidebar.success(f"✅ {model_config['name']}")
        else:
            st.sidebar.warning(f"⚠️ {model_config['name']} (will auto-download)")

def main():
    """Main application"""
    initialize_session_state()

    # Page navigation
    st.sidebar.markdown("# 🏠 Navigation")
    page = st.sidebar.selectbox(
        "Choose Application Mode:",
        ["Player Tracking", "Keypoint Detection", "Multi-Camera Analysis", "Tactical Analysis"],
        index=0 if st.session_state.current_page == "Player Tracking" else
              (1 if st.session_state.current_page == "Keypoint Detection" else
               (2 if st.session_state.current_page == "Multi-Camera Analysis" else 3))
    )
    st.session_state.current_page = page

    if page == "Player Tracking":
        show_player_tracking_page()
    elif page == "Keypoint Detection":
        show_keypoint_detection_page()
    elif page == "Multi-Camera Analysis":
        show_homography_analysis_page()
    elif page == "Tactical Analysis":
        show_tactical_analysis_page()

def show_player_tracking_page():
    """Show player tracking page"""
    st.markdown('<h1 class="main-header">⚽ Football Player Re-ID System</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    config = create_sidebar()

    # Show model status
    show_model_status()

    # Main interface
    uploaded_file = create_main_interface()

    # Processing section
    if uploaded_file is not None:
        st.session_state.video_uploaded = True

        # Validate file
        is_valid, validation_message = validate_video_file(uploaded_file)

        if is_valid:
            # Show video info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "Status": "✅ Valid video file"
            }

            st.markdown("### 📋 Video Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Filename:** {file_details['Filename']}")
            with col2:
                st.write(f"**File size:** {file_details['File size']}")
            with col3:
                st.write(f"**Status:** {file_details['Status']}")

            # Configuration summary
            with st.expander("🔧 Processing Configuration"):
                st.write(f"**Model:** {YOLO_MODELS[config['model']]['name']}")
                st.write(f"**Tracker:** {TRACKER_CONFIGS[config['tracker']]['name']}")
                st.write(f"**Confidence:** {config['confidence_threshold']}")
                st.write(f"**Re-ID Threshold:** {config['reid_threshold']}")

            # Process button
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                with st.spinner("🔄 Initializing processing..."):
                    try:
                        results = process_video(uploaded_file, config)
                        if results:
                            st.balloons()
                            st.success("🎉 Processing completed successfully!")
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
                        st.exception(e)
        else:
            st.error(f"❌ {validation_message}")

    # Analytics section
    if st.session_state.results_data:
        st.markdown("---")
        show_analytics()

    # Performance metrics
    if hasattr(st.session_state, 'performance_monitor'):
        perf_stats = st.session_state.performance_monitor.get_performance_stats()
        if perf_stats['avg_fps'] > 0:
            st.sidebar.markdown("### ⚡ Performance")
            st.sidebar.metric("Processing FPS", f"{perf_stats['avg_fps']:.1f}")

def show_keypoint_detection_page():
    """Show keypoint detection and tracking page"""
    st.markdown('<h1 class="main-header">🤸 Keypoint Detection & Tracking</h1>', unsafe_allow_html=True)

    # Keypoint detection configuration
    keypoint_config = create_keypoint_sidebar()

    # Main interface for keypoint detection
    uploaded_file = create_keypoint_interface()

    # Processing section
    if uploaded_file is not None:
        # Validate file
        is_valid, validation_message = validate_video_file(uploaded_file)

        if is_valid:
            # Show video info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / (1024*1024):.2f} MB",
                "Status": "✅ Valid video file"
            }

            st.markdown("### 📋 Video Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Filename:** {file_details['Filename']}")
            with col2:
                st.write(f"**File size:** {file_details['File size']}")
            with col3:
                st.write(f"**Status:** {file_details['Status']}")

            # Configuration summary
            with st.expander("🔧 Keypoint Configuration"):
                detection_types = st.session_state.keypoint_processor.get_detection_types()
                selected_type = detection_types[keypoint_config['detection_type']]
                st.write(f"**Detection Type:** {selected_type['name']}")
                st.write(f"**Description:** {selected_type['description']}")
                st.write(f"**Keypoints:** {selected_type['keypoints']}")
                st.write(f"**Confidence:** {keypoint_config['confidence_threshold']}")

            # Process button
            if st.button("🚀 Start Keypoint Detection", type="primary", use_container_width=True):
                with st.spinner("🔄 Initializing keypoint detection..."):
                    try:
                        results = process_keypoint_video(uploaded_file, keypoint_config)
                        if results:
                            st.balloons()
                            st.success("🎉 Keypoint detection completed successfully!")
                    except Exception as e:
                        st.error(f"❌ Error processing video: {str(e)}")
                        st.exception(e)
        else:
            st.error(f"❌ {validation_message}")

    # Keypoint analytics section
    if st.session_state.keypoint_results_data:
        st.markdown("---")
        show_keypoint_analytics()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Keypoint Detection & Tracking System | Built with MediaPipe & Streamlit</p>
        <p><small>Upload a video to start detecting and tracking keypoints</small></p>
    </div>
    """, unsafe_allow_html=True)

def show_homography_analysis_page():
    """Show multi-camera homography analysis page"""
    st.markdown('<h1 class="main-header">🏟️ Multi-Camera Field Analysis</h1>', unsafe_allow_html=True)
    st.markdown("### Analyze football field from two synchronized cameras with overlapping views")

    # Sidebar configuration
    config = create_sidebar()

    # Create tabs for different stages
    tab1, tab2, tab3, tab4 = st.tabs(["📹 Upload Videos", "🎯 Calibration", "🎬 Processing", "📊 Analytics"])

    with tab1:
        show_video_upload_section()

    with tab2:
        show_calibration_section(config)

    with tab3:
        show_dual_processing_section(config)

    with tab4:
        show_homography_analytics()

def show_video_upload_section():
    """Show video upload section for dual cameras"""
    st.markdown("## 📹 Upload Camera Videos")
    st.info("Upload videos from both cameras. Ensure they are synchronized and have overlapping views of the center field.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📹 Camera 1 (Left Half + Center)")
        st.markdown("*Should cover from left goal to center line with overlap*")
        camera1_file = st.file_uploader(
            "Choose Camera 1 video file",
            type=VIDEO_SETTINGS["supported_formats"],
            help="Left side camera covering A-D corners + center overlap",
            key="camera1_uploader"
        )

        if camera1_file:
            st.success(f"✅ Camera 1: {camera1_file.name} ({camera1_file.size / (1024*1024):.1f} MB)")

    with col2:
        st.markdown("### 📹 Camera 2 (Right Half + Center)")
        st.markdown("*Should cover from center line to right goal with overlap*")
        camera2_file = st.file_uploader(
            "Choose Camera 2 video file",
            type=VIDEO_SETTINGS["supported_formats"],
            help="Right side camera covering B-C corners + center overlap",
            key="camera2_uploader"
        )

        if camera2_file:
            st.success(f"✅ Camera 2: {camera2_file.name} ({camera2_file.size / (1024*1024):.1f} MB)")

    # Store files in session state
    if camera1_file:
        st.session_state.camera1_file = camera1_file
    if camera2_file:
        st.session_state.camera2_file = camera2_file

    # Show field diagram
    st.markdown("### 🏟️ Camera Coverage Diagram")
    st.markdown("""
    ```
    Field Layout (Top View):

    D ←────────── Center Line ──────────→ C
    │                │                   │
    │   Camera 1     │     Camera 2      │
    │   Coverage     │     Coverage      │
    │                │                   │
    │        ←─── Overlap ───→           │
    │                │                   │
    A ←────────── Center Line ──────────→ B

    Camera 1: Covers A-D side + center overlap
    Camera 2: Covers B-C side + center overlap
    Overlap: Ensures continuous tracking across field
    ```
    """)

def show_calibration_section(config):
    """Show camera calibration section"""
    st.markdown("## 🎯 Camera Calibration")

    if not hasattr(st.session_state, 'camera1_file') or not hasattr(st.session_state, 'camera2_file'):
        st.warning("⚠️ Please upload both camera videos first")
        return

    st.info("Click on corresponding field points in both camera views to establish homography mapping")

    # Calibration points selection
    st.markdown("### 📍 Select Calibration Points")

    # Group points by priority
    high_priority = [k for k, v in FIELD_CALIBRATION_POINTS.items() if v['priority'] == 'high']
    medium_priority = [k for k, v in FIELD_CALIBRATION_POINTS.items() if v['priority'] == 'medium']
    low_priority = [k for k, v in FIELD_CALIBRATION_POINTS.items() if v['priority'] == 'low']

    # Display point options with descriptions
    st.markdown("**High Priority Points (Recommended):**")
    for point in high_priority:
        st.write(f"• {FIELD_CALIBRATION_POINTS[point]['name']}: {FIELD_CALIBRATION_POINTS[point]['description']}")

    selected_points = st.multiselect(
        "Choose field points for calibration (minimum 4 required):",
        list(FIELD_CALIBRATION_POINTS.keys()),
        default=high_priority,
        format_func=lambda x: FIELD_CALIBRATION_POINTS[x]['name'],
        help="Select at least 4 corresponding points visible in both cameras"
    )

    if len(selected_points) >= 4:
        st.success(f"✅ {len(selected_points)} calibration points selected")

        # Manual point input (simplified for demo)
        st.markdown("### 🖱️ Point Coordinates")
        st.info("In a full implementation, this would be an interactive point selection interface")

        if st.button("🎯 Auto-Calibrate (Demo)", type="primary"):
            # Simulate calibration for demo
            st.session_state.homography_calibrated = True
            st.success("✅ Cameras calibrated successfully!")
            st.balloons()
    else:
        st.warning(f"⚠️ Need at least 4 points, currently selected: {len(selected_points)}")

def show_dual_processing_section(config):
    """Show dual camera processing section"""
    st.markdown("## 🎬 Dual Camera Processing")

    if not st.session_state.homography_calibrated:
        st.warning("⚠️ Please calibrate cameras first")
        return

    if not hasattr(st.session_state, 'camera1_file') or not hasattr(st.session_state, 'camera2_file'):
        st.warning("⚠️ Please upload both camera videos first")
        return

    # Processing configuration
    st.markdown("### ⚙️ Processing Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Model:** {YOLO_MODELS[config['model']]['name']}")
    with col2:
        st.write(f"**Tracker:** {TRACKER_CONFIGS[config['tracker']]['name']}")
    with col3:
        st.write(f"**Confidence:** {config['confidence_threshold']}")

    # Start processing
    if st.button("🚀 Start Dual Camera Analysis", type="primary", use_container_width=True):
        with st.spinner("🔄 Processing dual camera videos..."):
            try:
                homography_processor = st.session_state.homography_processor
                results = homography_processor.process_dual_camera_videos(
                    st.session_state.camera1_file,
                    st.session_state.camera2_file,
                    config
                )

                if results:
                    st.session_state.homography_results_data = results
                    st.balloons()
                    st.success("🎉 Dual camera analysis completed successfully!")
                else:
                    st.error("❌ Processing failed")

            except Exception as e:
                st.error(f"❌ Error during processing: {str(e)}")

def show_homography_analytics():
    """Show homography analysis results"""
    st.markdown("## 📊 Multi-Camera Analytics")

    if not st.session_state.homography_results_data:
        st.info("📊 Process dual camera videos to see analytics")
        return

    # Convert to DataFrame
    df = pd.DataFrame(st.session_state.homography_results_data)

    # Enhanced metrics
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Players", df['total_field_players'].max())
    with col2:
        st.metric("Avg Players", f"{df['total_field_players'].mean():.1f}")
    with col3:
        st.metric("Overlap Players", df['overlap_players'].max())
    with col4:
        st.metric("Left Half Avg", f"{df['left_half_players'].mean():.1f}")
    with col5:
        st.metric("Right Half Avg", f"{df['right_half_players'].mean():.1f}")

    # Create analytics tabs
    tab1, tab2, tab3 = st.tabs(["📈 Player Distribution", "🏟️ Field Coverage", "📊 Camera Performance"])

    with tab1:
        # Player distribution over time
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['total_field_players'],
                                mode='lines', name='Total Players', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['left_half_players'],
                                mode='lines', name='Left Half', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['right_half_players'],
                                mode='lines', name='Right Half', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['overlap_players'],
                                mode='lines', name='Overlap Region', line=dict(color='orange')))

        fig.update_layout(title='Player Distribution Across Field',
                         xaxis_title='Time (seconds)',
                         yaxis_title='Number of Players',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Field coverage heatmap (simplified)
        st.markdown("### 🏟️ Field Coverage Analysis")

        coverage_data = {
            'Region': ['Left Half', 'Center Overlap', 'Right Half'],
            'Avg Players': [df['left_half_players'].mean(),
                           df['overlap_players'].mean(),
                           df['right_half_players'].mean()],
            'Max Players': [df['left_half_players'].max(),
                           df['overlap_players'].max(),
                           df['right_half_players'].max()]
        }

        coverage_df = pd.DataFrame(coverage_data)

        fig = px.bar(coverage_df, x='Region', y=['Avg Players', 'Max Players'],
                    title='Player Distribution by Field Region',
                    barmode='group')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Camera performance comparison
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['camera1_players'],
                                mode='lines', name='Camera 1', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['camera2_players'],
                                mode='lines', name='Camera 2', line=dict(color='blue')))

        fig.update_layout(title='Camera Performance Comparison',
                         xaxis_title='Time (seconds)',
                         yaxis_title='Players Detected',
                         height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Performance summary
        st.markdown("### 📋 Performance Summary")
        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            st.markdown("**Camera 1 Statistics:**")
            st.write(f"- Average Players: {df['camera1_players'].mean():.1f}")
            st.write(f"- Max Players: {df['camera1_players'].max()}")
            st.write(f"- Detection Rate: {(df['camera1_players'] > 0).mean() * 100:.1f}%")

        with perf_col2:
            st.markdown("**Camera 2 Statistics:**")
            st.write(f"- Average Players: {df['camera2_players'].mean():.1f}")
            st.write(f"- Max Players: {df['camera2_players'].max()}")
            st.write(f"- Detection Rate: {(df['camera2_players'] > 0).mean() * 100:.1f}%")

def show_tactical_analysis_page():
    """Show tactical analysis page with individual player tracking and Hawk-Eye integration"""
    st.markdown('<h1 class="main-header">⚽ Tactical Analysis & Player Insights</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced tactical mapping with individual player tracking and Hawk-Eye homography analysis")

    # Sidebar configuration
    config = create_sidebar()

    # Create tabs for different analysis types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📹 Video Upload",
        "🎯 Hawk-Eye Analysis",
        "👤 Player Selection",
        "📊 Tactical Maps",
        "📈 Analytics"
    ])

    with tab1:
        show_tactical_video_upload(config)

    with tab2:
        show_hawkeye_analysis_section(config)

    with tab3:
        show_player_selection_section()

    with tab4:
        show_tactical_maps_section()

    with tab5:
        show_tactical_analytics_section()

def show_tactical_video_upload(config):
    """Show video upload section for tactical analysis"""
    st.markdown("## 📹 Upload Video for Tactical Analysis")
    st.info("Upload a football match video for comprehensive tactical analysis including player tracking, formation analysis, and Hawk-Eye homography mapping.")

    uploaded_file = st.file_uploader(
        "Choose a video file for tactical analysis",
        type=VIDEO_SETTINGS["supported_formats"],
        help="Upload a football match video for tactical analysis",
        key="tactical_uploader"
    )

    if uploaded_file:
        st.success(f"✅ Video uploaded: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.1f} MB)")
        st.session_state.tactical_video = uploaded_file

        # Analysis options
        st.markdown("### ⚙️ Analysis Options")
        col1, col2, col3 = st.columns(3)

        with col1:
            enable_tactical = st.checkbox("Enable Tactical Analysis", value=True,
                                        help="Analyze formations, player movements, and tactical patterns")
        with col2:
            enable_hawkeye = st.checkbox("Enable Hawk-Eye Mapping", value=True,
                                       help="Generate top-down field view using homography")
        with col3:
            enable_individual = st.checkbox("Enable Individual Tracking", value=True,
                                          help="Track individual players for detailed statistics")

        # Store analysis preferences
        st.session_state.enable_tactical_analysis = enable_tactical
        st.session_state.enable_hawkeye_analysis = enable_hawkeye
        st.session_state.enable_individual_tracking = enable_individual

        # Processing button
        if st.button("🚀 Start Tactical Analysis", type="primary", use_container_width=True):
            with st.spinner("🔄 Processing tactical analysis..."):
                try:
                    results = process_tactical_video(uploaded_file, config)
                    if results:
                        st.balloons()
                        st.success("🎉 Tactical analysis completed successfully!")
                except Exception as e:
                    st.error(f"❌ Error during tactical analysis: {str(e)}")

def show_hawkeye_analysis_section(config):
    """Show Hawk-Eye homography analysis section"""
    st.markdown("## 🎯 Hawk-Eye Homography Analysis")

    if not hasattr(st.session_state, 'tactical_video'):
        st.warning("⚠️ Please upload a video first")
        return

    st.info("Hawk-Eye analysis creates a top-down tactical view of the football field using computer vision homography techniques.")

    # Hawk-Eye configuration
    st.markdown("### 🔧 Hawk-Eye Configuration")

    col1, col2 = st.columns(2)
    with col1:
        edge_detection_method = st.selectbox(
            "Edge Detection Method",
            ["Canny", "Sobel", "Laplacian"],
            help="Method for generating edge maps"
        )

        homography_method = st.selectbox(
            "Homography Method",
            ["RANSAC", "Least Squares", "LMedS"],
            help="Method for calculating homography matrices"
        )

    with col2:
        augmentation_level = st.slider(
            "Data Augmentation Level",
            min_value=1, max_value=5, value=3,
            help="Level of data augmentation (pan, tilt, zoom variations)"
        )

        matching_threshold = st.slider(
            "Matching Threshold",
            min_value=0.1, max_value=1.0, value=0.7,
            help="Threshold for edge map matching"
        )

    # Show field template
    st.markdown("### 🏟️ Field Template")
    hawkeye_analyzer = st.session_state.hawkeye_analyzer
    field_template = hawkeye_analyzer.field_template

    if field_template is not None:
        st.image(cv2.cvtColor(field_template, cv2.COLOR_BGR2RGB),
                caption="Football Field Template for Homography Mapping",
                use_container_width=True)

    # Calibration status
    if hasattr(st.session_state, 'hawkeye_calibrated') and st.session_state.hawkeye_calibrated:
        st.success("✅ Hawk-Eye system calibrated and ready")
    else:
        st.warning("⚠️ Hawk-Eye system needs calibration with training data")

def show_player_selection_section():
    """Show player selection section for individual analysis"""
    st.markdown("## 👤 Individual Player Analysis")

    if not hasattr(st.session_state, 'results_data') or not st.session_state.results_data:
        st.warning("⚠️ Please process a video first to see player data")
        return

    # Get unique player IDs from results
    all_player_ids = set()
    for frame_data in st.session_state.results_data:
        if 'active_tracks' in frame_data:
            # Extract player IDs from tracking data
            for track_id in range(frame_data.get('active_players', 0)):
                all_player_ids.add(track_id + 1)  # Assuming 1-based IDs

    if not all_player_ids:
        st.info("📊 No player tracking data available")
        return

    # Player selection
    st.markdown("### 🎯 Select Player for Analysis")
    selected_player = st.selectbox(
        "Choose a player to analyze:",
        sorted(list(all_player_ids)),
        help="Select a player ID to view detailed tactical analysis"
    )

    if selected_player:
        st.session_state.selected_player_id = selected_player

        # Player analysis options
        col1, col2, col3 = st.columns(3)

        with col1:
            show_heatmap = st.checkbox("Show Heatmap", value=True)
        with col2:
            show_movement_trail = st.checkbox("Show Movement Trail", value=True)
        with col3:
            show_statistics = st.checkbox("Show Statistics", value=True)

        # Generate player analysis
        if st.button(f"📊 Analyze Player {selected_player}", type="primary"):
            with st.spinner(f"Analyzing Player {selected_player}..."):
                show_individual_player_analysis(selected_player, show_heatmap, show_movement_trail, show_statistics)

def show_tactical_maps_section():
    """Show tactical maps and formation analysis"""
    st.markdown("## 📊 Tactical Maps & Formation Analysis")

    if not hasattr(st.session_state, 'tactical_analyzer'):
        st.warning("⚠️ Tactical analyzer not initialized")
        return

    tactical_analyzer = st.session_state.tactical_analyzer

    # Formation analysis
    st.markdown("### 🏃 Team Formation Analysis")

    # Mock team data for demonstration
    if hasattr(st.session_state, 'results_data') and st.session_state.results_data:
        # Create formation visualization
        mock_team_data = [
            {'track_id': i, 'field_pos': np.array([np.random.uniform(-50, 50), np.random.uniform(-30, 30)])}
            for i in range(1, 12)  # 11 players
        ]

        formation_fig = tactical_analyzer.create_formation_visualization(mock_team_data, "Team Analysis")
        st.plotly_chart(formation_fig, use_container_width=True)

        # Formation statistics
        formation_analysis = tactical_analyzer.analyze_team_formation(mock_team_data)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Formation", formation_analysis.get('formation', 'Unknown'))
        with col2:
            st.metric("Confidence", f"{formation_analysis.get('confidence', 0):.2f}")
        with col3:
            st.metric("Style", formation_analysis.get('style', 'Unknown'))
        with col4:
            st.metric("Compactness", f"{formation_analysis.get('compactness', 0):.1f}m")
    else:
        st.info("📊 Process a video to see tactical formations")

def show_tactical_analytics_section():
    """Show comprehensive tactical analytics"""
    st.markdown("## 📈 Tactical Analytics Dashboard")

    if not hasattr(st.session_state, 'results_data') or not st.session_state.results_data:
        st.info("📊 Process a video to see tactical analytics")
        return

    # Create analytics tabs
    analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
        "🏃 Player Performance",
        "⚽ Team Tactics",
        "🎯 Hawk-Eye Insights"
    ])

    with analytics_tab1:
        st.markdown("### 👤 Individual Player Performance")

        if hasattr(st.session_state, 'selected_player_id'):
            player_id = st.session_state.selected_player_id
            tactical_analyzer = st.session_state.tactical_analyzer

            # Get player statistics
            player_stats = tactical_analyzer.get_player_statistics(player_id)

            if player_stats:
                # Display key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Distance", f"{player_stats.get('total_distance', 0):.1f}m")
                with col2:
                    st.metric("Avg Speed", f"{player_stats.get('avg_speed', 0):.1f} m/s")
                with col3:
                    st.metric("Max Speed", f"{player_stats.get('max_speed', 0):.1f} m/s")
                with col4:
                    st.metric("Sprint Count", player_stats.get('sprint_count', 0))

                # Zone distribution
                zone_dist = player_stats.get('zone_distribution', {})
                if zone_dist:
                    st.markdown("#### Zone Distribution")
                    zone_df = pd.DataFrame(list(zone_dist.items()), columns=['Zone', 'Time %'])
                    zone_df['Time %'] = zone_df['Time %'] * 100

                    fig = px.bar(zone_df, x='Zone', y='Time %',
                               title='Time Spent in Different Field Zones')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👤 Select a player in the Player Selection tab to see performance metrics")

    with analytics_tab2:
        st.markdown("### ⚽ Team Tactical Analysis")
        st.info("Team tactical analysis will be displayed here after processing")

        # Placeholder for team analytics
        if st.session_state.results_data:
            df = pd.DataFrame(st.session_state.results_data)

            # Team performance over time
            if 'active_players' in df.columns:
                fig = px.line(df, x='timestamp', y='active_players',
                            title='Team Activity Over Time')
                st.plotly_chart(fig, use_container_width=True)

    with analytics_tab3:
        st.markdown("### 🎯 Hawk-Eye Analysis Results")

        if hasattr(st.session_state, 'hawkeye_results'):
            st.success("✅ Hawk-Eye analysis completed")
            # Display Hawk-Eye specific results
        else:
            st.info("🎯 Enable Hawk-Eye analysis to see homography mapping results")

def show_individual_player_analysis(player_id: int, show_heatmap: bool,
                                  show_movement_trail: bool, show_statistics: bool):
    """Show detailed analysis for individual player"""
    tactical_analyzer = st.session_state.tactical_analyzer

    # Get player statistics
    player_stats = tactical_analyzer.get_player_statistics(player_id)

    if not player_stats:
        st.warning(f"⚠️ No data available for Player {player_id}")
        return

    st.markdown(f"### 📊 Player {player_id} Analysis")

    if show_heatmap:
        st.markdown("#### 🔥 Player Heatmap")
        heatmap_fig = tactical_analyzer.create_tactical_heatmap(player_id)
        st.plotly_chart(heatmap_fig, use_container_width=True)

    if show_statistics:
        st.markdown("#### 📈 Performance Statistics")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Distance", f"{player_stats.get('total_distance', 0):.1f}m")
            st.metric("Average Speed", f"{player_stats.get('avg_speed', 0):.2f} m/s")
        with col2:
            st.metric("Maximum Speed", f"{player_stats.get('max_speed', 0):.2f} m/s")
            st.metric("Sprint Count", player_stats.get('sprint_count', 0))
        with col3:
            st.metric("Time in Attack", f"{player_stats.get('time_in_attacking_third', 0)*100:.1f}%")
            st.metric("Time in Defense", f"{player_stats.get('time_in_defensive_third', 0)*100:.1f}%")

def process_tactical_video(video_file, config):
    """Process video with tactical analysis enabled"""
    try:
        # Use existing video processor with tactical enhancements
        video_processor = st.session_state.video_processor

        # Enable tactical analysis
        video_processor.enable_tactical_analysis = st.session_state.get('enable_tactical_analysis', False)
        video_processor.enable_hawkeye_analysis = st.session_state.get('enable_hawkeye_analysis', False)

        # Process video
        results = video_processor.process_uploaded_video(video_file, config)

        # Store results
        st.session_state.results_data = results
        st.session_state.tactical_results_data = results

        return results

    except Exception as e:
        st.error(f"❌ Error processing tactical video: {str(e)}")
        return []

if __name__ == "__main__":
    main()
