"""
Tactical Analysis System for Football Player Tracking
Provides tactical mapping, formation analysis, and individual player insights
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, deque
import math

class TacticalAnalyzer:
    """Advanced tactical analysis for football player tracking"""
    
    def __init__(self):
        # Field dimensions (FIFA standard)
        self.field_length = 105.0  # meters
        self.field_width = 68.0    # meters
        
        # Tactical zones
        self.zones = self._define_tactical_zones()
        
        # Player tracking history
        self.player_histories = defaultdict(lambda: {
            'positions': deque(maxlen=100),
            'timestamps': deque(maxlen=100),
            'speeds': deque(maxlen=50),
            'zones': deque(maxlen=50),
            'actions': deque(maxlen=30)
        })
        
        # Formation detection
        self.formation_history = deque(maxlen=20)
        
        # Tactical metrics
        self.team_metrics = {
            'team_1': defaultdict(list),
            'team_2': defaultdict(list)
        }
        
    def _define_tactical_zones(self) -> Dict[str, Dict]:
        """Define tactical zones on the football field"""
        zones = {
            # Defensive zones
            'left_defense': {
                'bounds': (-self.field_length/2, -self.field_length/6, -self.field_width/2, self.field_width/2),
                'color': 'red',
                'priority': 'defensive'
            },
            'right_defense': {
                'bounds': (self.field_length/6, self.field_length/2, -self.field_width/2, self.field_width/2),
                'color': 'red',
                'priority': 'defensive'
            },
            
            # Midfield zones
            'left_midfield': {
                'bounds': (-self.field_length/6, self.field_length/6, -self.field_width/2, 0),
                'color': 'yellow',
                'priority': 'midfield'
            },
            'right_midfield': {
                'bounds': (-self.field_length/6, self.field_length/6, 0, self.field_width/2),
                'color': 'yellow',
                'priority': 'midfield'
            },
            
            # Attacking zones
            'left_attack': {
                'bounds': (self.field_length/6, self.field_length/2, -self.field_width/2, self.field_width/2),
                'color': 'green',
                'priority': 'attacking'
            },
            'right_attack': {
                'bounds': (-self.field_length/2, -self.field_length/6, -self.field_width/2, self.field_width/2),
                'color': 'green',
                'priority': 'attacking'
            },
            
            # Special zones
            'left_penalty_area': {
                'bounds': (-self.field_length/2, -self.field_length/2 + 16.5, -20.15, 20.15),
                'color': 'orange',
                'priority': 'critical'
            },
            'right_penalty_area': {
                'bounds': (self.field_length/2 - 16.5, self.field_length/2, -20.15, 20.15),
                'color': 'orange',
                'priority': 'critical'
            },
            
            'center_circle': {
                'bounds': (-9.15, 9.15, -9.15, 9.15),
                'color': 'blue',
                'priority': 'neutral'
            }
        }
        
        return zones
    
    def update_player_tracking(self, player_data: List[Dict], timestamp: float):
        """Update player tracking data with new frame information"""
        for player in player_data:
            player_id = player.get('track_id', player.get('id'))
            if player_id is None:
                continue
            
            # Get field position
            field_pos = player.get('field_pos')
            if field_pos is None:
                continue
            
            # Update position history
            history = self.player_histories[player_id]
            history['positions'].append(field_pos)
            history['timestamps'].append(timestamp)
            
            # Calculate speed if we have previous position
            if len(history['positions']) >= 2 and len(history['timestamps']) >= 2:
                prev_pos = history['positions'][-2]
                curr_pos = history['positions'][-1]
                prev_time = history['timestamps'][-2]
                curr_time = history['timestamps'][-1]
                
                if curr_time > prev_time:
                    distance = np.linalg.norm(curr_pos - prev_pos)
                    time_diff = curr_time - prev_time
                    speed = distance / time_diff  # m/s
                    history['speeds'].append(speed)
            
            # Determine current zone
            current_zone = self._get_player_zone(field_pos)
            history['zones'].append(current_zone)
            
            # Detect actions based on movement patterns
            action = self._detect_player_action(history)
            if action:
                history['actions'].append({
                    'action': action,
                    'timestamp': timestamp,
                    'position': field_pos.copy()
                })
    
    def _get_player_zone(self, position: np.ndarray) -> str:
        """Determine which tactical zone a player is in"""
        x, y = position[0], position[1]
        
        for zone_name, zone_info in self.zones.items():
            min_x, max_x, min_y, max_y = zone_info['bounds']
            if min_x <= x <= max_x and min_y <= y <= max_y:
                return zone_name
        
        return 'unknown'
    
    def _detect_player_action(self, history: Dict) -> Optional[str]:
        """Detect player actions based on movement patterns"""
        if len(history['speeds']) < 3:
            return None
        
        recent_speeds = list(history['speeds'])[-3:]
        avg_speed = np.mean(recent_speeds)
        speed_variance = np.var(recent_speeds)
        
        # Action detection based on speed patterns
        if avg_speed > 7.0:  # High speed
            return 'sprinting'
        elif avg_speed > 4.0:
            return 'running'
        elif avg_speed < 1.0:
            return 'stationary'
        elif speed_variance > 2.0:  # High variance in speed
            return 'changing_pace'
        else:
            return 'walking'
    
    def analyze_team_formation(self, team_players: List[Dict]) -> Dict:
        """Analyze team formation based on player positions"""
        if len(team_players) < 7:  # Need minimum players for formation analysis
            return {'formation': 'unknown', 'confidence': 0.0}
        
        # Extract positions
        positions = []
        for player in team_players:
            if 'field_pos' in player:
                positions.append(player['field_pos'])
        
        if len(positions) < 7:
            return {'formation': 'unknown', 'confidence': 0.0}
        
        positions = np.array(positions)
        
        # Sort by x-coordinate (defensive to attacking)
        sorted_indices = np.argsort(positions[:, 0])
        sorted_positions = positions[sorted_indices]
        
        # Analyze formation structure
        formation_analysis = self._classify_formation(sorted_positions)
        
        return formation_analysis
    
    def _classify_formation(self, positions: np.ndarray) -> Dict:
        """Classify team formation based on player positions"""
        # Divide field into thirds
        field_thirds = [
            -self.field_length/2 + self.field_length/3,
            -self.field_length/2 + 2*self.field_length/3
        ]
        
        # Count players in each third
        defensive_third = np.sum(positions[:, 0] < field_thirds[0])
        middle_third = np.sum((positions[:, 0] >= field_thirds[0]) & (positions[:, 0] < field_thirds[1]))
        attacking_third = np.sum(positions[:, 0] >= field_thirds[1])
        
        # Classify formation
        formation_string = f"{defensive_third}-{middle_third}-{attacking_third}"
        
        # Common formation patterns
        formation_patterns = {
            '4-4-2': {'confidence': 0.9, 'style': 'balanced'},
            '4-3-3': {'confidence': 0.9, 'style': 'attacking'},
            '3-5-2': {'confidence': 0.8, 'style': 'midfield_heavy'},
            '5-3-2': {'confidence': 0.8, 'style': 'defensive'},
            '4-5-1': {'confidence': 0.8, 'style': 'defensive_midfield'},
            '3-4-3': {'confidence': 0.8, 'style': 'attacking'}
        }
        
        if formation_string in formation_patterns:
            result = formation_patterns[formation_string].copy()
            result['formation'] = formation_string
        else:
            result = {
                'formation': formation_string,
                'confidence': 0.5,
                'style': 'custom'
            }
        
        # Add positional analysis
        result['defensive_line'] = defensive_third
        result['midfield_line'] = middle_third
        result['attacking_line'] = attacking_third
        result['width'] = np.max(positions[:, 1]) - np.min(positions[:, 1])
        result['compactness'] = self._calculate_team_compactness(positions)
        
        return result
    
    def _calculate_team_compactness(self, positions: np.ndarray) -> float:
        """Calculate team compactness (lower values = more compact)"""
        if len(positions) < 2:
            return 0.0
        
        # Calculate average distance between all players
        total_distance = 0
        count = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distance = np.linalg.norm(positions[i] - positions[j])
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    def get_player_statistics(self, player_id: int) -> Dict:
        """Get comprehensive statistics for a specific player"""
        if player_id not in self.player_histories:
            return {}
        
        history = self.player_histories[player_id]
        
        if not history['positions']:
            return {}
        
        positions = np.array(list(history['positions']))
        speeds = list(history['speeds'])
        zones = list(history['zones'])
        actions = list(history['actions'])
        
        stats = {
            'total_distance': self._calculate_total_distance(positions),
            'avg_speed': np.mean(speeds) if speeds else 0.0,
            'max_speed': np.max(speeds) if speeds else 0.0,
            'avg_position': np.mean(positions, axis=0) if len(positions) > 0 else np.array([0, 0]),
            'position_variance': np.var(positions, axis=0) if len(positions) > 0 else np.array([0, 0]),
            'zone_distribution': self._calculate_zone_distribution(zones),
            'action_distribution': self._calculate_action_distribution(actions),
            'heat_map_data': positions.tolist() if len(positions) > 0 else [],
            'sprint_count': len([s for s in speeds if s > 7.0]),
            'time_in_attacking_third': self._calculate_time_in_zone(zones, ['left_attack', 'right_attack']),
            'time_in_defensive_third': self._calculate_time_in_zone(zones, ['left_defense', 'right_defense'])
        }
        
        return stats
    
    def _calculate_total_distance(self, positions: np.ndarray) -> float:
        """Calculate total distance covered by player"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[i-1])
            total_distance += distance
        
        return total_distance
    
    def _calculate_zone_distribution(self, zones: List[str]) -> Dict[str, float]:
        """Calculate percentage of time spent in each zone"""
        if not zones:
            return {}
        
        zone_counts = defaultdict(int)
        for zone in zones:
            zone_counts[zone] += 1
        
        total_count = len(zones)
        return {zone: count / total_count for zone, count in zone_counts.items()}
    
    def _calculate_action_distribution(self, actions: List[Dict]) -> Dict[str, int]:
        """Calculate distribution of player actions"""
        if not actions:
            return {}
        
        action_counts = defaultdict(int)
        for action_data in actions:
            action_counts[action_data['action']] += 1
        
        return dict(action_counts)
    
    def _calculate_time_in_zone(self, zones: List[str], target_zones: List[str]) -> float:
        """Calculate percentage of time spent in specific zones"""
        if not zones:
            return 0.0
        
        time_in_zones = sum(1 for zone in zones if zone in target_zones)
        return time_in_zones / len(zones)
    
    def create_tactical_heatmap(self, player_id: int) -> go.Figure:
        """Create tactical heatmap for specific player"""
        stats = self.get_player_statistics(player_id)
        
        if not stats or not stats['heat_map_data']:
            return go.Figure()
        
        positions = np.array(stats['heat_map_data'])
        
        # Create heatmap
        fig = go.Figure()
        
        # Add field background
        self._add_field_background(fig)
        
        # Add player heatmap
        fig.add_trace(go.Histogram2d(
            x=positions[:, 0],
            y=positions[:, 1],
            nbinsx=20,
            nbinsy=15,
            colorscale='Reds',
            opacity=0.7,
            name=f'Player {player_id} Heatmap'
        ))
        
        # Add average position
        avg_pos = stats['avg_position']
        fig.add_trace(go.Scatter(
            x=[avg_pos[0]],
            y=[avg_pos[1]],
            mode='markers',
            marker=dict(size=15, color='blue', symbol='star'),
            name='Average Position'
        ))
        
        fig.update_layout(
            title=f'Tactical Heatmap - Player {player_id}',
            xaxis_title='Field Length (m)',
            yaxis_title='Field Width (m)',
            width=800,
            height=600
        )
        
        return fig
    
    def create_formation_visualization(self, team_players: List[Dict], team_name: str = "Team") -> go.Figure:
        """Create formation visualization"""
        fig = go.Figure()
        
        # Add field background
        self._add_field_background(fig)
        
        # Add players
        for player in team_players:
            if 'field_pos' in player:
                pos = player['field_pos']
                player_id = player.get('track_id', player.get('id', 'Unknown'))
                
                fig.add_trace(go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode='markers+text',
                    marker=dict(size=12, color='red'),
                    text=[str(player_id)],
                    textposition='middle center',
                    name=f'Player {player_id}'
                ))
        
        # Analyze formation
        formation_analysis = self.analyze_team_formation(team_players)
        
        fig.update_layout(
            title=f'{team_name} Formation: {formation_analysis.get("formation", "Unknown")} '
                  f'(Confidence: {formation_analysis.get("confidence", 0):.2f})',
            xaxis_title='Field Length (m)',
            yaxis_title='Field Width (m)',
            width=800,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _add_field_background(self, fig: go.Figure):
        """Add football field background to plotly figure"""
        # Field boundary
        fig.add_shape(
            type="rect",
            x0=-self.field_length/2, y0=-self.field_width/2,
            x1=self.field_length/2, y1=self.field_width/2,
            line=dict(color="white", width=2),
            fillcolor="green",
            opacity=0.3
        )
        
        # Center line
        fig.add_shape(
            type="line",
            x0=0, y0=-self.field_width/2,
            x1=0, y1=self.field_width/2,
            line=dict(color="white", width=2)
        )
        
        # Center circle
        fig.add_shape(
            type="circle",
            x0=-9.15, y0=-9.15,
            x1=9.15, y1=9.15,
            line=dict(color="white", width=2)
        )
        
        # Penalty areas
        fig.add_shape(
            type="rect",
            x0=-self.field_length/2, y0=-20.15,
            x1=-self.field_length/2 + 16.5, y1=20.15,
            line=dict(color="white", width=2)
        )
        
        fig.add_shape(
            type="rect",
            x0=self.field_length/2 - 16.5, y0=-20.15,
            x1=self.field_length/2, y1=20.15,
            line=dict(color="white", width=2)
        )
        
        # Set axis properties
        fig.update_xaxes(range=[-self.field_length/2 - 5, self.field_length/2 + 5])
        fig.update_yaxes(range=[-self.field_width/2 - 5, self.field_width/2 + 5])
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
