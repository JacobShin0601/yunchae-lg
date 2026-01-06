"""
Evolution Animator - Visualization of NSGA-II Evolution Process

This module provides comprehensive visualization of the NSGA-II evolutionary
optimization process, showing how the Pareto front evolves across generations.

Features:
- Animated Pareto front evolution
- Hypervolume improvement tracking
- Diversity metrics visualization
- Generation-by-generation comparison
- Export capabilities (HTML, video)
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import json


class EvolutionAnimator:
    """
    Visualizes NSGA-II evolution process with animated Pareto fronts

    Tracks and visualizes:
    - Pareto front evolution across generations
    - Hypervolume improvement over time
    - Diversity metrics progression
    - Convergence indicators
    """

    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize animator

        Args:
            save_dir: Directory to save animations (default: log/animations/)
        """
        if save_dir is None:
            save_dir = Path('log/animations')

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.pareto_history = []
        self.hypervolume_history = []
        self.diversity_history = []
        self.generation_count = 0

    def load_from_nsga2(self, nsga2_optimizer) -> None:
        """
        Load evolution data from NSGA2Optimizer instance

        Args:
            nsga2_optimizer: NSGA2Optimizer instance with history data
        """
        self.pareto_history = nsga2_optimizer.pareto_history
        self.hypervolume_history = nsga2_optimizer.hypervolume_history
        self.diversity_history = nsga2_optimizer.diversity_history
        self.generation_count = len(self.pareto_history)

        print(f"📊 Loaded evolution data: {self.generation_count} generations")

    def create_animated_pareto_front(
        self,
        title: str = "NSGA-II Evolution: Pareto Front Animation",
        carbon_label: str = "Total Carbon Emissions",
        cost_label: str = "Total Cost ($)"
    ) -> go.Figure:
        """
        Create animated visualization of Pareto front evolution

        Args:
            title: Chart title
            carbon_label: X-axis label
            cost_label: Y-axis label

        Returns:
            Plotly Figure with animation
        """
        if not self.pareto_history:
            raise ValueError("No evolution data available. Call load_from_nsga2() first.")

        print(f"🎬 Creating animated Pareto front...")

        # Prepare data for animation
        frames = []
        all_carbons = []
        all_costs = []

        for gen_idx, generation_data in enumerate(self.pareto_history):
            carbons = [pt['carbon'] for pt in generation_data]
            costs = [pt['cost'] for pt in generation_data]

            all_carbons.extend(carbons)
            all_costs.extend(costs)

            # Create frame for this generation
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=carbons,
                        y=costs,
                        mode='markers',
                        marker=dict(
                            size=10,
                            color=list(range(len(carbons))),
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Point Index")
                        ),
                        text=[f"Point {i+1}<br>Carbon: {c:.2f}<br>Cost: ${co:,.0f}"
                              for i, (c, co) in enumerate(zip(carbons, costs))],
                        hovertemplate='%{text}<extra></extra>'
                    )
                ],
                name=f"Gen {gen_idx + 1}",
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=f"Generation {gen_idx + 1}/{self.generation_count}<br>"
                                 f"Points: {len(carbons)}<br>"
                                 f"Hypervolume: {self.hypervolume_history[gen_idx]:.2e}<br>"
                                 f"Diversity: {self.diversity_history[gen_idx]:.4f}",
                            xref="paper", yref="paper",
                            x=0.02, y=0.98,
                            showarrow=False,
                            xanchor='left', yanchor='top',
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='black',
                            borderwidth=1,
                            font=dict(size=12)
                        )
                    ]
                )
            )
            frames.append(frame)

        # Create initial figure
        initial_carbons = [pt['carbon'] for pt in self.pareto_history[0]]
        initial_costs = [pt['cost'] for pt in self.pareto_history[0]]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=initial_carbons,
                    y=initial_costs,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=list(range(len(initial_carbons))),
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Point Index")
                    ),
                    text=[f"Point {i+1}<br>Carbon: {c:.2f}<br>Cost: ${co:,.0f}"
                          for i, (c, co) in enumerate(zip(initial_carbons, initial_costs))],
                    hovertemplate='%{text}<extra></extra>'
                )
            ],
            frames=frames
        )

        # Add animation controls
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis=dict(
                title=carbon_label,
                range=[min(all_carbons) * 0.95, max(all_carbons) * 1.05]
            ),
            yaxis=dict(
                title=cost_label,
                range=[min(all_costs) * 0.95, max(all_costs) * 1.05]
            ),
            hovermode='closest',
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='▶ Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=500, redraw=True),
                                fromcurrent=True,
                                mode='immediate',
                                transition=dict(duration=300)
                            )]
                        ),
                        dict(
                            label='⏸ Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate',
                                transition=dict(duration=0)
                            )]
                        )
                    ],
                    x=0.1, y=0,
                    xanchor='left', yanchor='top'
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            args=[
                                [f"Gen {i + 1}"],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode='immediate',
                                    transition=dict(duration=0)
                                )
                            ],
                            label=f"Gen {i + 1}",
                            method='animate'
                        )
                        for i in range(self.generation_count)
                    ],
                    x=0.1, y=0,
                    len=0.85,
                    xanchor='left', yanchor='top',
                    pad=dict(t=50, b=10)
                )
            ],
            annotations=[
                dict(
                    text=f"Generation 1/{self.generation_count}<br>"
                         f"Points: {len(initial_carbons)}<br>"
                         f"Hypervolume: {self.hypervolume_history[0]:.2e}<br>"
                         f"Diversity: {self.diversity_history[0]:.4f}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    xanchor='left', yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=12)
                )
            ],
            height=600,
            template='plotly_white'
        )

        print(f"   ✅ Animation created with {self.generation_count} frames")
        return fig

    def create_metrics_dashboard(self) -> go.Figure:
        """
        Create dashboard showing evolution metrics over time

        Returns:
            Plotly Figure with 3 subplots (hypervolume, diversity, front size)
        """
        if not self.pareto_history:
            raise ValueError("No evolution data available.")

        print(f"📊 Creating metrics dashboard...")

        generations = list(range(1, self.generation_count + 1))
        front_sizes = [len(gen_data) for gen_data in self.pareto_history]

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                'Hypervolume (Coverage)',
                'Diversity (Spacing)',
                'Pareto Front Size'
            ),
            vertical_spacing=0.12
        )

        # Hypervolume plot
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=self.hypervolume_history,
                mode='lines+markers',
                name='Hypervolume',
                line=dict(color='blue', width=2),
                marker=dict(size=6),
                hovertemplate='Gen %{x}<br>Hypervolume: %{y:.2e}<extra></extra>'
            ),
            row=1, col=1
        )

        # Diversity plot
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=self.diversity_history,
                mode='lines+markers',
                name='Diversity',
                line=dict(color='green', width=2),
                marker=dict(size=6),
                hovertemplate='Gen %{x}<br>Diversity: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )

        # Front size plot
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=front_sizes,
                mode='lines+markers',
                name='Front Size',
                line=dict(color='red', width=2),
                marker=dict(size=6),
                hovertemplate='Gen %{x}<br>Points: %{y}<extra></extra>'
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Generation", row=3, col=1)
        fig.update_yaxes(title_text="Hypervolume", row=1, col=1)
        fig.update_yaxes(title_text="Spacing", row=2, col=1)
        fig.update_yaxes(title_text="Number of Points", row=3, col=1)

        fig.update_layout(
            title=dict(
                text="NSGA-II Evolution Metrics Dashboard",
                x=0.5,
                xanchor='center'
            ),
            height=900,
            showlegend=False,
            template='plotly_white'
        )

        print(f"   ✅ Dashboard created")
        return fig

    def create_convergence_analysis(self, patience: int = 20) -> go.Figure:
        """
        Create convergence analysis visualization

        Args:
            patience: Window size for convergence check

        Returns:
            Plotly Figure showing hypervolume improvement rates
        """
        if not self.hypervolume_history:
            raise ValueError("No hypervolume history available.")

        print(f"📈 Creating convergence analysis...")

        generations = list(range(1, self.generation_count + 1))

        # Calculate improvement rates
        improvement_rates = []
        for i in range(self.generation_count):
            if i < patience:
                improvement_rates.append(None)
            else:
                window_start = self.hypervolume_history[i - patience]
                window_end = self.hypervolume_history[i]

                if window_start > 0:
                    rate = (window_end - window_start) / window_start * 100
                    improvement_rates.append(rate)
                else:
                    improvement_rates.append(None)

        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Hypervolume Over Time',
                f'Improvement Rate (Rolling {patience} generations)'
            ),
            vertical_spacing=0.15
        )

        # Hypervolume trace
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=self.hypervolume_history,
                mode='lines+markers',
                name='Hypervolume',
                line=dict(color='blue', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

        # Improvement rate trace
        fig.add_trace(
            go.Scatter(
                x=generations,
                y=improvement_rates,
                mode='lines+markers',
                name='Improvement Rate',
                line=dict(color='green', width=2),
                marker=dict(size=6),
                hovertemplate='Gen %{x}<br>Rate: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # Add convergence threshold line
        fig.add_hline(
            y=1.0,
            line_dash="dash",
            line_color="red",
            annotation_text="1% Convergence Threshold",
            row=2, col=1
        )

        fig.add_hline(
            y=-1.0,
            line_dash="dash",
            line_color="red",
            row=2, col=1
        )

        fig.update_xaxes(title_text="Generation", row=2, col=1)
        fig.update_yaxes(title_text="Hypervolume", row=1, col=1)
        fig.update_yaxes(title_text="Improvement Rate (%)", row=2, col=1)

        fig.update_layout(
            title=dict(
                text="NSGA-II Convergence Analysis",
                x=0.5,
                xanchor='center'
            ),
            height=700,
            showlegend=False,
            template='plotly_white'
        )

        print(f"   ✅ Convergence analysis created")
        return fig

    def save_animation(
        self,
        fig: go.Figure,
        filename: str,
        format: str = 'html'
    ) -> Path:
        """
        Save animation to file

        Args:
            fig: Plotly Figure to save
            filename: Output filename (without extension)
            format: Output format ('html' or 'json')

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if format == 'html':
            filepath = self.save_dir / f"{filename}_{timestamp}.html"
            fig.write_html(str(filepath))
        elif format == 'json':
            filepath = self.save_dir / f"{filename}_{timestamp}.json"
            fig.write_json(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"💾 Animation saved: {filepath}")
        return filepath

    def generate_full_report(self) -> Dict[str, Path]:
        """
        Generate complete evolution report with all visualizations

        Returns:
            Dictionary mapping visualization type to saved file path
        """
        if not self.pareto_history:
            raise ValueError("No evolution data available.")

        print(f"\n📊 Generating Full Evolution Report")
        print("=" * 60)

        saved_files = {}

        # 1. Animated Pareto front
        print("\n1️⃣ Creating animated Pareto front...")
        pareto_fig = self.create_animated_pareto_front()
        saved_files['pareto_animation'] = self.save_animation(
            pareto_fig, 'nsga2_pareto_evolution', 'html'
        )

        # 2. Metrics dashboard
        print("\n2️⃣ Creating metrics dashboard...")
        metrics_fig = self.create_metrics_dashboard()
        saved_files['metrics_dashboard'] = self.save_animation(
            metrics_fig, 'nsga2_metrics_dashboard', 'html'
        )

        # 3. Convergence analysis
        print("\n3️⃣ Creating convergence analysis...")
        convergence_fig = self.create_convergence_analysis()
        saved_files['convergence_analysis'] = self.save_animation(
            convergence_fig, 'nsga2_convergence_analysis', 'html'
        )

        print("\n" + "=" * 60)
        print("✅ Full report generated successfully")
        print(f"📁 Files saved to: {self.save_dir}")

        return saved_files

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of the evolution process

        Returns:
            Dictionary with summary statistics
        """
        if not self.pareto_history:
            return {}

        summary = {
            'total_generations': self.generation_count,
            'final_front_size': len(self.pareto_history[-1]),
            'initial_hypervolume': self.hypervolume_history[0],
            'final_hypervolume': self.hypervolume_history[-1],
            'hypervolume_improvement': (
                (self.hypervolume_history[-1] - self.hypervolume_history[0]) /
                self.hypervolume_history[0] * 100
            ),
            'initial_diversity': self.diversity_history[0],
            'final_diversity': self.diversity_history[-1],
            'diversity_change': (
                (self.diversity_history[-1] - self.diversity_history[0]) /
                self.diversity_history[0] * 100
            ),
            'avg_front_size': np.mean([len(gen_data) for gen_data in self.pareto_history]),
            'max_front_size': max([len(gen_data) for gen_data in self.pareto_history]),
            'min_front_size': min([len(gen_data) for gen_data in self.pareto_history])
        }

        return summary

    def print_summary(self) -> None:
        """Print formatted summary of evolution process"""
        summary = self.get_summary_statistics()

        if not summary:
            print("No evolution data available.")
            return

        print("\n" + "=" * 60)
        print("NSGA-II EVOLUTION SUMMARY")
        print("=" * 60)
        print(f"\n📊 General Statistics:")
        print(f"   Total Generations: {summary['total_generations']}")
        print(f"   Final Pareto Front Size: {summary['final_front_size']}")
        print(f"   Average Front Size: {summary['avg_front_size']:.1f}")
        print(f"   Front Size Range: {summary['min_front_size']} - {summary['max_front_size']}")

        print(f"\n📈 Hypervolume (Coverage):")
        print(f"   Initial: {summary['initial_hypervolume']:.2e}")
        print(f"   Final: {summary['final_hypervolume']:.2e}")
        print(f"   Improvement: {summary['hypervolume_improvement']:+.2f}%")

        print(f"\n🎯 Diversity (Spacing):")
        print(f"   Initial: {summary['initial_diversity']:.4f}")
        print(f"   Final: {summary['final_diversity']:.4f}")
        print(f"   Change: {summary['diversity_change']:+.2f}%")
        print(f"   (Lower is better - more uniform distribution)")

        print("=" * 60 + "\n")
