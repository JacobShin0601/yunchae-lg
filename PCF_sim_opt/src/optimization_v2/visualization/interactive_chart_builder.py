"""
Interactive Chart Builder - Advanced Plotly visualizations for optimization results

This module provides comprehensive interactive visualization capabilities:
- 3D Pareto front exploration (carbon, cost, robustness)
- Animated 2D charts with transitions
- Method comparison visualizations
- Scenario analysis charts
- Export to HTML/PNG/PDF

Features:
- Hover tooltips with detailed information
- Interactive filtering and selection
- Drill-down capabilities
- Synchronized multi-chart views
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import json


class InteractiveChartBuilder:
    """
    Advanced interactive chart builder using Plotly

    Provides rich visualization capabilities for optimization results
    with full interactivity and export options.
    """

    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize chart builder

        Args:
            theme: Plotly template theme
                - 'plotly_white': Clean white background
                - 'plotly_dark': Dark theme
                - 'seaborn': Seaborn-style
                - 'ggplot2': ggplot2-style
        """
        self.theme = theme
        self.default_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'optimization_chart',
                'height': 800,
                'width': 1200,
                'scale': 2
            }
        }

    def create_3d_pareto_front(
        self,
        results: List[Dict[str, Any]],
        z_metric: str = 'method',
        title: str = "3D Pareto Front Exploration",
        color_by: str = 'method'
    ) -> go.Figure:
        """
        Create 3D Pareto front visualization

        Args:
            results: List of optimization results
            z_metric: Third dimension metric
                - 'method': Categorical by method type
                - 'robustness_score': Robustness metric
                - 'solver_time': Computation time
                - 'carbon_reduction_pct': Carbon reduction %
            title: Chart title
            color_by: Color encoding variable

        Returns:
            Plotly 3D scatter figure
        """
        if not results:
            raise ValueError("No results provided")

        print(f"🎨 Creating 3D Pareto Front (z_metric: {z_metric})...")

        # Extract data
        carbons = []
        costs = []
        z_values = []
        hover_texts = []
        colors = []
        methods = []

        for i, result in enumerate(results):
            summary = result.get('summary', {})
            carbon = summary.get('total_carbon', 0)
            cost = summary.get('total_cost', 0)

            carbons.append(carbon)
            costs.append(cost)

            # Z-axis value
            if z_metric == 'method':
                method = result.get('method', 'unknown')
                methods.append(method)
                # Encode method as numeric for 3D
                method_map = {
                    'weight_sweep': 0,
                    'epsilon_constraint': 1,
                    'nsga2': 2,
                    'robust': 3,
                    'stochastic': 4
                }
                z_values.append(method_map.get(method, -1))
            elif z_metric == 'robustness_score':
                z_values.append(result.get('robustness_score', 0))
            elif z_metric == 'solver_time':
                z_values.append(result.get('solver_time', 0))
            elif z_metric == 'carbon_reduction_pct':
                z_values.append(summary.get('carbon_reduction_pct', 0))
            else:
                z_values.append(0)

            # Color value
            if color_by == 'method':
                colors.append(result.get('method', 'unknown'))
            elif color_by == 'pareto_rank':
                colors.append(result.get('pareto_rank', 0))
            else:
                colors.append(carbon)

            # Hover text
            hover_text = f"<b>Point {i+1}</b><br>"
            hover_text += f"Carbon: {carbon:.2f}<br>"
            hover_text += f"Cost: ${cost:,.0f}<br>"

            if z_metric == 'method':
                hover_text += f"Method: {result.get('method', 'unknown')}<br>"
            elif z_metric == 'robustness_score':
                hover_text += f"Robustness: {result.get('robustness_score', 0):.3f}<br>"
            elif z_metric == 'solver_time':
                hover_text += f"Time: {result.get('solver_time', 0):.1f}s<br>"

            if 'carbon_reduction_pct' in summary:
                hover_text += f"Carbon Reduction: {summary['carbon_reduction_pct']:.1f}%<br>"

            if 'pareto_rank' in result:
                hover_text += f"Pareto Rank: {result['pareto_rank']}"

            hover_texts.append(hover_text)

        # Create 3D scatter
        fig = go.Figure()

        if color_by == 'method':
            # Group by method for separate traces
            unique_methods = list(set(colors))

            for method in unique_methods:
                mask = [c == method for c in colors]

                fig.add_trace(go.Scatter3d(
                    x=[carbons[i] for i, m in enumerate(mask) if m],
                    y=[costs[i] for i, m in enumerate(mask) if m],
                    z=[z_values[i] for i, m in enumerate(mask) if m],
                    mode='markers',
                    name=method.replace('_', ' ').title(),
                    marker=dict(
                        size=8,
                        opacity=0.8,
                        line=dict(width=1, color='white')
                    ),
                    text=[hover_texts[i] for i, m in enumerate(mask) if m],
                    hovertemplate='%{text}<extra></extra>'
                ))
        else:
            # Single trace with color scale
            fig.add_trace(go.Scatter3d(
                x=carbons,
                y=costs,
                z=z_values,
                mode='markers',
                marker=dict(
                    size=8,
                    color=colors if isinstance(colors[0], (int, float)) else list(range(len(colors))),
                    colorscale='Viridis',
                    showscale=True,
                    opacity=0.8,
                    line=dict(width=1, color='white'),
                    colorbar=dict(title=color_by.replace('_', ' ').title())
                ),
                text=hover_texts,
                hovertemplate='%{text}<extra></extra>'
            ))

        # Update layout
        z_label = z_metric.replace('_', ' ').title()
        if z_metric == 'method':
            z_label = "Method Type"

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis=dict(title='Total Carbon Emissions'),
                yaxis=dict(title='Total Cost ($)'),
                zaxis=dict(title=z_label),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.3)
                )
            ),
            hovermode='closest',
            template=self.theme,
            height=700,
            margin=dict(l=0, r=0, t=50, b=0)
        )

        print(f"   ✅ 3D chart created with {len(results)} points")
        return fig

    def create_animated_pareto_comparison(
        self,
        method_results: Dict[str, List[Dict[str, Any]]],
        title: str = "Animated Pareto Front Comparison"
    ) -> go.Figure:
        """
        Create animated comparison of Pareto fronts across methods

        Args:
            method_results: Dict mapping method name to results list
            title: Chart title

        Returns:
            Plotly figure with animation
        """
        print(f"🎬 Creating animated Pareto comparison...")

        if not method_results:
            raise ValueError("No method results provided")

        # Prepare frames (one per method)
        frames = []
        all_carbons = []
        all_costs = []

        for method_name, results in method_results.items():
            if not results:
                continue

            carbons = [r['summary']['total_carbon'] for r in results]
            costs = [r['summary']['total_cost'] for r in results]

            all_carbons.extend(carbons)
            all_costs.extend(costs)

            # Create frame
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=carbons,
                        y=costs,
                        mode='markers+lines',
                        name=method_name,
                        marker=dict(size=10, opacity=0.7),
                        line=dict(width=2),
                        text=[f"Point {i+1}<br>Carbon: {c:.2f}<br>Cost: ${co:,.0f}"
                              for i, (c, co) in enumerate(zip(carbons, costs))],
                        hovertemplate='%{text}<extra></extra>'
                    )
                ],
                name=method_name,
                layout=go.Layout(
                    annotations=[
                        dict(
                            text=f"<b>{method_name.replace('_', ' ').title()}</b><br>"
                                 f"Points: {len(results)}",
                            xref="paper", yref="paper",
                            x=0.02, y=0.98,
                            showarrow=False,
                            xanchor='left', yanchor='top',
                            bgcolor='rgba(255, 255, 255, 0.8)',
                            bordercolor='black',
                            borderwidth=1,
                            font=dict(size=14)
                        )
                    ]
                )
            )
            frames.append(frame)

        # Create initial figure (first method)
        first_method = list(method_results.keys())[0]
        first_results = method_results[first_method]
        first_carbons = [r['summary']['total_carbon'] for r in first_results]
        first_costs = [r['summary']['total_cost'] for r in first_results]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=first_carbons,
                    y=first_costs,
                    mode='markers+lines',
                    name=first_method,
                    marker=dict(size=10, opacity=0.7),
                    line=dict(width=2),
                    text=[f"Point {i+1}<br>Carbon: {c:.2f}<br>Cost: ${co:,.0f}"
                          for i, (c, co) in enumerate(zip(first_carbons, first_costs))],
                    hovertemplate='%{text}<extra></extra>'
                )
            ],
            frames=frames
        )

        # Add animation controls
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis=dict(
                title='Total Carbon Emissions',
                range=[min(all_carbons) * 0.95, max(all_carbons) * 1.05]
            ),
            yaxis=dict(
                title='Total Cost ($)',
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
                                frame=dict(duration=1000, redraw=True),
                                fromcurrent=True,
                                mode='immediate',
                                transition=dict(duration=500)
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
                    x=0.1, y=1.15,
                    xanchor='left', yanchor='top'
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[
                        dict(
                            args=[
                                [method_name],
                                dict(
                                    frame=dict(duration=0, redraw=True),
                                    mode='immediate',
                                    transition=dict(duration=500)
                                )
                            ],
                            label=method_name.replace('_', ' ').title(),
                            method='animate'
                        )
                        for method_name in method_results.keys()
                    ],
                    x=0.1, y=0,
                    len=0.85,
                    xanchor='left', yanchor='top',
                    pad=dict(t=50, b=10)
                )
            ],
            annotations=[
                dict(
                    text=f"<b>{first_method.replace('_', ' ').title()}</b><br>"
                         f"Points: {len(first_results)}",
                    xref="paper", yref="paper",
                    x=0.02, y=0.98,
                    showarrow=False,
                    xanchor='left', yanchor='top',
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='black',
                    borderwidth=1,
                    font=dict(size=14)
                )
            ],
            template=self.theme,
            height=600
        )

        print(f"   ✅ Animated comparison created for {len(method_results)} methods")
        return fig

    def create_scenario_comparison_chart(
        self,
        scenario_results: Dict[str, List[Dict[str, Any]]],
        chart_type: str = 'box'
    ) -> go.Figure:
        """
        Create scenario comparison visualization

        Args:
            scenario_results: Dict mapping scenario name to results
            chart_type: Type of chart
                - 'box': Box plot
                - 'violin': Violin plot
                - 'strip': Strip plot with jitter

        Returns:
            Plotly figure
        """
        print(f"📊 Creating scenario comparison ({chart_type} plot)...")

        # Prepare data
        data = []
        for scenario_name, results in scenario_results.items():
            for result in results:
                summary = result.get('summary', {})
                data.append({
                    'Scenario': scenario_name,
                    'Carbon': summary.get('total_carbon', 0),
                    'Cost': summary.get('total_cost', 0),
                    'Carbon Reduction %': summary.get('carbon_reduction_pct', 0)
                })

        df = pd.DataFrame(data)

        # Create subplots
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Carbon Emissions', 'Total Cost', 'Carbon Reduction %'),
            horizontal_spacing=0.1
        )

        scenarios = df['Scenario'].unique()

        if chart_type == 'box':
            for scenario in scenarios:
                scenario_df = df[df['Scenario'] == scenario]

                fig.add_trace(
                    go.Box(y=scenario_df['Carbon'], name=scenario, showlegend=True),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Box(y=scenario_df['Cost'], name=scenario, showlegend=False),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Box(y=scenario_df['Carbon Reduction %'], name=scenario, showlegend=False),
                    row=1, col=3
                )

        elif chart_type == 'violin':
            for scenario in scenarios:
                scenario_df = df[df['Scenario'] == scenario]

                fig.add_trace(
                    go.Violin(y=scenario_df['Carbon'], name=scenario, showlegend=True),
                    row=1, col=1
                )
                fig.add_trace(
                    go.Violin(y=scenario_df['Cost'], name=scenario, showlegend=False),
                    row=1, col=2
                )
                fig.add_trace(
                    go.Violin(y=scenario_df['Carbon Reduction %'], name=scenario, showlegend=False),
                    row=1, col=3
                )

        fig.update_yaxes(title_text="Carbon (kg CO2e)", row=1, col=1)
        fig.update_yaxes(title_text="Cost ($)", row=1, col=2)
        fig.update_yaxes(title_text="Reduction (%)", row=1, col=3)

        fig.update_layout(
            title=dict(
                text="Scenario Comparison Analysis",
                x=0.5,
                xanchor='center'
            ),
            template=self.theme,
            height=500,
            showlegend=True
        )

        print(f"   ✅ Scenario comparison created for {len(scenarios)} scenarios")
        return fig

    def create_sensitivity_heatmap(
        self,
        sensitivity_data: Dict[str, Dict[str, float]],
        title: str = "Sensitivity Analysis"
    ) -> go.Figure:
        """
        Create sensitivity analysis heatmap

        Args:
            sensitivity_data: Nested dict {parameter: {metric: sensitivity_value}}
            title: Chart title

        Returns:
            Plotly heatmap figure
        """
        print(f"🔥 Creating sensitivity heatmap...")

        # Convert to matrix format
        parameters = list(sensitivity_data.keys())
        metrics = list(sensitivity_data[parameters[0]].keys()) if parameters else []

        z_values = []
        for metric in metrics:
            row = [sensitivity_data[param].get(metric, 0) for param in parameters]
            z_values.append(row)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=z_values,
            x=parameters,
            y=metrics,
            colorscale='RdBu_r',
            zmid=0,
            text=[[f"{val:.3f}" for val in row] for row in z_values],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Sensitivity")
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            xaxis=dict(title="Parameters", tickangle=-45),
            yaxis=dict(title="Metrics"),
            template=self.theme,
            height=500
        )

        print(f"   ✅ Heatmap created ({len(parameters)}x{len(metrics)})")
        return fig

    def create_tradeoff_surface(
        self,
        results: List[Dict[str, Any]],
        title: str = "Carbon-Cost Trade-off Surface"
    ) -> go.Figure:
        """
        Create 3D surface plot for trade-off visualization

        Args:
            results: Optimization results
            title: Chart title

        Returns:
            Plotly 3D surface figure
        """
        print(f"🏔️ Creating trade-off surface...")

        # Extract data
        carbons = np.array([r['summary']['total_carbon'] for r in results])
        costs = np.array([r['summary']['total_cost'] for r in results])

        # Create grid
        carbon_range = np.linspace(carbons.min(), carbons.max(), 50)
        cost_range = np.linspace(costs.min(), costs.max(), 50)
        carbon_grid, cost_grid = np.meshgrid(carbon_range, cost_range)

        # Interpolate surface (simplified - use actual trade-off function in practice)
        # For demonstration, create a smooth surface through points
        from scipy.interpolate import griddata

        # Use carbon reduction % as z-value
        z_values = np.array([
            r['summary'].get('carbon_reduction_pct', 0) for r in results
        ])

        z_grid = griddata(
            (carbons, costs),
            z_values,
            (carbon_grid, cost_grid),
            method='cubic',
            fill_value=0
        )

        # Create surface
        fig = go.Figure(data=[
            go.Surface(
                x=carbon_grid,
                y=cost_grid,
                z=z_grid,
                colorscale='Viridis',
                colorbar=dict(title="Carbon<br>Reduction %")
            )
        ])

        # Add actual points as scatter
        fig.add_trace(go.Scatter3d(
            x=carbons,
            y=costs,
            z=z_values,
            mode='markers',
            marker=dict(
                size=5,
                color='red',
                symbol='circle'
            ),
            name='Actual Solutions',
            hovertemplate='Carbon: %{x:.2f}<br>Cost: $%{y:,.0f}<br>Reduction: %{z:.1f}%<extra></extra>'
        ))

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center'),
            scene=dict(
                xaxis=dict(title='Carbon Emissions'),
                yaxis=dict(title='Cost ($)'),
                zaxis=dict(title='Carbon Reduction %'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
            ),
            template=self.theme,
            height=700
        )

        print(f"   ✅ Trade-off surface created")
        return fig

    def save_figure(
        self,
        fig: go.Figure,
        filename: str,
        format: str = 'html',
        save_dir: Path = None
    ) -> Path:
        """
        Save figure to file

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            format: Output format ('html', 'png', 'pdf', 'svg', 'json')
            save_dir: Save directory (default: log/charts/)

        Returns:
            Path to saved file
        """
        if save_dir is None:
            save_dir = Path('log/charts')

        save_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = save_dir / f"{filename}_{timestamp}.{format}"

        if format == 'html':
            fig.write_html(str(filepath), config=self.default_config)
        elif format == 'png':
            fig.write_image(str(filepath), width=1200, height=800)
        elif format == 'pdf':
            fig.write_image(str(filepath), width=1200, height=800)
        elif format == 'svg':
            fig.write_image(str(filepath))
        elif format == 'json':
            fig.write_json(str(filepath))
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"💾 Chart saved: {filepath}")
        return filepath

    def create_dashboard(
        self,
        results: List[Dict[str, Any]],
        method_results: Dict[str, List[Dict[str, Any]]] = None,
        scenario_results: Dict[str, List[Dict[str, Any]]] = None
    ) -> go.Figure:
        """
        Create comprehensive dashboard with multiple visualizations

        Args:
            results: All optimization results
            method_results: Results grouped by method
            scenario_results: Results grouped by scenario

        Returns:
            Plotly figure with subplots
        """
        print(f"📊 Creating comprehensive dashboard...")

        # Create 2x2 subplot grid
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Pareto Front',
                'Carbon vs Cost Distribution',
                'Method Comparison',
                'Solution Quality Metrics'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'box'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Pareto Front (top-left)
        carbons = [r['summary']['total_carbon'] for r in results]
        costs = [r['summary']['total_cost'] for r in results]

        fig.add_trace(
            go.Scatter(
                x=carbons,
                y=costs,
                mode='markers',
                marker=dict(size=8, color='blue', opacity=0.6),
                name='Solutions'
            ),
            row=1, col=1
        )

        # 2. Distribution (top-right)
        fig.add_trace(
            go.Box(y=carbons, name='Carbon', marker_color='green'),
            row=1, col=2
        )
        fig.add_trace(
            go.Box(y=costs, name='Cost', marker_color='orange'),
            row=1, col=2
        )

        # 3. Method Comparison (bottom-left)
        if method_results:
            methods = list(method_results.keys())
            avg_carbons = [
                np.mean([r['summary']['total_carbon'] for r in method_results[m]])
                for m in methods
            ]

            fig.add_trace(
                go.Bar(x=methods, y=avg_carbons, name='Avg Carbon'),
                row=2, col=1
            )

        # 4. Quality Metrics (bottom-right)
        solver_times = [r.get('solver_time', 0) for r in results]

        fig.add_trace(
            go.Scatter(
                x=carbons,
                y=solver_times,
                mode='markers',
                marker=dict(size=8, color='red', opacity=0.6),
                name='Solver Time'
            ),
            row=2, col=2
        )

        # Update axes
        fig.update_xaxes(title_text="Carbon", row=1, col=1)
        fig.update_yaxes(title_text="Cost", row=1, col=1)
        fig.update_xaxes(title_text="Method", row=2, col=1)
        fig.update_yaxes(title_text="Avg Carbon", row=2, col=1)
        fig.update_xaxes(title_text="Carbon", row=2, col=2)
        fig.update_yaxes(title_text="Time (s)", row=2, col=2)

        fig.update_layout(
            title=dict(
                text="Optimization Results Dashboard",
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            showlegend=False,
            template=self.theme,
            height=900
        )

        print(f"   ✅ Dashboard created")
        return fig
