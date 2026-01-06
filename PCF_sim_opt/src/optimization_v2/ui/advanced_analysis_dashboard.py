"""
Advanced Analysis Dashboard - Comprehensive 4-section analysis interface

Provides integrated analysis dashboard for optimization results with:
1. Pareto Front Comparison (3 methods side-by-side)
2. Sensitivity Analysis
3. Scenario Comparison
4. Robust/Stochastic Results

Designed for Streamlit integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import plotly.graph_objects as go
from datetime import datetime

from ..pareto import (
    ParetoMethodComparator,
    ParetoMethodRecommender,
    EpsilonConstraintOptimizer,
    NSGA2Optimizer,
    WeightSweepOptimizer
)
from ..visualization import InteractiveChartBuilder, EvolutionAnimator
from ..robust import RobustStochasticOptimizer, UncertaintyType, RiskMeasure
from ..scenarios import ScenarioLibrary, ScenarioManager


class AdvancedAnalysisDashboard:
    """
    Advanced Analysis Dashboard for comprehensive optimization analysis

    Provides 4 integrated sections:
    1. Pareto Method Comparison
    2. Sensitivity Analysis
    3. Scenario Comparison
    4. Robust/Stochastic Analysis
    """

    def __init__(self):
        """Initialize dashboard"""
        self.chart_builder = InteractiveChartBuilder(theme='plotly_white')
        self.method_comparator = ParetoMethodComparator()
        self.method_recommender = ParetoMethodRecommender()

    def render(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator: Any,
        baseline_case: str = 'case1'
    ) -> None:
        """
        Render complete dashboard

        Args:
            optimization_data: Optimization input data
            cost_calculator: Cost calculator instance
            baseline_case: Baseline case identifier
        """
        st.markdown("# 🔬 Advanced Analysis Dashboard")
        st.markdown("---")

        # Create tabs for 4 sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Pareto Method Comparison",
            "🎯 Sensitivity Analysis",
            "🎭 Scenario Comparison",
            "🛡️ Robust/Stochastic Analysis"
        ])

        with tab1:
            self._render_pareto_comparison(optimization_data, cost_calculator, baseline_case)

        with tab2:
            self._render_sensitivity_analysis(optimization_data, cost_calculator)

        with tab3:
            self._render_scenario_comparison(optimization_data, cost_calculator)

        with tab4:
            self._render_robust_stochastic_analysis(optimization_data, cost_calculator)

    def _render_pareto_comparison(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator: Any,
        baseline_case: str
    ) -> None:
        """Render Section 1: Pareto Method Comparison"""
        st.markdown("## 📊 Pareto Method Comparison")
        st.markdown("""
        Compare three Pareto optimization methods side-by-side:
        - **Weight Sweep**: Fast, reliable weighted sum approach
        - **Epsilon Constraint**: Systematic exploration with constraints
        - **NSGA-II**: Evolutionary algorithm for complex problems
        """)

        # Method recommendation
        with st.expander("🤖 Get Method Recommendation", expanded=False):
            st.markdown("### Problem Characteristics")

            col1, col2, col3 = st.columns(3)

            with col1:
                n_materials = st.number_input(
                    "Number of Materials",
                    min_value=1,
                    max_value=200,
                    value=len(optimization_data.get('material_classification', {})),
                    help="Total number of materials to optimize"
                )

                complexity = st.selectbox(
                    "Problem Complexity",
                    options=['simple', 'medium', 'high', 'very_high'],
                    index=1
                )

            with col2:
                time_limit = st.slider(
                    "Available Time (minutes)",
                    min_value=5,
                    max_value=120,
                    value=30
                )

                has_integer = st.checkbox(
                    "Has Integer Variables",
                    value=False
                )

            with col3:
                priority = st.selectbox(
                    "Priority",
                    options=['quality', 'speed', 'balanced', 'exploration'],
                    index=2
                )

                risk_tolerance = st.selectbox(
                    "Risk Tolerance",
                    options=['conservative', 'moderate', 'aggressive'],
                    index=1
                )

            if st.button("🎯 Get Recommendation", key="method_recommendation"):
                with st.spinner("Analyzing problem characteristics..."):
                    recommendation = self.method_recommender.recommend(
                        problem_characteristics={
                            'n_materials': n_materials,
                            'complexity': complexity,
                            'time_limit_min': time_limit,
                            'has_integer_vars': has_integer
                        },
                        user_preferences={
                            'priority': priority,
                            'risk_tolerance': risk_tolerance
                        }
                    )

                    st.success(f"**Recommended:** {recommendation['recommended_method'].replace('_', ' ').title()}")
                    st.info(f"**Confidence:** {recommendation['confidence']:.1%}")

                    with st.expander("📋 Detailed Reasoning"):
                        st.text(recommendation['reasoning'])

                    if recommendation['alternatives']:
                        st.markdown("**Alternative Methods:**")
                        for alt in recommendation['alternatives']:
                            st.markdown(f"- {alt.replace('_', ' ').title()}")

        # Run comparison
        st.markdown("---")
        st.markdown("### Run Comparison")

        col1, col2 = st.columns([2, 1])

        with col1:
            selected_methods = st.multiselect(
                "Select Methods to Compare",
                options=['weight_sweep', 'epsilon_constraint', 'nsga2'],
                default=['weight_sweep', 'epsilon_constraint'],
                format_func=lambda x: x.replace('_', ' ').title()
            )

        with col2:
            constraint_preset = st.selectbox(
                "Constraint Preset",
                options=['relaxed', 'medium', 'strict'],
                index=1
            )

        if st.button("▶️ Run Comparison", key="run_pareto_comparison", type="primary"):
            if not selected_methods:
                st.error("Please select at least one method")
                return

            method_results = {}

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, method in enumerate(selected_methods):
                status_text.text(f"Running {method.replace('_', ' ').title()}...")

                try:
                    if method == 'weight_sweep':
                        optimizer = WeightSweepOptimizer()
                        results = optimizer.run_adaptive_sweep(
                            optimization_data,
                            cost_calculator,
                            baseline_case,
                            constraint_preset
                        )
                    elif method == 'epsilon_constraint':
                        optimizer = EpsilonConstraintOptimizer()
                        results = optimizer.run_epsilon_sweep(
                            optimization_data,
                            cost_calculator,
                            baseline_case,
                            constraint_preset
                        )
                    elif method == 'nsga2':
                        optimizer = NSGA2Optimizer()
                        results = optimizer.run_nsga2(
                            optimization_data,
                            cost_calculator,
                            baseline_case,
                            constraint_preset
                        )
                    else:
                        continue

                    method_results[method] = results

                except Exception as e:
                    st.error(f"Error in {method}: {str(e)}")
                    continue

                progress_bar.progress((i + 1) / len(selected_methods))

            status_text.text("Comparison complete!")

            if method_results:
                # Store in session state
                st.session_state['pareto_comparison_results'] = method_results

                # Run comparison
                comparison = self.method_comparator.compare_methods(
                    epsilon_results=method_results.get('epsilon_constraint'),
                    nsga2_results=method_results.get('nsga2'),
                    weight_results=method_results.get('weight_sweep')
                )

                # Display results
                st.markdown("---")
                st.markdown("### Comparison Results")

                st.success(f"**Best Method:** {comparison['best_method'].replace('_', ' ').title()}")
                st.info(comparison['recommendation'])

                # Metrics table
                st.markdown("#### Method Metrics")
                metrics_data = []
                for method_name, analysis in comparison['methods'].items():
                    if analysis.get('valid'):
                        metrics_data.append({
                            'Method': method_name.replace('_', ' ').title(),
                            'Points': analysis['n_points'],
                            'Coverage (Hypervolume)': f"{analysis['coverage']:.2e}",
                            'Diversity (Spacing)': f"{analysis['diversity']:.4f}",
                            'Quality Score': f"{analysis['quality_score']:.1f}",
                            'Est. Time (s)': f"{analysis['computation_time']:.1f}"
                        })

                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

                # Visualizations
                st.markdown("#### Visual Comparison")

                viz_col1, viz_col2 = st.columns(2)

                with viz_col1:
                    # 3D Pareto Front
                    all_results = []
                    for method, results in method_results.items():
                        for r in results:
                            r['method'] = method
                            all_results.append(r)

                    fig_3d = self.chart_builder.create_3d_pareto_front(
                        all_results,
                        z_metric='method',
                        title="3D Pareto Front by Method",
                        color_by='method'
                    )
                    st.plotly_chart(fig_3d, use_container_width=True)

                with viz_col2:
                    # Animated comparison
                    fig_animated = self.chart_builder.create_animated_pareto_comparison(
                        method_results,
                        title="Animated Method Comparison"
                    )
                    st.plotly_chart(fig_animated, use_container_width=True)

                # Download results
                if st.button("💾 Download Comparison Report"):
                    report = self.method_comparator.get_summary_report()
                    st.download_button(
                        label="Download TXT Report",
                        data=report,
                        file_name=f"pareto_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )

        # Display cached results if available
        elif 'pareto_comparison_results' in st.session_state:
            st.info("Previous comparison results are available. Click 'Run Comparison' to refresh.")

    def _render_sensitivity_analysis(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator: Any
    ) -> None:
        """Render Section 2: Sensitivity Analysis"""
        st.markdown("## 🎯 Sensitivity Analysis")
        st.markdown("""
        Analyze how changes in parameters affect optimization results.
        """)

        st.info("🚧 Sensitivity Analysis - Coming in next update")

        # Placeholder for future implementation
        with st.expander("Preview: Parameter Sensitivity"):
            st.markdown("""
            **Planned Features:**
            - Emission factor sensitivity
            - Cost parameter sensitivity
            - Constraint relaxation analysis
            - One-at-a-time (OAT) analysis
            - Tornado diagrams
            """)

    def _render_scenario_comparison(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator: Any
    ) -> None:
        """Render Section 3: Scenario Comparison"""
        st.markdown("## 🎭 Scenario Comparison")
        st.markdown("""
        Compare optimization results across different scenarios and constraints.
        """)

        # Load scenario templates
        templates = ScenarioLibrary.get_all_templates()

        st.markdown("### Available Scenario Templates")

        scenario_col1, scenario_col2 = st.columns(2)

        with scenario_col1:
            selected_scenarios = st.multiselect(
                "Select Scenarios to Compare",
                options=list(templates.keys()),
                default=['baseline', 'balanced'],
                format_func=lambda x: x.replace('_', ' ').title()
            )

        with scenario_col2:
            n_samples = st.slider(
                "Monte Carlo Samples (per scenario)",
                min_value=10,
                max_value=200,
                value=50,
                help="Number of samples for stochastic analysis"
            )

        # Display scenario details
        if selected_scenarios:
            st.markdown("#### Scenario Details")

            for scenario_name in selected_scenarios:
                scenario = templates[scenario_name]

                with st.expander(f"📋 {scenario.name}"):
                    st.markdown(f"**Description:** {scenario.description}")
                    st.markdown(f"**Type:** {scenario.scenario_type.value}")
                    st.markdown(f"**Constraint Level:** {scenario.constraint_level.value}")
                    st.markdown(f"**Weights:** Carbon {scenario.carbon_weight:.1%}, Cost {scenario.cost_weight:.1%}")

        if st.button("▶️ Run Scenario Comparison", key="run_scenario_comparison", type="primary"):
            if not selected_scenarios:
                st.error("Please select at least one scenario")
                return

            st.info("🚧 Full scenario comparison implementation coming soon")

            # Placeholder visualization
            st.markdown("#### Scenario Comparison (Preview)")

            # Generate sample data for demonstration
            scenario_results = {}
            for scenario_name in selected_scenarios:
                results = []
                for i in range(10):
                    results.append({
                        'summary': {
                            'total_carbon': np.random.uniform(1000, 2000),
                            'total_cost': np.random.uniform(50000, 100000),
                            'carbon_reduction_pct': np.random.uniform(10, 40)
                        }
                    })
                scenario_results[scenario_name] = results

            fig = self.chart_builder.create_scenario_comparison_chart(
                scenario_results,
                chart_type='box'
            )

            st.plotly_chart(fig, use_container_width=True)

    def _render_robust_stochastic_analysis(
        self,
        optimization_data: Dict[str, Any],
        cost_calculator: Any
    ) -> None:
        """Render Section 4: Robust/Stochastic Analysis"""
        st.markdown("## 🛡️ Robust & Stochastic Analysis")
        st.markdown("""
        Optimization under uncertainty using:
        - **Robust Optimization**: Worst-case scenario protection
        - **Stochastic Optimization**: Probabilistic scenario analysis
        - **Risk Measures**: CVaR, Expected Value, Variance
        """)

        # Analysis type selection
        analysis_type = st.radio(
            "Select Analysis Type",
            options=['Robust', 'Stochastic', 'Hybrid'],
            horizontal=True
        )

        st.markdown("---")

        if analysis_type in ['Robust', 'Hybrid']:
            st.markdown("### Robust Optimization Settings")

            col1, col2 = st.columns(2)

            with col1:
                uncertainty_type = st.selectbox(
                    "Uncertainty Type",
                    options=['Box', 'Ellipsoidal', 'Polyhedral'],
                    index=0
                )

                emission_uncertainty = st.slider(
                    "Emission Factor Uncertainty (%)",
                    min_value=0,
                    max_value=50,
                    value=10,
                    help="Percentage uncertainty in emission factors"
                )

            with col2:
                cost_uncertainty = st.slider(
                    "Cost Uncertainty (%)",
                    min_value=0,
                    max_value=50,
                    value=15,
                    help="Percentage uncertainty in costs"
                )

        if analysis_type in ['Stochastic', 'Hybrid']:
            st.markdown("### Stochastic Optimization Settings")

            col1, col2, col3 = st.columns(3)

            with col1:
                n_scenarios = st.slider(
                    "Number of Scenarios",
                    min_value=10,
                    max_value=500,
                    value=100
                )

            with col2:
                sampling_method = st.selectbox(
                    "Sampling Method",
                    options=['Monte Carlo', 'Latin Hypercube'],
                    index=1
                )

            with col3:
                risk_measure = st.selectbox(
                    "Risk Measure",
                    options=['Expected Value', 'CVaR', 'Worst Case', 'Variance'],
                    index=0
                )

        if st.button("▶️ Run Uncertainty Analysis", key="run_uncertainty_analysis", type="primary"):
            with st.spinner("Running uncertainty analysis..."):
                # Initialize optimizer
                optimizer = RobustStochasticOptimizer()

                # Add uncertain parameters
                optimizer.add_uncertain_parameter(
                    name="emission_factor_scale",
                    nominal_value=1.0,
                    uncertainty_type=UncertaintyType.BOX,
                    min_value=1.0 - emission_uncertainty/100,
                    max_value=1.0 + emission_uncertainty/100
                )

                try:
                    if analysis_type == 'Robust':
                        result = optimizer.optimize_robust(
                            optimization_data,
                            cost_calculator
                        )

                        st.success("✅ Robust optimization complete!")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Worst-Case Carbon", f"{result['summary']['total_carbon']:.2f}")
                        with col2:
                            st.metric("Worst-Case Cost", f"${result['summary']['total_cost']:,.0f}")
                        with col3:
                            st.metric("Robustness Score", f"{result.get('robustness_score', 1.0):.2%}")

                    elif analysis_type == 'Stochastic':
                        # Convert risk measure
                        risk_map = {
                            'Expected Value': RiskMeasure.EXPECTED_VALUE,
                            'CVaR': RiskMeasure.CVAR,
                            'Worst Case': RiskMeasure.WORST_CASE,
                            'Variance': RiskMeasure.VARIANCE
                        }

                        result = optimizer.optimize_stochastic(
                            optimization_data,
                            cost_calculator,
                            n_scenarios=n_scenarios,
                            risk_measure=risk_map[risk_measure]
                        )

                        st.success("✅ Stochastic optimization complete!")

                        stats = result.get('statistics', {})

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Expected Carbon",
                                f"{stats.get('expected_carbon', 0):.2f}",
                                delta=f"±{stats.get('carbon_std', 0):.2f}"
                            )
                        with col2:
                            st.metric(
                                "Expected Cost",
                                f"${stats.get('expected_cost', 0):,.0f}",
                                delta=f"±${stats.get('cost_std', 0):,.0f}"
                            )

                        # Show distribution
                        st.markdown("#### Carbon Distribution")
                        st.info(f"Range: [{stats.get('carbon_min', 0):.2f}, {stats.get('carbon_max', 0):.2f}]")
                        st.info(f"Median: {stats.get('carbon_median', 0):.2f}")

                    else:  # Hybrid
                        st.info("🚧 Hybrid robust-stochastic optimization coming soon")

                except Exception as e:
                    st.error(f"❌ Optimization failed: {str(e)}")
                    st.exception(e)


def render_dashboard(
    optimization_data: Dict[str, Any],
    cost_calculator: Any,
    baseline_case: str = 'case1'
) -> None:
    """
    Convenience function to render dashboard

    Args:
        optimization_data: Optimization input data
        cost_calculator: Cost calculator instance
        baseline_case: Baseline case identifier
    """
    dashboard = AdvancedAnalysisDashboard()
    dashboard.render(optimization_data, cost_calculator, baseline_case)
