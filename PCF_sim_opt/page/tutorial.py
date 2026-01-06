"""
Tutorial Page - 5-Level Progressive Guide

Provides comprehensive step-by-step tutorials for optimization system:
- Level 1 (Beginner): Single objective optimization
- Level 2 (Intermediate): Weight sweep Pareto analysis
- Level 3 (Advanced): Epsilon constraint & NSGA-II
- Level 4 (Expert): Robust-stochastic optimization
- Level 5 (Master): Custom scenarios and advanced analysis

Each level includes:
- Conceptual explanation
- Step-by-step instructions
- Interactive examples
- Best practices
- Common pitfalls to avoid
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime


def tutorial_page():
    """Main tutorial page entry point"""
    st.set_page_config(page_title="Tutorial", page_icon="🎓", layout="wide")

    # Custom CSS
    st.markdown("""
    <style>
    .tutorial-level {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    .beginner { border-left-color: #2ecc71; }
    .intermediate { border-left-color: #3498db; }
    .advanced { border-left-color: #f39c12; }
    .expert { border-left-color: #e74c3c; }
    .master { border-left-color: #9b59b6; }

    .step-box {
        background-color: #ffffff;
        border: 1px solid #ddd;
        padding: 10px;
        margin: 5px 0;
        border-radius: 4px;
    }

    .tip-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }

    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }

    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px;
        margin: 10px 0;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🎓 Optimization Tutorial")
    st.markdown("""
    Welcome to the comprehensive optimization tutorial! This guide will take you from basic concepts
    to advanced optimization techniques in a progressive, hands-on manner.

    **Select your level below to begin learning.**
    """)

    # Progress tracking
    if 'tutorial_progress' not in st.session_state:
        st.session_state.tutorial_progress = {
            'beginner': False,
            'intermediate': False,
            'advanced': False,
            'expert': False,
            'master': False
        }

    # Level overview
    st.markdown("---")
    st.markdown("## 📚 Tutorial Levels")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        status_beginner = "✅" if st.session_state.tutorial_progress['beginner'] else "⭕"
        st.markdown(f"### {status_beginner} Level 1")
        st.markdown("**Beginner**")
        st.caption("Single Objective")

    with col2:
        status_intermediate = "✅" if st.session_state.tutorial_progress['intermediate'] else "⭕"
        st.markdown(f"### {status_intermediate} Level 2")
        st.markdown("**Intermediate**")
        st.caption("Weight Sweep")

    with col3:
        status_advanced = "✅" if st.session_state.tutorial_progress['advanced'] else "⭕"
        st.markdown(f"### {status_advanced} Level 3")
        st.markdown("**Advanced**")
        st.caption("Epsilon & NSGA-II")

    with col4:
        status_expert = "✅" if st.session_state.tutorial_progress['expert'] else "⭕"
        st.markdown(f"### {status_expert} Level 4")
        st.markdown("**Expert**")
        st.caption("Robust/Stochastic")

    with col5:
        status_master = "✅" if st.session_state.tutorial_progress['master'] else "⭕"
        st.markdown(f"### {status_master} Level 5")
        st.markdown("**Master**")
        st.caption("Custom Scenarios")

    # Level selection
    st.markdown("---")

    level = st.selectbox(
        "Select Tutorial Level",
        options=[
            "1️⃣ Level 1: Beginner - Single Objective Optimization",
            "2️⃣ Level 2: Intermediate - Weight Sweep Pareto Analysis",
            "3️⃣ Level 3: Advanced - Epsilon Constraint & NSGA-II",
            "4️⃣ Level 4: Expert - Robust & Stochastic Optimization",
            "5️⃣ Level 5: Master - Custom Scenarios & Advanced Analysis"
        ],
        index=0
    )

    st.markdown("---")

    # Render selected level
    if "Level 1" in level:
        render_level_1_beginner()
    elif "Level 2" in level:
        render_level_2_intermediate()
    elif "Level 3" in level:
        render_level_3_advanced()
    elif "Level 4" in level:
        render_level_4_expert()
    elif "Level 5" in level:
        render_level_5_master()


def render_level_1_beginner():
    """Level 1: Beginner - Single Objective Optimization"""
    st.markdown('<div class="tutorial-level beginner">', unsafe_allow_html=True)
    st.markdown("# 1️⃣ Level 1: Beginner")
    st.markdown("## Single Objective Optimization")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Learning Objectives
    - Understand what optimization is
    - Learn how to minimize carbon emissions
    - Run your first optimization
    - Interpret basic results

    **Estimated Time:** 15-20 minutes
    """)

    # Section 1: Concept
    with st.expander("📖 1. What is Optimization?", expanded=True):
        st.markdown("""
        **Optimization** is the process of finding the best solution from all possible solutions.

        In our context:
        - **Goal**: Minimize carbon emissions from battery production
        - **Decision Variables**: Material sourcing choices (recycled, low-carbon, virgin)
        - **Constraints**: Budget limits, material availability, quality requirements

        Think of it like shopping for groceries with a budget - you want the best quality
        within your price limit, but now we're optimizing for environmental impact!
        """)

        st.image("https://via.placeholder.com/600x300?text=Optimization+Concept+Diagram",
                 caption="Optimization finds the best solution among many options")

    # Section 2: Step-by-Step Guide
    with st.expander("🚀 2. Step-by-Step Guide", expanded=True):
        st.markdown("### Follow these steps to run your first optimization:")

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 1: Load Data**
        1. Go to the "⚙️ 설정 및 실행" tab
        2. Click "데이터 로드" button
        3. Wait for the success message ✅
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 2: Choose Objective**
        1. In the "목적 함수" section, select **"탄소 최소화"** (Minimize Carbon)
        2. This tells the system to focus purely on reducing emissions
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 3: Set Constraints (Optional for beginners)**
        1. For your first run, you can skip adding constraints
        2. The system will use reasonable defaults
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 4: Run Optimization**
        1. Click the big green **"▶️ 최적화 실행"** button
        2. Wait for the optimization to complete (usually 30-60 seconds)
        3. You'll see a success message when done!
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 5: View Results**
        1. Switch to the "📊 결과 분석" tab
        2. Look at:
           - **Total Carbon**: How much CO2 your solution produces
           - **Carbon Reduction %**: How much better than baseline
           - **Material Breakdown**: Which materials changed
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 3: Interactive Example
    with st.expander("💡 3. Interactive Example", expanded=False):
        st.markdown("### Try it yourself!")
        st.markdown("This is a simplified example to understand the concept:")

        # Simple interactive example
        st.markdown("**Scenario**: Choose materials for battery production")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Material Options:**")
            nickel_type = st.radio(
                "Nickel Source",
                options=["Virgin (high carbon)", "Low-carbon", "Recycled (low carbon)"],
                index=0
            )

            cobalt_type = st.radio(
                "Cobalt Source",
                options=["Virgin (high carbon)", "Low-carbon", "Recycled (low carbon)"],
                index=0
            )

        with col2:
            st.markdown("**Carbon Impact:**")

            # Calculate carbon
            carbon_values = {
                "Virgin (high carbon)": 100,
                "Low-carbon": 60,
                "Recycled (low carbon)": 30
            }

            total_carbon = carbon_values[nickel_type] + carbon_values[cobalt_type]
            baseline_carbon = 200  # All virgin

            reduction = (baseline_carbon - total_carbon) / baseline_carbon * 100

            st.metric("Total Carbon", f"{total_carbon} kg CO2e",
                     delta=f"-{reduction:.1f}% vs baseline")

            if total_carbon <= 60:
                st.success("🌟 Excellent! Very low carbon footprint!")
            elif total_carbon <= 120:
                st.info("👍 Good! Moderate carbon reduction")
            else:
                st.warning("⚠️ High carbon - consider recycled materials")

    # Section 4: Best Practices
    with st.expander("✨ 4. Best Practices", expanded=False):
        st.markdown('<div class="tip-box">', unsafe_allow_html=True)
        st.markdown("""
        **💡 Tips for Success:**
        - Start simple: minimize carbon first, add constraints later
        - Check the results carefully: does the solution make sense?
        - Compare to baseline: how much improvement did you achieve?
        - Save good solutions: use the "결과 저장" button
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.markdown("""
        **⚠️ Common Pitfalls:**
        - Don't add too many constraints initially - the problem may become infeasible
        - If optimization fails, try relaxing constraints
        - Pure carbon minimization ignores cost - you may get expensive solutions
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 5: Quiz
    with st.expander("📝 5. Knowledge Check", expanded=False):
        st.markdown("### Test your understanding:")

        q1 = st.radio(
            "Q1: What is the goal of single-objective optimization?",
            options=[
                "Minimize one thing (e.g., carbon emissions)",
                "Maximize profits while minimizing costs",
                "Find multiple good solutions"
            ],
            index=None
        )

        if q1 == "Minimize one thing (e.g., carbon emissions)":
            st.success("✅ Correct! Single-objective optimization focuses on one goal.")
        elif q1 is not None:
            st.error("❌ Try again. Single-objective means one optimization goal.")

        q2 = st.radio(
            "Q2: What are decision variables?",
            options=[
                "The results of optimization",
                "Things we can control (material choices)",
                "Fixed parameters"
            ],
            index=None
        )

        if q2 == "Things we can control (material choices)":
            st.success("✅ Correct! Decision variables are what we optimize.")
        elif q2 is not None:
            st.error("❌ Try again. Think about what the optimizer changes.")

    # Completion
    if st.button("✅ Mark Level 1 as Complete"):
        st.session_state.tutorial_progress['beginner'] = True
        st.balloons()
        st.success("🎉 Congratulations! You've completed Level 1!")
        st.info("👉 Ready for Level 2? Learn about Pareto optimization!")


def render_level_2_intermediate():
    """Level 2: Intermediate - Weight Sweep Pareto Analysis"""
    st.markdown('<div class="tutorial-level intermediate">', unsafe_allow_html=True)
    st.markdown("# 2️⃣ Level 2: Intermediate")
    st.markdown("## Weight Sweep Pareto Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Learning Objectives
    - Understand multi-objective optimization
    - Learn about Pareto frontiers
    - Run weight sweep analysis
    - Interpret trade-off curves

    **Estimated Time:** 25-30 minutes
    """)

    # Section 1: Multi-Objective Concept
    with st.expander("📖 1. Multi-Objective Optimization", expanded=True):
        st.markdown("""
        **The Challenge**: In real life, we often have conflicting goals.

        For battery production:
        - **Goal 1**: Minimize carbon emissions 🌱
        - **Goal 2**: Minimize costs 💰

        **The Problem**: These goals conflict!
        - Lower carbon often means higher cost (recycled materials are expensive)
        - Lower cost often means higher carbon (virgin materials are cheaper)

        **The Solution**: Find the **Pareto Frontier** - the set of best trade-offs.
        """)

        # Simple visualization
        import plotly.graph_objects as go

        # Generate sample Pareto curve
        carbon_vals = np.linspace(1000, 2000, 20)
        cost_vals = 50000 + 30000 * ((2000 - carbon_vals) / 1000) ** 2

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=carbon_vals,
            y=cost_vals,
            mode='markers+lines',
            name='Pareto Frontier',
            marker=dict(size=10, color='blue'),
            line=dict(width=2)
        ))

        fig.update_layout(
            title="Example Pareto Frontier: Carbon vs Cost",
            xaxis_title="Carbon Emissions (kg CO2e)",
            yaxis_title="Total Cost ($)",
            template='plotly_white',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        **Key Insight**: Every point on the Pareto frontier is a valid solution.
        You can't improve one objective without worsening the other.
        """)

    # Section 2: Weight Sweep Method
    with st.expander("🔄 2. How Weight Sweep Works", expanded=True):
        st.markdown("""
        **Weight Sweep Method** explores the Pareto frontier by varying weights:

        ```
        Objective = w1 × Carbon + w2 × Cost

        Where: w1 + w2 = 1
        ```

        **Example Weights:**
        - `w1=1.0, w2=0.0`: Pure carbon minimization
        - `w1=0.7, w2=0.3`: Mostly carbon, some cost consideration
        - `w1=0.5, w2=0.5`: Balanced trade-off
        - `w1=0.3, w2=0.7`: Mostly cost, some carbon consideration
        - `w1=0.0, w2=1.0`: Pure cost minimization
        """)

        # Interactive weight slider
        st.markdown("### 🎛️ Try It: Adjust Weights")

        carbon_weight = st.slider(
            "Carbon Weight (w1)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

        cost_weight = 1.0 - carbon_weight

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Carbon Weight", f"{carbon_weight:.1f}")
        with col2:
            st.metric("Cost Weight", f"{cost_weight:.1f}")
        with col3:
            if carbon_weight > 0.7:
                st.info("🌱 Carbon-focused")
            elif carbon_weight < 0.3:
                st.info("💰 Cost-focused")
            else:
                st.info("⚖️ Balanced")

    # Section 3: Step-by-Step Guide
    with st.expander("🚀 3. Running Weight Sweep", expanded=True):
        st.markdown("### Step-by-Step Instructions:")

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 1: Navigate to Advanced Dashboard**
        1. Go to "🔬 고급 분석" tab
        2. Expand "▶ 8. 📊 통합 분석 대시보드 (Phase 4 - NEW)"
        3. Select the "📊 Pareto Method Comparison" tab
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 2: Select Weight Sweep Method**
        1. In "Select Methods to Compare", choose **"Weight Sweep"**
        2. Optionally add other methods for comparison
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 3: Run the Analysis**
        1. Click **"▶️ Run Comparison"**
        2. Wait for completion (2-5 minutes depending on problem size)
        3. View the Pareto frontier visualization
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 4: Interpret Results**
        1. Look at the Pareto curve: each point is a trade-off
        2. Find your preferred balance point
        3. Download the solution that matches your needs
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 4: Best Practices
    with st.expander("✨ 4. Best Practices", expanded=False):
        st.markdown('<div class="tip-box">', unsafe_allow_html=True)
        st.markdown("""
        **💡 Tips:**
        - Start with 5-10 weight combinations for quick exploration
        - Use adaptive weight scanning for better coverage
        - Check for dominated solutions (should be filtered automatically)
        - Consider stakeholder preferences when choosing final solution
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        **🎯 Decision-Making Guide:**
        - **Conservative approach**: Choose point with 70% carbon weight
        - **Balanced approach**: Choose point with 50-50 weights
        - **Cost-conscious**: Choose point with 70% cost weight
        - **Best Practice**: Present top 3 options to stakeholders
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Completion
    if st.button("✅ Mark Level 2 as Complete"):
        st.session_state.tutorial_progress['intermediate'] = True
        st.balloons()
        st.success("🎉 Great job! You've mastered Pareto analysis!")
        st.info("👉 Ready for Level 3? Learn advanced Pareto methods!")


def render_level_3_advanced():
    """Level 3: Advanced - Epsilon Constraint & NSGA-II"""
    st.markdown('<div class="tutorial-level advanced">', unsafe_allow_html=True)
    st.markdown("# 3️⃣ Level 3: Advanced")
    st.markdown("## Epsilon Constraint & NSGA-II")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Learning Objectives
    - Understand advanced Pareto methods
    - Learn Epsilon Constraint systematic approach
    - Master NSGA-II evolutionary algorithm
    - Compare method performance

    **Estimated Time:** 30-40 minutes
    """)

    # Section 1: Method Comparison
    with st.expander("📊 1. Pareto Methods Comparison", expanded=True):
        st.markdown("""
        ### Three Approaches to Multi-Objective Optimization:
        """)

        comparison_data = pd.DataFrame({
            'Method': ['Weight Sweep', 'Epsilon Constraint', 'NSGA-II'],
            'Speed': ['⚡⚡⚡ Fast', '⚡⚡ Medium', '⚡ Slow'],
            'Coverage': ['⭐⭐ Good', '⭐⭐⭐ Excellent', '⭐⭐⭐ Excellent'],
            'Best For': ['Quick exploration', 'Thorough search', 'Complex problems'],
            'Advantages': [
                'Fast, reliable',
                'Complete coverage',
                'Handles complexity'
            ],
            'Disadvantages': [
                'May miss non-convex regions',
                'Many infeasible solves',
                'Stochastic, needs tuning'
            ]
        })

        st.dataframe(comparison_data, use_container_width=True)

    # Section 2: Epsilon Constraint
    with st.expander("🎯 2. Epsilon Constraint Method", expanded=True):
        st.markdown("""
        ### How It Works:

        Instead of combining objectives, we:
        1. **Optimize** one objective (e.g., minimize carbon)
        2. **Constrain** the other objective (e.g., cost ≤ ε)
        3. **Vary ε** systematically to explore trade-offs

        **Mathematical Formulation:**
        ```
        Minimize: Carbon
        Subject to: Cost ≤ ε
                   (other constraints)
        ```

        **Key Advantage**: Guarantees finding true Pareto points, even in non-convex regions.
        """)

        # Visualization
        st.markdown("### 📈 Visualization:")
        st.info("""
        Epsilon Constraint systematically sweeps through cost budgets:
        - ε₁ = $50,000 → Find minimum carbon with this budget
        - ε₂ = $60,000 → Find minimum carbon with this budget
        - ε₃ = $70,000 → ...and so on
        """)

    # Section 3: NSGA-II
    with st.expander("🧬 3. NSGA-II Evolutionary Algorithm", expanded=True):
        st.markdown("""
        ### Genetic Algorithm Approach:

        **NSGA-II** (Non-dominated Sorting Genetic Algorithm II) uses evolution-inspired optimization:

        **Basic Concept:**
        1. Start with a **population** of random solutions
        2. **Evaluate** each solution's fitness (carbon & cost)
        3. **Select** the best solutions (Pareto ranking)
        4. **Crossover** (combine) and **mutate** to create offspring
        5. **Repeat** for multiple generations

        **Key Features:**
        - Finds multiple Pareto points simultaneously
        - Good for complex, non-linear problems
        - Handles many constraints naturally
        """)

        st.markdown("### 🧬 Evolution Process:")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Generation 1 (Initial)**
            ```
            🔴 Random solutions
            Wide spread
            Many dominated
            ```
            """)

        with col2:
            st.markdown("""
            **Generation 100 (Final)**
            ```
            🟢 Optimized solutions
            Concentrated on Pareto front
            All non-dominated
            ```
            """)

    # Section 4: Practical Guide
    with st.expander("🚀 4. Running Advanced Methods", expanded=True):
        st.markdown("### Using the Method Recommender:")

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 1: Get Recommendation**
        1. In Advanced Dashboard, expand "🤖 Get Method Recommendation"
        2. Enter your problem characteristics:
           - Number of materials
           - Complexity level
           - Available time
        3. Click "🎯 Get Recommendation"
        4. System will suggest best method with confidence score
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 2: Compare Methods**
        1. Select multiple methods (e.g., Weight Sweep + Epsilon + NSGA-II)
        2. Run comparison
        3. View side-by-side results:
           - Coverage (Hypervolume)
           - Diversity (Spacing)
           - Computation time
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        **🎯 Method Selection Guide:**
        - **Small problem (<20 materials), plenty of time**: Epsilon Constraint
        - **Large problem (>50 materials)**: NSGA-II
        - **Quick exploration needed**: Weight Sweep with adaptive scanning
        - **Not sure**: Use the Method Recommender!
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Completion
    if st.button("✅ Mark Level 3 as Complete"):
        st.session_state.tutorial_progress['advanced'] = True
        st.balloons()
        st.success("🎉 Excellent! You're now an advanced user!")
        st.info("👉 Ready for Level 4? Master optimization under uncertainty!")


def render_level_4_expert():
    """Level 4: Expert - Robust & Stochastic Optimization"""
    st.markdown('<div class="tutorial-level expert">', unsafe_allow_html=True)
    st.markdown("# 4️⃣ Level 4: Expert")
    st.markdown("## Robust & Stochastic Optimization")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Learning Objectives
    - Understand optimization under uncertainty
    - Learn robust optimization (worst-case)
    - Master stochastic optimization (probabilistic)
    - Apply risk measures (CVaR)

    **Estimated Time:** 40-50 minutes
    """)

    # Section 1: Uncertainty Concept
    with st.expander("❓ 1. Why Uncertainty Matters", expanded=True):
        st.markdown("""
        ### The Real World is Uncertain!

        In practice, many parameters are not known exactly:
        - **Emission factors** may vary by ±10-20%
        - **Material costs** fluctuate with market prices
        - **Recycling rates** depend on supply chain reliability
        - **Technology performance** varies in real conditions

        **Problem**: Optimal solution for average case may perform poorly in reality.

        **Solution**: Optimize considering uncertainty!
        """)

        # Simple example
        st.markdown("### 📊 Example:")
        st.info("""
        **Scenario**: You optimize for emission factor = 5.0 kg CO2/kg

        **Reality**: Actual factor ranges from 4.5 to 5.5 kg CO2/kg

        **Result**: Your "optimal" solution may violate constraints!
        """)

    # Section 2: Robust Optimization
    with st.expander("🛡️ 2. Robust Optimization", expanded=True):
        st.markdown("""
        ### Worst-Case Protection

        **Approach**: Optimize for the worst possible scenario.

        **Mathematical Concept:**
        ```
        Minimize: max(Carbon under all scenarios)
        ```

        **Uncertainty Sets:**
        - **Box Uncertainty**: Parameter ∈ [min, max]
        - **Ellipsoidal**: Parameters lie within ellipsoid
        - **Polyhedral**: Linear constraints on parameters

        **Advantage**: Guaranteed performance no matter what happens.
        **Disadvantage**: May be too conservative (expensive).
        """)

        st.markdown("### 🎛️ Try It: Uncertainty Level")

        uncertainty_pct = st.slider(
            "Emission Factor Uncertainty (%)",
            min_value=0,
            max_value=30,
            value=10,
            help="How much can the emission factor vary?"
        )

        nominal_emission = 5.0
        min_emission = nominal_emission * (1 - uncertainty_pct/100)
        max_emission = nominal_emission * (1 + uncertainty_pct/100)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nominal", f"{nominal_emission:.2f}")
        with col2:
            st.metric("Worst Case", f"{max_emission:.2f}")
        with col3:
            increase_pct = (max_emission - nominal_emission) / nominal_emission * 100
            st.metric("Impact", f"+{increase_pct:.1f}%")

    # Section 3: Stochastic Optimization
    with st.expander("🎲 3. Stochastic Optimization", expanded=True):
        st.markdown("""
        ### Probabilistic Approach

        **Idea**: Parameters follow probability distributions.

        **Approach**:
        1. Define distributions (e.g., Normal, Uniform)
        2. Generate scenarios via sampling
        3. Optimize expected value or risk measure

        **Sampling Methods:**
        - **Monte Carlo**: Random sampling
        - **Latin Hypercube**: Stratified sampling (better coverage)

        **Risk Measures:**
        - **Expected Value**: E[Carbon]
        - **CVaR (95%)**: Average of worst 5% cases
        - **Variance**: Risk of deviation
        """)

        st.markdown("### 📊 CVaR Example:")

        # Generate sample data
        np.random.seed(42)
        carbon_samples = np.random.normal(1500, 200, 1000)
        carbon_samples = np.sort(carbon_samples)

        # Calculate CVaR (95%)
        alpha = 0.95
        var_index = int(len(carbon_samples) * alpha)
        var_value = carbon_samples[var_index]
        cvar_value = np.mean(carbon_samples[var_index:])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Expected Value", f"{np.mean(carbon_samples):.0f}")
        with col2:
            st.metric("VaR (95%)", f"{var_value:.0f}")
        with col3:
            st.metric("CVaR (95%)", f"{cvar_value:.0f}")

        st.caption("""
        **CVaR interpretation**: On average, in the worst 5% of cases,
        carbon emissions will be {:.0f} kg CO2e.
        """.format(cvar_value))

    # Section 4: Practical Guide
    with st.expander("🚀 4. Running Uncertainty Analysis", expanded=True):
        st.markdown("### Step-by-Step Guide:")

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 1: Navigate to Dashboard**
        1. Go to Advanced Dashboard
        2. Select "🛡️ Robust & Stochastic Analysis" tab
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 2: Choose Analysis Type**
        - **Robust**: For guaranteed worst-case protection
        - **Stochastic**: For probabilistic average-case optimization
        - **Hybrid**: Combination of both
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 3: Configure Uncertainty**
        1. Set uncertainty levels (e.g., ±10% emission factors)
        2. Choose number of scenarios (100+ recommended)
        3. Select sampling method (Latin Hypercube preferred)
        4. Pick risk measure (Expected Value or CVaR)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="step-box">', unsafe_allow_html=True)
        st.markdown("""
        **Step 4: Run & Interpret**
        1. Click "▶️ Run Uncertainty Analysis"
        2. View results:
           - Expected carbon ± standard deviation
           - Confidence intervals
           - Risk metrics (CVaR)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Completion
    if st.button("✅ Mark Level 4 as Complete"):
        st.session_state.tutorial_progress['expert'] = True
        st.balloons()
        st.success("🎉 Outstanding! You're now an expert optimizer!")
        st.info("👉 One more level! Master custom scenarios!")


def render_level_5_master():
    """Level 5: Master - Custom Scenarios & Advanced Analysis"""
    st.markdown('<div class="tutorial-level master">', unsafe_allow_html=True)
    st.markdown("# 5️⃣ Level 5: Master")
    st.markdown("## Custom Scenarios & Advanced Analysis")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Learning Objectives
    - Create custom optimization scenarios
    - Build advanced constraint configurations
    - Perform comprehensive sensitivity analysis
    - Export and document results professionally

    **Estimated Time:** 50-60 minutes
    """)

    # Section 1: Scenario Framework
    with st.expander("🎭 1. Scenario Framework", expanded=True):
        st.markdown("""
        ### Building Custom Scenarios

        **Scenario Components:**
        1. **Element Constraints**: Ni, Co, Li sourcing rules
        2. **RE100 Constraints**: Renewable energy requirements
        3. **Cost Constraints**: Budget limits and premiums
        4. **Objective Weights**: Carbon vs cost priorities

        **Pre-built Templates:**
        - **Baseline**: Business as usual, no constraints
        - **Aggressive**: Maximum recycling (40%+), high low-carbon (20%+)
        - **Balanced**: Moderate requirements across all elements
        - **Cost-Focused**: Minimal requirements, cost priority
        """)

        st.markdown("### 📋 Scenario Template Comparison:")

        template_comparison = pd.DataFrame({
            'Template': ['Baseline', 'Aggressive', 'Balanced', 'Cost-Focused'],
            'Recycling Min': ['0%', '40%', '20%', '0%'],
            'Low-Carbon Min': ['0%', '20%', '10%', '0%'],
            'Carbon Weight': ['100%', '90%', '50%', '10%'],
            'Best For': [
                'Reference case',
                'Sustainability goals',
                'Trade-off exploration',
                'Budget constraints'
            ]
        })

        st.dataframe(template_comparison, use_container_width=True)

    # Section 2: Advanced Constraints
    with st.expander("🔧 2. Advanced Constraint Configuration", expanded=True):
        st.markdown("""
        ### Constraint Types:

        **1. Element-Level Constraints**
        ```python
        Nickel:
          - Recycle ratio: 30-60%
          - Low-carbon ratio: 10-40%
          - Virgin ratio: computed (must sum to 100%)
        ```

        **2. Material-Level Constraints**
        ```python
        Tier1 RE: 50-100%
        Tier2 RE: 30-80%
        ```

        **3. Budget Constraints**
        ```python
        Premium limit: ≤15% above baseline
        Absolute budget: ≤$100,000
        ```
        """)

        st.markdown("### 🎛️ Constraint Builder:")
        st.info("""
        **Tips for Building Constraints:**
        - Start loose, tighten gradually
        - Check feasibility with test runs
        - Use constraint relaxation analysis to find bottlenecks
        - Document rationale for each constraint
        """)

    # Section 3: Sensitivity Analysis
    with st.expander("🎯 3. Advanced Sensitivity Analysis", expanded=True):
        st.markdown("""
        ### What is Sensitivity Analysis?

        **Goal**: Understand how changes in input parameters affect results.

        **Applications:**
        - Identify critical parameters (high sensitivity)
        - Quantify risk from parameter uncertainty
        - Guide data collection priorities
        - Support robust decision-making

        **Methods:**
        1. **One-at-a-Time (OAT)**: Vary one parameter at a time
        2. **Tornado Diagram**: Visualize relative sensitivities
        3. **Sobol Indices**: Global sensitivity analysis
        """)

        st.markdown("### 📊 Example Tornado Diagram:")
        st.caption("(Shows which parameters have biggest impact on carbon emissions)")

        # Simple tornado diagram example
        parameters = ['Emission Factor', 'Recycling Rate', 'RE100 Price', 'Material Cost', 'Quality Constraint']
        sensitivity = [45, 30, 15, 7, 3]

        import plotly.graph_objects as go

        fig = go.Figure(go.Bar(
            x=sensitivity,
            y=parameters,
            orientation='h',
            marker=dict(color=['red', 'orange', 'yellow', 'lightblue', 'lightgreen'])
        ))

        fig.update_layout(
            title="Parameter Sensitivity (% impact on carbon)",
            xaxis_title="Sensitivity (%)",
            yaxis_title="Parameter",
            template='plotly_white',
            height=300
        )

        st.plotly_chart(fig, use_container_width=True)

    # Section 4: Professional Documentation
    with st.expander("📄 4. Professional Documentation", expanded=True):
        st.markdown("""
        ### Best Practices for Documentation:

        **1. Executive Summary**
        - Key findings (1-2 paragraphs)
        - Recommended solution
        - Expected impact (carbon reduction, cost increase)

        **2. Methodology**
        - Optimization method used
        - Constraints applied
        - Assumptions made

        **3. Results**
        - Baseline vs optimized comparison
        - Pareto frontier if multi-objective
        - Sensitivity analysis

        **4. Recommendations**
        - Preferred solution with rationale
        - Implementation considerations
        - Risk assessment

        **5. Appendix**
        - Detailed material breakdown
        - Technical parameters
        - Validation checks
        """)

        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("""
        **📥 Export Checklist:**
        - [ ] Results CSV with all solutions
        - [ ] Pareto front visualization (PNG/HTML)
        - [ ] Comparison report (TXT/PDF)
        - [ ] Sensitivity analysis charts
        - [ ] Scenario configuration (JSON)
        - [ ] Executive summary document
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # Section 5: Capstone Project
    with st.expander("🏆 5. Capstone Project", expanded=True):
        st.markdown("""
        ### Master-Level Challenge

        **Project**: Complete end-to-end optimization analysis

        **Requirements:**
        1. Define a custom scenario with specific business goals
        2. Configure appropriate constraints (minimum 5)
        3. Run 3 different Pareto methods and compare
        4. Perform robust optimization with ±15% uncertainty
        5. Conduct sensitivity analysis on top 3 parameters
        6. Create professional documentation
        7. Present recommendation with supporting evidence

        **Success Criteria:**
        - Feasible solutions found
        - Clear trade-off analysis
        - Risk assessment included
        - Professional presentation quality

        **Time Estimate**: 2-3 hours
        """)

        if st.checkbox("I have completed the capstone project"):
            st.balloons()
            st.success("🏆 Congratulations, Master Optimizer!")
            st.markdown("""
            You have successfully completed all 5 levels!

            **Your Skills:**
            - ✅ Single & Multi-objective optimization
            - ✅ Pareto analysis (3 methods)
            - ✅ Robust & Stochastic optimization
            - ✅ Custom scenario development
            - ✅ Professional documentation

            **What's Next:**
            - Apply these skills to real projects
            - Explore advanced features (parallel computing, custom algorithms)
            - Share knowledge with colleagues
            - Contribute to optimization community
            """)

    # Final Completion
    if st.button("🏆 Mark Level 5 as Complete"):
        st.session_state.tutorial_progress['master'] = True
        st.balloons()
        st.success("🎉🎉🎉 CONGRATULATIONS, MASTER! 🎉🎉🎉")
        st.markdown("""
        You have mastered the entire optimization system!

        **Achievement Unlocked**: 🏆 Master Optimizer

        You are now equipped to tackle the most complex optimization challenges!
        """)


if __name__ == "__main__":
    tutorial_page()
