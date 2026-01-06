"""
Centralized CSS styling for PCF optimization Streamlit application.
This module consolidates all CSS styles to avoid duplication across pages.
"""

# Common styles used across all pages
COMMON_STYLES = """
<style>
/* Define CSS variables for theme-aware colors */
@media (prefers-color-scheme: dark) {
    :root {
        --primary-color: #4CAF50;
        --secondary-color: #2196F3;
        --warning-color: #ff9800;
        --error-color: #F44336;
        --text-color: #ffffff;
        --text-secondary: #cccccc;
        --text-muted: #888888;
        --bg-primary: #1e1e1e;
        --bg-secondary: #2d2d2d;
        --bg-tertiary: #333333;
        --bg-hover: #3d3d3d;
        --border-color: #333333;
        --border-secondary: #444444;
        --shadow-color: rgba(0, 0, 0, 0.3);
        --info-bg: #2d3d2d;
        --warning-bg: #3d2d2d;
        --error-bg: #4d2d2d;
        --success-bg: #2d3d2d;
    }
}

@media (prefers-color-scheme: light) {
    :root {
        --primary-color: #4CAF50;
        --secondary-color: #2196F3;
        --warning-color: #ff9800;
        --error-color: #F44336;
        --text-color: #212121;
        --text-secondary: #666666;
        --text-muted: #999999;
        --bg-primary: #ffffff;
        --bg-secondary: #f5f5f5;
        --bg-tertiary: #e0e0e0;
        --bg-hover: #eeeeee;
        --border-color: #dddddd;
        --border-secondary: #cccccc;
        --shadow-color: rgba(0, 0, 0, 0.1);
        --info-bg: #e8f5e9;
        --warning-bg: #fff3e0;
        --error-bg: #ffebee;
        --success-bg: #e8f5e9;
    }
}

/* Force light mode colors for Streamlit's light theme */
[data-theme="light"] {
    --primary-color: #4CAF50;
    --secondary-color: #2196F3;
    --warning-color: #ff9800;
    --error-color: #F44336;
    --text-color: #212121;
    --text-secondary: #666666;
    --text-muted: #999999;
    --bg-primary: #ffffff;
    --bg-secondary: #f5f5f5;
    --bg-tertiary: #e0e0e0;
    --bg-hover: #eeeeee;
    --border-color: #dddddd;
    --border-secondary: #cccccc;
    --shadow-color: rgba(0, 0, 0, 0.1);
    --info-bg: #e8f5e9;
    --warning-bg: #fff3e0;
    --error-bg: #ffebee;
    --success-bg: #e8f5e9;
}

/* Force dark mode colors for Streamlit's dark theme */
[data-theme="dark"] {
    --primary-color: #4CAF50;
    --secondary-color: #2196F3;
    --warning-color: #ff9800;
    --error-color: #F44336;
    --text-color: #ffffff;
    --text-secondary: #cccccc;
    --text-muted: #888888;
    --bg-primary: #1e1e1e;
    --bg-secondary: #2d2d2d;
    --bg-tertiary: #333333;
    --bg-hover: #3d3d3d;
    --border-color: #333333;
    --border-secondary: #444444;
    --shadow-color: rgba(0, 0, 0, 0.3);
    --info-bg: #2d3d2d;
    --warning-bg: #3d2d2d;
    --error-bg: #4d2d2d;
    --success-bg: #2d3d2d;
}

.config-section {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}

.result-section {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
}

.config-title {
    color: var(--text-color);
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 15px;
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 5px;
}

.result-title {
    color: var(--text-color);
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 15px;
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 5px;
}

.info-box {
    background-color: var(--info-bg);
    border-left: 4px solid var(--primary-color);
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.warning-box {
    background-color: var(--warning-bg);
    border-left: 4px solid var(--warning-color);
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.error-box {
    background-color: var(--error-bg);
    border-left: 4px solid var(--error-color);
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.success-box {
    background-color: var(--success-bg);
    border-left: 4px solid var(--primary-color);
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
}

.metric-card {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 15px;
    margin: 5px;
    text-align: center;
}

.metric-value {
    font-size: 2rem;
    font-weight: bold;
    color: var(--secondary-color);
}

.metric-label {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 5px;
}

/* Streamlit specific overrides */
.stAlert {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
}

.stTabs [aria-selected="true"] {
    background-color: var(--bg-secondary);
    border-color: var(--primary-color);
}

/* Introduction page specific styles */
.intro-container {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 30px;
    margin-bottom: 20px;
}

.intro-title {
    color: var(--text-color);
    font-size: 2rem;
    font-weight: bold;
    margin-bottom: 20px;
    text-align: center;
}

.intro-description {
    color: var(--text-secondary);
    font-size: 1.1rem;
    line-height: 1.6;
    margin-bottom: 30px;
}

.process-container {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}

.process-title {
    color: var(--text-color);
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 20px;
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 10px;
}

.process-step {
    background-color: var(--bg-secondary);
    border-left: 4px solid var(--secondary-color);
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
}

.process-step-title {
    color: var(--text-color);
    font-size: 1.2rem;
    font-weight: bold;
    margin-bottom: 5px;
}

.process-step-description {
    color: var(--text-secondary);
    font-size: 1rem;
}

/* Sidebar styling */
.sidebar-section {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 15px;
    margin-bottom: 15px;
}

.sidebar-title {
    color: var(--text-color);
    font-size: 1.1rem;
    font-weight: bold;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--primary-color);
    padding-bottom: 5px;
}

/* Data table styling */
.dataframe {
    background-color: var(--bg-primary);
}

.dataframe td, .dataframe th {
    border: 1px solid var(--border-color);
    padding: 8px;
}

.dataframe th {
    background-color: var(--bg-secondary);
    font-weight: bold;
}

/* Button styling */
button[kind="primary"] {
    background-color: var(--primary-color);
    border: none;
}

button[kind="secondary"] {
    background-color: var(--secondary-color);
    border: none;
}

/* Expander styling */
.streamlit-expanderHeader {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: 5px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 20px;
    color: var(--text-secondary);
    font-size: 0.9rem;
    border-top: 1px solid var(--border-color);
    margin-top: 50px;
}
</style>
"""

# Scenario configuration page specific styles
SCENARIO_CONFIG_STYLES = """
<style>
.scenario-container {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}

.scenario-header {
    color: var(--text-color);
    font-size: 1.3rem;
    font-weight: bold;
    margin-bottom: 15px;
    border-bottom: 2px solid var(--warning-color);
    padding-bottom: 5px;
}

.scenario-description {
    color: var(--text-secondary);
    font-size: 1rem;
    line-height: 1.5;
    margin-bottom: 15px;
}

.scenario-config-box {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-secondary);
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}

.tier-config-section {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-secondary);
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}

.tier-title {
    color: var(--warning-color);
    font-size: 1.1rem;
    font-weight: bold;
    margin-bottom: 10px;
}
</style>
"""

# PCF simulation page specific styles
PCF_SIMULATION_STYLES = """
<style>
.simulation-container {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
}

.simulation-title {
    color: var(--text-color);
    font-size: 1.5rem;
    font-weight: bold;
    margin-bottom: 20px;
    border-bottom: 2px solid var(--secondary-color);
    padding-bottom: 10px;
}

.simulation-step {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-secondary);
    border-radius: 5px;
    padding: 15px;
    margin: 10px 0;
}

.simulation-results {
    background-color: var(--bg-primary);
    border: 2px solid var(--primary-color);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
}

.result-highlight {
    background-color: var(--success-bg);
    border-left: 4px solid var(--primary-color);
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
    font-weight: bold;
}

.optimization-section {
    background-color: var(--bg-primary);
    border: 1px solid var(--secondary-color);
    border-radius: 10px;
    padding: 20px;
    margin: 20px 0;
}

.optimization-title {
    color: var(--secondary-color);
    font-size: 1.3rem;
    font-weight: bold;
    margin-bottom: 15px;
}
</style>
"""


# Introduction page specific styles
INTRODUCTION_STYLES = """
<style>
/* Introduction page specific styles */
.intro-section {
    background-color: var(--bg-primary);
    border: 1px solid var(--border-color);
    border-radius: 10px;
    padding: 20px;
    margin: 15px 0;
    box-shadow: 0 2px 8px var(--shadow-color);
}

.intro-section h2 {
    color: var(--primary-color);
    margin-bottom: 15px;
    border-bottom: 2px solid var(--primary-color);
    padding-bottom: 10px;
}

.intro-section p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    line-height: 1.6;
}

.feature-section {
    background-color: var(--bg-secondary);
    border-left: 4px solid var(--secondary-color);
    border-radius: 5px;
    padding: 20px;
    margin: 20px 0;
}

.feature-section h3 {
    color: var(--secondary-color);
    margin-bottom: 15px;
}

.feature-section h4 {
    color: var(--text-color);
    margin-top: 20px;
    margin-bottom: 10px;
    border-bottom: 1px solid var(--border-secondary);
    padding-bottom: 5px;
}

.feature-section ul {
    color: var(--text-secondary);
    padding-left: 25px;
}

.feature-section li {
    margin-bottom: 8px;
    line-height: 1.5;
}

.workflow-section {
    background-color: var(--bg-secondary);
    border-left: 4px solid var(--warning-color);
    border-radius: 5px;
    padding: 20px;
    margin: 20px 0;
}

.workflow-section h3 {
    color: var(--warning-color);
    margin-bottom: 15px;
}

.workflow-section ol {
    color: var(--text-secondary);
    padding-left: 25px;
}

.workflow-section li {
    margin-bottom: 10px;
    line-height: 1.5;
}

.workflow-section strong {
    color: var(--text-color);
}
</style>
"""

def get_page_styles(page_name: str) -> str:
    """
    Get the appropriate styles for a specific page.
    
    Args:
        page_name: Name of the page (e.g., 'introduction', 'cathode_configuration', 
                   'scenario_configuration', 'pcf_simulation')
    
    Returns:
        Combined CSS styles for the page
    """
    page_specific_styles = {
        'introduction': INTRODUCTION_STYLES,
        'scenario_configuration': SCENARIO_CONFIG_STYLES,
        'pcf_simulation': PCF_SIMULATION_STYLES
    }
    
    # Always include common styles
    styles = COMMON_STYLES
    
    # Add page-specific styles if available
    if page_name in page_specific_styles:
        styles += page_specific_styles[page_name]
    
    return styles