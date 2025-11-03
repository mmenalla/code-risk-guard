"""
Streamlit UI Utilities for MaintSight Dashboard
Contains all reusable UI components and helper functions
"""
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any, Tuple
import os
from bson.objectid import ObjectId


# ============= STYLING & DISPLAY HELPERS =============

def render_app_header():
    """Render the main application header"""
    st.markdown(
        """
        <div style='background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%); 
                    padding: 30px 20px; 
                    border-radius: 15px; 
                    margin-bottom: 30px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
            <div style='display: flex; align-items: center; justify-content: center; gap: 15px;'>
                <span style='font-size: 48px;'>🤖</span>
                <h1 style='color: white; 
                           margin: 0; 
                           font-size: 42px; 
                           font-weight: 700;
                           text-shadow: 2px 2px 4px rgba(0,0,0,0.2);'>
                    MaintSight
                </h1>
            </div>
            <p style='color: rgba(255,255,255,0.9); 
                      text-align: center; 
                      margin: 10px 0 0 0; 
                      font-size: 16px;'>
                AI-Powered Technical Debt Risk Management
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_risk_info(val: float) -> Dict[str, str]:
    """Get risk styling information based on risk score"""
    if not isinstance(val, (int, float)):
        return {"color": "#E0E0E0", "label": "N/A", "icon": "⚪", "border": "#CCCCCC"}
    if val >= 0.7:
        return {"color": "#FFCDD2", "label": "HIGH", "icon": "🔴", "border": "#E57373"}
    if val >= 0.4:
        return {"color": "#FFF9C4", "label": "MEDIUM", "icon": "🟡", "border": "#FFD54F"}
    return {"color": "#C8E6C9", "label": "LOW", "icon": "🟢", "border": "#81C784"}


def get_status_styling(status: str, risk_info: Dict[str, str]) -> Tuple[str, str, str]:
    """Get card background, border color, and status badge based on status"""
    if status == "approved":
        return "#E8F5E9", "#4CAF50", "✅ APPROVED"
    elif status == "deleted":
        return "#FFEBEE", "#F44336", "🗑️ DELETED"
    else:
        return "#FFFFFF", risk_info["border"], ""


def color_for_risk(val: float) -> Tuple[str, str]:
    """Get color and label for risk value"""
    if val >= 0.7: 
        return ("#FFCDD2", "HIGH")
    if val >= 0.4: 
        return ("#FFF9C4", "MEDIUM")
    return ("#C8E6C9", "LOW")


# ============= TAB 1: TICKET REVIEW COMPONENTS =============

def render_ticket_summary(tickets: List[Dict]) -> None:
    """Render summary dashboard with ticket statistics"""
    valid_risks = [t.get("risk_score") for t in tickets if isinstance(t.get("risk_score"), (int, float))]
    avg_risk = sum(valid_risks) / len(valid_risks) if valid_risks else 0.0
    high_count = sum(1 for r in valid_risks if r >= 0.7)
    medium_count = sum(1 for r in valid_risks if 0.4 <= r < 0.7)
    low_count = sum(1 for r in valid_risks if r < 0.4)

    st.markdown("### 📊 Ticket Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📋 Total Tickets", len(tickets))
    with col2:
        st.metric("📈 Avg Risk", f"{avg_risk:.2f}")
    with col3:
        st.metric("🔴 High Risk", high_count, help="Risk ≥ 0.7")
    with col4:
        st.metric("🟡 Medium Risk", medium_count, help="0.4 ≤ Risk < 0.7")
    with col5:
        st.metric("🟢 Low Risk", low_count, help="Risk < 0.4")
    
    st.markdown("---")


def render_ticket_filters(tickets: List[Dict]) -> Tuple[str, str, str, str]:
    """Render filter controls and return filter values"""
    st.markdown("### 🔍 Filters & Search")
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([3, 2, 2, 2])
    
    with filter_col1:
        search_term = st.text_input("🔎 Search", placeholder="Search by title, module, or repo...", label_visibility="collapsed")
    
    with filter_col2:
        risk_filter = st.selectbox("Risk Level", ["All", "🔴 High (≥0.7)", "🟡 Medium (0.4-0.7)", "🟢 Low (<0.4)"], label_visibility="collapsed")
    
    with filter_col3:
        repos = list(set([t.get("repo_name", "Unknown") for t in tickets]))
        repo_filter = st.selectbox("Repository", ["All"] + sorted(repos), label_visibility="collapsed")
    
    with filter_col4:
        sort_by = st.selectbox("Sort By", ["Risk ↓", "Risk ↑", "Title A-Z", "Title Z-A"], label_visibility="collapsed")
    
    return search_term, risk_filter, repo_filter, sort_by


def apply_ticket_filters(tickets: List[Dict], search_term: str, risk_filter: str, repo_filter: str, sort_by: str) -> List[Dict]:
    """Apply filters and sorting to ticket list"""
    filtered_tickets = tickets.copy()
    
    # Search filter
    if search_term:
        search_lower = search_term.lower()
        filtered_tickets = [t for t in filtered_tickets if 
            search_lower in t.get("title", "").lower() or 
            search_lower in t.get("module", "").lower() or 
            search_lower in t.get("repo_name", "").lower()]
    
    # Risk filter
    if risk_filter != "All":
        if "High" in risk_filter:
            filtered_tickets = [t for t in filtered_tickets if isinstance(t.get("risk_score"), (int, float)) and t.get("risk_score") >= 0.7]
        elif "Medium" in risk_filter:
            filtered_tickets = [t for t in filtered_tickets if isinstance(t.get("risk_score"), (int, float)) and 0.4 <= t.get("risk_score") < 0.7]
        elif "Low" in risk_filter:
            filtered_tickets = [t for t in filtered_tickets if isinstance(t.get("risk_score"), (int, float)) and t.get("risk_score") < 0.4]
    
    # Repo filter
    if repo_filter != "All":
        filtered_tickets = [t for t in filtered_tickets if t.get("repo_name") == repo_filter]
    
    # Sort
    if sort_by == "Risk ↓":
        filtered_tickets.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
    elif sort_by == "Risk ↑":
        filtered_tickets.sort(key=lambda x: x.get("risk_score", 0))
    elif sort_by == "Title A-Z":
        filtered_tickets.sort(key=lambda x: x.get("title", ""))
    elif sort_by == "Title Z-A":
        filtered_tickets.sort(key=lambda x: x.get("title", ""), reverse=True)
    
    return filtered_tickets


def render_ticket_card_html(ticket: Dict, status: str, card_bg: str, border_color: str, status_badge: str, risk_info: Dict) -> str:
    """Generate HTML for ticket card content"""
    repo_name = ticket.get("repo_name", "Unknown")
    prediction_id = ticket.get("prediction_id")
    
    return f"""<div style='background:{card_bg}; border-left: 6px solid {border_color}; border: 1px solid {border_color}40; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>{f'<div style="text-align: right; margin-bottom: 10px;"><span style="background: #E3F2FD; color: #1976D2; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600;">{status_badge}</span></div>' if status_badge else ''}<div style='display: flex; gap: 20px; flex-wrap: wrap; padding: 10px; background: #F8F9FA; border-radius: 8px; margin-bottom: 15px;'><div style='display: flex; align-items: center; gap: 6px;'><span style='font-size: 14px;'>📁</span><span style='font-size: 13px; color: #666;'>Module:</span><code style='background: #E3F2FD; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{ticket.get("module", "N/A")}</code></div><div style='display: flex; align-items: center; gap: 6px;'><span style='font-size: 14px;'>🏢</span><span style='font-size: 13px; color: #666;'>Repo:</span><code style='background: #E3F2FD; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{repo_name}</code></div>{f'<div style="display: flex; align-items: center; gap: 6px;"><span style="font-size: 14px;">🔗</span><span style="font-size: 13px; color: #666;">ID:</span><code style="background: #E3F2FD; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{prediction_id}</code></div>' if prediction_id else ''}</div><div style='background: white; padding: 15px; border-radius: 8px; border: 1px solid #E0E0E0; margin-bottom: 15px;'><div style='font-weight: 600; color: #424242; margin-bottom: 8px; font-size: 13px;'>📝 DESCRIPTION</div><div style='color: #616161; line-height: 1.6; font-size: 14px;'>{ticket.get("description", "No description provided.")}</div></div>{f'<div style="background: #FFF9C4; padding: 12px; border-radius: 8px; border: 1px solid #FBC02D; margin-bottom: 15px;"><div style="font-weight: 600; color: #F57F17; margin-bottom: 6px; font-size: 12px;">⚠️ WHY IS THIS RISKY?</div><div style="color: #F57F17; font-size: 13px; line-height: 1.5;">Recent churn: {ticket.get("context", {}).get("recent_churn", "N/A")} lines • Bug ratio: {ticket.get("context", {}).get("bug_ratio", "N/A")} • Recent PRs: {ticket.get("context", {}).get("recent_prs", "N/A")}</div></div>' if ticket.get('context') else ''}<hr style='border: none; border-top: 2px solid #E0E0E0; margin: 20px 0 15px 0;'><div style='font-weight: 600; color: #424242; margin-bottom: 12px; font-size: 13px;'>⚙️ ACTIONS</div></div>"""


def determine_jira_priority(risk_score: float) -> str:
    """Determine Jira priority based on risk score"""
    if risk_score >= 0.7:
        return "High"
    elif risk_score >= 0.4:
        return "Medium"
    else:
        return "Low"


def generate_jira_labels(module_path: str) -> List[str]:
    """Generate Jira labels from module path"""
    filename = module_path.split("/")[-1] if module_path else "unknown"
    return [
        module_path.replace("/", "-").replace(".", "-") if module_path else "unknown-module",
        filename.replace(".", "-") if filename else "unknown-file",
        "MaintSightGenerated"
    ]


# ============= TAB 2: EDIT RISK SCORES COMPONENTS =============

def render_risk_overview(predictions: List[Dict]) -> None:
    """Render risk overview statistics"""
    total_files = len(predictions)
    valid_risks = [p.get("current_risk") for p in predictions if isinstance(p.get("current_risk"), (int, float))]
    avg_risk = sum(valid_risks) / len(valid_risks) if valid_risks else 0.0
    high_files = sum(1 for r in valid_risks if r >= 0.7)
    med_files = sum(1 for r in valid_risks if 0.4 <= r < 0.7)
    low_files = sum(1 for r in valid_risks if r < 0.4)
    
    st.markdown("### 📊 Risk Overview")
    col_stat1, col_stat2, col_stat3, col_stat4, col_stat5 = st.columns(5)
    with col_stat1:
        st.metric("Total Files", total_files)
    with col_stat2:
        st.metric("Avg Risk", f"{avg_risk:.2f}")
    with col_stat3:
        st.metric("High Risk", high_files, delta=None, delta_color="inverse")
    with col_stat4:
        st.metric("Medium Risk", med_files)
    with col_stat5:
        st.metric("Low Risk", low_files)
    
    st.markdown("---")


def render_risk_browser_controls() -> Tuple[float, float, str, str]:
    """Render risk browser controls and return values"""
    st.markdown("### 🔧 Risk Browser")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2,2,2])
    with col_ctrl1:
        min_display_risk = st.slider("Min risk threshold", 0.0, 1.0, 0.0, 0.01, help="Only show files with risk >= this value")
    with col_ctrl2:
        max_display_risk = st.slider("Max risk threshold", 0.0, 1.0, 1.0, 0.01, help="Only show files with risk <= this value")
    with col_ctrl3:
        sort_mode = st.selectbox("Sort modules by", ["Avg Risk ↓", "Avg Risk ↑", "Name", "Latest First"], index=0)
    
    search_term = st.text_input("🔍 Search files/modules", placeholder="Type to filter by filename or path...", help="Search by filename or module path")
    
    return min_display_risk, max_display_risk, sort_mode, search_term


def render_risk_legend() -> None:
    """Render risk level legend"""
    st.markdown(
        """
        <div style='display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; padding:10px; background:#f8f9fa; border-radius:8px;'>
          <span style='background:#C8E6C9; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:600;'>LOW < 0.4</span>
          <span style='background:#FFF9C4; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:600;'>MEDIUM 0.4–0.69</span>
          <span style='background:#FFCDD2; padding:4px 10px; border-radius:12px; font-size:12px; font-weight:600;'>HIGH ≥ 0.7</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_full_module(pred: Dict) -> str:
    """Get the full module path from prediction record"""
    module = pred.get("module") or pred.get("features", {}).get("filename") or "unknown_module"
    return module.strip()


def get_file_name(full_path: str) -> str:
    """Extract filename from full path, handling edge cases"""
    if not full_path or full_path == "unknown_module":
        return "unknown_module"
    
    parts = full_path.split('/')
    filename = parts[-1] if parts else full_path
    
    if not filename and len(parts) > 1:
        filename = parts[-2]
    
    if not filename or (len(filename) < 3 and '.' not in filename):
        return full_path
    
    return filename


def get_group_module(pred: Dict) -> str:
    """Get the directory path for grouping files"""
    full = get_full_module(pred)
    if '/' in full:
        directory = full[:full.rfind('/')]
        return directory if directory else '(root)'
    return '(root)'


def render_module_progress_bar(avg_risk: float, high_count: int, med_count: int, low_count: int, total_files: int) -> str:
    """Generate HTML for module progress bar and badges"""
    avg_color, _ = color_for_risk(avg_risk)
    bar_width = f"{avg_risk*100:.1f}%"
    progress_html = f"""
      <div style='background:#eee; height:12px; border-radius:6px; overflow:hidden; position:relative;'>
        <div style='background:{avg_color}; width:{bar_width}; height:100%; transition:width .4s;'></div>
      </div>
    """
    badge_html = f"""
      <div style='display:flex; gap:6px; font-size:11px; margin-top:4px;'>
        <span style='background:#FFCDD2; padding:2px 6px; border-radius:10px;'>High: {high_count}</span>
        <span style='background:#FFF9C4; padding:2px 6px; border-radius:10px;'>Med: {med_count}</span>
        <span style='background:#C8E6C9; padding:2px 6px; border-radius:10px;'>Low: {low_count}</span>
        <span style='background:#E0E0E0; padding:2px 6px; border-radius:10px;'>Files: {total_files}</span>
      </div>
    """
    return progress_html + badge_html


# ============= TAB 3: MODEL PERFORMANCE COMPONENTS =============

def render_performance_header() -> None:
    """Render model performance dashboard header"""
    st.markdown("### 📈 Model Performance Dashboard")
    st.markdown("Track model accuracy, correction patterns, and disagreement rates over time.")


def render_agreement_pie_chart(agreement_count: int, disagreement_count: int, agreement_rate: float) -> None:
    """Render agreement vs disagreement as a pie chart"""
    import plotly.graph_objects as go
    
    fig = go.Figure(data=[go.Pie(
        labels=['✅ Agreed (No Feedback)', '⚠️ Disagreed (Gave Feedback)'],
        values=[agreement_count, disagreement_count],
        hole=0.4,
        marker=dict(colors=['#4CAF50', '#FF9800']),
        textinfo='label+percent',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        showlegend=True,
        height=400,
        margin=dict(t=40, b=40, l=40, r=40),
        annotations=[dict(
            text=f'{agreement_rate:.1f}%<br>Agreement',
            x=0.5, y=0.5,
            font_size=20,
            showarrow=False
        )]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Goal: High agreement rate (>75%) means model is accurate")


def render_kpi_metrics(feedback_df, predictions: List[Dict]) -> None:
    """Render key performance indicators"""
    import numpy as np
    
    st.markdown("---")
    st.markdown("#### 🎯 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_feedback = len(feedback_df)
        st.metric(
            "Total Feedback",
            total_feedback,
            help="Number of manager corrections received"
        )
    
    with col2:
        avg_correction = feedback_df['correction'].mean()
        st.metric(
            "Avg Correction",
            f"{avg_correction:.3f}",
            delta=f"{-avg_correction:.3f}" if avg_correction < 0.3 else None,
            delta_color="inverse",
            help="Average absolute difference between prediction and manager correction"
        )
    
    with col3:
        mae = feedback_df['correction'].mean()
        st.metric(
            "MAE",
            f"{mae:.3f}",
            help="Mean Absolute Error - same as average correction"
        )
    
    with col4:
        rmse = np.sqrt((feedback_df['correction'] ** 2).mean())
        st.metric(
            "RMSE",
            f"{rmse:.3f}",
            help="Root Mean Squared Error - penalizes large errors more"
        )
    
    with col5:
        total_predictions = len(predictions)
        disagreement_count = len(feedback_df)
        disagreement_rate = (disagreement_count / total_predictions * 100) if total_predictions > 0 else 0
        
        st.metric(
            "Disagreement Rate",
            f"{disagreement_rate:.1f}%",
            delta=f"{disagreement_count} of {total_predictions}",
            delta_color="inverse",
            help=f"% of predictions that received manager feedback (implicit agreement if no feedback)"
        )


def render_model_metrics_progress(metrics_df) -> None:
    """Render model metrics progress/regression analysis"""
    import plotly.graph_objects as go
    
    st.markdown("---")
    st.markdown("#### 📊 Model Training Progress")
    
    if len(metrics_df) == 0:
        st.info("No training metrics available yet.")
        return
    
    # Sort by timestamp
    metrics_df = metrics_df.sort_values('timestamp')
    
    # Get latest and earliest metrics for comparison
    latest = metrics_df.iloc[-1]
    if len(metrics_df) > 1:
        earliest = metrics_df.iloc[0]
        
        # Calculate improvements
        mae_change = latest['mae'] - earliest['mae']
        mse_change = latest['mse'] - earliest['mse']
        r2_change = latest['r2'] - earliest['r2']
        
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric(
                "Latest MAE",
                f"{latest['mae']:.4f}",
                delta=f"{mae_change:.4f}",
                delta_color="inverse",
                help="Lower is better. Mean Absolute Error"
            )
        
        with col_metric2:
            st.metric(
                "Latest MSE",
                f"{latest['mse']:.4f}",
                delta=f"{mse_change:.4f}",
                delta_color="inverse",
                help="Lower is better. Mean Squared Error"
            )
        
        with col_metric3:
            st.metric(
                "Latest R²",
                f"{latest['r2']:.4f}",
                delta=f"{r2_change:.4f}",
                delta_color="normal",
                help="Higher is better. R-squared score (max 1.0)"
            )
    else:
        # Only one record
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric("Latest MAE", f"{latest['mae']:.4f}", help="Mean Absolute Error")
        
        with col_metric2:
            st.metric("Latest MSE", f"{latest['mse']:.4f}", help="Mean Squared Error")
        
        with col_metric3:
            st.metric("Latest R²", f"{latest['r2']:.4f}", help="R-squared score")
    
    # Progress trend charts
    st.markdown("##### 📈 Training Metrics Over Time")
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # MAE and MSE trends (lower is better)
        fig_error = go.Figure()
        
        fig_error.add_trace(go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['mae'],
            mode='lines+markers',
            name='MAE',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=8)
        ))
        
        fig_error.add_trace(go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['mse'],
            mode='lines+markers',
            name='MSE',
            line=dict(color='#4ECDC4', width=2),
            marker=dict(size=8)
        ))
        
        fig_error.update_layout(
            title="Error Metrics (Lower is Better)",
            xaxis_title="Training Date",
            yaxis_title="Error Value",
            hovermode='x unified',
            height=350,
            margin=dict(t=50, b=40, l=40, r=40)
        )
        
        st.plotly_chart(fig_error, use_container_width=True)
    
    with col_chart2:
        # R² trend (higher is better)
        fig_r2 = go.Figure()
        
        fig_r2.add_trace(go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['r2'],
            mode='lines+markers',
            name='R²',
            line=dict(color='#95E1D3', width=2),
            marker=dict(size=8),
            fill='tozeroy',
            fillcolor='rgba(149, 225, 211, 0.2)'
        ))
        
        fig_r2.update_layout(
            title="R² Score (Higher is Better)",
            xaxis_title="Training Date",
            yaxis_title="R² Score",
            hovermode='x unified',
            height=350,
            margin=dict(t=50, b=40, l=40, r=40),
            yaxis=dict(range=[0, 1])
        )
        
        st.plotly_chart(fig_r2, use_container_width=True)
    
    # Model comparison table
    if len(metrics_df) > 1:
        st.markdown("##### 📋 All Training Runs")
        
        display_df = metrics_df[['timestamp', 'model_name', 'mae', 'mse', 'r2']].copy()
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
        display_df.columns = ['Timestamp', 'Model', 'MAE', 'MSE', 'R²']
        
        # Add improvement indicators
        display_df['MAE Trend'] = ''
        display_df['MSE Trend'] = ''
        display_df['R² Trend'] = ''
        
        for i in range(1, len(display_df)):
            prev_idx = display_df.index[i-1]
            curr_idx = display_df.index[i]
            
            # MAE (lower is better)
            mae_diff = metrics_df.loc[curr_idx, 'mae'] - metrics_df.loc[prev_idx, 'mae']
            display_df.loc[curr_idx, 'MAE Trend'] = '📉' if mae_diff < 0 else '📈'
            
            # MSE (lower is better)
            mse_diff = metrics_df.loc[curr_idx, 'mse'] - metrics_df.loc[prev_idx, 'mse']
            display_df.loc[curr_idx, 'MSE Trend'] = '📉' if mse_diff < 0 else '📈'
            
            # R² (higher is better)
            r2_diff = metrics_df.loc[curr_idx, 'r2'] - metrics_df.loc[prev_idx, 'r2']
            display_df.loc[curr_idx, 'R² Trend'] = '📈' if r2_diff > 0 else '📉'
        
        st.dataframe(
            display_df.sort_values('Timestamp', ascending=False),
            use_container_width=True,
            hide_index=True
        )
        
        # Progress assessment
        if len(metrics_df) >= 2:
            recent_mae = metrics_df.tail(2)['mae'].values
            recent_r2 = metrics_df.tail(2)['r2'].values
            
            if recent_mae[-1] < recent_mae[0] and recent_r2[-1] > recent_r2[0]:
                st.success("✅ **Model is improving!** Latest training shows better metrics.")
            elif recent_mae[-1] > recent_mae[0] and recent_r2[-1] < recent_r2[0]:
                st.warning("⚠️ **Model is regressing.** Latest training shows worse metrics.")
            else:
                st.info("📊 **Model is stable.** Metrics show mixed changes.")


def generate_actionable_insights(feedback_df, predictions: List[Dict], over_predictions: int, under_predictions: int) -> List[str]:
    """Generate actionable insights based on metrics"""
    insights = []
    avg_correction = feedback_df['correction'].mean()
    total_feedback = len(feedback_df)
    total_predictions = len(predictions)
    disagreement_count = len(feedback_df)
    disagreement_rate = (disagreement_count / total_predictions * 100) if total_predictions > 0 else 0
    
    # Insight 1: Overall accuracy
    if avg_correction < 0.15:
        insights.append("✅ **Excellent**: Model is highly accurate (avg correction < 0.15)")
    elif avg_correction < 0.3:
        insights.append("⚠️ **Good**: Model is performing well but can be improved (avg correction < 0.3)")
    else:
        insights.append("❌ **Needs Improvement**: Model has significant errors (avg correction > 0.3) - consider fine-tuning")
    
    # Insight 2: Bias detection
    if over_predictions > under_predictions * 1.5:
        insights.append(f"📈 **Over-prediction Bias**: Model tends to over-estimate risk ({over_predictions} vs {under_predictions})")
    elif under_predictions > over_predictions * 1.5:
        insights.append(f"📉 **Under-prediction Bias**: Model tends to under-estimate risk ({under_predictions} vs {over_predictions})")
    else:
        insights.append("⚖️ **Balanced**: Model shows no significant bias in predictions")
    
    # Insight 3: Disagreement rate
    if disagreement_rate > 50:
        insights.append(f"⚠️ **High Disagreement**: {disagreement_rate:.0f}% of predictions needed correction ({disagreement_count}/{total_predictions}) - recommend retraining")
    elif disagreement_rate > 25:
        insights.append(f"📊 **Moderate Disagreement**: {disagreement_rate:.0f}% needed correction ({disagreement_count}/{total_predictions}) - model learning but needs more feedback")
    elif disagreement_rate > 10:
        insights.append(f"✅ **Good Agreement**: Only {disagreement_rate:.0f}% needed correction ({disagreement_count}/{total_predictions}) - model is well-calibrated")
    else:
        insights.append(f"🎯 **Excellent Agreement**: Only {disagreement_rate:.0f}% needed correction ({disagreement_count}/{total_predictions}) - manager agrees with most predictions")
    
    # Insight 4: Data sufficiency
    if total_feedback < 10:
        insights.append(f"📝 **Need More Data**: Only {total_feedback} corrections - collect at least 20-30 for meaningful fine-tuning")
    elif total_feedback < 30:
        insights.append(f"📊 **Good Sample**: {total_feedback} corrections - sufficient for initial fine-tuning")
    else:
        insights.append(f"✅ **Excellent Dataset**: {total_feedback} corrections - great for model improvement")
    
    return insights


def generate_recommendations(avg_correction: float, disagreement_rate: float, total_feedback: int, over_predictions: int, under_predictions: int) -> List[str]:
    """Generate actionable recommendations"""
    recommendations = []
    
    if avg_correction > 0.3:
        recommendations.append("1. **Trigger Fine-tuning**: Run `airflow dags trigger risk_model_finetune_dag` to improve model with feedback")
    
    if disagreement_rate > 40:
        recommendations.append("2. **Review High-disagreement Cases**: Focus on modules with corrections > 0.3 to understand patterns")
    
    if total_feedback >= 20 and avg_correction > 0.25:
        recommendations.append("3. **Consider Feature Engineering**: High error rate with sufficient data suggests missing features")
    
    if over_predictions > under_predictions * 2:
        recommendations.append("4. **Adjust Model Threshold**: Model is too conservative - consider lowering risk thresholds")
    elif under_predictions > over_predictions * 2:
        recommendations.append("4. **Adjust Model Threshold**: Model is too lenient - consider raising risk thresholds")
    
    if total_feedback < 20:
        recommendations.append("5. **Continue Collecting Feedback**: Review more predictions to build a robust training dataset")
    
    if not recommendations:
        recommendations.append("✅ **Keep Monitoring**: Model is performing well - continue regular reviews")
    
    return recommendations


def render_feature_importance(model_path: str = None) -> None:
    """
    Render XGBoost feature importance visualization
    
    Parameters:
    - model_path: Path to the trained model file (optional, will auto-detect latest)
    """
    import joblib
    from pathlib import Path
    import plotly.graph_objects as go
    
    st.markdown("---")
    st.markdown("#### 🎯 Feature Importance Analysis")
    st.markdown("Which features drive the risk predictions the most?")
    
    try:
        # Auto-detect model directory
        models_dir = Path("dags/src/models/artifacts")
        if not models_dir.exists():
            st.warning("⚠️ Model directory not found. Train a model first.")
            return
        
        # Get all model files (sorted by version number, latest first)
        model_files = sorted(
            models_dir.glob("xgboost_risk_model_v*.pkl"),
            key=lambda x: int(x.stem.split('_v')[-1]),
            reverse=True
        )
        
        if not model_files:
            st.warning("⚠️ No trained models found. Run training DAG first.")
            st.code("airflow dags trigger risk_model_training_dag")
            return
        
        # Model Selection Dropdown
        col_model_select, col_spacer = st.columns([2, 2])
        with col_model_select:
            # Create display names for dropdown
            model_options = {str(f): f.stem for f in model_files}
            model_display_names = [f"{model_options[str(f)]} (Latest)" if i == 0 else model_options[str(f)] 
                                   for i, f in enumerate(model_files)]
            
            selected_display = st.selectbox(
                "📦 Select Model",
                options=model_display_names,
                index=0,  # Default to latest
                help="Choose which trained model to analyze"
            )
            
            # Get the actual file path from selection
            selected_index = model_display_names.index(selected_display)
            model_path = str(model_files[selected_index])
            model_version = model_files[selected_index].stem
        
        # Load the model
        model = joblib.load(model_path)
        
        # Check if it's an XGBoost model
        if not hasattr(model, 'get_booster'):
            st.error("❌ Model is not an XGBoost model. Feature importance only available for XGBoost.")
            return
        
        # Get feature importance
        booster = model.get_booster()
        
        # Try different importance types
        importance_types = {
            'weight': 'Number of times feature used in splits',
            'gain': 'Average gain when feature is used',
            'cover': 'Average coverage of feature when used'
        }
        
        # Select importance type
        col_info, col_selector = st.columns([3, 1])
        with col_selector:
            importance_type = st.selectbox(
                "Importance Type",
                options=list(importance_types.keys()),
                index=1,  # Default to 'gain'
                help="Different ways to measure feature importance"
            )
        
        with col_info:
            st.info(f"**{importance_type.upper()}**: {importance_types[importance_type]}")
            st.caption(f"📦 Model: `{model_version}` | 🔢 Total Features: {len(model.feature_names_in_)}")
        
        # Get importance scores
        importance_dict = booster.get_score(importance_type=importance_type)
        
        if not importance_dict:
            st.warning("⚠️ No feature importance data available in this model.")
            return
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare data for visualization
        features = [f[0] for f in sorted_features]
        scores = [f[1] for f in sorted_features]
        
        # Normalize scores to percentages for better readability
        total_score = sum(scores)
        percentages = [(score / total_score * 100) if total_score > 0 else 0 for score in scores]
        
        # Create two columns for charts
        col_chart, col_table = st.columns([2, 1])
        
        with col_chart:
            # Create horizontal bar chart with Plotly
            fig = go.Figure(go.Bar(
                x=scores,
                y=features,
                orientation='h',
                marker=dict(
                    color=scores,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Importance")
                ),
                text=[f"{p:.1f}%" for p in percentages],
                textposition='auto',
                hovertemplate='<b>%{y}</b><br>Score: %{x:.2f}<br>%{text}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f"Feature Importance ({importance_type.upper()})",
                xaxis_title="Importance Score",
                yaxis_title="Feature",
                height=max(400, len(features) * 25),  # Dynamic height based on feature count
                yaxis={'categoryorder': 'total ascending'},
                showlegend=False,
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col_table:
            st.markdown("**📊 Top 10 Features**")
            
            # Create dataframe for top features
            import pandas as pd
            top_features_df = pd.DataFrame({
                'Feature': features[:10],
                'Score': scores[:10],
                'Share': [f"{p:.1f}%" for p in percentages[:10]]
            })
            
            st.dataframe(
                top_features_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Feature importance insights
            st.markdown("**💡 Insights**")
            
            # Top feature
            top_feature = features[0]
            top_pct = percentages[0]
            st.markdown(f"• **Most Important**: `{top_feature}` ({top_pct:.1f}%)")
            
            # Top 3 combined
            top3_pct = sum(percentages[:3])
            st.markdown(f"• **Top 3 Combined**: {top3_pct:.1f}% of total importance")
            
            # Feature diversity
            if len(features) > 5:
                top5_pct = sum(percentages[:5])
                if top5_pct > 80:
                    st.markdown(f"• ⚠️ **High Concentration**: Top 5 features account for {top5_pct:.1f}% - model heavily relies on few features")
                elif top5_pct < 50:
                    st.markdown(f"• ✅ **Good Diversity**: Top 5 features only {top5_pct:.1f}% - model uses many features")
                else:
                    st.markdown(f"• 📊 **Balanced**: Top 5 features account for {top5_pct:.1f}%")
        
        # Feature descriptions (expandable)
        with st.expander("📖 Feature Descriptions"):
            st.markdown("""
            **Common Features:**
            - `bug_ratio`: Proportion of PRs that were bug fixes (0-1)
            - `churn_per_pr`: Average lines changed per PR
            - `lines_per_pr`: Average total lines modified per PR
            - `lines_per_author`: Average lines per developer
            - `author_concentration`: Inverse of unique authors (1/authors)
            - `add_del_ratio`: Ratio of additions to deletions
            - `deletion_ratio`: Proportion of changes that are deletions
            - `bug_density`: Bugs per line of code
            - `collaboration_complexity`: Authors × churn per PR
            
            See `FEATURES_DOCUMENTATION.md` for complete details.
            """)
        
        # Download option
        st.markdown("---")
        col_download, col_note = st.columns([1, 3])
        with col_download:
            # Prepare CSV
            feature_importance_df = pd.DataFrame({
                'Feature': features,
                'Score': scores,
                'Percentage': percentages
            })
            csv = feature_importance_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"feature_importance_{model_version}.csv",
                mime="text/csv"
            )
        with col_note:
            st.caption("💡 Use feature importance to understand model behavior and improve feature engineering")
        
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: `{model_path}`")
        st.info("Run the training DAG to create a model first.")
    except Exception as e:
        st.error(f"❌ Error loading feature importance: {e}")
        import traceback
        with st.expander("🔍 Debug Info"):
            st.code(traceback.format_exc())
