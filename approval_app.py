"""
MaintSight - AI-Powered Technical Debt Risk Management Dashboard
Main Streamlit application entry point
"""
import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
from datetime import datetime
import pandas as pd
import numpy as np

from dags.src.data.save_incremental_labeled_data import log_human_feedback
from dags.src.jira.jira_client import JiraClient
from dags.src.utils.config import Config
import st_utils as stu

load_dotenv()

# ============= CONFIGURATION =============

MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:admin@localhost:27017/risk_model_db?authSource=admin")
DB_NAME = os.getenv("MONGO_DB", "risk_model_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "ticket_drafts")
PREDICTIONS_COLLECTION = os.getenv("MONGO_PREDICTIONS_COLLECTION", "model_predictions")
FEEDBACK_COLLECTION = os.getenv("MONGO_FEEDBACK_COLLECTION", "risk_feedback")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
pred_collection = client[DB_NAME][PREDICTIONS_COLLECTION]

jira_client = JiraClient()

# ============= PAGE SETUP =============

st.set_page_config(page_title="MaintSight - AI Risk Model Dashboard", layout="wide", page_icon="🤖")

stu.render_app_header()

tab1, tab2, tab3 = st.tabs(["🎫 Ticket Review", "📊 Edit Risk Scores", "📈 Model Performance"])


# ============= TAB 1: TICKET REVIEW =============

with tab1:
    tickets = list(collection.find({"is_deleted": False}))

    # Initialize session state
    if "ticket_status" not in st.session_state:
        st.session_state.ticket_status = {}
    if "selected_tickets" not in st.session_state:
        st.session_state.selected_tickets = set()

    # Summary Dashboard
    if tickets:
        stu.render_ticket_summary(tickets)

        # Filters & Search
        search_term, risk_filter, repo_filter, sort_by = stu.render_ticket_filters(tickets)

        # Apply filters
        filtered_tickets = stu.apply_ticket_filters(tickets, search_term, risk_filter, repo_filter, sort_by)

        st.markdown(f"**Showing {len(filtered_tickets)} of {len(tickets)} tickets**")
        st.markdown("---")

    if not tickets:
        st.info("✅ No pending tickets to review.")
    elif not filtered_tickets:
        st.warning("🔍 No tickets match your filters. Try adjusting the search criteria.")
    else:
        # Ticket Cards
        st.markdown("### 🎫 Tickets")
        
        for ticket in filtered_tickets:
            ticket_id = str(ticket["_id"])
            status = st.session_state.ticket_status.get(ticket_id, None)
            repo_name = ticket.get("repo_name", "Unknown")
            prediction_id = ticket.get("prediction_id")
            risk_score = ticket.get("risk_score", 0.0)
            
            risk_info = stu.get_risk_info(risk_score)
            card_bg, border_color, status_badge = stu.get_status_styling(status, risk_info)

            # Create expander title with key info
            expander_label = f"{risk_info['icon']} {ticket.get('title', 'Untitled')} - {risk_info['label']}: {float(risk_score):.2f}"
            
            # Use expander for collapsible tickets
            with st.expander(expander_label, expanded=False):
                with st.container():
                    # Render card HTML
                    card_html = stu.render_ticket_card_html(ticket, status, card_bg, border_color, status_badge, risk_info)
                    st.markdown(card_html, unsafe_allow_html=True)

                    # Action buttons
                    action_cols = st.columns([2, 2, 1, 1])
                
                with action_cols[0]:
                    new_score = st.number_input(
                        "Adjust Risk Score",
                        key=f"risk_score_{ticket_id}",
                        value=float(risk_score),
                        step=0.05,
                        min_value=0.0,
                        max_value=1.0,
                        help="Modify the risk score based on your assessment"
                    )
                
                with action_cols[1]:
                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    if st.button("💾 Update Risk", key=f"update_{ticket_id}", width="stretch"):
                        log_human_feedback(
                            module=ticket["module"],
                            repo_name=repo_name,
                            predicted_risk=risk_score,
                            manager_risk=new_score,
                            prediction_id=str(prediction_id) if prediction_id else None,
                            user_id=os.getenv("CURRENT_USER", "manager_ui"),
                        )
                        collection.update_one(
                            {"_id": ObjectId(ticket["_id"])},
                            {"$set": {"risk_score": new_score}},
                        )
                        st.success(f"✅ Updated to {new_score:.2f}")
                        st.rerun()
                
                with action_cols[2]:
                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    if st.button("📤 Send to Jira", key=f"approve_{ticket_id}", width="stretch"):
                        try:
                            priority = stu.determine_jira_priority(risk_score)
                            module_path = ticket.get("module", "")
                            labels = stu.generate_jira_labels(module_path)
                            
                            jira_issue = jira_client.create_ticket(
                                title=ticket["title"],
                                description=ticket["description"],
                                project_key=os.getenv("JIRA_PROJECT_KEY", "TECH"),
                                priority=priority,
                                labels=labels
                            )
                            collection.update_one(
                                {"_id": ObjectId(ticket["_id"])},
                                {"$set": {
                                    "is_deleted": True,
                                    "jira_key": jira_issue["key"],
                                    "jira_url": jira_issue["url"],
                                    "approved_at": datetime.utcnow()
                                }},
                            )
                            st.session_state.ticket_status[ticket_id] = "approved"
                            st.success(f"🎉 Jira ticket created successfully!")
                            st.markdown(f"**[{jira_issue['key']}]({jira_issue['url']})** - Click to open in Jira")
                            st.info(f"📊 Priority: **{priority}** | 🏷️ Labels: {', '.join(labels)}")
                        except Exception as e:
                            st.error(f"❌ Error: {e}")
                
                with action_cols[3]:
                    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                    if st.button("🗑️ Reject", key=f"delete_{ticket_id}", width="stretch", type="secondary"):
                        collection.update_one(
                            {"_id": ObjectId(ticket["_id"])},
                            {"$set": {"is_deleted": True, "rejected_at": datetime.utcnow()}},
                        )
                        st.session_state.ticket_status[ticket_id] = "deleted"
                        log_human_feedback(
                            module=ticket["module"],
                            repo_name=repo_name,
                            predicted_risk=risk_score,
                            manager_risk=0.0,
                            prediction_id=str(prediction_id) if prediction_id else None,
                            user_id=os.getenv("CURRENT_USER", "manager_ui"),
                        )
                        st.warning("🗑️ Ticket rejected")
                        st.rerun()
                    
                    # Close the card div
                    st.markdown("</div>", unsafe_allow_html=True)


# ============= TAB 2: EDIT RISK SCORES =============

with tab2:
    predictions = list(pred_collection.find({}))
    if not predictions:
        st.info("✅ No predictions found.")
    else:
        # Fetch latest risk scores from risk_feedback table
        feedback_collection = client[DB_NAME][FEEDBACK_COLLECTION]
        
        # Merge feedback data with predictions
        for pred in predictions:
            pred_id = str(pred.get("_id"))
            latest_feedback = feedback_collection.find_one(
                {"prediction_id": pred_id},
                sort=[("created_at", -1)]
            )
            if latest_feedback:
                pred["current_risk"] = latest_feedback.get("manager_risk")
                pred["has_feedback"] = True
                pred["original_risk"] = pred.get("predicted_risk")
            else:
                pred["current_risk"] = pred.get("predicted_risk")
                pred["has_feedback"] = False
                pred["original_risk"] = pred.get("predicted_risk")
        
        # Risk Overview
        stu.render_risk_overview(predictions)
        
        # Risk Browser Controls
        min_display_risk, max_display_risk, sort_mode, search_term = stu.render_risk_browser_controls()
        
        # Legend
        stu.render_risk_legend()
        
        st.markdown("---")

        # Apply filters and group by module
        groups = {}
        for pred in predictions:
            risk = pred.get("current_risk")
            if not isinstance(risk, (int, float)):
                continue
            if risk < min_display_risk or risk > max_display_risk:
                continue
            
            full_module = stu.get_full_module(pred)
            if search_term and search_term.lower() not in full_module.lower():
                continue
                
            group_key = stu.get_group_module(pred)
            groups.setdefault(group_key, []).append(pred)
        
        if "pred_score_status" not in st.session_state:
            st.session_state.pred_score_status = {}
        if "expanded_file" not in st.session_state:
            st.session_state.expanded_file = None

        # Prepare module ordering
        module_items = []
        for key, preds in groups.items():
            vals = [p.get("current_risk", 0.0) for p in preds if isinstance(p.get("current_risk"), (int, float))]
            avg = sum(vals)/len(vals) if vals else 0.0
            latest_created = max((p.get("created_at") for p in preds if p.get("created_at")), default=None)
            module_items.append((key, avg, preds, latest_created))
        
        if sort_mode == "Avg Risk ↓":
            module_items.sort(key=lambda x: x[1], reverse=True)
        elif sort_mode == "Avg Risk ↑":
            module_items.sort(key=lambda x: x[1])
        elif sort_mode == "Latest First":
            module_items.sort(key=lambda x: x[3] if x[3] else datetime.min, reverse=True)
        else:  # Name
            module_items.sort(key=lambda x: x[0])
        
        # Show filtered count
        filtered_files = sum(len(preds) for _, _, preds, _ in module_items)
        st.info(f"📂 Showing {len(module_items)} modules with {filtered_files} files")

        for group_key, avg_risk, preds, latest_created in module_items:
            avg_color, avg_level = stu.color_for_risk(avg_risk)
            high_count = sum(1 for p in preds if isinstance(p.get("current_risk"), (int, float)) and p.get("current_risk") >= 0.7)
            med_count = sum(1 for p in preds if isinstance(p.get("current_risk"), (int, float)) and 0.4 <= p.get("current_risk") < 0.7)
            low_count = sum(1 for p in preds if isinstance(p.get("current_risk"), (int, float)) and p.get("current_risk") < 0.4)
            
            header_md = f"{avg_risk:.2f} | {avg_level}"
            latest_md = f" | Latest: {latest_created.strftime('%Y-%m-%d %H:%M:%S')}" if (latest_created and hasattr(latest_created,'strftime')) else ""
            
            with st.expander(f"📁 {group_key} — {header_md}{latest_md}", expanded=False):
                progress_badges = stu.render_module_progress_bar(avg_risk, high_count, med_count, low_count, len(preds))
                st.markdown(progress_badges, unsafe_allow_html=True)
                st.markdown("")  # spacing
                
                # Sort files within module by risk descending
                sorted_preds = sorted(preds, key=lambda p: p.get("current_risk", 0.0), reverse=True)
                for pred in sorted_preds:
                    pred_id = str(pred["_id"])
                    status = st.session_state.pred_score_status.get(pred_id, None)
                    full_module = stu.get_full_module(pred)
                    file_name = stu.get_file_name(full_module)
                    
                    risk_score = pred.get("current_risk")
                    original_risk = pred.get("original_risk")
                    has_feedback = pred.get("has_feedback", False)
                    
                    file_color, file_level = stu.color_for_risk(risk_score if isinstance(risk_score,(int,float)) else 0.0)
                    risk_score_display = f"{risk_score:.2f}" if isinstance(risk_score,(int,float)) else "N/A"
                    
                    card_bg = "#e8f5e9" if has_feedback else "#ffffff"
                    model_name_raw = pred.get("model_name", "unknown_model")
                    model_name = model_name_raw.replace("_v2_v3_v2", "").replace("_v3_v2", "")
                    created_at = pred.get("created_at")
                    created_str = created_at.strftime('%Y-%m-%d %H:%M') if (created_at and hasattr(created_at,'strftime')) else "N/A"
                    
                    with st.container():
                        expander_title = f"📁 {file_name}"
                        if has_feedback:
                            expander_title += " 🔄"
                        
                        is_expanded = st.session_state.expanded_file == pred_id
                        
                        with st.expander(expander_title, expanded=is_expanded):
                            if is_expanded:
                                pass
                            else:
                                st.session_state.expanded_file = pred_id
                            
                            # File metadata in columns
                            meta_col1, meta_col2 = st.columns(2)
                            
                            with meta_col1:
                                st.markdown(f"**📄 File Name:** `{file_name}`")
                                st.markdown(f"**📂 Path:** `{full_module}`")
                                st.markdown(f"**🎯 Current Risk:** {risk_score_display}")
                                if has_feedback and isinstance(original_risk, (int, float)):
                                    st.markdown(f"**📊 Original Risk:** {original_risk:.2f}")
                                    if abs(original_risk - risk_score) > 0.01:
                                        change = risk_score - original_risk
                                        arrow = "↑" if change > 0 else "↓"
                                        st.markdown(f"**📈 Change:** {arrow} {abs(change):.2f}")
                            
                            with meta_col2:
                                st.markdown(f"**🤖 Model:** {model_name}")
                                st.markdown(f"**📅 Date:** {created_str}")
                                st.markdown(f"**⚠️ Risk Level:** {file_level}")
                                repo = pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO
                                st.markdown(f"**📦 Repository:** {repo}")
                            
                            st.markdown("---")
                            
                            # Feature Details
                            st.markdown("### 📊 Feature Details")
                            feat = pred.get("features", {})
                            if feat:
                                cols = st.columns(3)
                                feature_items = [(k, v) for k, v in feat.items() if k not in ("repo_name", "created_at")]
                                for idx, (k, v) in enumerate(feature_items):
                                    with cols[idx % 3]:
                                        if isinstance(v, float):
                                            st.metric(k.replace("_", " ").title(), f"{v:.3f}")
                                        else:
                                            st.metric(k.replace("_", " ").title(), v)
                            else:
                                st.info("No feature details available.")
                        
                        # Action row
                        col1, col2, col3 = st.columns([3, 1, 1])
                        with col1:
                            new_score = st.number_input(
                                "Adjust Risk Score",
                                key=f"pred_risk_{pred_id}",
                                value=float(pred.get("current_risk") or 0.0),
                                step=0.01,
                                min_value=0.0,
                                max_value=1.0,
                                help="Update the risk score based on your assessment"
                            )
                        
                        with col2:
                            if st.button("💾 Save", key=f"update_pred_{pred_id}", help="Save updated risk score", width="stretch"):
                                log_human_feedback(
                                    module=full_module,
                                    repo_name=pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO,
                                    predicted_risk=pred.get("original_risk", 0.0),
                                    manager_risk=new_score,
                                    prediction_id=str(pred.get("_id")),
                                    user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                )
                                st.session_state.pred_score_status[pred_id] = "updated"
                                st.success(f"✅ Updated to {new_score:.2f} - Refresh page to see in stats")
                        
                        with col3:
                            # Quick set buttons using a popover
                            quick_menu = st.popover("⚡ Quick", help="Quick risk presets", width="stretch")
                            with quick_menu:
                                st.markdown("**Set Risk to:**")
                                quick_values = [(0.0, "No Risk"), (0.25, "Low"), (0.5, "Medium"), (0.75, "High"), (1.0, "Critical")]
                                
                                for val, label in quick_values:
                                    if st.button(f"{val} ({label})", key=f"quick_{int(val*100)}_{pred_id}", width="stretch"):
                                        log_human_feedback(
                                            module=full_module,
                                            repo_name=pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO,
                                            predicted_risk=pred.get("original_risk", 0.0),
                                            manager_risk=val,
                                            prediction_id=str(pred.get("_id")),
                                            user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                        )
                                        st.session_state.pred_score_status[pred_id] = "updated"
                                        st.success(f"✅ Set to {val:.2f} - Refresh to update stats")
                        
                        st.markdown("---")
                        st.markdown("")  # spacing


# ============= TAB 3: MODEL PERFORMANCE =============

with tab3:
    stu.render_performance_header()
    
    # Load data from MongoDB
    feedback_collection = client[DB_NAME][FEEDBACK_COLLECTION]
    predictions = list(pred_collection.find({}))
    feedback_records = list(feedback_collection.find({}))
    
    # Feature Importance Analysis (always show, independent of predictions/feedback)
    stu.render_feature_importance()
    
    if not feedback_records:
        st.info("📊 No feedback data yet. Manager corrections will appear here once you adjust risk scores.")
    else:
        # Prepare feedback dataframe
        feedback_df = pd.DataFrame(feedback_records)
        feedback_df['correction'] = abs(feedback_df['manager_risk'] - feedback_df['predicted_risk'])
        feedback_df['direction'] = feedback_df['manager_risk'] - feedback_df['predicted_risk']
        feedback_df['created_at'] = pd.to_datetime(feedback_df['created_at'])
        
        # KPI Metrics at the top
        stu.render_kpi_metrics(feedback_df, predictions)
        
        # Agreement vs Disagreement Overview with Pie Chart
        st.markdown("---")
        st.markdown("#### 🎯 Agreement vs Disagreement Overview")
        
        total_predictions = len(predictions)
        disagreement_count = len(feedback_df)
        disagreement_rate = (disagreement_count / total_predictions * 100) if total_predictions > 0 else 0
        agreement_count = total_predictions - disagreement_count
        agreement_rate = 100 - disagreement_rate
        
        col_pie, col_metrics = st.columns([1, 1])
        
        with col_pie:
            stu.render_agreement_pie_chart(agreement_count, disagreement_count, agreement_rate)
        
        with col_metrics:
            st.markdown("<div style='padding-top: 60px;'></div>", unsafe_allow_html=True)
            st.metric(
                "✅ Total Agreed",
                agreement_count,
                help="Predictions where manager didn't provide feedback = implicit agreement"
            )
            st.metric(
                "⚠️ Total Disagreed",
                disagreement_count,
                help="Predictions where manager provided corrections"
            )
            st.metric(
                "📊 Agreement Rate",
                f"{agreement_rate:.1f}%",
                help="% of predictions manager agreed with (no feedback needed)"
            )
        
        # Model Bias Analysis
        st.markdown("---")
        st.markdown("#### 📊 Model Bias Analysis")
        
        over_predictions = (feedback_df['direction'] < 0).sum()
        under_predictions = (feedback_df['direction'] > 0).sum()
        exact_match = (feedback_df['direction'] == 0).sum()
        
        col_bias1, col_bias2 = st.columns(2)
        
        with col_bias1:
            direction_data = pd.DataFrame({
                'Category': ['Over-predicted', 'Under-predicted', 'Exact'],
                'Count': [over_predictions, under_predictions, exact_match]
            })
            st.bar_chart(direction_data.set_index('Category')['Count'], use_container_width=True)
            st.caption(f"Model prediction bias")
        
        with col_bias2:
            st.markdown("<div style='padding-top: 40px;'></div>", unsafe_allow_html=True)
            st.metric("🔼 Over-predicted", over_predictions, help="Model predicted higher risk than manager")
            st.metric("🔽 Under-predicted", under_predictions, help="Model predicted lower risk than manager")
            st.metric("🎯 Exact Match", exact_match, help="Model and manager agreed exactly")
        
        # Model Training Progress
        metrics_collection = client[DB_NAME]["model_metrics"]
        metrics_records = list(metrics_collection.find({}).sort("timestamp", -1))
        
        if metrics_records:
            metrics_df = pd.DataFrame(metrics_records)
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            stu.render_model_metrics_progress(metrics_df)
        
        # Top Disagreement Files
        st.markdown("---")
        st.markdown("#### ⚠️ Top 10 Disagreement Cases")
        
        top_disagreements = feedback_df.nlargest(10, 'correction')[
            ['module', 'predicted_risk', 'manager_risk', 'correction']
        ].copy()
        top_disagreements.columns = ['Module', 'Predicted', 'Manager', 'Correction']
        
        st.dataframe(
            top_disagreements,
            use_container_width=True,
            hide_index=True
        )
        st.caption("Files where model predictions differed most from manager assessment")
        
        # Actionable Insights
        st.markdown("---")
        st.markdown("#### 💡 Actionable Insights")
        
        insights = stu.generate_actionable_insights(feedback_df, predictions, over_predictions, under_predictions)
        for insight in insights:
            st.markdown(insight)
        
        # Recommendations
        st.markdown("---")
        st.markdown("#### 🎯 Recommendations")
        
        avg_correction = feedback_df['correction'].mean()
        recommendations = stu.generate_recommendations(avg_correction, disagreement_rate, len(feedback_df), over_predictions, under_predictions)
        for rec in recommendations:
            st.markdown(rec)


