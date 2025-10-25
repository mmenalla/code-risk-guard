import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
from datetime import datetime

from dags.src.data.save_incremental_labeled_data import log_human_feedback
from dags.src.jira.jira_client import JiraClient
from dags.src.utils.config import Config

load_dotenv()
MONGO_URI = "mongodb://admin:admin@localhost:27017/risk_model_db?authSource=admin"

DB_NAME = os.getenv("MONGO_DB", "risk_model_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "ticket_drafts")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

jira_client = JiraClient()

from dags.src.data.save_incremental_labeled_data import log_human_feedback
from dags.src.utils.config import Config


load_dotenv()
MONGO_URI = "mongodb://admin:admin@localhost:27017/risk_model_db?authSource=admin"

DB_NAME = os.getenv("MONGO_DB", "risk_model_db")
PREDICTIONS_COLLECTION = os.getenv("MONGO_PREDICTIONS_COLLECTION", "model_predictions")

client = MongoClient(MONGO_URI)
pred_collection = client[DB_NAME][PREDICTIONS_COLLECTION]

st.set_page_config(page_title="AI Risk Model Dashboard", layout="wide")
st.markdown("---")

st.markdown(
    """
    <div style='display: flex; justify-content: center; align-items: center;'>
        <h1 style='text-align: center;'>MaintAI</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")
tab1, tab2 = st.tabs(["Ticket Review", "Edit Risk Scores"])


with tab1:
    tickets = list(collection.find({"is_deleted": False}))

    if not tickets:
        st.info("✅ No pending tickets to review.")
    else:
        if "ticket_status" not in st.session_state:
            st.session_state.ticket_status = {}

        for ticket in tickets:
            ticket_id = str(ticket["_id"])
            status = st.session_state.ticket_status.get(ticket_id, None)

            if status == "approved":
                color = "#d4edda"  # light green
            elif status == "deleted":
                color = "#f8d7da"  # light red
            else:
                color = "#ffffff"  # white

            with st.expander(f"📄 {ticket['title']} — ({ticket['module']})", expanded=True):
                # Enhanced ticket card layout
                def ticket_color(val):
                    if not isinstance(val, (int, float)):
                        return "#E0E0E0"
                    if val >= 0.7:
                        return "#FFCDD2"  # high
                    if val >= 0.4:
                        return "#FFF9C4"  # medium
                    return "#C8E6C9"      # low
                pill_bg = ticket_color(ticket.get("risk_score"))
                risk_display = f"{float(ticket.get('risk_score', 0.0)):.2f}" if isinstance(ticket.get('risk_score'), (int,float)) else "N/A"
                pill_html = f"<span style='background:{pill_bg}; padding:6px 12px; border-radius:18px; font-weight:600;'>Risk: {risk_display}</span>"
                repo_name = ticket.get("repo_name", "repo?")
                prediction_id = ticket.get("prediction_id")

                st.markdown(
                    f"""
                    <div style='background:#fafafa; border:1px solid #ddd; padding:12px 16px; border-radius:10px; display:flex; flex-direction:column; gap:10px;'>
                      <div style='display:flex; align-items:center; gap:12px; flex-wrap:wrap;'>
                        {pill_html}
                        <span style='font-size:12px; color:#555;'>Module: <strong>{ticket['module']}</strong></span>
                        <span style='font-size:12px; color:#555;'>Repo: <strong>{repo_name}</strong></span>
                        {f"<span style='font-size:12px; color:#555;'>Prediction ID: <strong>{prediction_id}</strong></span>" if prediction_id else ''}
                      </div>
                      <div style='font-size:13px; line-height:1.4;'><strong>Description:</strong><br>{ticket['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Layout: left side risk edit, right side actions
                edit_col, action_col = st.columns([2,2])
                with edit_col:
                    new_score = st.number_input(
                        "Edit Risk Score",
                        key=f"risk_score_{ticket_id}",
                        value=float(ticket['risk_score']),
                        step=0.01,
                        min_value=0.0,
                        max_value=1.0,
                    )
                    if st.button("💾 Update Risk Score", key=f"update_{ticket_id}"):
                        log_human_feedback(
                            module=ticket["module"],
                            repo_name=repo_name,
                            predicted_risk=ticket["risk_score"],
                            manager_risk=new_score,
                            prediction_id=str(prediction_id) if prediction_id else None,
                            user_id=os.getenv("CURRENT_USER", "manager_ui"),
                        )
                        ticket["risk_score"] = new_score
                        st.success(f"Updated risk to {new_score:.2f}")

                with action_col:
                    a1, a2, a3 = st.columns(3)
                    with a1:
                        if st.button("✅ Approve", key=f"approve_{ticket_id}"):
                            try:
                                jira_issue = jira_client.create_ticket(
                                    title=ticket["title"],
                                    description=ticket["description"],
                                    project_key=os.getenv("JIRA_PROJECT_KEY", "TECH")
                                )
                                collection.update_one(
                                    {"_id": ObjectId(ticket["_id"])},
                                    {"$set": {
                                        "is_deleted": True,
                                        "jira_key": jira_issue["key"],
                                        "jira_url": jira_issue["url"]
                                    }},
                                )
                                st.session_state.ticket_status[ticket_id] = "approved"
                                st.success(f"Jira: {jira_issue['key']}")
                            except Exception as e:
                                st.error(f"Jira error: {e}")
                    with a2:
                        if st.button("🗑️ Delete", key=f"delete_{ticket_id}"):
                            collection.update_one(
                                {"_id": ObjectId(ticket["_id"])},
                                {"$set": {"is_deleted": True}},
                            )
                            st.session_state.ticket_status[ticket_id] = "deleted"
                            log_human_feedback(
                                module=ticket["module"],
                                repo_name=repo_name,
                                predicted_risk=ticket["risk_score"],
                                manager_risk=0.0,
                                prediction_id=str(prediction_id) if prediction_id else None,
                                user_id=os.getenv("CURRENT_USER", "manager_ui"),
                            )
                            st.warning("Ticket deleted.")
                    with a3:
                        if st.button("↩ Refresh", key=f"refresh_{ticket_id}"):
                            st.experimental_rerun()


with tab2:
    predictions = list(pred_collection.find({}))
    if not predictions:
        st.info("✅ No predictions found.")
    else:
        # Fetch latest risk scores from risk_feedback table
        feedback_collection = client[DB_NAME]["risk_feedback"]
        
        # Merge feedback data with predictions
        for pred in predictions:
            pred_id = str(pred.get("_id"))
            # Get the most recent feedback for this prediction
            latest_feedback = feedback_collection.find_one(
                {"prediction_id": pred_id},
                sort=[("created_at", -1)]
            )
            if latest_feedback:
                # Use the updated risk score from feedback
                pred["current_risk"] = latest_feedback.get("manager_risk")
                pred["has_feedback"] = True
                pred["original_risk"] = pred.get("predicted_risk")
            else:
                # No feedback yet, use original prediction
                pred["current_risk"] = pred.get("predicted_risk")
                pred["has_feedback"] = False
                pred["original_risk"] = pred.get("predicted_risk")
        
        # --- Statistics Summary (using current_risk) ---
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
        
        # --- Controls / Legend ---
        st.markdown("### 🔧 Risk Browser")
        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2,2,2])
        with col_ctrl1:
            min_display_risk = st.slider("Min risk threshold", 0.0, 1.0, 0.0, 0.01, help="Only show files with risk >= this value")
        with col_ctrl2:
            max_display_risk = st.slider("Max risk threshold", 0.0, 1.0, 1.0, 0.01, help="Only show files with risk <= this value")
        with col_ctrl3:
            sort_mode = st.selectbox("Sort modules by", ["Avg Risk ↓", "Avg Risk ↑", "Name", "Latest First"], index=0)
        
        # Search box
        search_term = st.text_input("🔍 Search files/modules", placeholder="Type to filter by filename or path...", help="Search by filename or module path")
        
        # Legend
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
        
        st.markdown("---")

        def get_full_module(pred):
            return (pred.get("module") or pred.get("features", {}).get("filename") or "unknown_module").strip()
        def get_group_module(pred):
            full = get_full_module(pred)
            if '/' in full:
                return full[:full.rfind('/')]
            return '(root)'
        
        # Apply filters
        groups = {}
        for pred in predictions:
            risk = pred.get("current_risk")
            if not isinstance(risk, (int, float)):
                continue
            if risk < min_display_risk or risk > max_display_risk:
                continue
            
            # Apply search filter
            full_module = get_full_module(pred)
            if search_term and search_term.lower() not in full_module.lower():
                continue
                
            group_key = get_group_module(pred)
            groups.setdefault(group_key, []).append(pred)
        
        if "pred_score_status" not in st.session_state:
            st.session_state.pred_score_status = {}
        if "show_features" not in st.session_state:
            st.session_state.show_features = {}

        def color_for(val: float):
            if val >= 0.7: return ("#FFCDD2", "HIGH")
            if val >= 0.4: return ("#FFF9C4", "MEDIUM")
            return ("#C8E6C9", "LOW")

        # Prepare module ordering (using current_risk)
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
            avg_color, avg_level = color_for(avg_risk)
            high_count = sum(1 for p in preds if isinstance(p.get("current_risk"), (int, float)) and p.get("current_risk") >= 0.7)
            med_count = sum(1 for p in preds if isinstance(p.get("current_risk"), (int, float)) and 0.4 <= p.get("current_risk") < 0.7)
            low_count = sum(1 for p in preds if isinstance(p.get("current_risk"), (int, float)) and p.get("current_risk") < 0.4)
            bar_width = f"{avg_risk*100:.1f}%"
            progress_html = f"""
              <div style='background:#eee; height:12px; border-radius:6px; overflow:hidden; position:relative;'>
                <div style='background:{avg_color}; width:{bar_width}; height:100%; transition:width .4s;'></div>
              </div>
            """
            header_md = f"{avg_risk:.2f} | {avg_level}"
            latest_md = f" | Latest: {latest_created.strftime('%Y-%m-%d %H:%M:%S')}" if (latest_created and hasattr(latest_created,'strftime')) else ""
            badge_html = f"""
              <div style='display:flex; gap:6px; font-size:11px; margin-top:4px;'>
                <span style='background:#FFCDD2; padding:2px 6px; border-radius:10px;'>High: {high_count}</span>
                <span style='background:#FFF9C4; padding:2px 6px; border-radius:10px;'>Med: {med_count}</span>
                <span style='background:#C8E6C9; padding:2px 6px; border-radius:10px;'>Low: {low_count}</span>
                <span style='background:#E0E0E0; padding:2px 6px; border-radius:10px;'>Files: {len(preds)}</span>
              </div>
            """
            with st.expander(f"📁 {group_key} — {header_md}{latest_md}", expanded=False):
                st.markdown(progress_html + badge_html, unsafe_allow_html=True)
                st.markdown("")  # spacing
                
                # Sort files within module by risk descending (using current_risk)
                sorted_preds = sorted(preds, key=lambda p: p.get("current_risk", 0.0), reverse=True)
                for pred in sorted_preds:
                    pred_id = str(pred["_id"])
                    status = st.session_state.pred_score_status.get(pred_id, None)
                    full_module = get_full_module(pred)
                    file_name = full_module.split('/')[-1]
                    
                    # Use current_risk for display
                    risk_score = pred.get("current_risk")
                    original_risk = pred.get("original_risk")
                    has_feedback = pred.get("has_feedback", False)
                    
                    file_color, file_level = color_for(risk_score if isinstance(risk_score,(int,float)) else 0.0)
                    risk_score_display = f"{risk_score:.2f}" if isinstance(risk_score,(int,float)) else "N/A"
                    
                    # Highlight if the risk has been updated
                    card_bg = "#e8f5e9" if has_feedback else "#ffffff"
                    card_border = "#4caf50" if has_feedback else "#ddd"
                    model_name_raw = pred.get("model_name", "unknown_model")
                    # Clean up model name
                    model_name = model_name_raw.replace("_v2_v3_v2", "").replace("_v3_v2", "")
                    created_at = pred.get("created_at")
                    created_str = created_at.strftime('%Y-%m-%d %H:%M') if (created_at and hasattr(created_at,'strftime')) else "N/A"
                    
                    # Show original vs current if different
                    risk_change_badge = ""
                    if has_feedback and isinstance(original_risk, (int, float)) and isinstance(risk_score, (int, float)):
                        if abs(original_risk - risk_score) > 0.01:
                            change = risk_score - original_risk
                            arrow = "↑" if change > 0 else "↓"
                            change_color = "#ff5252" if change > 0 else "#4caf50"
                            risk_change_badge = f"<span style='background:{change_color}; color:white; padding:2px 6px; border-radius:8px; font-size:11px; margin-left:6px;'>{arrow} {abs(change):.2f}</span>"
                    
                    with st.container():
                        # Use columns for better layout instead of HTML
                        header_col1, header_col2 = st.columns([2, 2])
                        
                        with header_col1:
                            # Risk score badge and filename
                            badge_col, name_col = st.columns([1, 4])
                            with badge_col:
                                st.markdown(f"<div style='background:{file_color}; padding:8px; border-radius:16px; font-weight:700; text-align:center;'>{risk_score_display}</div>", unsafe_allow_html=True)
                            with name_col:
                                display_name = f"**{file_name}**"
                                if risk_change_badge:
                                    st.markdown(risk_change_badge + " " + display_name, unsafe_allow_html=True)
                                else:
                                    st.markdown(display_name)
                                if has_feedback:
                                    st.markdown("<span style='background:#2196F3; color:white; padding:2px 6px; border-radius:8px; font-size:10px;'>UPDATED</span>", unsafe_allow_html=True)
                        
                        with header_col2:
                            st.caption(f"**Model:** {model_name}")
                            st.caption(f"**Date:** {created_str}")
                            st.caption(f"**Path:** `{full_module}`")
                        
                        
                        
                        # Action row
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
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
                            if st.button("💾 Save", key=f"update_pred_{pred_id}", help="Save updated risk score", use_container_width=True):
                                log_human_feedback(
                                    module=full_module,
                                    repo_name=pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO,
                                    predicted_risk=pred.get("original_risk", 0.0),
                                    manager_risk=new_score,
                                    prediction_id=str(pred.get("_id")),
                                    user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                )
                                st.session_state.pred_score_status[pred_id] = "updated"
                                st.success(f"✅ Updated to {new_score:.2f}")
                                st.rerun()
                        
                        with col3:
                            if st.button("🔍 Details", key=f"toggle_features_{pred_id}", help="Show/hide feature details", use_container_width=True):
                                st.session_state.show_features[pred_id] = not st.session_state.show_features.get(pred_id, False)
                                st.rerun()
                        
                        with col4:
                            # Quick set buttons using a popover/menu approach
                            quick_menu = st.popover("⚡ Quick", help="Quick risk presets", use_container_width=True)
                            with quick_menu:
                                st.markdown("**Set Risk to:**")
                                if st.button("0.0 (No Risk)", key=f"quick_0_{pred_id}", use_container_width=True):
                                    new_val = 0.0
                                    log_human_feedback(
                                        module=full_module,
                                        repo_name=pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO,
                                        predicted_risk=pred.get("original_risk", 0.0),
                                        manager_risk=new_val,
                                        prediction_id=str(pred.get("_id")),
                                        user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                    )
                                    st.session_state.pred_score_status[pred_id] = "updated"
                                    st.rerun()
                                
                                if st.button("0.25 (Low)", key=f"quick_25_{pred_id}", use_container_width=True):
                                    new_val = 0.25
                                    log_human_feedback(
                                        module=full_module,
                                        repo_name=pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO,
                                        predicted_risk=pred.get("original_risk", 0.0),
                                        manager_risk=new_val,
                                        prediction_id=str(pred.get("_id")),
                                        user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                    )
                                    st.session_state.pred_score_status[pred_id] = "updated"
                                    st.rerun()
                                
                                if st.button("0.5 (Medium)", key=f"quick_50_{pred_id}", use_container_width=True):
                                    new_val = 0.5
                                    log_human_feedback(
                                        module=full_module,
                                        repo_name=pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO,
                                        predicted_risk=pred.get("original_risk", 0.0),
                                        manager_risk=new_val,
                                        prediction_id=str(pred.get("_id")),
                                        user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                    )
                                    st.session_state.pred_score_status[pred_id] = "updated"
                                    st.rerun()
                                
                                if st.button("0.75 (High)", key=f"quick_75_{pred_id}", use_container_width=True):
                                    new_val = 0.75
                                    log_human_feedback(
                                        module=full_module,
                                        repo_name=pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO,
                                        predicted_risk=pred.get("original_risk", 0.0),
                                        manager_risk=new_val,
                                        prediction_id=str(pred.get("_id")),
                                        user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                    )
                                    st.session_state.pred_score_status[pred_id] = "updated"
                                    st.rerun()
                                
                                if st.button("1.0 (Critical)", key=f"quick_100_{pred_id}", use_container_width=True):
                                    new_val = 1.0
                                    log_human_feedback(
                                        module=full_module,
                                        repo_name=pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO,
                                        predicted_risk=pred.get("original_risk", 0.0),
                                        manager_risk=new_val,
                                        prediction_id=str(pred.get("_id")),
                                        user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                    )
                                    st.session_state.pred_score_status[pred_id] = "updated"
                                    st.rerun()
                        st.markdown("---")
                        # Show features if toggled
                        if st.session_state.show_features.get(pred_id):
                            with st.expander("📊 Feature Details", expanded=True):
                                feat = pred.get("features", {})
                                if feat:
                                    # Display in a more readable format
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
                        
                        st.markdown("")  # spacing
