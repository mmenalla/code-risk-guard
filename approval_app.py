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

# MongoDB configuration from environment variables
MONGO_URI = os.getenv("MONGO_URI", "mongodb://admin:admin@localhost:27017/risk_model_db?authSource=admin")
DB_NAME = os.getenv("MONGO_DB", "risk_model_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "ticket_drafts")
PREDICTIONS_COLLECTION = os.getenv("MONGO_PREDICTIONS_COLLECTION", "model_predictions")
FEEDBACK_COLLECTION = os.getenv("MONGO_FEEDBACK_COLLECTION", "risk_feedback")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
pred_collection = client[DB_NAME][PREDICTIONS_COLLECTION]

jira_client = JiraClient()

st.set_page_config(page_title="MaintAI - AI Risk Model Dashboard", layout="wide", page_icon="🤖")

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
                MaintAI
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

tab1, tab2, tab3 = st.tabs(["🎫 Ticket Review", "📊 Edit Risk Scores", "📈 Model Performance"])


with tab1:
    tickets = list(collection.find({"is_deleted": False}))

    # Initialize session state
    if "ticket_status" not in st.session_state:
        st.session_state.ticket_status = {}
    if "selected_tickets" not in st.session_state:
        st.session_state.selected_tickets = set()

    # === Summary Dashboard ===
    if tickets:
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

        # === Filters & Search ===
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

        # Apply filters
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

        st.markdown(f"**Showing {len(filtered_tickets)} of {len(tickets)} tickets**")
        st.markdown("---")

    if not tickets:
        st.info("✅ No pending tickets to review.")
    elif not filtered_tickets:
        st.warning("🔍 No tickets match your filters. Try adjusting the search criteria.")
    else:
        # === Ticket Cards ===
        st.markdown("### 🎫 Tickets")
        
        for ticket in filtered_tickets:
            ticket_id = str(ticket["_id"])
            status = st.session_state.ticket_status.get(ticket_id, None)
            repo_name = ticket.get("repo_name", "Unknown")
            prediction_id = ticket.get("prediction_id")
            risk_score = ticket.get("risk_score", 0.0)
            
            # Determine colors and borders
            def get_risk_info(val):
                if not isinstance(val, (int, float)):
                    return {"color": "#E0E0E0", "label": "N/A", "icon": "⚪", "border": "#CCCCCC"}
                if val >= 0.7:
                    return {"color": "#FFCDD2", "label": "HIGH", "icon": "🔴", "border": "#E57373"}
                if val >= 0.4:
                    return {"color": "#FFF9C4", "label": "MEDIUM", "icon": "🟡", "border": "#FFD54F"}
                return {"color": "#C8E6C9", "label": "LOW", "icon": "🟢", "border": "#81C784"}
            
            risk_info = get_risk_info(risk_score)
            
            # Status styling
            if status == "approved":
                card_bg = "#E8F5E9"
                border_color = "#4CAF50"
                status_badge = "✅ APPROVED"
            elif status == "deleted":
                card_bg = "#FFEBEE"
                border_color = "#F44336"
                status_badge = "🗑️ DELETED"
            else:
                card_bg = "#FFFFFF"
                border_color = risk_info["border"]
                status_badge = ""

            # Create expander title with key info
            expander_label = f"{risk_info['icon']} {ticket.get('title', 'Untitled')} - {risk_info['label']}: {float(risk_score):.2f}"
            
            # Use expander for collapsible tickets
            with st.expander(expander_label, expanded=False):
                # Card container inside expander
                with st.container():
                    # Metadata section
                    st.markdown(f"""<div style='background:{card_bg}; border-left: 6px solid {border_color}; border: 1px solid {border_color}40; border-radius: 12px; padding: 20px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>{f'<div style="text-align: right; margin-bottom: 10px;"><span style="background: #E3F2FD; color: #1976D2; padding: 4px 12px; border-radius: 12px; font-size: 11px; font-weight: 600;">{status_badge}</span></div>' if status_badge else ''}<div style='display: flex; gap: 20px; flex-wrap: wrap; padding: 10px; background: #F8F9FA; border-radius: 8px; margin-bottom: 15px;'><div style='display: flex; align-items: center; gap: 6px;'><span style='font-size: 14px;'>📁</span><span style='font-size: 13px; color: #666;'>Module:</span><code style='background: #E3F2FD; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{ticket.get("module", "N/A")}</code></div><div style='display: flex; align-items: center; gap: 6px;'><span style='font-size: 14px;'>🏢</span><span style='font-size: 13px; color: #666;'>Repo:</span><code style='background: #E3F2FD; padding: 2px 8px; border-radius: 4px; font-size: 12px;'>{repo_name}</code></div>{f'<div style="display: flex; align-items: center; gap: 6px;"><span style="font-size: 14px;">🔗</span><span style="font-size: 13px; color: #666;">ID:</span><code style="background: #E3F2FD; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{prediction_id}</code></div>' if prediction_id else ''}</div><div style='background: white; padding: 15px; border-radius: 8px; border: 1px solid #E0E0E0; margin-bottom: 15px;'><div style='font-weight: 600; color: #424242; margin-bottom: 8px; font-size: 13px;'>📝 DESCRIPTION</div><div style='color: #616161; line-height: 1.6; font-size: 14px;'>{ticket.get("description", "No description provided.")}</div></div>{f'<div style="background: #FFF9C4; padding: 12px; border-radius: 8px; border: 1px solid #FBC02D; margin-bottom: 15px;"><div style="font-weight: 600; color: #F57F17; margin-bottom: 6px; font-size: 12px;">⚠️ WHY IS THIS RISKY?</div><div style="color: #F57F17; font-size: 13px; line-height: 1.5;">Recent churn: {ticket.get("context", {}).get("recent_churn", "N/A")} lines • Bug ratio: {ticket.get("context", {}).get("bug_ratio", "N/A")} • Recent PRs: {ticket.get("context", {}).get("recent_prs", "N/A")}</div></div>' if ticket.get('context') else ''}<hr style='border: none; border-top: 2px solid #E0E0E0; margin: 20px 0 15px 0;'><div style='font-weight: 600; color: #424242; margin-bottom: 12px; font-size: 13px;'>⚙️ ACTIONS</div></div>""", unsafe_allow_html=True)

                    # Action buttons - INSIDE the card
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
                            # Determine priority based on risk score
                            if risk_score >= 0.7:
                                priority = "High"
                            elif risk_score >= 0.4:
                                priority = "Medium"
                            else:
                                priority = "Low"
                            
                            # Extract filename from module path
                            module_path = ticket.get("module", "")
                            filename = module_path.split("/")[-1] if module_path else "unknown"
                            
                            # Create labels: module path, filename, and MaintAIGenerated
                            labels = [
                                module_path.replace("/", "-").replace(".", "-") if module_path else "unknown-module",
                                filename.replace(".", "-") if filename else "unknown-file",
                                "MaintAIGenerated"
                            ]
                            
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
            """Get the full module path from prediction record."""
            module = pred.get("module") or pred.get("features", {}).get("filename") or "unknown_module"
            return module.strip()
        
        def get_file_name(full_path):
            """Extract filename from full path, handling edge cases."""
            if not full_path or full_path == "unknown_module":
                return "unknown_module"
            
            # Split by / to get last component
            parts = full_path.split('/')
            filename = parts[-1] if parts else full_path
            
            # If filename is empty (path ends with /), try previous part
            if not filename and len(parts) > 1:
                filename = parts[-2]
            
            # If still empty or suspicious (no extension and very short), might be a directory
            if not filename or (len(filename) < 3 and '.' not in filename):
                # Return the full path instead
                return full_path
            
            return filename
        
        def get_group_module(pred):
            """Get the directory path for grouping files."""
            full = get_full_module(pred)
            if '/' in full:
                # Get everything except the last part (filename)
                directory = full[:full.rfind('/')]
                # If directory is empty, return root
                return directory if directory else '(root)'
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
        if "expanded_file" not in st.session_state:
            st.session_state.expanded_file = None

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
                    file_name = get_file_name(full_module)
                    
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
                        # Expandable file details section
                        expander_title = f"📁 {file_name}"
                        if has_feedback:
                            expander_title += " 🔄"
                        
                        # Check if this file should be expanded
                        is_expanded = st.session_state.expanded_file == pred_id
                        
                        with st.expander(expander_title, expanded=is_expanded):
                            # Track that this expander is being viewed
                            if is_expanded:
                                # Keep it tracked as expanded
                                pass
                            else:
                                # User manually expanded it, track it
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
                            # Quick set buttons using a popover/menu approach
                            quick_menu = st.popover("⚡ Quick", help="Quick risk presets", width="stretch")
                            with quick_menu:
                                st.markdown("**Set Risk to:**")
                                if st.button("0.0 (No Risk)", key=f"quick_0_{pred_id}", width="stretch"):
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
                                    st.success(f"✅ Set to {new_val:.2f} - Refresh to update stats")
                                
                                if st.button("0.25 (Low)", key=f"quick_25_{pred_id}", width="stretch"):
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
                                    st.success(f"✅ Set to {new_val:.2f} - Refresh to update stats")
                                
                                if st.button("0.5 (Medium)", key=f"quick_50_{pred_id}", width="stretch"):
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
                                    st.success(f"✅ Set to {new_val:.2f} - Refresh to update stats")
                                
                                if st.button("0.75 (High)", key=f"quick_75_{pred_id}", width="stretch"):
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
                                    st.success(f"✅ Set to {new_val:.2f} - Refresh to update stats")
                                
                                if st.button("1.0 (Critical)", key=f"quick_100_{pred_id}", width="stretch"):
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
                                    st.success(f"✅ Set to {new_val:.2f} - Refresh to update stats")
                        
                        st.markdown("---")
                        st.markdown("")  # spacing


with tab3:
    st.markdown("### 📈 Model Performance Dashboard")
    st.markdown("Track model accuracy, correction patterns, and disagreement rates over time.")
    
    # Load data from MongoDB
    feedback_collection = client[DB_NAME][FEEDBACK_COLLECTION]
    predictions = list(pred_collection.find({}))
    feedback_records = list(feedback_collection.find({}))
    
    # Get model metrics from model_metrics collection
    metrics_collection = client[DB_NAME]["model_metrics"]
    metrics_records = list(metrics_collection.find({}).sort("timestamp", -1))
    
    if not feedback_records:
        st.info("📊 No feedback data yet. Manager corrections will appear here once you adjust risk scores.")
    else:
        import numpy as np
        import pandas as pd
        
        # Prepare feedback dataframe
        feedback_df = pd.DataFrame(feedback_records)
        feedback_df['correction'] = abs(feedback_df['manager_risk'] - feedback_df['predicted_risk'])
        feedback_df['direction'] = feedback_df['manager_risk'] - feedback_df['predicted_risk']
        feedback_df['created_at'] = pd.to_datetime(feedback_df['created_at'])
        
        # ========== KEY METRICS SECTION ==========
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
            # Disagreement rate: feedback given / total predictions
            # If manager gives feedback = disagreed, no feedback = agreed
            total_predictions = len(predictions)
            disagreement_count = len(feedback_df)
            disagreement_rate = (disagreement_count / total_predictions * 100) if total_predictions > 0 else 0
            agreement_count = total_predictions - disagreement_count
            
            st.metric(
                "Disagreement Rate",
                f"{disagreement_rate:.1f}%",
                delta=f"{disagreement_count} of {total_predictions}",
                delta_color="inverse",
                help=f"% of predictions that received manager feedback (implicit agreement if no feedback)"
            )
        
        # ========== AGREEMENT VS DISAGREEMENT ==========
        st.markdown("---")
        st.markdown("#### 🎯 Agreement vs Disagreement Overview")
        st.markdown("""
        **Logic**: If manager provides feedback = disagreement. No feedback = implicit agreement.
        """)
        
        col_agree1, col_agree2, col_agree3 = st.columns(3)
        
        with col_agree1:
            st.metric(
                "✅ Agreed (No Feedback)",
                agreement_count,
                help="Predictions where manager didn't provide feedback = implicit agreement"
            )
        
        with col_agree2:
            st.metric(
                "⚠️ Disagreed (Gave Feedback)",
                disagreement_count,
                delta=f"{disagreement_rate:.1f}%",
                delta_color="inverse",
                help="Predictions where manager provided corrections"
            )
        
        with col_agree3:
            agreement_rate = 100 - disagreement_rate
            st.metric(
                "Agreement Rate",
                f"{agreement_rate:.1f}%",
                help="% of predictions manager agreed with (no feedback needed)"
            )
        
        # Visual breakdown
        agree_disagree_data = pd.DataFrame({
            'Status': ['Agreed\n(No Feedback)', 'Disagreed\n(Gave Feedback)'],
            'Count': [agreement_count, disagreement_count]
        })
        
        st.bar_chart(agree_disagree_data.set_index('Status')['Count'], width="stretch")
        st.caption(f"Goal: High agreement rate (>75%) means model is accurate")
        
        # ========== CORRECTION DISTRIBUTION ==========
        st.markdown("---")
        st.markdown("#### 📊 Correction Magnitude Distribution")
        st.markdown("For the predictions where manager disagreed, how large were the corrections?")
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Histogram of corrections
            bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
            labels = ['0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5+']
            feedback_df['correction_bin'] = pd.cut(feedback_df['correction'], bins=bins, labels=labels)
            correction_counts = feedback_df['correction_bin'].value_counts().sort_index()
            
            st.bar_chart(correction_counts, width="stretch")
            st.caption("Distribution of correction magnitudes - lower is better")
        
        with col_chart2:
            # Direction of corrections
            over_predictions = (feedback_df['direction'] < 0).sum()
            under_predictions = (feedback_df['direction'] > 0).sum()
            exact_match = (feedback_df['direction'] == 0).sum()
            
            direction_data = pd.DataFrame({
                'Category': ['Over-predicted', 'Under-predicted', 'Exact'],
                'Count': [over_predictions, under_predictions, exact_match]
            })
            
            st.bar_chart(direction_data.set_index('Category')['Count'], width="stretch")
            st.caption(f"Model bias: {over_predictions} over, {under_predictions} under, {exact_match} exact")
        
        # ========== CORRECTION TRENDS OVER TIME ==========
        st.markdown("---")
        st.markdown("#### 📉 Correction Trends Over Time")
        
        # Group by date
        feedback_df['date'] = feedback_df['created_at'].dt.date
        daily_corrections = feedback_df.groupby('date').agg({
            'correction': ['mean', 'count', 'max']
        }).reset_index()
        daily_corrections.columns = ['date', 'avg_correction', 'count', 'max_correction']
        daily_corrections = daily_corrections.set_index('date')
        
        col_trend1, col_trend2 = st.columns(2)
        
        with col_trend1:
            st.line_chart(daily_corrections['avg_correction'], width="stretch")
            st.caption("Average correction magnitude over time - should trend downward")
        
        with col_trend2:
            st.line_chart(daily_corrections['count'], width="stretch")
            st.caption("Number of corrections per day")
        
        # ========== MODEL COMPARISON ==========
        st.markdown("---")
        st.markdown("#### 🤖 Model Version Comparison")
        
        # Try to extract model version from predictions
        pred_df = pd.DataFrame(predictions)
        if 'model_name' in pred_df.columns and len(pred_df) > 0:
            # Join predictions with feedback
            pred_df['prediction_id'] = pred_df['_id'].astype(str)
            feedback_with_model = feedback_df.merge(
                pred_df[['prediction_id', 'model_name']],
                left_on='prediction_id',
                right_on='prediction_id',
                how='left'
            )
            
            if 'model_name' in feedback_with_model.columns:
                # Count predictions per model
                model_pred_counts = pred_df['model_name'].value_counts().to_dict()
                
                model_performance = feedback_with_model.groupby('model_name').agg({
                    'correction': ['mean', 'count'],
                    'predicted_risk': 'mean',
                    'manager_risk': 'mean'
                }).reset_index()
                model_performance.columns = ['model_name', 'avg_correction', 'feedback_count', 'avg_pred_risk', 'avg_mgr_risk']
                
                # Add total predictions for this model
                model_performance['total_predictions'] = model_performance['model_name'].map(model_pred_counts)
                
                # Disagreement rate = feedback_count / total_predictions
                model_performance['disagreement_rate'] = (
                    model_performance['feedback_count'] / model_performance['total_predictions'] * 100
                ).round(1)
                
                # Agreement rate (implicit)
                model_performance['agreement_count'] = model_performance['total_predictions'] - model_performance['feedback_count']
                
                # Reorder columns
                model_performance = model_performance[[
                    'model_name', 'total_predictions', 'feedback_count', 'agreement_count',
                    'disagreement_rate', 'avg_correction', 'avg_pred_risk', 'avg_mgr_risk'
                ]]
                
                st.dataframe(
                    model_performance.sort_values('disagreement_rate'),
                    width="stretch",
                    hide_index=True
                )
                st.caption("Lower disagreement rate = better model (manager agrees more often)")
        
        # ========== TOP DISAGREEMENT FILES ==========
        st.markdown("---")
        st.markdown("#### ⚠️ Top Disagreement Cases")
        st.markdown("Files where model predictions differed most from manager assessment")
        
        top_disagreements = feedback_df.nlargest(10, 'correction')[
            ['module', 'predicted_risk', 'manager_risk', 'correction', 'created_at']
        ].copy()
        top_disagreements['created_at'] = top_disagreements['created_at'].dt.strftime('%Y-%m-%d %H:%M')
        top_disagreements.columns = ['Module', 'Predicted', 'Manager', 'Correction', 'Date']
        
        st.dataframe(
            top_disagreements,
            width="stretch",
            hide_index=True
        )
        
        # ========== RISK CALIBRATION ==========
        st.markdown("---")
        st.markdown("#### 🎯 Risk Calibration Analysis")
        st.markdown("How well are predicted risks aligned with manager assessments?")
        
        col_cal1, col_cal2 = st.columns(2)
        
        with col_cal1:
            # Scatter plot data
            scatter_data = feedback_df[['predicted_risk', 'manager_risk']].copy()
            scatter_data.columns = ['Predicted Risk', 'Manager Risk']
            
            st.scatter_chart(
                data=scatter_data,
                x='Predicted Risk',
                y='Manager Risk',
                width="stretch"
            )
            st.caption("Perfect calibration would be a diagonal line from (0,0) to (1,1)")
        
        with col_cal2:
            # Calibration by risk bucket
            feedback_df['pred_bucket'] = pd.cut(
                feedback_df['predicted_risk'],
                bins=[0, 0.25, 0.5, 0.75, 1.0],
                labels=['Low (0-0.25)', 'Medium (0.25-0.5)', 'High (0.5-0.75)', 'Critical (0.75-1.0)']
            )
            
            calibration = feedback_df.groupby('pred_bucket').agg({
                'predicted_risk': 'mean',
                'manager_risk': 'mean',
                'correction': 'mean'
            }).round(3)
            calibration.columns = ['Avg Predicted', 'Avg Manager', 'Avg Correction']
            
            st.dataframe(calibration, width="stretch")
            st.caption("Check if each risk bucket is well-calibrated")
        
        # ========== MODEL METRICS HISTORY ==========
        if metrics_records:
            st.markdown("---")
            st.markdown("#### 📊 Training Metrics History")
            st.markdown("Performance metrics from model training runs")
            
            metrics_df = pd.DataFrame(metrics_records)
            metrics_df['timestamp'] = pd.to_datetime(metrics_df['timestamp'])
            metrics_df = metrics_df.sort_values('timestamp')
            
            # Display recent metrics
            recent_metrics = metrics_df.tail(10)[['model_name', 'mae', 'mse', 'r2', 'timestamp']].copy()
            recent_metrics['timestamp'] = recent_metrics['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            recent_metrics.columns = ['Model', 'MAE', 'MSE', 'R²', 'Timestamp']
            
            st.dataframe(
                recent_metrics,
                width="stretch",
                hide_index=True
            )
            
            # Plot metrics over time
            col_metric1, col_metric2 = st.columns(2)
            
            with col_metric1:
                metrics_chart = metrics_df.set_index('timestamp')[['mae', 'mse']]
                st.line_chart(metrics_chart, width="stretch")
                st.caption("MAE and MSE over time - should decrease with improvements")
            
            with col_metric2:
                r2_chart = metrics_df.set_index('timestamp')[['r2']]
                st.line_chart(r2_chart, width="stretch")
                st.caption("R² score over time - higher is better (max 1.0)")
        
        # ========== ACTIONABLE INSIGHTS ==========
        st.markdown("---")
        st.markdown("#### 💡 Actionable Insights")
        
        insights = []
        
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
        
        # Insight 3: Disagreement rate (updated logic)
        total_predictions = len(predictions)
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
        
        # Insight 5: Trend analysis
        if len(daily_corrections) >= 3:
            recent_avg = daily_corrections.tail(3)['avg_correction'].mean()
            older_avg = daily_corrections.head(max(1, len(daily_corrections) - 3))['avg_correction'].mean()
            if recent_avg < older_avg * 0.8:
                insights.append("📈 **Improving**: Recent corrections are smaller - model is getting better over time")
            elif recent_avg > older_avg * 1.2:
                insights.append("📉 **Degrading**: Recent corrections are larger - model may need retraining")
        
        for insight in insights:
            st.markdown(insight)
        
        # ========== RECOMMENDATIONS ==========
        st.markdown("---")
        st.markdown("#### 🎯 Recommendations")
        
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
        
        for rec in recommendations:
            st.markdown(rec)
        
        # ========== EXPORT DATA ==========
        st.markdown("---")
        st.markdown("#### 💾 Export Data")
        
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            # Export feedback data
            feedback_export = feedback_df[['module', 'predicted_risk', 'manager_risk', 'correction', 'created_at']].copy()
            feedback_export['created_at'] = feedback_export['created_at'].dt.strftime('%Y-%m-%d %H:%M:%S')
            csv_feedback = feedback_export.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Feedback Data (CSV)",
                data=csv_feedback,
                file_name=f"model_feedback_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                width="stretch"
            )
        
        with col_export2:
            # Export metrics data
            if metrics_records:
                metrics_export = metrics_df[['model_name', 'mae', 'mse', 'r2', 'timestamp']].copy()
                metrics_export['timestamp'] = metrics_export['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                csv_metrics = metrics_export.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Metrics History (CSV)",
                    data=csv_metrics,
                    file_name=f"model_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    width="stretch"
                )

