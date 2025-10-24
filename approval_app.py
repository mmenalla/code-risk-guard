import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os

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
                # --- Ticket info card ---
                st.markdown(
                    f"""
                    <div style="background-color:{color}; padding:15px; border-radius:10px;">
                        <strong>Module:</strong> {ticket['module']}<br>
                        <strong>Description:</strong> {ticket['description']}<br>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                col_score, col_button = st.columns([3, 1])
                with col_score:
                    new_score = st.number_input(
                        "Risk Score (Editable)",
                        key=f"risk_score_{ticket_id}",
                        value=float(ticket['risk_score']),
                        step=0.01,
                        min_value=0.0,
                        max_value=1.0,
                    )

                with col_button:
                    if st.button(f"💾 Update", key=f"update_{ticket_id}"):
                        # Removed persistence to ticket_drafts; only log feedback now
                        log_human_feedback(
                            module=ticket["module"],
                            repo_name=ticket.get("repo_name", Config.GITHUB_REPO),
                            predicted_risk=ticket["risk_score"],
                            manager_risk=new_score,
                            prediction_id=str(ticket.get("prediction_id")) if ticket.get("prediction_id") else None,
                            user_id=os.getenv("CURRENT_USER", "manager_ui"),
                        )
                        # Update in-memory ticket for immediate UI reflect
                        ticket["risk_score"] = new_score
                        st.success(f"✅ Logged feedback for {ticket['module']} (new risk: {new_score:.2f})")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Approve & Send to Jira", key=f"approve_{ticket_id}"):
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
                            st.success(f"✅ Ticket created in Jira: [{jira_issue['key']}]({jira_issue['url']})")

                        except Exception as e:
                            st.error(f"❌ Failed to create Jira ticket: {e}")

                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{ticket_id}"):
                        collection.update_one(
                            {"_id": ObjectId(ticket["_id"])},
                            {"$set": {"is_deleted": True}},
                        )
                        st.session_state.ticket_status[ticket_id] = "deleted"
                        st.warning("🗑️ Ticket marked as deleted.")
                        log_human_feedback(
                            module=ticket["module"],
                            repo_name=ticket.get("repo_name", Config.GITHUB_REPO),
                            predicted_risk=ticket["risk_score"],
                            manager_risk=0.0,  # Deleted ticket treated as risk 0
                            prediction_id=str(ticket.get("prediction_id")) if ticket.get("prediction_id") else None,
                            user_id=os.getenv("CURRENT_USER", "manager_ui"),
                        )


with tab2:
    predictions = list(pred_collection.find({}))
    if not predictions:
        st.info("✅ No predictions found.")
    else:
        def get_full_module(pred):
            return (pred.get("module") or pred.get("features", {}).get("filename") or "unknown_module").strip()
        def get_group_module(pred):
            full = get_full_module(pred)
            return full.rsplit('/', 1)[0] if '/' in full else '(root)'
        groups = {}
        for pred in predictions:
            group_key = get_group_module(pred)
            groups.setdefault(group_key, []).append(pred)
        if "pred_score_status" not in st.session_state:
            st.session_state.pred_score_status = {}
        if "show_features" not in st.session_state:
            st.session_state.show_features = {}
        for group_key, preds in sorted(groups.items()):
            valid_scores = [p.get("predicted_risk", 0.0) for p in preds if isinstance(p.get("predicted_risk"), (int, float))]
            avg_risk = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            latest_created = max((p.get("created_at") for p in preds if p.get("created_at")), default=None)
            header_extra = f"Avg Risk: {avg_risk:.2f}"
            if avg_risk >= 0.7:
                header_extra = f":red[Avg Risk: {avg_risk:.2f}]"
            if latest_created and hasattr(latest_created, 'strftime'):
                header_extra += f" | Latest: {latest_created.strftime('%Y-%m-%d %H:%M:%S')}"
            with st.expander(f"📁 Module: {group_key} — {header_extra}", expanded=False):
                sorted_preds = sorted(preds, key=lambda p: p.get("created_at"), reverse=True)
                for pred in sorted_preds:
                    pred_id = str(pred["_id"])
                    status = st.session_state.pred_score_status.get(pred_id, None)
                    full_module = get_full_module(pred)
                    file_name = full_module.split('/')[-1]
                    risk_score = pred.get("predicted_risk")
                    risk_score_display = f"{risk_score:.2f}" if isinstance(risk_score, (int, float)) else "N/A"
                    # --- Risk styling ---
                    def risk_style(val):
                        if not isinstance(val, (int, float)):
                            return ("❔", "#E0E0E0", "#000", "UNKNOWN")
                        if val <= 0.3:
                            return ("✅", "#A5D6A7", "#000", "LOW")  # low risk green (changed icon)
                        if val <= 0.6:
                            return ("⚠️", "#FFEB99", "#000", "MEDIUM")  # medium risk yellow
                        return ("❗", "#EF9A9A", "#000", "HIGH")  # high risk red
                    icon, pill_bg, pill_fg, risk_level = risk_style(risk_score)
                    model_name = pred.get("model_name", "unknown_model")
                    created_at = pred.get("created_at")
                    created_str = created_at.strftime('%Y-%m-%d %H:%M:%S') if (created_at and hasattr(created_at, 'strftime')) else "N/A"
                    color = "#fff3cd" if status == "updated" else "#ffffff"
                    # Use icon & level in expander label for quick glance
                    with st.expander(f"{icon} {file_name} — {risk_level} {risk_score_display} | Model {model_name} | {created_str}", expanded=False):
                        # Visual bar (simple CSS div) representing risk proportion
                        bar_html = f"""
                        <div style='background:#eee; border-radius:6px; height:10px; position:relative; margin-top:4px;'>
                            <div style='background:{pill_bg}; width:{min(max(risk_score or 0,0),1)*100:.1f}%; height:100%; border-radius:6px;'></div>
                        </div>
                        """
                        st.markdown(
                            f"""
                            <div style=\"background-color:{color}; padding:10px; border-radius:6px;\">\n                            <strong>Path:</strong> {full_module}<br>\n                            <strong>Model:</strong> {model_name}<br>\n                            <strong>Risk:</strong> <span style=\"background-color:{pill_bg}; color:{pill_fg}; padding:4px 10px; border-radius:14px; font-weight:600;\">{risk_level} {risk_score_display}</span><br>\n                            <strong>Timestamp:</strong> {created_str}<br>\n                            {bar_html}\n                            </div>\n                            """,
                            unsafe_allow_html=True,
                        )
                        col_score, col_button = st.columns([3, 1])
                        # REPLACED: put save next to features inside same column and remove separate col_button
                        col_score = st.container()
                        with col_score:
                            new_score = st.number_input(
                                "Edit Risk Score",
                                key=f"pred_risk_{pred_id}",
                                value=float(pred.get("predicted_risk") or 0.0),
                                step=0.01,
                                min_value=0.0,
                                max_value=1.0,
                            )
                            feat_col, save_col = st.columns([1,1])
                            with feat_col:
                                if st.button("🔍 Features", key=f"toggle_features_{pred_id}"):
                                    st.session_state.show_features[pred_id] = not st.session_state.show_features.get(pred_id, False)
                            if st.session_state.show_features.get(pred_id):
                                feat = pred.get("features", {})
                                if feat:
                                    st.json({k: v for k, v in feat.items() if k not in ("repo_name",)})
                                else:
                                    st.info("No feature details available.")
                            with save_col:
                                if st.button("💾 Update Risk Score", key=f"update_pred_{pred_id}"):
                                    log_human_feedback(
                                        module=full_module,
                                        repo_name=pred.get("repo_name") or pred.get("features", {}).get("repo_name") or Config.GITHUB_REPO,
                                        predicted_risk=pred.get("predicted_risk", 0.0),
                                        manager_risk=new_score,
                                        prediction_id=str(pred.get("_id")),
                                        user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                    )
                                    pred["predicted_risk"] = new_score
                                    st.session_state.pred_score_status[pred_id] = "updated"
                                    st.success(f"✅ Logged feedback for {full_module} (new risk: {new_score:.2f})")
                        # removed old col_button usage
                        st.markdown("---")
