import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib
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
                        collection.update_one(
                            {"_id": ObjectId(ticket["_id"])},
                            {"$set": {"risk_score": new_score}}
                        )

                        # --- Log manager feedback ---
                        log_human_feedback(
                            module=ticket["module"],
                            repo_name=ticket.get("repo_name", Config.GITHUB_REPO),
                            predicted_risk=ticket["risk_score"],
                            manager_risk=new_score,
                            prediction_id=str(ticket.get("prediction_id")) if ticket.get("prediction_id") else None,
                            user_id=os.getenv("CURRENT_USER", "manager_ui"),
                        )

                        st.success(f"✅ Updated risk score for {ticket['module']} (new: {new_score:.2f})")

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
        grouped = {}
        for pred in predictions:
            folder = pred.get("folder") or pred.get("module", "unknown_folder")
            grouped.setdefault(folder, []).append(pred)

        if "pred_score_status" not in st.session_state:
            st.session_state.pred_score_status = {}

        for folder, preds in grouped.items():
            valid_scores = [p.get("predicted_risk", 0.0) for p in preds if isinstance(p.get("predicted_risk"), (int, float))]
            avg_risk = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

            with st.expander(f"📁 Folder: {folder} — Avg Risk: {avg_risk:.2f}", expanded=False):
                for pred in preds:
                    pred_id = str(pred["_id"])
                    status = st.session_state.pred_score_status.get(pred_id, None)
                    color = "#fff3cd" if status == "updated" else "#ffffff"

                    risk_score = pred.get("predicted_risk")
                    risk_score_display = f"{risk_score:.2f}" if isinstance(risk_score, (int, float)) else "N/A"

                    with st.expander(f"📄 File: {pred.get('module', 'N/A')} — Risk: {risk_score_display}", expanded=False):
                        st.markdown(
                            f"""
                            <div style="background-color:{color}; padding:15px; border-radius:10px;">
                                <strong>File:</strong> {pred.get('module', 'N/A')}<br>
                                <strong>Predicted Risk:</strong> {risk_score_display}<br>
                                <strong>Additional Info:</strong> {pred.get('additional_info', 'N/A')}<br>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        col_score, col_button = st.columns([3, 1])
                        with col_score:
                            new_score = st.number_input(
                                "Risk Score (Editable)",
                                key=f"pred_risk_{pred_id}",
                                value=float(pred.get("predicted_risk") or 0.0),
                                step=0.01,
                                min_value=0.0,
                                max_value=1.0,
                            )

                        with col_button:
                            if st.button(f"💾 Update Score", key=f"update_pred_{pred_id}"):
                                pred_collection.update_one(
                                    {"_id": ObjectId(pred["_id"])},
                                    {"$set": {"predicted_risk": new_score}}
                                )

                                # Log human feedback
                                log_human_feedback(
                                    module=pred.get("module"),
                                    repo_name=pred.get("repo_name", Config.GITHUB_REPO),
                                    predicted_risk=pred.get("predicted_risk", 0.0),
                                    manager_risk=new_score,
                                    prediction_id=str(pred.get("_id")),
                                    user_id=os.getenv("CURRENT_USER", "manager_ui"),
                                )

                                st.session_state.pred_score_status[pred_id] = "updated"
                                st.success(f"✅ Updated risk score for {pred.get('module')} to {new_score:.2f}")
