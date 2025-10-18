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

st.set_page_config(page_title="AI Risk Ticket Review", layout="wide")
st.title("AI Risk Model — Ticket Review Dashboard")

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

            # --- Editable risk score with update button beside it ---
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
                        repo_name=ticket.get(Config().GITHUB_REPO, "unknown_repo"),
                        predicted_risk=ticket["risk_score"],
                        manager_risk=new_score,
                        prediction_id=ticket.get("prediction_id"),
                        user_id=os.getenv("CURRENT_USER", "manager_ui"),
                    )

                    st.success(f"✅ Updated risk score for {ticket['module']} (new: {new_score:.2f})")

            # --- Approve or delete buttons ---
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
                        repo_name=ticket.get(Config().GITHUB_REPO, "unknown_repo"),
                        predicted_risk=ticket["risk_score"],
                        manager_risk=0.0,  # Deleted ticket treated as risk 0
                        prediction_id=ticket.get("prediction_id"),
                        user_id=os.getenv("CURRENT_USER", "manager_ui"),
                    )
