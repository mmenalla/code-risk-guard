import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv
import os
from dags.src.jira.jira_client import JiraClient

load_dotenv()
MONGO_URI = "mongodb://admin:admin@localhost:27017/risk_model_db?authSource=admin"

# --- MongoDB setup ---
DB_NAME = os.getenv("MONGO_DB", "risk_model_db")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION", "ticket_drafts")

client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]

# --- Initialize Jira client ---
jira_client = JiraClient()

# --- Streamlit UI setup ---
st.set_page_config(page_title="AI Risk Ticket Review", layout="wide")
st.title("🧠 AI Risk Model — Ticket Review Dashboard")

# --- Load tickets ---
tickets = list(collection.find({"is_deleted": False}))

if not tickets:
    st.info("✅ No pending tickets to review.")
else:
    # Track ticket statuses in session state
    if "ticket_status" not in st.session_state:
        st.session_state.ticket_status = {}

    for ticket in tickets:
        ticket_id = str(ticket["_id"])
        status = st.session_state.ticket_status.get(ticket_id, None)

        # Apply visual feedback
        if status == "approved":
            color = "#d4edda"  # light green
        elif status == "deleted":
            color = "#f8d7da"  # light red
        else:
            color = "#ffffff"  # default white

        with st.expander(f"📄 {ticket['title']} — ({ticket['module']})", expanded=True):
            st.markdown(
                f"""
                <div style="background-color:{color}; padding:15px; border-radius:10px;">
                    <strong>Risk Score:</strong> {ticket['risk_score']:.3f}<br>
                    <strong>Module:</strong> {ticket['module']}<br>
                    <strong>Description:</strong> {ticket['description']}
                </div>
                """,
                unsafe_allow_html=True,
            )

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
