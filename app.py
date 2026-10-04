import streamlit as st
import json
import os
from tools.weather import get_weather_data
from tools.geodata import get_location_data
from tools.retrieval import search_emergency_knowledge
from agents.crew_setup import run_rescue_mission

# Page Configuration
st.set_page_config(
    page_title="RescueAI — Emergency Mission Control",
    page_icon="🚨",
    layout="wide"
)

# Header Banner
st.title("🚨 RescueAI — Emergency Decision-Support System")
st.caption("Multi-Agent Emergency Response System • Hackathon MVP v1.0")

# Disclaimer Banner (PRD Section 5 & 12)
st.warning("⚠️ **DECISION-SUPPORT SYSTEM ONLY:** RescueAI does not replace official emergency services, medical diagnosis, or automated dispatch. In real emergencies, immediately dial 112/911.")

# Sidebar Configuration
st.sidebar.header("🔑 Credentials & Settings")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""))

st.sidebar.markdown("---")
st.sidebar.header("📋 Demo Scenarios")

# Load Preset Scenarios
scenarios_path = "data/scenarios.json"
scenarios = []
if os.path.exists(scenarios_path):
    with open(scenarios_path, "r") as f:
        scenarios = json.load(f)

selected_scenario_name = st.sidebar.selectbox(
    "Load Hackathon Demo Scenario",
    options=["Custom Input"] + [s["name"] for s in scenarios]
)

default_desc = ""
default_loc = ""

if selected_scenario_name != "Custom Input":
    for sc in scenarios:
        if sc["name"] == selected_scenario_name:
            default_desc = sc["description"]
            default_loc = sc["location"]

# Input Panel
st.subheader("📥 Emergency Intake Panel")
col1, col2 = st.columns([3, 1])

with col1:
    emergency_text = st.text_area("Emergency Incident Report", value=default_desc, height=120, placeholder="Describe the situation, estimated injuries, visible hazards...")

with col2:
    location_text = st.text_input("Location / Region", value=default_loc, placeholder="e.g., Bahawalpur, Punjab")

run_btn = st.button("🚀 Coordinate Response Plan", type="primary", use_container_width=True)

# Processing Logic
if run_btn:
    if not gemini_api_key:
        st.error("Please provide a valid Gemini API Key in the sidebar.")
    elif not emergency_text or not location_text:
        st.error("Please enter both emergency description and location.")
    else:
        with st.spinner("Multi-Agent Crew orchestrating context, geodata, RAG, and safety planning..."):
            try:
                results = run_rescue_mission(emergency_text, location_text, gemini_api_key)
                
                st.success("Mission Coordination Complete!")

                # Layout: Split Mission Control Display
                left_col, right_col = st.columns([2, 1])

                with left_col:
                    st.subheader("📋 Prioritized Response Action Plan")
                    st.markdown(results["final_plan"])

                with right_col:
                    st.subheader("🌐 Context & Knowledge Evidence")
                    
                    # Geodata Card
                    st.markdown("#### 📍 Geodata Context")
                    geo = results["geodata"]
                    st.json({
                        "Location": geo.get("display_name", location_text),
                        "Latitude": geo.get("latitude"),
                        "Longitude": geo.get("longitude"),
                        "Status": geo.get("status")
                    })

                    # Weather Card
                    st.markdown("#### 🌤️ Weather Conditions (Open-Meteo)")
                    w = results["weather"]
                    st.json({
                        "Temperature": f"{w.get('temperature_c', 'N/A')} °C",
                        "Wind Speed": f"{w.get('windspeed_kmh', 'N/A')} km/h",
                        "Status": w.get("status")
                    })

                    # RAG Knowledge Base Sources
                    st.markdown("#### 📚 Retrieved RAG Documents (FAISS)")
                    for i, source in enumerate(results["rag_sources"], 1):
                        meta = source.get("metadata", {})
                        st.info(f"**Source {i}: {meta.get('title', 'SOP Guidelines')}**\n\n{source.get('content')}")

            except Exception as e:
                st.error(f"Error executing mission workflow: {str(e)}")
