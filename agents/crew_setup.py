import os
from crewai import Agent, Task, Crew, Process, LLM
from tools.weather import get_weather_data
from tools.geodata import get_location_data
from tools.retrieval import search_emergency_knowledge

def initialize_llm(api_key: str):
    """Initialize LLM using CrewAI's native LLM wrapper with proper Gemini model naming."""
    # Environment variable explicitly set karein taake LiteLLM backend read kar sake
    os.environ["GEMINI_API_KEY"] = api_key
    
    return LLM(
        model="gemini/gemini-1.5-flash-latest",
        api_key=api_key,
        temperature=0.2
    )

def run_rescue_mission(report: str, location_query: str, api_key: str):
    llm = initialize_llm(api_key)
    
    # Tool collection step
    geo_res = get_location_data(location_query)
    lat, lon = geo_res.get("latitude", 29.3956), geo_res.get("longitude", 71.6836)
    weather_res = get_weather_data(lat, lon)
    rag_res = search_emergency_knowledge(report)

    # CrewAI Agents Setup
    incident_analyst = Agent(
        role="Incident Analyst",
        goal="Extract incident facts, casualties, hazards, and missing details accurately.",
        backstory="An expert triage evaluator skilled at extracting structured facts from emergency text.",
        llm=llm,
        verbose=True
    )

    location_agent = Agent(
        role="Location & Context Agent",
        goal="Analyze geographical positioning, risk terrain, and location details.",
        backstory="A spatial logistics analyst verifying coordinates and surrounding terrain.",
        llm=llm,
        verbose=True
    )

    environment_agent = Agent(
        role="Environment Agent",
        goal="Evaluate weather conditions and environmental impact on rescue safety.",
        backstory="A meteorological and environmental responder assessing scene safety.",
        llm=llm,
        verbose=True
    )

    knowledge_agent = Agent(
        role="Knowledge & RAG Agent",
        goal="Synthesize formal safety protocols and standard operating procedures.",
        backstory="A domain specialist matching emergency situations against verified SOP manuals.",
        llm=llm,
        verbose=True
    )

    planner = Agent(
        role="Response Planner",
        goal="Synthesize all gathered evidence into an actionable, prioritized action plan.",
        backstory="A senior incident commander synthesizing multi-agent data into step-by-step actions.",
        llm=llm,
        verbose=True
    )

    safety_reviewer = Agent(
        role="Safety Reviewer",
        goal="Audit response plans, eliminate hallucinations/unsafe assumptions, and flag uncertainties.",
        backstory="A rigorous compliance safety reviewer ensuring decision support remains within safe bounds.",
        llm=llm,
        verbose=True
    )

    # Define Tasks
    t1 = Task(
        description=f"Analyze emergency report: '{report}'. Structured facts required: Type, Casualties, Severity, Hazards.",
        expected_output="Structured summary of incident severity, hazards, and facts.",
        agent=incident_analyst
    )

    t2 = Task(
        description=f"Evaluate location context with geodata: {geo_res}.",
        expected_output="Geographic assessment and positional risks.",
        agent=location_agent
    )

    t3 = Task(
        description=f"Evaluate weather hazards using weather data: {weather_res}.",
        expected_output="Weather impact report for emergency response teams.",
        agent=environment_agent
    )

    t4 = Task(
        description=f"Evaluate matching protocols retrieved: {rag_res}.",
        expected_output="Applicable guidelines and protocols summarized.",
        agent=knowledge_agent
    )

    t5 = Task(
        description="Synthesize input from incident analyst, location, environment, and knowledge agents into a prioritized action plan.",
        expected_output="A step-by-step prioritized emergency action plan.",
        agent=planner
    )

    t6 = Task(
        description="Review the generated response plan. Remove unverified claims, insert uncertainty disclaimers, and format as safe decision support.",
        expected_output="Final, safe, multi-agent emergency response plan with disclaimers.",
        agent=safety_reviewer
    )

    rescue_crew = Crew(
        agents=[incident_analyst, location_agent, environment_agent, knowledge_agent, planner, safety_reviewer],
        tasks=[t1, t2, t3, t4, t5, t6],
        process=Process.sequential,
        verbose=True
    )

    final_plan = rescue_crew.kickoff()
    
    return {
        "final_plan": str(final_plan),
        "geodata": geo_res,
        "weather": weather_res,
        "rag_sources": rag_res
    }
