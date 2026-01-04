import truststore
truststore.inject_into_ssl()
import streamlit as st
import os
import json
import logging
import time
from typing import List, Dict
from dotenv import load_dotenv

# Import our agents from the src directory
from src.clarifier import Clarifier
from src.planner import Planner
from src.splitter import Splitter
from src.coordinator import Coordinator
from src.reviewer import Reviewer

# Configuration and Secrets
load_dotenv()

# Strategy to support both Hugging Face Spaces (st.secrets) and Local (.env/os.getenv)
def get_secret(key):
    try:
        return st.secrets.get(key)
    except Exception:
        return os.getenv(key)

HF_KEY = get_secret("HF_KEY")
TAVILY_API_KEY = get_secret("TAVILY_API_KEY")

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- UI STYLING ---
st.set_page_config(page_title="Deep Research Agent", page_icon="🧬", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }

    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
        background: linear-gradient(90deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #4338ca 0%, #2563eb 100%);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        transform: translateY(-1px);
    }
    .report-container {
        padding: 25px;
        border-radius: 12px;
        background-color: #1a1c23;
        border: 1px solid #2d3748;
    }
    .status-box {
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 12px;
        border-left: 5px solid #4f46e5;
        background-color: #242731;
    }
    </style>
""", unsafe_allow_html=True)

# --- SESSION STATE INITIALIZATION ---
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'initial_topic' not in st.session_state:
    st.session_state.initial_topic = ""
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = []
if 'final_topic' not in st.session_state:
    st.session_state.final_topic = ""
if 'research_plan' not in st.session_state:
    st.session_state.research_plan = ""
if 'subtasks' not in st.session_state:
    st.session_state.subtasks = []
if 'findings' not in st.session_state:
    st.session_state.findings = {}
if 'final_report' not in st.session_state:
    st.session_state.final_report = ""

# --- MODELS ---
CLARIFIER_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
PLANNER_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
SPLITTER_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
COORDINATOR_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
SUBAGENT_MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVIEWER_MODEL = 'Qwen/Qwen2.5-7B-Instruct'

# --- HEADERS ---
st.title("🧬 Deep Research Agent")
st.markdown("### The ultimate AI research pipeline that browses the web for you.")

# --- SIDEBAR CONFIG ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Allow local override of keys if not in secrets
    if not HF_KEY:
        hf_input = st.text_input("Hugging Face Token", type="password")
        if hf_input: HF_KEY = hf_input
    
    if not TAVILY_API_KEY:
        tavily_input = st.text_input("Tavily API Key", type="password")
        if tavily_input: TAVILY_API_KEY = tavily_input
        
    if not HF_KEY or not TAVILY_API_KEY:
        st.error("Please provide both HF_KEY and TAVILY_API_KEY to start.")
        st.stop()
    
    st.success("API Keys connected.")
    
    st.divider()
    if st.button("🔄 Reset Global Research"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# --- STEP 1: INITIAL TOPIC ---
if st.session_state.step == 1:
    st.header("1️⃣ What are you researching?")
    topic_input = st.text_input("Enter a broad topic or research question:", placeholder="e.g., The impact of AI on specialized legal services")
    
    if st.button("Start Clarification"):
        if topic_input:
            st.session_state.initial_topic = topic_input
            with st.spinner("Analyzing topic and generating directions..."):
                clarifier = Clarifier(model_name=CLARIFIER_MODEL, hf_key=HF_KEY)
                st.session_state.suggestions = clarifier.get_suggestions(topic_input)
                st.session_state.step = 2
                st.rerun()
        else:
            st.error("Please enter a topic.")

# --- STEP 2: CLARIFICATION SUGGESTIONS ---
elif st.session_state.step == 2:
    st.header("2️⃣ Refine Your Topic")
    st.write(f"Based on: **{st.session_state.initial_topic}**")
    
    if not st.session_state.suggestions:
        st.warning("No suggestions generated. You can proceed with the original topic.")
        st.session_state.final_topic = st.session_state.initial_topic
        if st.button("Use Original Topic"):
            st.session_state.step = 3
            st.rerun()
    else:
        st.markdown("### Choose a Direction:")
        cols = st.columns(len(st.session_state.suggestions))
        for i, sug in enumerate(st.session_state.suggestions):
            with cols[i]:
                st.subheader(sug['title'])
                st.write(sug['description'])
                if st.button(f"Select Option {i+1}", key=f"sel_{i}"):
                    st.session_state.final_topic = f"{sug['title']}: {sug['description']}"
                    st.session_state.step = 3
                    st.rerun()
        
        st.divider()
        custom_topic = st.text_area("Or type your own refined topic:", value=st.session_state.initial_topic)
        if st.button("Use Custom Topic"):
            st.session_state.final_topic = custom_topic
            st.session_state.step = 3
            st.rerun()

# --- STEP 3: PLAN & SPLIT ---
elif st.session_state.step == 3:
    st.header("3️⃣ Strategy & Task Splitting")
    st.info(f"Targeting: **{st.session_state.final_topic}**")
    
    if not st.session_state.research_plan:
        if st.button("Generate Strategy"):
            with st.spinner("Building research plan..."):
                planner = Planner(model_name=PLANNER_MODEL, hf_key=HF_KEY)
                st.session_state.research_plan = planner.plan(st.session_state.final_topic)
                st.rerun()
    else:
        with st.expander("📝 View Research Strategy", expanded=True):
            st.markdown(st.session_state.research_plan)
        
        if not st.session_state.subtasks:
            if st.button("Decompose into Subtasks"):
                with st.spinner("Splitting plan into independent task modules..."):
                    splitter = Splitter(model_name=SPLITTER_MODEL, hf_key=HF_KEY)
                    st.session_state.subtasks = splitter.split(st.session_state.research_plan)
                    st.rerun()
        else:
            st.markdown("### 📋 Generated Subtasks")
            for task in st.session_state.subtasks:
                st.markdown(f"- **{task['title']}** (ID: `{task['id']}`)")
            
            if st.button("🚀 Execute Research Agents"):
                st.session_state.step = 4
                st.rerun()

# --- STEP 4: COORDINATOR & RESEARCH ---
elif st.session_state.step == 4:
    st.header("4️⃣ Agentic Research in Progress")
    st.warning("Agents are browsing the web using Tavily. This may take several minutes.")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if not st.session_state.final_report:
        # We need to run the coordination manually in the app context to show progress
        coordinator = Coordinator(
            model_name=COORDINATOR_MODEL, 
            subagent_model_id=SUBAGENT_MODEL,
            hf_key=HF_KEY
        )
        # Note: We can't easily call coordinator.coordinate exactly because it loops internally.
        # We'll re-implement the loop here to update the UI.
        
        findings = []
        # Import ToolCallingAgent and web_search logic for the UI loop
        from smolagents import ToolCallingAgent, tool
        from src.prompts import SUBAGENT_DIRECTION, COORDINATOR_DIRECTION
        from tavily import TavilyClient
        
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

        @tool
        def web_search(query: str) -> str:
            """
            Search the web for real-time information using Tavily.
            
            Args:
                query: The search query to look up.
            """
            try:
                response = tavily_client.search(query=query, search_depth="advanced", max_results=5)
                results = response.get("results", [])
                formatted = [f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}\n" for r in results]
                return "\n---\n".join(formatted) if formatted else "No results."
            except Exception as e:
                return f"Search failed: {e}"

        current_findings = []
        for i, task in enumerate(st.session_state.subtasks):
            t_id = task['id']
            t_title = task['title']
            t_desc = task['description']
            
            status_text.markdown(f"**Agent working on:** {t_title}...")
            
            with st.status(f"🔍 Researching: {t_title}", expanded=False) as status:
                subagent = ToolCallingAgent(
                    tools=[web_search],
                    model=coordinator.subagent_model,
                    add_base_tools=False,
                    name=f"subagent_{t_id}",
                    max_steps=2 # Optimized for speed in UI
                )
                prompt = SUBAGENT_DIRECTION.format(
                    user_query=st.session_state.final_topic,
                    research_plan=st.session_state.research_plan,
                    subtask_id=t_id,
                    subtask_title=t_title,
                    subtask_description=t_desc
                )
                try:
                    finding = subagent.run(prompt)
                    current_findings.append(f"FINDINGS FOR TASK {t_id}: {t_title}\n\n{finding}")
                    status.update(label=f"✅ {t_title} complete!", state="complete")
                    st.markdown(finding)
                except Exception as e:
                    status.update(label=f"❌ {t_title} failed", state="error")
                    st.error(f"Error: {e}")
            
            progress_bar.progress((i + 1) / len(st.session_state.subtasks))
        
        status_text.markdown("### ✨ Synthesis: Generating Final Report...")
        with st.spinner("Synthesizing all agent findings into a cohesive document..."):
            sys_prompt = COORDINATOR_DIRECTION.format(
                user_query=st.session_state.final_topic,
                research_plan=st.session_state.research_plan,
                subtasks_json=json.dumps(st.session_state.subtasks, indent=2)
            )
            user_prompt = f"Synthesize these findings:\n\n" + "\n\n".join(current_findings)
            
            try:
                response = coordinator.coordinator_model(messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ])
                final_report = response.content
                # Clean think tags
                if "<think>" in final_report and "</think>" in final_report:
                    final_report = final_report.split("</think>")[-1].strip()
                elif "<think>" in final_report:
                    final_report = final_report.split("<think>")[-1].strip()
                    if "\n\n" in final_report: final_report = final_report.split("\n\n", 1)[-1]
                
                st.session_state.final_report = final_report
                
                # NEW: Review Step
                status_text.markdown("### 🖋️ Review: Polishing and Finalizing...")
                with st.spinner("Reviewer agent is refining the report and preparing PDF..."):
                    reviewer = Reviewer(model_name=REVIEWER_MODEL, hf_key=HF_KEY)
                    polished_report = reviewer.review(final_report)
                    st.session_state.final_report = polished_report
                    
                    # Generate PDF data for session state
                    os.makedirs("temp_outputs", exist_ok=True)
                    pdf_path = f"temp_outputs/research_{int(time.time())}.pdf"
                    if reviewer.generate_pdf(polished_report, pdf_path):
                        with open(pdf_path, "rb") as f:
                            st.session_state.pdf_data = f.read()
                        os.remove(pdf_path) # Clean up
                
                st.session_state.step = 5
                st.rerun()
            except Exception as e:
                st.error(f"Synthesis failed: {e}")

# --- STEP 5: FINAL REPORT ---
elif st.session_state.step == 5:
    st.header("🏁 Final Research Report")
    st.success("Research mission accomplished.")
    
    st.markdown("---")
    st.markdown(st.session_state.final_report)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download Report (Markdown)",
            st.session_state.final_report,
            file_name="final_research_report.md",
            mime="text/markdown"
        )
    with col2:
        if 'pdf_data' in st.session_state:
            st.download_button(
                "📄 Download Report (PDF)",
                st.session_state.pdf_data,
                file_name="final_research_report.pdf",
                mime="application/pdf"
            )
    
    if st.button("Start New Research"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
