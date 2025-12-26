import truststore
truststore.inject_into_ssl()
import streamlit as st
import os
from dotenv import load_dotenv
from src.clarifier import Clarifier
from src.planner import Planner

# Load environment variables
load_dotenv()
HF_KEY = os.getenv("HF_KEY")

# Page Config
st.set_page_config(page_title="Deep Research Agent", page_icon="🔍", layout="centered")

# Custom CSS for better aesthetics
# Custom CSS for Premium Aesthetics
st.markdown("""
    <style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: #3b3b55;
        color: #ffffff;
        border: 1px solid #555;
        border-radius: 8px;
        padding: 10px;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.4);
    }
    
    /* Headers */
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff8a00, #e52e71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-bottom: 1rem;
    }
    
    /* Success/Info Messages */
    .stAlert {
        background-color: rgba(60, 60, 80, 0.8);
        border: 1px solid #555;
        color: #eee;
        border-radius: 10px;
    }
    
    /* Sub-headers */
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ff8a00;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #333;
        padding-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">Deep Research Agent 🔍</div>', unsafe_allow_html=True)
st.markdown("Generate comprehensive research plans using AI agents.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    if not HF_KEY:
        HF_KEY = st.text_input("Enter HuggingFace API Key", type="password")
        if HF_KEY:
             os.environ["HF_KEY"] = HF_KEY
    
    model_name = st.text_input("Model Name", value='deepseek-ai/DeepSeek-R1-Distill-Llama-8B')

if not HF_KEY:
    st.warning("Please provide a HuggingFace API Token in the sidebar or .env file to proceed.")
    st.stop()

# Session State
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = None
if 'final_topic' not in st.session_state:
    st.session_state.final_topic = ""

# Step 1: Input Topic
st.markdown('<div class="sub-header">1. Define Topic</div>', unsafe_allow_html=True)
initial_topic = st.text_input("Enter your broad research topic:", placeholder="e.g., The future of renewable energy in Southeast Asia")

if st.button("Clarify Topic"):
    if initial_topic:
        with st.spinner("Consulting with Clarifier Agent..."):
            clarifier = Clarifier(model_name=model_name, hf_key=HF_KEY)
            suggestions = clarifier.get_suggestions(initial_topic)
            st.session_state.suggestions = suggestions
    else:
        st.error("Please enter a topic first.")

# Step 2: Refine & Select
if st.session_state.suggestions:
    st.markdown('<div class="sub-header">2. Refine Logic</div>', unsafe_allow_html=True)
    st.markdown("The agent has analyzed your topic and suggested the following directions:")
    
    for i, sug in enumerate(st.session_state.suggestions, 1):
        with st.expander(f"Option {i}: {sug.get('title')}", expanded=True):
            st.write(sug.get('description'))
            if st.button(f"Use Option {i}", key=f"btn_{i}"):
                st.session_state.final_topic = f"{sug.get('title')} - {sug.get('description')}"
                st.rerun()
    
    st.session_state.final_topic = st.text_area("Refined Research Topic (Edit below or use suggestions):", value=initial_topic if not st.session_state.final_topic else st.session_state.final_topic, height=100)

    # Step 3: Generate Plan
    if st.button("Generate Research Plan"):
        if st.session_state.final_topic:
            with st.spinner("Planner Agent is creating your strategy..."):
                planner = Planner(model_name=model_name, hf_key=HF_KEY)
                plan = planner.plan(st.session_state.final_topic)
                
                st.markdown('<div class="sub-header">3. Research Plan</div>', unsafe_allow_html=True)
                st.markdown(plan)
                
                # Download button
                st.download_button(
                    label="Download Plan",
                    data=plan,
                    file_name="research_plan.md",
                    mime="text/markdown"
                )
        else:
            st.error("Please define a final topic.")
