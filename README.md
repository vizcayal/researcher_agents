---
title: Deep Research Agent
emoji: 🧬
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.31.0
app_file: app.py
pinned: false
---

# Deep Research Agents 🔍

An agentic research pipeline that helps you clarify topics, plan research, and automatically gather information from the web to generate comprehensive Markdown reports.

## 🚀 Features

- **Topic Clarification**: Iteratively refines broad research questions into specific, actionable topics.
- **Strategic Planning**: Generates structured research plans to cover all necessary aspects of a topic.
- **Agentic Coordination**: Uses `smolagents` and **Tavily Search** to orchestrate specialized agents that browse the web and synthesize findings.
- **Robust Model Support**: Specifically optimized for "Reasoning" models (like DeepSeek-R1) and stable tool-calling models (like Qwen-2.5-Coder and MiniMax).
- **Corporate Network Ready**: Includes automated SSL certificate handling via `truststore` to bypass common proxy errors.

## 🛠️ Project Structure

```text
researcher-agents/
├── main.py              # CLI Entry point
├── app.py               # Streamlit UI
├── requirements.txt     # Dependencies
└── src/                 # Core Agents
    ├── clarifier.py     # Topic refinement
    ├── planner.py       # Strategic planning
    ├── splitter.py      # Task decomposition
    ├── coordinator.py   # Agent orchestration
    └── prompts.py       # LLM Instructions
```

## ⚙️ Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vizcayal/researcher_agents.git
   cd researcher_agents
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   HF_KEY=your_huggingface_token
   FIRECRAWL_KEY=your_firecrawl_api_key
   ```

## 📖 Usage

### Command Line Interface
Run the full automated research pipeline:
```bash
python main.py
```

### Streamlit Web App
Run the interactive UI (currently supports Clarification and Planning):
```bash
streamlit run app.py
```

## 🤖 Recommended Models
The project is configured to work with the Hugging Face Serverless Inference API:
- **Reasoning**: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`
- **Orchestration**: `Qwen/Qwen2.5-Coder-32B-Instruct`
- **Output**: `meta-llama/Llama-3.3-70B-Instruct`

## 📄 License
MIT
