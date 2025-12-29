import truststore
truststore.inject_into_ssl()
import logging
import os
import json
os.environ['SSL_CERT_FILE'] = ""
os.environ['REQUESTS_CA_BUNDLE'] = ""
os.environ['CURL_CA_BUNDLE'] = ""
import time
from dotenv import load_dotenv
from src.planner import Planner
from src.clarifier import Clarifier
from src.splitter import Splitter
from src.coordinator import Coordinator

# Load environment variables from .env file
load_dotenv()
HF_KEY = os.getenv("HF_KEY")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CLARIFIER_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
PLANNER_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
SPLITTER_MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
COORDINATOR_MODEL = 'MiniMaxAI/MiniMax-M1-80k'
SUBAGENT_MODEL = 'MiniMaxAI/MiniMax-M1-80k'

if __name__ == "__main__":
    start_time = time.perf_counter()
    
    print("\n\033[93m--- Deep Research Agent ---\033[0m")

    initial_topic = input("Enter a research topic: ")
    
    # Clarify Topic
    clarifier = Clarifier(model_name=CLARIFIER_MODEL, hf_key=HF_KEY)
    final_topic = clarifier.clarify(topic=initial_topic)

    planner = Planner(model_name=PLANNER_MODEL, hf_key=HF_KEY)
    plan = planner.plan(topic=final_topic)
    
    # Save research plan
    os.makedirs("research_outputs", exist_ok=True)
    with open("research_outputs/research_plan.txt", "w", encoding="utf-8") as f:
        f.write(plan)
    logger.info("Research plan saved to research_outputs/research_plan.txt")

    splitter = Splitter(model_name=SPLITTER_MODEL, hf_key=HF_KEY)
    subtasks = splitter.split(plan)

    if not subtasks:
        logger.error("No subtasks generated. Exiting.")
        exit(1)
        
    # Save subtasks
    with open("research_outputs/subtasks.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(subtasks, indent=2))
    logger.info("Subtasks saved to research_outputs/subtasks.txt")

    coordinator = Coordinator(
        model_name=COORDINATOR_MODEL, 
        subagent_model_id=SUBAGENT_MODEL,
        hf_key=HF_KEY
    )
    report = coordinator.coordinate(user_query=final_topic, research_plan=plan, subtasks=subtasks)

    print("\n\033[93m--- Final Research Report ---\033[0m")
    print(report)
    print("\033[93m-----------------------------\033[0m")
    
    elapsed_time = time.perf_counter() - start_time
    print(f"\n\033[93m--- Research complete ({elapsed_time:.2f}s) ---\033[0m")
    logger.info(f"Research planning took {elapsed_time:.2f} seconds.")