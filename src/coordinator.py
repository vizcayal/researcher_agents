import os
import json
import logging
from typing import List, Dict
import truststore
truststore.inject_into_ssl()

# Set environment variables to help with corporate proxies
os.environ['SSL_CERT_FILE'] = ""  # Force use of system store
os.environ['REQUESTS_CA_BUNDLE'] = ""
os.environ['CURL_CA_BUNDLE'] = ""

from smolagents import InferenceClientModel, MCPClient, tool, CodeAgent
from .prompts import COORDINATOR_DIRECTION, SUBAGENT_DIRECTION

logger = logging.getLogger(__name__)

class Coordinator:
    def __init__(
        self, 
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        subagent_model_id: str = "meta-llama/Llama-3.3-70B-Instruct",
        hf_key: str = None
    ):
        self.hf_key = hf_key or os.getenv("HF_KEY") or os.getenv("HF_TOKEN")
        self.coordinator_model = InferenceClientModel(
            model_id=model_name,
            api_key=self.hf_key,
        )
        self.subagent_model = InferenceClientModel(
            model_id=subagent_model_id,
            api_key=self.hf_key,
        )
        
        firecrawl_key = os.getenv("FIRECRAWL_KEY")
        if not firecrawl_key:
            raise ValueError("FIRECRAWL_KEY environment variable is missing.")
        
        self.mcp_url = f"https://mcp.firecrawl.dev/{firecrawl_key}/v2/mcp"

    def coordinate(self, user_query: str, research_plan: str, subtasks: List[Dict]) -> str:
        logger.info("Initializing Coordinator and sub-agents...")
        
        with MCPClient({"url": self.mcp_url, "transport": "streamable-http"}) as mcp_tools:
            
            @tool
            def research_subtask(subtask_id: str, title: str, description: str) -> str:
                """
                A tool that performs deep research on a specific subtask. 
                It spawns a specialized agent to handle just this part of the plan.

                Args:
                    subtask_id (str): The unique ID of the subtask.
                    title (str): The title of the subtask.
                    description (str): Detailed research instructions for this subtask.
                """
                logger.info(f"Starting sub-agent for task: {subtask_id}")

                subagent = CodeAgent(
                    tools=mcp_tools,
                    model=self.subagent_model,
                    add_base_tools=True,
                    name=f"subagent_{subtask_id}",
                    max_steps=5
                )
                
                subagent_prompt = SUBAGENT_DIRECTION.format(
                    user_query=user_query,
                    research_plan=research_plan,
                    subtask_id=subtask_id,
                    subtask_title=title,
                    subtask_description=description,
                )

                return subagent.run(subagent_prompt)

            coordinator_agent = CodeAgent(
                tools=[research_subtask],
                model=self.coordinator_model,
                add_base_tools=True,
                name="coordinator_agent"
            )

            # Define the task for the coordinator
            coordinator_task = (
                f"Your mission is to complete the research report for: '{user_query}'.\n"
                "The 'subtasks' variable contains a list of research components to be investigated.\n"
                "STEP-BY-STEP PROCESS:\n"
                "1. Loop through every item in the `subtasks` list.\n"
                "2. For each item, call the `research_subtask` function using its 'id', 'title', and 'description'.\n"
                "3. Append the result of each call into a list of findings.\n"
                "4. Once all subtasks are finished, combine all captured findings into a single, polished Markdown report.\n"
                "5. Ensure the report follows the GUIDELINES (Headings, Bibliography, Cohesion).\n\n"
                "DO NOT define any new classes or complex structures. Just use a simple loop and the provided `research_subtask` function."
            )

            # Run with variables passed directly to the interpreter scope
            final_report = coordinator_agent.run(
                coordinator_task,
                additional_args={
                    "user_query": user_query,
                    "research_plan": research_plan,
                    "subtasks": subtasks
                }
            )
            return final_report

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Example usage (would require valid keys and plan)
    # coord = Coordinator()
    # report = coord.coordinate(user_query="...", research_plan="...", subtasks=[...])
    # print(report)