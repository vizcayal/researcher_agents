import os
import json
import logging
import time
from typing import List, Dict
import truststore
truststore.inject_into_ssl()
import winsound

# Set environment variables to help with corporate proxies
os.environ['SSL_CERT_FILE'] = ""  # Force use of system store
os.environ['REQUESTS_CA_BUNDLE'] = ""
os.environ['CURL_CA_BUNDLE'] = ""

from smolagents import InferenceClientModel, tool, ToolCallingAgent
from .prompts import COORDINATOR_DIRECTION, SUBAGENT_DIRECTION
from tavily import TavilyClient

logger = logging.getLogger(__name__)

class Coordinator:
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
        subagent_model_id: str = "Qwen/Qwen2.5-Coder-32B-Instruct",
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
        
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        if not self.tavily_key:
            raise ValueError("TAVILY_API_KEY environment variable is missing.")
        
        self.tavily_client = TavilyClient(api_key=self.tavily_key)

    def coordinate(self, user_query: str, research_plan: str, subtasks: List[Dict]) -> str:
        logger.info("Initializing Coordinator and sub-agents...")

        @tool
        def web_search(query: str) -> str:
            """
            Search the web for real-time information using Tavily.
            
            Args:
                query: The search query to look up.
            """
            try:
                response = self.tavily_client.search(query=query, search_depth="advanced", max_results=5)
                results = response.get("results", [])
                formatted_results = []
                logger.info(f"Tavily search results: {results}")
                for res in results:
                    formatted_results.append(f"Title: {res.get('title')}\nURL: {res.get('url')}\nContent: {res.get('content')}\n")
                return "\n---\n".join(formatted_results) if formatted_results else "No relevant results found."
            except Exception as e:
                logger.error(f"Tavily search error: {e}")
                return f"Search failed: {e}"

        findings = []
        
        for task in subtasks:
            subtask_id = task.get('id')
            title = task.get('title')
            description = task.get('description')
            
            logger.info(f"Starting sub-agent for task: {subtask_id}")

            subagent = ToolCallingAgent(
                tools=[web_search],
                model=self.subagent_model,
                add_base_tools=False,
                name=f"subagent_{subtask_id}",
                max_steps=1,
            )
            
            subagent_prompt = SUBAGENT_DIRECTION.format(
                user_query=user_query,
                research_plan=research_plan,
                subtask_id=subtask_id,
                subtask_title=title,
                subtask_description=description,
            )

            try:
                finding = subagent.run(subagent_prompt)
                findings.append(f"FINDINGS FOR TASK {subtask_id}: {title}\n\n{finding}")
                
                # Save individual sub-agent finding
                os.makedirs("research_outputs", exist_ok=True)
                with open(f"research_outputs/subtask_{subtask_id}.txt", "w", encoding="utf-8") as f:
                    f.write(finding)
                logger.info(f"Sub-agent for {subtask_id} complete. Finding saved to research_outputs/subtask_{subtask_id}.txt")
                winsound.Beep(1000, 500)
                
            except Exception as e:
                logger.error(f"Error in sub-agent {subtask_id}: {e}")
                findings.append(f"FINDINGS FOR TASK {subtask_id}: {title}\n\nERROR: Failed to complete task. {e}")

        # Final Synthesis using the model directly
        logger.info("Synthesizing final report...")
        
        system_prompt = COORDINATOR_DIRECTION.format(
            user_query=user_query,
            research_plan=research_plan,
            subtasks_json=json.dumps(subtasks, indent=2)
        )
        
        synthesis_input = "\n\n".join(findings)
        user_prompt = f"Here are the findings from the specialized sub-agents. Please synthesize them into a cohesive final research report as per the original project guidelines.\n\nSUB-AGENT FINDINGS:\n{synthesis_input}"
        
        try:
            response = self.coordinator_model(messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ])
            
            final_report = response.content
            
            # Clean up potential <think> tags if from a reasoning model
            if "<think>" in final_report and "</think>" in final_report:
                final_report = final_report.split("</think>")[-1].strip()
            elif "<think>" in final_report:
                final_report = final_report.split("<think>")[-1].strip()
                if "\n\n" in final_report:
                    final_report = final_report.split("\n\n", 1)[-1]
            
            # Save final report
            with open("research_outputs/final_report.md", "w", encoding="utf-8") as f:
                f.write(final_report)
            logger.info("Final report saved to research_outputs/final_report.md")
                    
            return final_report
        except Exception as e:
            logger.error(f"Error during final synthesis: {e}")
            return f"# Research Output\n\nFailed to synthesize final report. Error: {e}\n\n## Raw Findings\n\n{synthesis_input}"

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
