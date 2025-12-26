import os
import json
import logging
import time
from typing import List
from pydantic import Field, BaseModel
from huggingface_hub import InferenceClient
from .prompts import SPLITTER_DIRECTION
from pprint import pprint

logger = logging.getLogger(__name__)

class Subtask(BaseModel):
    id: str = Field(
        ...,
        description="Short identifier for the subtask (e.g. 'A', 'history', 'drivers').",
    )
    title: str = Field(
        ...,
        description="Short descriptive title of the subtask.",
    )
    description: str = Field(
        ...,
        description="Clear, detailed instructions for the sub-agent that will research this subtask.",
    )

class SubtaskList(BaseModel):
    subtasks: List[Subtask] = Field(
        ...,
        description="List of subtasks that together cover the whole research plan.",
    )

TASK_SPLITTER_SCHEMA = {
    "name": "subtaskList",
    "schema": SubtaskList.model_json_schema(),
    "strict": True,
}

class Splitter:
    def __init__(self, model_name: str = "moonshotai/Kimi-K2-Thinking", hf_key: str = None):
        self.model_name = model_name
        self.hf_key = hf_key or os.getenv("HF_KEY") or os.getenv("HF_TOKEN")
        self.client = InferenceClient(
            api_key=self.hf_key,
        )

    def split(self, research_plan: str) -> List[dict]:
        logger.info(f"Splitting the research plan into subtasks using {self.model_name}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                full_content = ""
                try:
                    # Using chat_completion with streaming
                    stream = self.client.chat_completion(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": SPLITTER_DIRECTION},
                            {"role": "user", "content": research_plan},
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": TASK_SPLITTER_SCHEMA,
                        } if attempt == 0 else None,
                        max_tokens=4000,
                        stream=True,
                        temperature=0.6,
                        top_p=0.95
                    )

                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            full_content += delta.content
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            if "<think>" not in full_content:
                                full_content += "<think>"
                            full_content += delta.reasoning_content
                        elif hasattr(delta, 'reasoning') and delta.reasoning:
                            if "<think>" not in full_content:
                                full_content += "<think>"
                            full_content += delta.reasoning
                            
                    if "<think>" in full_content and "</think>" not in full_content:
                        full_content += "</think>"
                
                except StopIteration:
                    if not full_content:
                        logger.error(f"Model {self.model_name} returned an empty stream. It may not be supported on the current Inference API endpoint.")
                    else:
                        logger.warning("Stream ended with StopIteration.")
                except Exception as stream_err:
                    logger.error(f"Stream error: {stream_err}")

                if not full_content:
                    logger.warning(f"Empty content on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return []

                subtasks = self._parse_subtasks(full_content)
                if subtasks:
                    return subtasks

            except Exception as e:
                logger.error(f"Error during split (Attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        return []

    def _parse_subtasks(self, content: str) -> List[dict]:
        if not content:
            return []
            
        # Basic cleaning of the response
        clean_content = content.strip()
        
        # Remove DeepSeek/NVIDIA thinking block if present
        if "<think>" in clean_content:
            if "</think>" in clean_content:
                clean_content = clean_content.split("</think>")[-1].strip()
            else:
                parts = clean_content.split("<think>")
                clean_content = parts[-1].strip()
                if "{" in clean_content:
                    clean_content = "{" + clean_content.split("{", 1)[1]

        if "```json" in clean_content:
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_content:
            block_content = clean_content.split("```")[1].split("```")[0].strip()
            if block_content.startswith("{") or "[" in block_content:
                clean_content = block_content

        if not (clean_content.startswith("{") or clean_content.startswith("[")):
            if "{" in clean_content:
                clean_content = "{" + clean_content.split("{", 1)[1]
            if "}" in clean_content:
                clean_content = clean_content.rsplit("}", 1)[0] + "}"

        try:
            content_json = json.loads(clean_content)
            if isinstance(content_json, dict):
                subtasks = content_json.get('subtasks', [])
            elif isinstance(content_json, list):
                subtasks = content_json
            else:
                subtasks = []
            
            if subtasks:
                print("\n\033[93m--- Generated Subtasks ---\033[0m")
                for task in subtasks:
                    print(f"\033[93mID: {task.get('id')} - {task.get('title')}\033[0m")
                    pprint(task.get('description'))
                    print()
                return subtasks
        except json.JSONDecodeError as je:
            logger.error(f"Failed to parse JSON content: {je}. Snippet: {clean_content[:150]}...")
            
        return []

if __name__ == "__main__":
    # Test block
    from dotenv import load_dotenv
    load_dotenv()
    
    test_plan = "Research the current state of Solid State Batteries, focusing on major players and technical hurdles."
    splitter = Splitter()
    result = splitter.split(test_plan)