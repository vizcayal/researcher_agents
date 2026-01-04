import logging
import time
import json
from huggingface_hub import InferenceClient
from .prompts import PLANNER_DIRECTION

logger = logging.getLogger(__name__)

class Planner:
    def __init__(self, model_name: str, hf_key: str):
        self.model_name = model_name
        self.client = InferenceClient(token=hf_key, timeout=120)

    def plan(self, topic: str) -> str:
        logger.info(f'Starting Planning: {topic} using model {self.model_name}')
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                full_content = ""
                # Use streaming for robustness with reasoning/large models
                try:
                    stream = self.client.chat_completion(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": PLANNER_DIRECTION},
                            {"role": "user", "content": topic}
                        ],
                        max_tokens=4000,
                        stream=True,
                        temperature=1.0,
                        top_p=1.0
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
                    logger.error(f"Streaming failed: {stream_err}")

                if not full_content:
                    logger.warning(f"Empty content received on attempt {attempt + 1}")
                    continue

                research_plan = self._parse_plan(full_content)
                
                if not research_plan:
                    logger.warning(f"Failed to parse plan from content on attempt {attempt + 1}")
                    continue

                logger.info("Generated research plan")
                return research_plan
            except Exception as e:
                logger.error(f"Error during API call (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                     return ""
        return ""

    def _parse_plan(self, content: str) -> str:
        if not content:
            return ""
            
        # Clean the content: remove thinking blocks and extract JSON
        clean_content = content.strip()
        
        # Remove thinking blocks if present
        if "<think>" in clean_content:
            if "</think>" in clean_content:
                clean_content = clean_content.split("</think>")[-1].strip()
            else:
                parts = clean_content.split("<think>")
                clean_content = parts[-1].strip()
                if "{" in clean_content:
                    clean_content = "{" + clean_content.split("{", 1)[1]

        # Extract JSON from potential code blocks
        if "```json" in clean_content:
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_content:
            block_content = clean_content.split("```")[1].split("```")[0].strip()
            if block_content.startswith("{") or "research_plan" in block_content:
                clean_content = block_content

        # Final attempt to find a JSON-like structure
        if not clean_content.startswith("{"):
            if "{" in clean_content:
                clean_content = "{" + clean_content.split("{", 1)[1]
            if "}" in clean_content:
                clean_content = clean_content.rsplit("}", 1)[0] + "}"

        try:
            data = json.loads(clean_content)
            if isinstance(data, dict):
                return data.get("research_plan", clean_content)
            return clean_content
        except json.JSONDecodeError:
            # Fallback for models that don't return valid JSON
            return content.split("</think>")[-1].strip() if "</think>" in content else content.strip()
