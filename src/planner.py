import logging
import time
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

                # Remove DeepSeek/NVIDIA thinking block if present
                research_plan = full_content.strip()
                if "<think>" in research_plan and "</think>" in research_plan:
                    logger.info("Detected thinking block in plan, extracting final content...")
                    research_plan = research_plan.split("</think>")[-1].strip()
                elif "<think>" in research_plan:
                    # Handle unclosed think blocks
                    research_plan = research_plan.split("<think>")[-1].strip()
                    if "\n\n" in research_plan: # Guess where text might start if logic is cut
                         research_plan = research_plan.split("\n\n", 1)[-1]

                logger.info("Generated research plan")
                print("\n\033[93m--- Research Plan ---\033[0m")
                print(research_plan)
                print("\033[93m---------------------\033[0m")
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
