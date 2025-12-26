import logging
import time
import json
from typing import List, Dict
from huggingface_hub import InferenceClient
from .prompts import CLARIFIER_DIRECTION

logger = logging.getLogger(__name__)

CLARIFICATION_SCHEMA = {
    "name": "clarification_suggestions",
    "schema": {
        "type": "object",
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["title", "description"]
                }
            }
        },
        "required": ["suggestions"]
    },
    "strict": True
}

class Clarifier:
    def __init__(self, model_name: str, hf_key: str):
        self.model_name = model_name
        self.client = InferenceClient(token=hf_key, timeout=120)

    def get_suggestions(self, topic: str) -> List[Dict[str, str]]:
        logger.info(f'Clarifying Topic: {topic} using model {self.model_name}')
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempt {attempt + 1}/{max_retries} using {self.model_name}")
                
                # Use streaming to be more resilient to StopIteration/timeout issues on thinking models
                full_content = ""
                try:
                    # Note: response_format is used to guide the model, but we'll parse manually for robustness
                    stream = self.client.chat_completion(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": CLARIFIER_DIRECTION},
                            {"role": "user", "content": topic}
                        ],
                        max_tokens=2000,
                        stream=True,
                        temperature=1.0,
                        top_p=1.0,
                        # Fallback for models that don't support JSON schema
                        response_format={
                            "type": "json_schema",
                            "json_schema": CLARIFICATION_SCHEMA,
                        } if attempt == 0 else None
                    )
                    
                    for chunk in stream:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            full_content += delta.content
                        # Capture DeepSeek-R1 style reasoning_content
                        if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                            # Wrap in think tags if not already present in the stream to satisfy our parsers
                            if "<think>" not in full_content:
                                full_content += "<think>"
                            full_content += delta.reasoning_content
                        elif hasattr(delta, 'reasoning') and delta.reasoning:
                            if "<think>" not in full_content:
                                full_content += "<think>"
                            full_content += delta.reasoning
                    
                    # Close think tag if it was opened but never closed by the model
                    if "<think>" in full_content and "</think>" not in full_content:
                        full_content += "</think>"
                
                except StopIteration:
                    if not full_content:
                        logger.error(f"Model {self.model_name} returned an empty stream. It may not be supported on the current Inference API endpoint.")
                    else:
                        logger.warning("Stream ended abruptly.")
                except Exception as stream_err:
                    logger.error(f"Streaming error from {self.model_name}: {stream_err}")
                
                if not full_content:
                    logger.warning(f"No content received on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                    return []

                suggestions = self._parse_suggestions(full_content)
                if suggestions:
                    return suggestions
                
                logger.warning(f"Failed to parse suggestions from content on attempt {attempt + 1}")

            except Exception as e:
                logger.error(f"Error during get_suggestions (Attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
        
        return []

    def _parse_suggestions(self, content: str) -> List[Dict[str, str]]:
        if not content:
            return []
            
        logger.debug(f"Raw response: {content}")
        
        # Clean the content: remove thinking blocks and extract JSON
        clean_content = content.strip()
        
        # Remove thinking blocks if present
        if "<think>" in clean_content:
            if "</think>" in clean_content:
                clean_content = clean_content.split("</think>")[-1].strip()
            else:
                # If think is open but not closed, try to find the first JSON object after it
                parts = clean_content.split("<think>")
                clean_content = parts[-1].strip()
                if "{" in clean_content:
                    clean_content = "{" + clean_content.split("{", 1)[1]

        # Extract JSON from potential code blocks
        if "```json" in clean_content:
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_content:
            # Check if it looks like JSON inside a generic code block
            block_content = clean_content.split("```")[1].split("```")[0].strip()
            if block_content.startswith("{") or "[" in block_content:
                clean_content = block_content

        # Final attempt to find a JSON-like structure if it's buried in text
        if not (clean_content.startswith("{") or clean_content.startswith("[")):
            if "{" in clean_content:
                clean_content = "{" + clean_content.split("{", 1)[1]
            if "}" in clean_content:
                clean_content = clean_content.rsplit("}", 1)[0] + "}"

        try:
            data = json.loads(clean_content)
            # Handle potential nested 'suggestions' key or direct list
            if isinstance(data, dict):
                suggestions = data.get("suggestions", [])
            elif isinstance(data, list):
                suggestions = data
            else:
                suggestions = []
                
            logger.info(f"Successfully extracted {len(suggestions)} suggestions")
            return suggestions
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}. Snippet: {clean_content[:150]}...")
            return []

    def clarify(self, topic: str) -> str:
        suggestions = self.get_suggestions(topic)
        
        if not suggestions:
            logger.warning("No suggestions generated, using original topic.")
            return topic

        print("\n\033[93m--- Research Topic Suggestions ---\033[0m")
        for i, sug in enumerate(suggestions, 1):
            print(f"[{i}] {sug.get('title')}")
            print(f"    {sug.get('description')}\n")
        print(f"[0] Use original: {topic}")
        print("\033[93m----------------------------------\033[0m")
        
        while True:
            try:
                user_input = input("Select a choice (number) or enter a custom topic: ").strip()
                
                if user_input == "0":
                    return topic
                
                if user_input.isdigit():
                    idx = int(user_input) - 1
                    if 0 <= idx < len(suggestions):
                        selected = suggestions[idx]
                        # Return the combined title and description as the refined topic
                        final_topic = f"{selected.get('title')}: {selected.get('description')}"
                        logger.info(f"User selected suggestion {user_input}")
                        return final_topic
                    else:
                        print(f"Please enter a number between 0 and {len(suggestions)}.")
                        continue
                
                if user_input:
                    # Treat non-numeric non-empty input as a custom topic
                    logger.info("User provided custom topic.")
                    return user_input
                else:
                    print("Input cannot be empty. Please select a number or type a topic.")

            except EOFError:
                return topic
