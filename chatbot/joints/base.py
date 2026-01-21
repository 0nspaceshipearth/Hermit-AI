
import json
import re
from typing import Dict, List, Optional, Any
from chatbot import config
from chatbot.model_manager import ModelManager

def debug_print(joint_name: str, msg: str):
    """Print debug message for a specific joint."""
    if config.DEBUG:
        print(f"[\033[94m{joint_name}\033[0m] {msg}")

def extract_json_from_text(text: str) -> Optional[Any]:
    """
    Robustly extract the first valid JSON object or array from text.
    Handles Markdown code blocks, conversational filler, and nested structures.
    """
    if not text:
        return None
        
    # Remove markdown code blocks if present
    text = re.sub(r'```(?:json)?\s*(.*?)\s*```', r'\1', text, flags=re.DOTALL)
    
    # Try to find the first '{' or '['
    start_match = re.search(r'[\{\[]', text)
    if not start_match:
        return None
        
    start_idx = start_match.start()
    
    # Try parsing from the start index onwards
    # We'll try to find the balancing closing bracket/brace
    # but simplest is to just try parsing increasingly smaller substrings
    content = text[start_idx:]
    
    # Match pairs of braces/brackets to find potential JSON end
    stack = []
    end_idx = -1
    for i, char in enumerate(content):
        if char in '{[':
            stack.append(char)
        elif char in '}]':
            if not stack: continue
            opener = stack.pop()
            if (opener == '{' and char == '}') or (opener == '[' and char == ']'):
                if not stack:
                    end_idx = i + 1
                    break
    
    if end_idx != -1:
        try:
            return json.loads(content[:end_idx])
        except json.JSONDecodeError:
            pass

    # Fallback to older method if stack-based fails or doesn't find a clean end
    # Match outermost braces
    try:
        # Try to find the last '}' or ']'
        last_brace = text.rfind('}')
        last_bracket = text.rfind(']')
        end_idx_fallback = max(last_brace, last_bracket)
        
        if end_idx_fallback > start_idx:
            return json.loads(text[start_idx:end_idx_fallback+1])
    except:
        pass
        
    return None

def local_inference(model: str, prompt: str, temperature: float = 0.0, timeout: int = 5, use_json_grammar: bool = False):
    """
    Run local inference using ModelManager.
    Uses chat completion to avoid KV cache contamination.
    """
    # Use larger context size for joints to handle retrieved content
    n_ctx = 4096  # Increased from 2048 to prevent overflow
    try:
        llm = ModelManager.get_model(model, n_ctx=n_ctx)
        
        # Use chat completion to avoid KV cache issues
        messages = [
            {"role": "system", "content": "You are a precise JSON extraction system. Output only valid JSON."},
            {"role": "user", "content": prompt}
        ]
        
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=temperature
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        debug_print("BASE:INFERENCE", f"Inference failed: {e}")
        raise e

