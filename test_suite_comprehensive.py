
import sys
import os
import time
from typing import List

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Enable Debug to see which ZIMs are hit
from chatbot import config
config.DEBUG = True

from chatbot import chat
from chatbot.models import Message

# The "Chef's Kiss" 15-Question Comprehensive Test Suite
TEST_CASES = [
    # --- Category 1: FACTUAL / WIKIPEDIA (The Baseline) ---
    {
        "category": "History - Wikipedia",
        "query": "Who was the last Emperor of the Western Roman Empire?"
    },
    {
        "category": "Literature - Wikipedia/Gutenberg",
        "query": "Who devised the 'Turing Test' and in what year?"
    },
    {
        "category": "Tech History - Wikipedia",
        "query": "Who is the author of the Python programming language?"
    },
    
    # --- Category 2: DOMAIN SPECIFIC (Single ZIM Tests) ---
    {
        "category": "Legal - law.stackexchange",
        "query": "What are the elements of a valid contract?"
    },
    {
        "category": "Medical - medicalsciences.stackexchange",
        "query": "What is the half-life of Caffeine in the human body?"
    },
    {
        "category": "Tech/Unix - unix.stackexchange",
        "query": "Explain the difference between 'chmod 777' and 'chmod 755'."
    },
    {
        "category": "Network/Tor - tor.stackexchange",
        "query": "How to configure Tor to use a specific exit node?"
    },

    # --- Category 3: CROSS-DOMAIN / MULTI-ZIM (The Real Test) ---
    {
        "category": "Cross-Domain (Law + Tech)",
        "query": "Is reverse engineering a proprietary driver for interoperability legal in the US?"
        # Needs 'reverseengineering' and 'law' ZIMs
    },
    {
        "category": "Cross-Domain (RPi + Tor)",
        "query": "Can I run a Tor relay on a Raspberry Pi Zero w? Will it be fast enough?"
        # Needs 'raspberrypi' and 'tor' ZIMs
    },
    {
        "category": "Cross-Domain (Sysadmin + General)",
        "query": "What is the difference between TCP and UDP?"
        # Needs 'superuser' or 'serverfault'
    },

    # --- Category 4: REASONING & MULTI-HOP ---
    {
        "category": "Multi-Hop Reasoning",
        "query": "Who was the doctoral advisor of the creator of Python?"
        # Hop 1: Creator of Python -> Guido van Rossum
        # Hop 2: Guido van Rossum -> Advisor
    },
    {
        "category": "Comparative Reasoning",
        "query": "Compare the symptoms of Vitamin D deficiency vs Vitamin B12 deficiency."
        # Needs retrieval of both, then synthesis
    },

    # --- Category 5: CREATIVE / SYNTHESIS ---
    {
        "category": "Creative - Tech Poetry",
        "query": "Write a poem about the struggle of a system administrator dealing with a DDoS attack."
    },
    {
        "category": "Creative - SciFi",
        "query": "Write a short debug log entry for a spaceship AI that is failing to boot due to a corrupted personality matrix."
    },
    
    # --- Category 6: INSTRUCTIONAL ---
    {
        "category": "Instructional - Raspberry Pi",
        "query": "How do I set up a static IP on a Raspberry Pi?"
    }
]

def run_test():
    print(f"\nSTARTING COMPREHENSIVE {len(TEST_CASES)}-QUESTION TEST SUITE")
    print("="*80)
    
    results = []
    
    for i, test in enumerate(TEST_CASES, 1):
        query = test['query']
        category = test['category']
        
        print(f"\n\nTEST #{i} [{category}]")
        print(f"QUERY: {query}")
        print("-" * 80)
        
        start_time = time.time()
        
        # Build context
        system_prompt = "You are a helpful AI assistant. Answer the user's questions based on the provided context."
        history = [Message(role="user", content=query)]
        
        try:
            # 1. Retrieval & Message Build
            messages = chat.build_messages(system_prompt, history, user_query=query)
            
            # 2. Generation
            response = chat.full_chat(config.DEFAULT_MODEL, messages)
            
            duration = time.time() - start_time
            
            print(f"\n[RESPONSE - {duration:.2f}s]")
            print(response)
            
            results.append({
                "id": i,
                "category": category,
                "query": query,
                "duration": duration,
                "status": "SUCCESS"
            })
            
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({
                "id": i,
                "category": category,
                "query": query,
                "duration": 0,
                "status": "FAILED",
                "error": str(e)
            })
            
        print("="*80)
        # Brief pause to allow logs to flush and prevent overheating/thrashing
        time.sleep(2)

    print("\n\nTEST SUITE COMPLETE")
    print(f"Run {len(results)} tests.")

if __name__ == "__main__":
    run_test()
