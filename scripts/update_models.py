#!/usr/bin/env python3
"""
Script to update model configurations in models.json
Fetches real model information from APIs when available.
"""

import json
import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

def fetch_live_models() -> Dict:
    """Fetch actual model information from APIs."""
    live_models = {}
    
    # Try OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            print("  Fetching OpenAI models from API...")
            headers = {"Authorization": f"Bearer {api_key}"}
            response = requests.get(
                "https://api.openai.com/v1/models",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    # Filter for relevant models
                    if any(x in model_id for x in ["gpt-5", "gpt-4", "gpt-3.5"]):
                        live_models[model_id] = {"found_via": "openai_api"}
                        print(f"    ✓ Found: {model_id}")
        except Exception as e:
            print(f"  Warning: Could not fetch OpenAI models: {e}")
    
    # Anthropic doesn't have a models endpoint, but we can test known models
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        print("  Checking Anthropic models...")
        test_models = [
            "claude-opus-4-1-20250805",
            "claude-opus-4-20250514", 
            "claude-sonnet-4-20250514",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022"
        ]
        
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        for model_id in test_models:
            try:
                response = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": "test"}],
                        "max_tokens": 1,
                        "temperature": 0
                    },
                    timeout=5
                )
                
                if response.status_code in [200, 400, 429]:  # Model exists
                    live_models[model_id] = {"found_via": "anthropic_api"}
                    print(f"    ✓ Verified: {model_id}")
            except:
                pass
    
    return live_models

def get_context_window(model_id: str) -> int:
    """Get context window size for a model."""
    # Claude Sonnet 4 has 1M in beta
    if "claude-sonnet-4" in model_id:
        return 1000000
    # All other Claude models at 200K
    if "claude" in model_id:
        return 200000
    # GPT-4.1 series has 1M
    if "gpt-4.1" in model_id:
        return 1000000
    # GPT-5 series has 200K (except nano)
    if "gpt-5" in model_id:
        return 128000 if "nano" in model_id else 200000
    # GPT-4o and turbo have 128K
    if any(x in model_id for x in ["gpt-4o", "turbo", "0125"]):
        return 128000
    # Old GPT-4 has 8K
    if "gpt-4-0613" in model_id:
        return 8192
    # GPT-3.5
    if "gpt-3.5" in model_id:
        return 16385
    
    return 8192  # Conservative default

def update_models():
    """Update model configurations with latest versions."""
    
    models_path = Path(__file__).parent.parent / "agentic_coder" / "models.json"
    
    # Load existing data
    existing_data = {}
    if models_path.exists():
        try:
            with open(models_path, 'r') as f:
                existing_data = json.load(f)
        except:
            pass
    
    # Try to fetch live models
    live_models = fetch_live_models()
    
    # Update context windows for any new models found
    context_windows = existing_data.get("context_windows", {})
    
    for model_id in live_models:
        if model_id not in context_windows:
            context_windows[model_id] = get_context_window(model_id)
            print(f"  Added context window for new model: {model_id}")
    
    # Build the configuration
    models_config = {
        "last_updated": datetime.now().strftime("%Y-%m-%d"),
        "context_windows": context_windows,
        "providers": {
            "anthropic": {
                "current_default": "claude-opus-4-1-20250805",
                "aliases": {
                    # Super simple aliases
                    "claude": "claude-opus-4-1-20250805",
                    
                    # Claude 4 models
                    "claude-4": "claude-opus-4-1-20250805",
                    "claude-4-opus": "claude-opus-4-1-20250805",
                    "claude-4-opus-latest": "claude-opus-4-1-20250805",
                    "claude-opus-4.1": "claude-opus-4-1-20250805",
                    "claude-opus-4": "claude-opus-4-20250514",
                    "claude-4-sonnet": "claude-sonnet-4-20250514",
                    "claude-4-sonnet-latest": "claude-sonnet-4-20250514",
                    "claude-sonnet-4": "claude-sonnet-4-20250514",
                    
                    # Claude 3.5 models
                    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
                    "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
                    "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
                    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
                    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
                    "claude-3.5-haiku": "claude-3-5-haiku-20241022",
                    
                    # Claude 3 models
                    "claude-3-opus": "claude-3-opus-20240229",
                    "claude-3-opus-latest": "claude-3-opus-20240229",
                    "claude-3-sonnet": "claude-3-sonnet-20240229",
                    "claude-3-sonnet-latest": "claude-3-sonnet-20240229",
                    "claude-3-haiku": "claude-3-haiku-20240307",
                    "claude-3-haiku-latest": "claude-3-haiku-20240307",
                    
                    # Short aliases
                    "opus": "claude-opus-4-1-20250805",
                    "sonnet": "claude-sonnet-4-20250514",
                    "haiku": "claude-3-5-haiku-20241022"
                },
                "failover_chains": {
                    # Claude 4 failover chains
                    "claude-opus-4-1-20250805": [
                        "claude-opus-4-1-20250805",
                        "claude-opus-4-20250514",
                        "claude-sonnet-4-20250514",
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022"
                    ],
                    "claude-opus-4-20250514": [
                        "claude-opus-4-20250514",
                        "claude-sonnet-4-20250514",
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022"
                    ],
                    "claude-sonnet-4-20250514": [
                        "claude-sonnet-4-20250514",
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022"
                    ],
                    
                    # Claude 3.5 failover chains
                    "claude-3-5-sonnet-20241022": [
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022"
                    ],
                    "claude-3-5-haiku-20241022": [
                        "claude-3-5-haiku-20241022"
                    ],
                    
                    # Claude 3 failover chains
                    "claude-3-opus-20240229": [
                        "claude-3-opus-20240229",
                        "claude-3-5-sonnet-20241022",
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307"
                    ],
                    "claude-3-sonnet-20240229": [
                        "claude-3-sonnet-20240229",
                        "claude-3-haiku-20240307"
                    ],
                    "claude-3-haiku-20240307": [
                        "claude-3-haiku-20240307"
                    ]
                }
            },
            "openai": {
                "current_default": "gpt-5",
                "aliases": {
                    # Super simple alias
                    "gpt": "gpt-5",
                    
                    # GPT-5 models
                    "gpt-5": "gpt-5",
                    "gpt-5-mini": "gpt-5-mini",
                    "gpt-5-nano": "gpt-5-nano",
                    "gpt-5-thinking": "gpt-5-thinking",
                    "gpt-5-thinking-mini": "gpt-5-thinking-mini",
                    "gpt-5-thinking-nano": "gpt-5-thinking-nano",
                    
                    # GPT-4.1 models (newest)
                    "gpt-4.1": "gpt-4.1",
                    "gpt-4.1-mini": "gpt-4.1-mini",
                    "gpt-4.1-nano": "gpt-4.1-nano",
                    
                    # GPT-4 models
                    "gpt-4": "gpt-4-0613",
                    "gpt-4o": "gpt-4o",
                    "gpt-4o-mini": "gpt-4o-mini",
                    "gpt-4-turbo": "gpt-4-turbo-2024-04-09",
                    "gpt-4-turbo-preview": "gpt-4-0125-preview",
                    
                    # GPT-3.5 models
                    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",
                    "gpt-3.5": "gpt-3.5-turbo-0125"
                },
                "failover_chains": {
                    # GPT-5 failover chains (5 -> 4 -> 3.5)
                    "gpt-5": [
                        "gpt-5",
                        "gpt-5-mini",
                        "gpt-5-nano",
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-5-mini": [
                        "gpt-5-mini",
                        "gpt-5-nano",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-5-nano": [
                        "gpt-5-nano",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-5-thinking": [
                        "gpt-5-thinking",
                        "gpt-5-thinking-mini",
                        "gpt-5-thinking-nano",
                        "gpt-5",
                        "gpt-4o",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-5-thinking-mini": [
                        "gpt-5-thinking-mini",
                        "gpt-5-thinking-nano",
                        "gpt-5-mini",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-5-thinking-nano": [
                        "gpt-5-thinking-nano",
                        "gpt-5-nano",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    
                    # GPT-4.1 failover chains (newest)
                    "gpt-4.1": [
                        "gpt-4.1",
                        "gpt-4.1-mini",
                        "gpt-4.1-nano",
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-4.1-mini": [
                        "gpt-4.1-mini",
                        "gpt-4.1-nano",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-4.1-nano": [
                        "gpt-4.1-nano",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    
                    # GPT-4o failover chains
                    "gpt-4o": [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo-2024-04-09",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-4-turbo-2024-04-09": [
                        "gpt-4-turbo-2024-04-09",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-4-0613": [
                        "gpt-4-0613",
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-4o-mini": [
                        "gpt-4o-mini",
                        "gpt-3.5-turbo-0125"
                    ],
                    "gpt-3.5-turbo-0125": [
                        "gpt-3.5-turbo-0125"
                    ]
                }
            }
        }
    }
    
    # Write to models.json in the agentic_coder directory
    models_path = Path(__file__).parent.parent / "agentic_coder" / "models.json"
    
    with open(models_path, 'w') as f:
        json.dump(models_config, f, indent=2)
    
    print(f"✓ Updated {models_path}")
    print(f"  Last updated: {models_config['last_updated']}")
    print(f"  Anthropic default: {models_config['providers']['anthropic']['current_default']}")
    print(f"  OpenAI default: {models_config['providers']['openai']['current_default']}")
    if models_config['context_windows']:
        print(f"  Context windows: {len(models_config['context_windows'])} models configured")
    
    return True

def should_check_for_updates(force: bool = False) -> bool:
    """Check if we should update models (monthly or forced)."""
    if force:
        return True
    
    models_path = Path(__file__).parent.parent / "agentic_coder" / "models.json"
    
    if not models_path.exists():
        return True
    
    try:
        # Get file modification time
        mtime = datetime.fromtimestamp(models_path.stat().st_mtime)
        
        # Check if more than 30 days have passed
        if datetime.now() - mtime > timedelta(days=30):
            return True
    except:
        return True
    
    return False

def update_models_if_needed(force: bool = False, silent: bool = False) -> bool:
    """Update models if needed, optionally in silent mode."""
    if not should_check_for_updates(force):
        return False
    
    if not silent:
        has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        
        if not has_anthropic and not has_openai:
            print("Checking for model updates (no API keys for live data)...")
        else:
            print("Checking for model updates...")
    
    try:
        update_models()
        return True
    except Exception as e:
        if not silent:
            print(f"Failed to update models: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    # Check for API keys
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    
    print("Model Configuration Updater")
    print("=" * 40)
    print(f"Anthropic API key: {'✓' if has_anthropic else '✗'}")
    print(f"OpenAI API key: {'✓' if has_openai else '✗'}")
    
    if not has_anthropic and not has_openai:
        print("\n⚠️  No API keys found. Using hardcoded fallback data.")
        print("Set ANTHROPIC_API_KEY or OPENAI_API_KEY for live model detection.")
    
    update_models()