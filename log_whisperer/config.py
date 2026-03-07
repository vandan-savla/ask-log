"""
Configuration management for log-whisperer
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import click


class Config:
    """Manages configuration for log-whisperer"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".log-whisperer"
        self.config_file = self.config_dir / "config.yaml"
        self.ensure_config_dir()
    
    def ensure_config_dir(self):
        """Ensure configuration directory exists"""
        self.config_dir.mkdir(exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        if not self.config_file.exists():
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            return {}
    
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            click.echo(f"Error saving config: {e}", err=True)
            raise
    
    def get_provider_config(self) -> Optional[Dict[str, Any]]:
        """Get the current provider configuration"""
        config = self.load_config()
        return config.get('provider')
    
    def set_provider_config(self, provider_config: Dict[str, Any]):
        """Set provider configuration"""
        config = self.load_config()
        config['provider'] = provider_config
        self.save_config(config)


# Supported LLM providers with their package requirements
SUPPORTED_PROVIDERS = {
    "openai": {
        "package": "langchain-openai", 
        "class": "ChatOpenAI",
        "required_params": ["api_key"],
        "optional_params": ["temperature", "max_tokens", "top_p"]
    },
    "anthropic": {
        "package": "langchain-anthropic", 
        "class": "ChatAnthropic",
        "required_params": ["api_key"],
        "optional_params": ["temperature", "max_tokens", "top_p"]
    },
    "google-genai": {
        "package": "langchain-google-genai", 
        "class": "ChatGoogleGenerativeAI",
        "required_params": ["google_api_key"],
        "optional_params": ["temperature", "max_output_tokens", "top_p"]
    },
    "google-vertexai": {
        "package": "langchain-google-vertexai", 
        "class": "ChatVertexAI",
        "required_params": ["project", "location"],
        "optional_params": ["temperature", "max_output_tokens", "top_p"]
    }
}


def get_provider_info(provider: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific provider"""
    return SUPPORTED_PROVIDERS.get(provider.lower())


def list_supported_providers() -> list:
    """Get list of supported provider names"""
    return list(SUPPORTED_PROVIDERS.keys())
