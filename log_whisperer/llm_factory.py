"""
LLM Factory for dynamic provider loading and chat model creation
"""
import importlib
import os
import subprocess
import sys
from typing import Dict, Any, Optional
import click
from rich.console import Console

from .config import get_provider_info, SUPPORTED_PROVIDERS

console = Console()


class LLMFactory:
    """Factory for creating LLM instances with dynamic provider loading"""
    
    def __init__(self):
        self._installed_packages = set()
    
    def _install_package(self, package_name: str) -> bool:
        """Install a package dynamically"""
        if package_name in self._installed_packages:
            return True
            
        try:
            console.print(f"[yellow]Installing {package_name}...[/yellow]")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name, "-q"
            ])
            self._installed_packages.add(package_name)
            console.print(f"[green]✓ {package_name} installed successfully[/green]")
            return True
        except subprocess.CalledProcessError as e:
            console.print(f"[red]✗ Failed to install {package_name}: {e}[/red]")
            return False
    
    def _import_provider_class(self, provider_info: Dict[str, Any]):
        """Dynamically import the provider class"""
        try:
            # Try to import the module first
            module_name = provider_info["package"].replace("-", "_")
            module = importlib.import_module(module_name)
            
            # Get the class from the module
            class_name = provider_info["class"]
            llm_class = getattr(module, class_name)
            return llm_class
            
        except ImportError:
            # Package not installed, try to install it
            if self._install_package(provider_info["package"]):
                # Try importing again after installation
                module_name = provider_info["package"].replace("-", "_")
                module = importlib.import_module(module_name)
                class_name = provider_info["class"]
                llm_class = getattr(module, class_name)
                return llm_class
            else:
                raise ImportError(f"Failed to import {provider_info['class']} from {provider_info['package']}")
        except AttributeError:
            raise ImportError(f"Class {provider_info['class']} not found in {provider_info['package']}")
    
    def create_llm(self, provider: str, model: str, config: Dict[str, Any]):
        """Create an LLM instance based on provider and configuration"""
        provider_info = get_provider_info(provider)
        if not provider_info:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Import the provider class
        llm_class = self._import_provider_class(provider_info)
        
        # Prepare initialization parameters
        init_params = {"model": model}
        
        # Map required parameters to environment variable fallbacks
        env_fallbacks = {
            "openai": {"api_key": "OPENAI_API_KEY"},
            "anthropic": {"api_key": "ANTHROPIC_API_KEY"},
            "google-genai": {"google_api_key": "GOOGLE_API_KEY"},
            "google-vertexai": {"project": "GOOGLE_CLOUD_PROJECT", "location": "GOOGLE_CLOUD_LOCATION"},
            "groq": {"groq_api_key": "GROQ_API_KEY"},
        }

        # Add required parameters with env var fallback (no secrets printed)
        for param in provider_info["required_params"]:
            value = config.get(param)
            if not value:
                env_map = env_fallbacks.get(provider, {})
                env_name = env_map.get(param)
                if env_name:
                    value = os.getenv(env_name)
            if value:
                init_params[param] = value
            else:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Add optional parameters if provided
        for param in provider_info["optional_params"]:
            if param in config and config[param] is not None:
                init_params[param] = config[param]
        
        # Special handling for different providers
        if provider == "openai":
            # OpenAI uses openai_api_key parameter name
            if "api_key" in init_params:
                init_params["openai_api_key"] = init_params.pop("api_key")
        elif provider == "anthropic":
            # Anthropic uses anthropic_api_key parameter name
            if "api_key" in init_params:
                init_params["anthropic_api_key"] = init_params.pop("api_key")
        
        try:
            return llm_class(**init_params)
        except Exception as e:
            raise RuntimeError(f"Failed to create {provider} LLM: {e}")
    
    def test_provider_connection(self, provider: str, model: str, config: Dict[str, Any]) -> bool:
        """Test if the provider configuration works"""
        try:
            llm = self.create_llm(provider, model, config)
            # Try a simple invocation to test the connection
            response = llm.invoke("Hello")
            return True
        except Exception as e:
            console.print(f"[red]Connection test failed: {e}[/red]")
            return False


# Global factory instance
llm_factory = LLMFactory()
