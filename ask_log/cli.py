"""
Command Line Interface for ask-log
"""
import click
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import questionary
from questionary import Choice

from .config import Config, list_supported_providers, get_provider_info
from .llm_factory import llm_factory
from .chat import LogAnalyzer

console = Console()

def ensure_configured():
    """
    Checks for configuration. If missing, prints an error 
    and exits the current process immediately.
    """
    config = Config()
    if not config.get_provider_config():
        console.print(Panel(
            "[bold red]Error: System Not Configured[/bold red]\n\n"
            "You must run [cyan]ask-log configure[/cyan] before using this command.",
            title="Access Denied",
            border_style="red"
        ))
        sys.exit(1)  # Exit with error code to stay in the same terminal

@click.group()
@click.version_option(version="0.1.0")
def main():
    """Log Whisperer - An AI log analyzer with chat interface"""
    pass


@main.command()
def configure():
    """Configure LLM provider settings"""
    console.print(Panel.fit(
        "[bold blue]Log Whisperer Configuration[/bold blue]",
        border_style="blue"
    ))
    
    config = Config()
    
    # Show supported providers
    providers = list_supported_providers()
    console.print("\n[bold]Supported LLM Providers:[/bold]")
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Provider", style="cyan")
    table.add_column("Description", style="white")
    
    provider_descriptions = {
        "openai": "OpenAI GPT models (GPT-3.5, GPT-4, etc.)",
        "anthropic": "Anthropic Claude models",
        "google-genai": "Google Gemini models",
        "google-vertexai": "Google Vertex AI models",
    }
    
    for provider in providers:
        description = provider_descriptions.get(provider, "")
        table.add_row(provider, description)
    
    console.print(table)
    
    # Get provider choice
    provider_choices = [
        Choice("OpenAI", "openai"),
        Choice("Anthropic", "anthropic"),
        Choice("Google AI Studio (Gemini)", "google-genai"),
        Choice("Google Vertex AI", "google-vertexai"),
    ]
    
    provider = questionary.select(
        "Select LLM provider:",
        choices=provider_choices,
        use_indicator=True,
    ).ask()
    
    if not provider:
        console.print("\n[yellow]Configuration cancelled.[/yellow]")
        return
        
    provider_info = get_provider_info(provider)
    
    # Get model name using litellm dynamically if available
    console.print(f"\n[bold]Configuring {provider.title()} provider[/bold]")
    
    try:
        from litellm import models_by_provider
        
        # map our provider ids to litellm provider prefixes
        litellm_provider_map = {
            "openai": "openai",
            "anthropic": "anthropic",
            "google-genai": "gemini",
            "google-vertexai": "vertex_ai",
        }
        
        litellm_provider = litellm_provider_map.get(provider, provider)
        raw_models = models_by_provider.get(litellm_provider, [])
        # Strip the litellm provider prefix (e.g. 'gemini/' -> '') for LangChain compatibility
        dynamic_models = []
        for m in raw_models:
            clean_m = m.split("/")[-1] if "/" in m else m
            if clean_m not in dynamic_models:
                dynamic_models.append(clean_m)
        
    except ImportError:
        dynamic_models = []
    
    # If dynamic models were found, offer them as a choice along with Manual Entry
    if dynamic_models:
        model_choices = [Choice(m, m) for m in dynamic_models[:50]] # Limit to 50
        model_choices.append(Choice("Enter manually...", "MANUAL"))
        
        model_selection = questionary.select(
            "Select model:",
            choices=model_choices,
            use_indicator=True
        ).ask()
        
        if not model_selection:
            console.print("\n[yellow]Configuration cancelled.[/yellow]")
            return
            
        if model_selection == "MANUAL":
            model = questionary.text("Enter model name:").ask()
        else:
            model = model_selection
    else:
        # Fallback to pure text input if litellm fails or returns empty list
        model_suggestions = {
            "openai": "gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4o, gpt-4o-mini",
            "anthropic": "claude-3-5-sonnet-20240620, claude-3-opus-20240229",
            "google-genai": "gemini-1.5-pro, gemini-1.5-flash",
            "google-vertexai": "gemini-1.5-pro, gemini-1.5-flash",
        }
        
        if provider in model_suggestions:
            console.print(f"[dim]Popular models: {model_suggestions[provider]}[/dim]")
            
        model = questionary.text("Enter model name:").ask()
        
    if not model:
        console.print("\n[yellow]Configuration cancelled.[/yellow]")
        return
        
    # Collect configuration
    provider_config = {
        "provider": provider,
        "model": model
    }
    
    # Get required parameters
    for param in provider_info["required_params"]:
        is_secret = param.endswith("_key") or param.endswith("_token")
        
        if is_secret:
            value = questionary.password(f"{param.replace('_', ' ').title()}:").ask()
        else:
            value = questionary.text(f"{param.replace('_', ' ').title()}:").ask()
            
        if value is None:
            console.print("\n[yellow]Configuration cancelled.[/yellow]")
            return
            
        provider_config[param] = value
    
    # Test the configuration
    console.print("\n[yellow]Testing configuration...[/yellow]")
    try:
        success = llm_factory.test_provider_connection(provider, model, provider_config)
        if success:
            # Save configuration
            config.set_provider_config(provider_config)
            console.print("\n[green]✓ Configuration saved successfully![/green]")
            console.print(f"[dim]Configuration saved to: {config.config_file}[/dim]")
        else:
            console.print("\n[red]✗ Configuration test failed. Please check your settings and try again.[/red]")
            return
    except Exception as e:
        console.print(f"\n[red]✗ Configuration test failed: {e}[/red]")
        console.print("[yellow]Configuration saved anyway. You can test it manually later.[/yellow]")
        config.set_provider_config(provider_config)


@main.command()
@click.option(
    "--log-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the log file to analyze"
)
@click.option(
    "--save",
    type=click.Path(path_type=Path),
    is_eager=False,
    help="Path to save the conversation and resuse the same chat session later. Can be used to have past context in later conversations."
)
def chat(log_file: Path, save: Path):
    """Start interactive chat session for log analysis"""
    try:
        ensure_configured()
        # Check if configuration exists
        config = Config()
        provider_config = config.get_provider_config()
        if not provider_config:
            console.print("[red]✗ No LLM provider configured.[/red]")
            console.print("Please run '[cyan]ask-log configure[/cyan]' first.")
            return
        
        # Initialize and start chat
        analyzer = LogAnalyzer(str(log_file), str(save) if save else None)
        analyzer.start_chat()
        
        # After chat ends, optionally save if not provided initially
        if not save and len(analyzer.conversation_history) > 0:
            if questionary.confirm("Do you want to save this conversation?").ask():
                history_dir = config.config_dir / "history"
                history_dir.mkdir(parents=True, exist_ok=True)
                
                default_name = f"chat_{log_file.stem}_{int(Path(log_file).stat().st_mtime)}.json"
                default_path = str(history_dir / default_name)
                
                save_filename = questionary.path("Enter save path:", default=default_path).ask()
                
                if save_filename:
                    analyzer.save_path = Path(save_filename)
                    analyzer._save_conversation()
                    console.print(f"[green]✓ Conversation saved to: {analyzer.save_path}[/green]")
        
    except Exception as e:
        console.print(f"[red]✗ Error starting chat: {e}[/red]")


@main.command()
def status():
    """Show current configuration status"""
    ensure_configured()
    config = Config()
    provider_config = config.get_provider_config()
    
    if not provider_config:
        console.print("[yellow]No LLM provider configured.[/yellow]")
        console.print("Run '[cyan]ask-log configure[/cyan]' to set up a provider.")
        return
    
    # Display current configuration
    table = Table(title="Current Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Provider", provider_config.get("provider", "Not set"))
    table.add_row("Model", provider_config.get("model", "Not set"))
    
    # Show optional parameters if set
    for key, value in provider_config.items():
        if key not in ["provider", "model"] and not key.endswith("_key") and not key.endswith("_token"):
            table.add_row(key.replace("_", " ").title(), str(value))
    
    # Hide sensitive information
    for key in provider_config:
        if key.endswith("_key") or key.endswith("_token"):
            table.add_row(key.replace("_", " ").title(), "***configured***")
    
    console.print(table)
    console.print(f"\n[dim]Configuration file: {config.config_file}[/dim]")


@main.command()
def reset():
    """Reset configuration"""
    config = Config()
    if click.confirm("Are you sure you want to reset all configuration?"):
        try:
            config.config_file.unlink(missing_ok=True)
            console.print("[green]✓ Configuration reset successfully. [/green]")
            console.print("[yellow]Reconfigure using ask-log configure[/yellow]")
        except Exception as e:
            console.print(f"[red]✗ Error resetting configuration: {e}[/red]")


if __name__ == "__main__":
    main()
