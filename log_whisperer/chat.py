"""
Chat interface with memory persistence for log analysis
"""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import rich.spinner   
from prompt_toolkit import prompt
from prompt_toolkit.history import FileHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
import hashlib
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import FileChatMessageHistory

from .config import Config
from .llm_factory import llm_factory

console = Console()


class LogAnalyzer:
    """Main chat interface for log analysis"""
    
    def __init__(self, log_file_path: str, save_path: Optional[str] = None):
        self.log_file_path = Path(log_file_path)
        self.save_path = Path(save_path) if save_path else None
        self.config = Config()
        self.llm = None
        self.conversation_history = []
        self.retriever = None
        self.rag_chain = None
        self.fallback_chain = None
        self.session_id = self._compute_session_id()
        
        # Load log file content
        self.log_content = self._load_log_file()
        
        # Initialize LLM
        self._initialize_llm()
        
        # RAG chain will be initialized lazily on first use to speed startup
        
        # Load previous conversation if save path exists
        if self.save_path and self.save_path.exists():
            self._load_conversation()
    
    def _load_log_file(self) -> str:
        """Load and return log file content"""
        try:
            with open(self.log_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            console.print(f"[green]✓ Loaded log file: {self.log_file_path}[/green]")
            console.print(f"[dim]Log file size: {len(content)} characters[/dim]")
            return content
        except Exception as e:
            console.print(f"[red]✗ Failed to load log file: {e}[/red]")
            raise
    
    def _initialize_llm(self):
        """Initialize the LLM from configuration"""
        provider_config = self.config.get_provider_config()
        if not provider_config:
            raise ValueError("No LLM provider configured. Please run 'log-whisperer configure' first.")
        
        try:
            self.llm = llm_factory.create_llm(
                provider_config["provider"],
                provider_config["model"],
                provider_config
            )
            console.print(f"[green]✓ Initialized {provider_config['provider']} with model {provider_config['model']}[/green]")
        except Exception as e:
            console.print(f"[red]✗ Failed to initialize LLM: {e}[/red]")
            raise
    
    def _load_conversation(self):
        """Load previous conversation from save file"""
        try:
            with open(self.save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.conversation_history = data.get('conversation', [])
            
            # Memory is derived on the fly from conversation_history; nothing else to do here
            
            console.print(f"[green]✓ Loaded previous conversation with {len(self.conversation_history)} messages[/green]")
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load previous conversation: {e}[/yellow]")
    
    def _save_conversation(self):
        """Save conversation to file (internal or user-specified)"""
        if not self.save_path:
            return
        
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'log_file': str(self.log_file_path),
                'session_id': self.session_id,
                'conversation': self.conversation_history
            }
            
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            console.print(f"[red]Warning: Could not save conversation: {e}[/red]")
    
    
    def _handle_exit_save(self):
        """Prompt user to save if they haven't specified a save path already"""
        # Only prompt if there is actually a conversation to save
        if self.save_path or not self.conversation_history:
            return

        if click.confirm("\n[bold cyan]? [/bold cyan]Would you like to save this conversation history?"):
            filename = click.prompt("[bold cyan]> [/bold cyan]Enter a filename (e.g., 'analysis_v1')")
            
            # Ensure it has a .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            # Define path: {config_dir}/history/{filename}.json
            history_dir = self.config.config_dir / "history"
            self.save_path = history_dir / filename
            
            # Execute the save
            self._save_conversation()
            console.print(f"[green]✓ Conversation saved to: {self.save_path}[/green]")
            
            
    def _get_system_instructions(self) -> str:
        """System instructions for the RAG chain (no full log in prompt)."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
       
        prompt_path = os.path.join(current_dir, "prompts", "system_prompt.txt")
        
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "Default system instructions..."
    
    def _compute_session_id(self) -> str:
        digest = hashlib.sha256(str(self.log_file_path.resolve()).encode("utf-8")).hexdigest()[:12]
        return f"session-{digest}"

    def _messages_store_path(self) -> Path:
        base = self.config.config_dir / "messages"
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{self.session_id}.json"

    def _get_chat_history(self, session_id: str) -> FileChatMessageHistory:
        # Single-session per log file. Persist messages for agent memory only.
        return FileChatMessageHistory(str(self._messages_store_path()))
    
    def _initialize_rag(self, force_rebuild: bool = False) -> None:
        """Create or load a vector store retriever and retrieval chain over the log file.

        Uses in-memory Chroma with FastEmbed for maximum efficiency and MMR for varied context.
        """
        with console.status("[yellow] Retrieving Embeddings...[/yellow]", spinner="material"):

            try:
                embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

                splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200, add_start_index=True)

                documents = splitter.create_documents(
                    [self.log_content], metadatas=[{"source": str(self.log_file_path)}]
                )
                
                # Use ephemeral in-memory Chroma vector store
                vector_store = Chroma.from_documents(documents, embeddings)

                # Silence ChromaDB warning when number of chunks is less than fetch_k
                doc_count = len(documents)
                fetch_k = min(20, doc_count)
                k = min(6, doc_count)

                # Use MMR constraint for diverse chunk selection
                self.retriever = vector_store.as_retriever(
                    search_type="mmr" if doc_count > 0 else "similarity", 
                    search_kwargs={"k": k, "fetch_k": fetch_k}
                )

                # Prompt and retrieval chain with persisted chat history
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "{system_instructions}\n\nRetrieved context:\n{context}"),
                    MessagesPlaceholder("history"),
                    ("human", "{input}")
                ])
                document_chain = create_stuff_documents_chain(self.llm, prompt)
                base_chain = create_retrieval_chain(self.retriever, document_chain)
                self.rag_chain = RunnableWithMessageHistory(
                    base_chain,
                    self._get_chat_history,
                    input_messages_key="input",
                    history_messages_key="history",
                    output_messages_key="answer",
                )
                console.print("[green]✓ Log are retrieved and ready to be analyzed[/green]")
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to initialize Log retriever: {e}[/yellow]")
                self.retriever = None
                self.rag_chain = None

    def _initialize_fallback_chain(self) -> None:
        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "{system_instructions}"),
                MessagesPlaceholder("history"),
                ("human", "{input}")
            ])
            base_chain = prompt | self.llm
            self.fallback_chain = RunnableWithMessageHistory(
                base_chain,
                self._get_chat_history,
                input_messages_key="input",
                history_messages_key="history",
            )
        except Exception:
            self.fallback_chain = None
    
    def _format_response(self, response: str) -> None:
        """Format and display AI response"""
        panel = Panel(
            Markdown(response),
            title="[bold blue]Agent[/bold blue]",
            border_style="blue",
            padding=(1, 2)
        )
        console.print(panel)
    
    def _add_to_history(self, message_type: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            'type': message_type,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
    
    def start_chat(self):
        """Start the interactive chat session"""
        # Welcome message
        welcome_msg = f"""🔍 **Welcome to Log Whisperer!**

            I'm ready to help you analyze your log file: `{self.log_file_path.name}`

            You can ask me questions like:
            - "What errors do you see in this log?"
            - "Summarize the main events"
            - "Are there any patterns or anomalies?"
            - "What happened around timestamp X?"

            Type '/quit', '/exit', or press Ctrl+C to end the session.
        """
        self._initialize_rag()
        
        self._format_response(welcome_msg)
        
        # Set up prompt history
        history_file = self.config.config_dir / "chat_history"
        history = FileHistory(str(history_file))
        
        try:
            while True:
                try:
                    # Get user input
                    user_input = prompt(
                        "You: ",
                        history=history
                    ).strip()
                    
                    if not user_input:
                        continue
                    
                    if user_input.lower() in ['/quit', '/exit']:
                        break
                    
                    # Add user message to history
                    self._add_to_history('human', user_input)
                    
                    # Get AI response (RAG if available; fallback to direct LLM) with transient status
                    with console.status("[dim]Analyzing...[/dim]", spinner="dots"):
                        if self.rag_chain is not None:
                            result = self.rag_chain.invoke(
                                {"input": user_input, "system_instructions": self._get_system_instructions()},
                                config={"configurable": {"session_id": self.session_id}},
                            )
                            ai_response = result.get("answer") or str(result)
                        else:
                            # Fallback chain with persisted chat history
                            if self.fallback_chain is None:
                                self._initialize_fallback_chain()
                            if self.fallback_chain is not None:
                                response_msg = self.fallback_chain.invoke(
                                    {"input": user_input, "system_instructions": self._get_system_instructions()},
                                    config={"configurable": {"session_id": self.session_id}},
                                )
                                ai_response = getattr(response_msg, "content", str(response_msg))
                            else:
                                response = self.llm.invoke([HumanMessage(content=user_input)])
                                ai_response = response.content if hasattr(response, "content") else str(response)
                    
                    # Add AI response to history
                    self._add_to_history('ai', ai_response)
                    
                    # Display response
                    self._format_response(ai_response)
                    
                    # Save conversation
                    
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                except Exception as e:
                    console.print(f"\n[red]Error: {e}[/red]")
                    continue
                finally:
                    self._handle_exit_save()
                    
        
        finally:
            console.print("\n[yellow]Goodbye! Your conversation has been saved.[/yellow]")
            self._save_conversation()
