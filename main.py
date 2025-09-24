import gradio as gr
import logging
from datetime import datetime
from pathlib import Path
from scripts.RepositoryHandler import RepositoryHandler
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# --- Setup Logging ---
def setup_logger():
    log_dir = Path("/data/home/sqamar/code-compass/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_file = log_dir / f"{timestamp}_code_compass.log"

    logger = logging.getLogger("code_compass")
    logger.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Formatter
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

setup_logger()
logger = logging.getLogger("code_compass")
# Global repository handler instance
repo_handler = RepositoryHandler()


def process_repository(input_type, github_url, zip_file):
    """Process repository based on input type"""
    
    # Clean up any previous repository
    repo_handler.cleanup()
    
    if input_type == "GitHub URL":
        if not github_url or not github_url.strip():
            return "❌ Please enter a GitHub repository URL", "", "disabled", "disabled"
        
        if not repo_handler.validate_github_url(github_url.strip()):
            return "❌ Invalid GitHub URL format. Please use: https://github.com/username/repository", "", "disabled", "disabled"
        
        success, message = repo_handler.download_github_repo(github_url.strip())
        
    else:  # ZIP File
        if zip_file is None:
            return "❌ Please upload a ZIP file", "", "disabled", "disabled"
        
        is_valid, validation_msg = repo_handler.validate_zip_file(zip_file)
        if not is_valid:
            return f"❌ {validation_msg}", "", "disabled", "disabled"
        
        success, message = repo_handler.extract_zip_file(zip_file)
    
    if success:
        structure = repo_handler.get_repo_structure()
        return message, structure, "🚀 Process Repository", "disabled"  # Enable process button, keep query disabled
    else:
        return message, "", "disabled", "disabled"

def process_chunks():
    """Process repository into chunks and store in vector database"""
    if not repo_handler.is_loaded:
        return "❌ No repository loaded", "disabled"
    
    # Run processing in background thread to avoid blocking UI
    def background_processing():
        return repo_handler.process_and_store_chunks()
    
    try:
        success, message = background_processing()
        if success:
            return message, "Ask AI"  # Enable query functionality
        else:
            return message, "disabled"
    except Exception as e:
        return f"❌ Error processing chunks: {str(e)}", "disabled"

def handle_query(query):
    """Handle user queries about the repository"""
    if not repo_handler.is_loaded:
        return "❌ No repository loaded. Please load a repository first."
    
    if not repo_handler.chunks:
        return "❌ Repository not processed yet. Please click 'Process Repository' first."
    
    if not query or not query.strip():
        return "Please enter a query about the repository."
    
    return repo_handler.query_repository(query.strip())

def get_repo_stats():
    """Get repository statistics for display"""
    if not repo_handler.is_loaded:
        return "No repository loaded"
    
    if repo_handler.vector_store and repo_handler.chunks:
        try:
            # Get repository overview from vector store
            overview = repo_handler.vector_store.get_repository_overview(repo_handler.repo_name)
            logger.debug(f"Repository overview: {overview}")
            if "error" not in overview:
                stats = f"""📊 **Repository Statistics**

🏷️ **Repository:** {overview['repo_name']}
📦 **Total Chunks:** {overview['total_chunks']}
📁 **Files:** {overview['files_count']}
🏛️ **Classes:** {overview['classes_count']}  
⚙️ **Functions:** {overview['functions_count']}
💻 **Languages:** {', '.join(overview['languages'])}

📋 **Chunk Distribution:**
"""
                for chunk_type, count in overview['chunk_distribution'].items():
                    stats += f"- {chunk_type.title()}: {count}\n"
                
                return stats
            else:
                return f"Error getting stats: {overview['error']}"
        except Exception as e:
            return f"Error getting repository stats: {str(e)}"
    
    return "Repository loaded but not processed yet"
# Additional handler functions for LLM integration
def initialize_llm():
    """Initialize LLM model loading"""
    return repo_handler.initialize_llm()

def handle_query_with_llm(query, use_llm):
    """Handle user queries with optional LLM processing"""
    if not repo_handler.is_loaded:
        return "❌ No repository loaded. Please load a repository first."
    
    if not repo_handler.chunks:
        return "❌ Repository not processed yet. Please click 'Process Repository' first."
    
    if not query or not query.strip():
        return "Please enter a query about the repository."
    
    return repo_handler.query_repository(query.strip(), use_llm=use_llm)

def clear_conversation():
    """Clear LLM conversation history"""
    if repo_handler.llm:
        repo_handler.llm.clear_conversation()
        return "🗑️ Conversation history cleared!"
    return "❌ LLM not initialized"

def export_conversation():
    """Export conversation history"""
    if repo_handler.llm and repo_handler.llm.is_model_ready():
        conversation = repo_handler.llm.export_conversation()
        if conversation:
            # Format for display
            export_text = "# Conversation Export\n\n"
            for msg in conversation:
                role_emoji = {"system": "⚙️", "user": "👤", "assistant": "🤖"}.get(msg["role"], "💬")
                export_text += f"## {role_emoji} {msg['role'].title()}\n"
                export_text += f"**Time:** {msg['timestamp']}\n\n"
                export_text += f"{msg['content']}\n\n---\n\n"
            return export_text
        else:
            return "No conversation to export"
    return "❌ LLM not ready or no conversation history"

def get_llm_status():
    """Get current LLM status"""
    if not repo_handler.llm_loading_started:
        return "🔄 LLM not initialized"
    elif repo_handler.llm.is_model_ready():
        model_info = repo_handler.llm.get_model_info()
        conversation_summary = repo_handler.llm.get_conversation_summary()
        return f"""✅ **LLM Ready!**
        
**Model:** Qwen2.5-Coder-7B-Instruct (Q4_K_M)
**Context Window:** {model_info['context_window']} tokens
**Temperature:** {model_info['temperature']}
**Status:** {conversation_summary}

🤖 Ready for intelligent code analysis!"""
    else:
        return "⏳ **LLM Loading...** Please wait for model initialization to complete."

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Code Compass", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("""
        # 🔍 Code Compass
        
        Upload your repository via GitHub URL or ZIP file, process it with AI-powered chunking, and query your codebase using semantic search!
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                
                # Input section
                with gr.Group():
                    gr.Markdown("### 📥 Repository Input")
                    
                    input_type = gr.Dropdown(
                        choices=["GitHub URL", "ZIP File"], 
                        value="GitHub URL",
                        label="Input Method",
                        info="Choose how you want to provide your repository"
                    )
                    
                    github_url = gr.Textbox(
                        label="GitHub Repository URL",
                        placeholder="https://github.com/username/repository",
                        visible=True
                    )
                    
                    zip_file = gr.File(
                        label="Upload ZIP File",
                        file_types=[".zip"],
                        visible=False
                    )
                    
                    load_btn = gr.Button("📁 Load Repository", variant="primary")
                
                # Processing section
                with gr.Group():
                    gr.Markdown("### ⚙️ Repository Processing")
                    gr.Markdown("After loading, process your repository to enable AI-powered search")
                    
                    process_btn = gr.Button("🚀 Process Repository", interactive=False, variant="secondary")
                    
                # Status section
                with gr.Group():
                    gr.Markdown("### 📊 Status")
                    status_output = gr.Textbox(
                        label="Status",
                        placeholder="Ready to load repository...",
                        interactive=False,
                        lines=3
                    )
            
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📁 Repository Structure")
                    structure_output = gr.Code(
                        label="Directory Structure",
                        # language="text",
                        interactive=False,
                        lines=10
                    )
                
                with gr.Group():
                    gr.Markdown("### 📊 Repository Stats")
                    stats_output = gr.Markdown(
                        value="Load and process a repository to see statistics"
                    )
                with gr.Group():
                    gr.Markdown("### 🤖 LLM Status")
                    llm_status = gr.Markdown(
                        value="🔄 LLM not initialized"
                    )
                    init_llm_btn = gr.Button("🚀 Initialize LLM", variant="secondary")
        # Query section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 💬 Query Repository")
                gr.Markdown("Ask questions about your code using natural language. The AI will search through your processed code chunks to find relevant information.")
                
                with gr.Row():
                    query_input = gr.Textbox(
                        label="Ask about your code",
                        placeholder="e.g., 'What does this repository do?', 'Show me authentication functions', 'How is error handling implemented?'",
                        lines=2,
                        scale=4
                    )
                    query_btn = gr.Button("🔍 Ask Question", interactive=False, scale=1)
                    use_llm_toggle = gr.Checkbox(
                            label="Use AI Analysis",
                            value=True,
                            info="Get intelligent responses using LLM"
                        )
                    # Conversation controls
                with gr.Row():
                    clear_chat_btn = gr.Button("🗑️ Clear Chat History", variant="secondary", interactive=False)
                    export_chat_btn = gr.Button("📥 Export Chat", variant="secondary", interactive=False)
                query_output = gr.Markdown(
                    value="Load and process a repository first to start querying...",
                    height=400
                )
        
        # Advanced options (collapsible)
        # with gr.Accordion("🛠️ Advanced Options", open=False):
        #     with gr.Row():
        #         with gr.Column():
        #             gr.Markdown("### 🔧 Pinecone Configuration")
        #             api_key_input = gr.Textbox(
        #                 label="Pinecone API Key",
        #                 placeholder="Enter your Pinecone API key (or set PINECONE_API_KEY env var)",
        #                 type="password"
        #             )
        #             environment_input = gr.Textbox(
        #                 label="Pinecone Environment",
        #                 value="us-west1-gcp-free",
        #                 placeholder="e.g., us-west1-gcp-free"
        #             )
                
        #         with gr.Column():
        #             gr.Markdown("### 📈 Processing Options")
        #             complexity_threshold = gr.Slider(
        #                 minimum=5,
        #                 maximum=50,
        #                 value=20,
        #                 step=5,
        #                 label="Complexity Threshold",
        #                 info="Functions above this complexity will be sub-chunked"
        #             )
        
        # Event handlers
        def toggle_inputs(choice):
            return (
                gr.update(visible=(choice == "GitHub URL")),
                gr.update(visible=(choice == "ZIP File"))
            )
        
        def update_buttons_after_load(status_text):
            # Enable process button if repository is successfully loaded
            is_loaded = "✅" in status_text and "successfully" in status_text.lower()
            return gr.update(interactive=is_loaded)
        
        def update_query_button_after_process(status_text):
            # Enable query button if processing is successful
            is_processed = "✅" in status_text and "complete" in status_text.lower()
            return gr.update(interactive=is_processed)
        
        def update_buttons_after_process(status_text):
            # Enable query button if processing is successful
            is_processed = "✅" in status_text and "complete" in status_text.lower()
            return (
                gr.update(interactive=is_processed),  # query_btn
                gr.update(interactive=is_processed),  # clear_chat_btn  
                gr.update(interactive=is_processed)   # export_chat_btn
            )
        
        def update_llm_status():
            return get_llm_status()
        
        def update_stats(status_output):
            return get_repo_stats(), update_buttons_after_load(status_output), update_query_button_after_process(status_output)
        
        # Wire up the interface
        input_type.change(
            fn=toggle_inputs,
            inputs=[input_type],
            outputs=[github_url, zip_file]
        )
        
        load_btn.click(
            fn=process_repository,
            inputs=[input_type, github_url, zip_file],
            outputs=[status_output, structure_output, process_btn, query_btn]
        ).then(
            fn=update_stats,
            inputs=[status_output],
            outputs=[stats_output, process_btn, query_btn]
        )
        
        process_btn.click(
            fn=process_chunks,
            outputs=[status_output, query_btn]
        ).then(
            fn=update_stats,
            inputs=[status_output],
            outputs=[stats_output, process_btn, query_btn]
        )
        
        # Query handling
        query_btn.click(
            fn=handle_query_with_llm,
            inputs=[query_input, use_llm_toggle],
            outputs=[query_output]
        ).then(
            fn=update_llm_status,
            outputs=[llm_status]
        )
        
        # Chat management
        clear_chat_btn.click(
            fn=clear_conversation,
            outputs=[query_output]
        ).then(
            fn=update_llm_status,
            outputs=[llm_status]
        )
        
        # Allow Enter key to submit query
        query_input.submit(
            fn=handle_query_with_llm,
            inputs=[query_input, use_llm_toggle],
            outputs=[query_output]
        )
         # LLM initialization
        init_llm_btn.click(
            fn=initialize_llm,
            outputs=[llm_status]
        ).then(
            fn=update_llm_status,
            outputs=[llm_status]
        )
        # Add some helpful examples
        gr.Markdown("""
        ### 📝 Example Repositories to Try:
        - `https://github.com/pallets/flask` - Popular Python web framework
        - `https://github.com/requests/requests` - HTTP library for Python
        - `https://github.com/fastapi/fastapi` - Modern Python web framework
        - `https://github.com/psf/black` - Python code formatter
        
        ### 💡 Example Queries:
        - "What is the main purpose of this repository?"
        - "Show me all the authentication functions"
        - "How is error handling implemented?"
        - "What are the main classes and their responsibilities?"  
        - "Find functions that handle file operations"
        - "Show me the configuration management code"
        
        ### ⚙️ Setup Requirements:
        1. **Pinecone API Key**: Get a free API key from [Pinecone.io](https://www.pinecone.io/)
        2. **Environment Variables**: Set `PINECONE_API_KEY` in your environment or enter it in Advanced Options
        3. **Internet Connection**: Required for downloading repositories and accessing Pinecone
        
        ### 🚀 How It Works:
        1. **Load**: Repository is downloaded/extracted and validated
        2. **Process**: Code is analyzed and split into hierarchical chunks (file → class → function → block)
        3. **Store**: Chunks are embedded using AI and stored in Pinecone vector database  
        4. **Query**: Your questions are semantically matched against stored code chunks
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    
    # Launch with some nice settings
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Standard port
        share=False,            # Set to True to create public link
        debug=True              # Enable debug mode for development
    )