import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import threading
from pathlib import Path

# llama-cpp-python for quantized model inference
from llama_cpp import Llama
import logging
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "7"

logger = logging.getLogger("code_compass")
@dataclass
class ChatMessage:
    """Represents a chat message in the conversation history"""
    role: str  # 'system', 'user', 'assistant'
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

class ConversationHistory:
    """Manages conversation history with context window management"""
    
    def __init__(self, max_messages: int = 20, max_tokens: int = 4000):
        self.messages: List[ChatMessage] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the conversation history"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {}
        )
        self.messages.append(message)
        self._trim_history()
    
    def _trim_history(self):
        """Trim history to stay within limits"""
        # Keep only the last max_messages
        if len(self.messages) > self.max_messages:
            # Always keep system messages
            system_messages = [msg for msg in self.messages if msg.role == 'system']
            recent_messages = [msg for msg in self.messages if msg.role != 'system'][-self.max_messages:]
            self.messages = system_messages + recent_messages
        
        # Estimate token count and trim if needed
        total_chars = sum(len(msg.content) for msg in self.messages)
        # Rough estimate: 4 characters per token
        estimated_tokens = total_chars // 4
        
        if estimated_tokens > self.max_tokens:
            # Keep system messages and trim from the oldest user/assistant messages
            system_messages = [msg for msg in self.messages if msg.role == 'system']
            other_messages = [msg for msg in self.messages if msg.role != 'system']
            
            # Remove oldest messages until we're under the limit
            while other_messages and (sum(len(msg.content) for msg in system_messages + other_messages) // 4) > self.max_tokens:
                other_messages.pop(0)
            
            self.messages = system_messages + other_messages
    
    def get_messages_for_llm(self) -> List[Dict[str, str]]:
        """Get messages in format expected by LLM"""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
    
    def get_summary(self) -> str:
        """Get a summary of the conversation"""
        if not self.messages:
            return "No conversation history"
        
        user_msgs = len([msg for msg in self.messages if msg.role == 'user'])
        assistant_msgs = len([msg for msg in self.messages if msg.role == 'assistant'])
        
        return f"Conversation: {user_msgs} questions, {assistant_msgs} responses"

class QwenCoderLLM:
    """
    Qwen2.5-Coder-7B-Instruct integration using llama-cpp-python
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 n_ctx: int = 8192,  # Context window
                 n_threads: int = -1,  # Auto-detect threads
                 n_gpu_layers: int = 0,  # CPU-only by default
                 temperature: float = 0.1,  # Low temperature for code tasks
                 max_tokens: int = 1024):
        
        self.model_path = model_path or self._get_model_path()
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Initialize conversation history
        self.conversation = ConversationHistory()
        
        # Model loading
        self.llm = None
        self.is_loaded = False
        self.loading_thread = None
        
        # System prompt for code analysis
        self.system_prompt = self._create_system_prompt()
        
        # Initialize system message
        self.conversation.add_message("system", self.system_prompt)
    
    def _get_model_path(self) -> str:
        """Get model path, with instructions for download if not found"""
        possible_paths = [
            "./models/qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            "./qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            os.path.expanduser("~/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Model not found - provide download instructions
        logger.info("🔍 Qwen2.5-Coder model not found!")
        logger.info("📥 Please download the quantized model:")
        logger.info("   wget https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf")
        logger.info("   mv Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf qwen2.5-coder-7b-instruct-q4_k_m.gguf")
        logger.info()
        
        # Return first path as placeholder
        return possible_paths[0]
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for code analysis tasks"""
        return """You are Qwen2.5-Coder, an expert AI assistant specialized in code analysis and software engineering. You have access to a codebase that has been analyzed and chunked hierarchically.

**Your Role:**
- Analyze code repositories with deep understanding
- Provide accurate, helpful responses about code structure, functionality, and best practices
- Maintain conversation context and refer to previous discussions
- Give practical, actionable advice

**Context Information:**
When answering questions, you'll be provided with:
1. **User Query**: The current question
2. **Retrieved Code Chunks**: Relevant code sections from the repository
3. **Conversation History**: Previous questions and answers in this session

**Response Guidelines:**
- Be concise but comprehensive
- Use code examples from the retrieved chunks when relevant
- Explain technical concepts clearly
- Suggest improvements or alternatives when appropriate
- If information is missing, say so rather than guessing
- Format code snippets with proper syntax highlighting

**Code Analysis Focus:**
- Understand code architecture and patterns
- Identify key functions, classes, and relationships  
- Explain implementation details and design decisions
- Highlight potential issues or improvements
- Provide context about how components work together

Always be helpful, accurate, and focused on the user's specific needs."""

    def load_model_async(self):
        """Load model asynchronously to avoid blocking the UI"""
        def _load():
            try:
                logger.info(f"🔄 Loading Qwen2.5-Coder model from {self.model_path}...")
                logger.info(f"⚙️  Configuration: n_ctx={self.n_ctx}, n_threads={self.n_threads}, n_gpu_layers={self.n_gpu_layers}")
                
                # self.llm = Llama(
                #     model_path=self.model_path,
                #     n_ctx=self.n_ctx,
                #     n_threads=self.n_threads,
                #     n_gpu_layers=self.n_gpu_layers,
                #     verbose=False,
                #     use_mlock=True,  # Keep model in memory
                #     use_mmap=True,   # Memory-map the model file
                # )
                self.llm = Llama(
                    model_path=self.model_path,
                    cache_dir=Path('models'),
                    seed=42,
                    n_ctx=self.n_ctx,
                    verbose=False,
                    n_gpu_layers=self.n_gpu_layers,
                    n_threads=self.n_threads,
                )
                
                self.is_loaded = True
                logger.info("✅ Qwen2.5-Coder model loaded successfully!")
                
                # Test the model with a simple query
                # test_response = self.llm.create_chat_completion(
                #     messages=[{"role": "user", "content": "Hello, are you working?"}],
                #     max_tokens=50,
                #     temperature=0.1
                # )
                # logger.info(f"🧪 Model test: {test_response['choices'][0]['message']['content'][:50]}...")
                
            except Exception as e:
                logger.info(f"❌ Error loading model: {str(e)}")
                self.is_loaded = False
                
        self.loading_thread = threading.Thread(target=_load)
        self.loading_thread.start()
    
    def wait_for_model(self, timeout: int = 300) -> bool:
        """Wait for model to load with timeout"""
        if self.loading_thread:
            self.loading_thread.join(timeout=timeout)
        return self.is_loaded
    
    def is_model_ready(self) -> bool:
        """Check if model is ready for inference"""
        return self.is_loaded and self.llm is not None
    
    def generate_response(self, 
                         user_query: str,
                         retrieved_chunks: List[Dict[str, Any]] = None,
                         use_history: bool = True) -> Dict[str, Any]:
        """
        Generate response using LLM with retrieved context and conversation history
        
        Args:
            user_query: User's question
            retrieved_chunks: Relevant code chunks from vector search
            use_history: Whether to include conversation history
            
        Returns:
            Dict with response and metadata
        """
        
        if not self.is_model_ready():
            return {
                "status": "error",
                "message": "❌ Model not loaded. Please wait for model initialization.",
                "response": ""
            }
        
        try:
            # Build context from retrieved chunks
            context = self._build_context_from_chunks(retrieved_chunks or [])
            
            # Create the current query with context
            query_with_context = self._format_query_with_context(user_query, context)
            
            # Add user query to conversation history
            self.conversation.add_message("user", user_query, {
                "chunks_count": len(retrieved_chunks) if retrieved_chunks else 0,
                "context_length": len(context)
            })
            
            # Prepare messages for LLM
            if use_history:
                messages = self.conversation.get_messages_for_llm()
                # Replace the last user message with the context-enhanced version
                messages[-1]["content"] = query_with_context
            else:
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query_with_context}
                ]
            
            logger.info(f"🤖 Generating response for query: '{user_query[:50]}...'")
            logger.info(f"📊 Context: {len(retrieved_chunks) if retrieved_chunks else 0} chunks, History: {len(self.conversation.messages)} messages")
            
            # Generate response
            start_time = time.time()
            
            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract response content
            assistant_response = response['choices'][0]['message']['content']
            
            # Add assistant response to conversation history
            self.conversation.add_message("assistant", assistant_response, {
                "generation_time": generation_time,
                "tokens_used": response.get('usage', {}).get('total_tokens', 0)
            })
            
            logger.info(f"✅ Response generated in {generation_time:.2f}s")
            
            return {
                "status": "success",
                "response": assistant_response,
                "metadata": {
                    "generation_time": generation_time,
                    "chunks_used": len(retrieved_chunks) if retrieved_chunks else 0,
                    "conversation_length": len(self.conversation.messages),
                    "tokens_used": response.get('usage', {}).get('total_tokens', 0)
                }
            }
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            logger.info(error_msg)
            
            return {
                "status": "error",
                "message": error_msg,
                "response": "I apologize, but I encountered an error while processing your request. Please try again."
            }
    
    def _build_context_from_chunks(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved code chunks"""
        if not chunks:
            return ""
        
        context_parts = ["**Retrieved Code Context:**\n"]
        
        for i, chunk in enumerate(chunks[:5], 1):  # Limit to top 5 chunks
            metadata = chunk.get('metadata', {})
            score = chunk.get('score', 0)
            
            chunk_type = metadata.get('chunk_type', 'code')
            file_path = metadata.get('file_path', 'unknown')
            
            context_parts.append(f"**{i}. {chunk_type.title()} from `{file_path}` (Similarity: {score:.2f})**")
            
            # Add specific context based on chunk type
            if chunk_type == 'function':
                func_name = metadata.get('function_name', 'unknown')
                signature = metadata.get('signature', func_name)
                class_name = metadata.get('class_name')
                
                if class_name:
                    context_parts.append(f"Function: `{class_name}.{signature}`")
                else:
                    context_parts.append(f"Function: `{signature}`")
                    
            elif chunk_type == 'class':
                class_name = metadata.get('class_name', 'unknown')
                methods = metadata.get('methods', [])
                context_parts.append(f"Class: `{class_name}`")
                if methods:
                    context_parts.append(f"Methods: {', '.join(methods[:5])}")
                    
            elif chunk_type == 'file':
                language = metadata.get('language', '')
                total_lines = metadata.get('total_lines', 'unknown')
                context_parts.append(f"File overview: {language} ({total_lines} lines)")
            
            # Add a separator
            context_parts.append("---\n")
        
        return "\n".join(context_parts)
    
    def _format_query_with_context(self, query: str, context: str) -> str:
        """Format user query with retrieved context"""
        if not context:
            return query
            
        return f"""**User Question:** {query}

{context}

**Instructions:** Using the retrieved code context above, please provide a comprehensive answer to the user's question. Reference specific code snippets, functions, or classes when relevant. If the context doesn't contain enough information to fully answer the question, please mention what additional information would be helpful."""
    
    def clear_conversation(self):
        """Clear conversation history but keep system prompt"""
        self.conversation.clear()
        self.conversation.add_message("system", self.system_prompt)
    
    def get_conversation_summary(self) -> str:
        """Get summary of current conversation"""
        return self.conversation.get_summary()
    
    def export_conversation(self) -> List[Dict[str, Any]]:
        """Export conversation history"""
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            }
            for msg in self.conversation.messages
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "context_window": self.n_ctx,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "conversation_messages": len(self.conversation.messages)
        }