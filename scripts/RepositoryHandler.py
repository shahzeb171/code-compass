import gradio as gr
import os
import zipfile
import tempfile
import shutil
import requests
import re
from pathlib import Path
from urllib.parse import urlparse
import subprocess
import threading
import time
import logging
# Import our custom modules
from .chunker import HierarchicalChunker
from .vectorstore import PineconeVectorStore
from .llm_service import QwenCoderLLM
from config import MODEL_PATH
from typing import List, Dict, Any
logger = logging.getLogger("code_compass")

class RepositoryHandler:
    def __init__(self):
        self.temp_dir = None
        self.repo_path = None
        self.is_loaded = False
        self.repo_name = None
        self.chunks = []
        
        # Initialize chunker and vector store
        self.chunker = HierarchicalChunker()
        self.vector_store = None  # Will be initialized when needed
        self.processing_status = {"status": "idle", "progress": 0, "message": ""}
        
        # Initialize LLM service
        self.llm = QwenCoderLLM(model_path=MODEL_PATH, n_gpu_layers=-1)  # Adjust n_gpu_layers based on your GPU memory
        self.llm_loading_started = False
    
    def validate_github_url(self, url):
        """Validate if URL is a proper GitHub repository URL"""
        github_pattern = r'https://github\.com/[\w\-\.]+/[\w\-\.]+'
        return bool(re.match(github_pattern, url))
    
    def validate_zip_file(self, zip_file):
        """Validate if uploaded file is a proper zip file"""
        if zip_file is None:
            return False, "No file uploaded"
        
        try:
            # Check if file exists and has .zip extension
            if not zip_file.name.lower().endswith('.zip'):
                return False, "File must be a .zip file"
            
            # Try to open and validate the zip file
            with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                # Test if zip file is valid
                zip_ref.testzip()
                
                # Check if it contains at least one file
                file_list = zip_ref.namelist()
                if not file_list:
                    return False, "Zip file is empty"
                
                # Check if it looks like a code repository
                code_extensions = ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb', '.ts']
                has_code_files = any(
                    any(fname.endswith(ext) for ext in code_extensions) 
                    for fname in file_list
                )
                
                if not has_code_files:
                    return False, "Zip file doesn't appear to contain code files"
                
                return True, f"Valid zip file with {len(file_list)} files"
                
        except zipfile.BadZipFile:
            return False, "Invalid or corrupted zip file"
        except Exception as e:
            return False, f"Error validating zip file: {str(e)}"
    
    def download_github_repo(self, github_url):
        """Download GitHub repository using git clone"""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="repo_")
            
            # Extract repo name for folder
            self.repo_name = github_url.split('/')[-1].replace('.git', '')
            self.repo_path = os.path.join(self.temp_dir, self.repo_name)
            
            # Clone the repository
            result = subprocess.run([
                'git', 'clone', github_url, self.repo_path
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                # If git clone fails, try downloading as zip
                return self._download_repo_as_zip(github_url)
            
            # Count files in repository
            total_files = sum(1 for _ in Path(self.repo_path).rglob('*') if _.is_file())
            
            self.is_loaded = True
            return True, f"✅ Repository successfully cloned! Found {total_files} files in {self.repo_name}"
            
        except subprocess.TimeoutExpired:
            return False, "❌ Download timeout - repository might be too large"
        except FileNotFoundError:
            # Git not installed, fallback to zip download
            return self._download_repo_as_zip(github_url)
        except Exception as e:
            return False, f"❌ Error downloading repository: {str(e)}"
    
    def _download_repo_as_zip(self, github_url):
        """Fallback method to download repo as zip if git is not available"""
        try:
            # Convert GitHub URL to zip download URL
            zip_url = github_url.rstrip('/') + '/archive/refs/heads/main.zip'
            
            # Try main branch, if fails try master
            for branch in ['main', 'master']:
                try:
                    zip_url = github_url.rstrip('/') + f'/archive/refs/heads/{branch}.zip'
                    response = requests.get(zip_url, timeout=60)
                    response.raise_for_status()
                    break
                except:
                    continue
            else:
                return False, "❌ Could not download repository - check if it's public and accessible"
            
            # Create temp directory and save zip
            self.temp_dir = tempfile.mkdtemp(prefix="repo_")
            zip_path = os.path.join(self.temp_dir, "repo.zip")
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            
            # Extract zip
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            
            # Find the extracted folder (usually repo-name-branch)
            extracted_folders = [d for d in os.listdir(self.temp_dir) 
                               if os.path.isdir(os.path.join(self.temp_dir, d))]
            
            if extracted_folders:
                self.repo_path = os.path.join(self.temp_dir, extracted_folders[0])
                total_files = sum(1 for _ in Path(self.repo_path).rglob('*') if _.is_file())
                self.is_loaded = True
                return True, f"✅ Repository successfully downloaded! Found {total_files} files"
            else:
                return False, "❌ Error extracting downloaded repository"
                
        except requests.RequestException as e:
            return False, f"❌ Network error downloading repository: {str(e)}"
        except Exception as e:
            return False, f"❌ Error downloading repository: {str(e)}"
    
    def extract_zip_file(self, zip_file):
        """Extract uploaded zip file"""
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="repo_")
            
            # Extract zip file
            with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                zip_ref.extractall(self.temp_dir)
            
            # Find the main folder or use temp_dir if files are in root
            extracted_items = os.listdir(self.temp_dir)
            
            # If there's only one folder, use it as repo_path
            if len(extracted_items) == 1 and os.path.isdir(os.path.join(self.temp_dir, extracted_items[0])):
                self.repo_path = os.path.join(self.temp_dir, extracted_items[0])
                self.repo_name = os.path.basename(self.repo_path)
            else:
                # Files are in root of zip
                self.repo_path = self.temp_dir
            
            # Count files
            total_files = sum(1 for _ in Path(self.repo_path).rglob('*') if _.is_file())
            
            self.is_loaded = True
            return True, f"✅ Zip file successfully extracted! Found {total_files} files"
            
        except Exception as e:
            return False, f"❌ Error extracting zip file: {str(e)}"
    
    def initialize_vector_store(self, namespace):
        """Initialize Pinecone vector store"""
        try:
            if self.vector_store is None:
                print("🔄 Initializing vector store...")
                self.vector_store = PineconeVectorStore(namespace=namespace)
                print("✅ Vector store initialized!")
            return True, "Vector store ready"
        except Exception as e:
            error_msg = f"❌ Error initializing vector store: {str(e)}"
            print(error_msg)
            return False, error_msg
    
    def process_and_store_chunks(self):
        """Process repository into chunks and store in vector database"""
        if not self.is_loaded or not self.repo_path:
            return False, "❌ No repository loaded"
        
        try:
            self.processing_status = {"status": "chunking", "progress": 10, "message": "Creating hierarchical chunks..."}
            namespace = self.repo_name + "_namespace"
            # Step 1: Create chunks
            logger.info(f"🔄 Creating chunks for {self.repo_name}...")
            self.chunks = self.chunker.chunk_repository(self.repo_path)
            
            if not self.chunks:
                return False, "❌ No chunks generated from repository"
            
            # self.processing_status = {"status": "embedding", "progress": 40, "message": f"Generating embeddings for {len(self.chunks)} chunks..."}
            
            # Step 2: Initialize vector store
            success, message = self.initialize_vector_store(namespace=namespace)
            if not success:
                return False, message
            
            # Step 3: Generate embeddings
            # print("🔄 Generating embeddings...")
            # self.chunks =  self.vector_store.generate_embeddings(self.chunks)
            
            self.processing_status = {"status": "storing", "progress": 70, "message": "Storing chunks in vector database..."}
            
            # Step 4: Store in Pinecone
            logger.info("🔄 Storing chunks in vector database...")
            result = self.vector_store.upsert_chunks(self.chunks)
            
            self.processing_status = {"status": "complete", "progress": 100, "message": "Processing complete!"}
            
            if result['status'] == 'success':
                summary = f"""✅ Repository processing complete!
                
📊 **Processing Summary:**
- Repository: {self.repo_name}
- Total chunks created: {len(self.chunks)}
- Successfully stored: {result['successful_upserts']}
- Failed: {result['failed_upserts']}

📁 **Chunk Distribution:**"""
                
                # Add chunk type distribution
                chunk_types = {}
                for chunk in self.chunks:
                    chunk_type = chunk.chunk_type
                    chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1
                
                for chunk_type, count in chunk_types.items():
                    summary += f"\n- {chunk_type.title()}: {count}"
                
                summary += f"\n\n🔍 **Ready for queries!** You can now ask questions about your code."
                
                return True, summary
            else:
                return False, f"❌ Error storing chunks: {result.get('message', 'Unknown error')}"
                
        except Exception as e:
            self.processing_status = {"status": "error", "progress": 0, "message": f"Error: {str(e)}"}
            return False, f"❌ Error processing repository: {str(e)}"
    
    def query_repository(self, query_text, search_type="hybrid",use_llm=True):
        """Query the repository using vector search"""
        if not self.vector_store or not self.chunks:
            return "❌ Repository not processed yet. Please load and process a repository first."
        
        if not query_text or not query_text.strip():
            return "Please enter a query about the repository."
        
        try:
            logger.info(f"🔍 Querying repository: {query_text}")
            
            # Perform hybrid search
            results = self.vector_store.hybrid_search(
                query_text=query_text.strip(),
                repo_names=[self.repo_name],
                top_k=10
            )
            
            if not results:
                return f"""🤖 No relevant results found for: "{query_text}"

Try rephrasing your question or asking about:
- Specific functions or classes
- Code patterns or algorithms  
- File structure or organization
- Dependencies or imports"""
            # Step 2: Use LLM for intelligent response if enabled and ready
            if use_llm:
                if not self.llm_loading_started:
                    self.initialize_llm()
                
                if self.llm.is_model_ready():
                    # Generate intelligent response using LLM
                    llm_response = self.llm.generate_response(
                        user_query=query_text.strip(),
                        retrieved_chunks=results,
                        use_history=True
                    )
                    
                    if llm_response["status"] == "success":
                        response = f"""🤖 **AI Analysis:**

        {llm_response["response"]}

        ---
        📊 **Query Details:**
        - Found {len(results)} relevant code sections
        - Response generated in {llm_response["metadata"]["generation_time"]:.2f}s
        - Conversation length: {llm_response["metadata"]["conversation_length"]} messages
        """
                        return response
                    else:
                        # Fall back to basic response if LLM fails
                        return self._generate_basic_response(query_text, results) + f"\n\n⚠️ LLM Error: {llm_response.get('message', 'Unknown error')}"
                else:
                    # LLM not ready, provide basic response with loading status
                    basic_response = self._generate_basic_response(query_text, results)
                    return basic_response + "\n\n⏳ **Note:** AI model is still loading. You'll get smarter responses once it's ready!"
            else:
                # Basic response without LLM
                return self._generate_basic_response(query_text, results)
            
        except Exception as e:
            return f"❌ Error querying repository: {str(e)}"
        # Format response
#         response = f"""🔍 **Query Results for:** "{query_text}"

#         📊 **Found {len(results)} relevant code sections:**

#         """
            
#             for i, result in enumerate(results[:5], 1):  # Show top 5 results
#                 metadata = result.get('metadata', {})
#                 score = result.get('score', 0)
                
#                 chunk_type = metadata.get('chunk_type', 'unknown')
#                 file_path = metadata.get('file_path', 'unknown')
                
#                 response += f"""**{i}. {chunk_type.title()} Match** (Similarity: {score:.2f})
# 📄 File: `{file_path}`
# """
                
#                 if chunk_type == 'function':
#                     func_name = metadata.get('function_name', 'unknown')
#                     class_name = metadata.get('class_name')
#                     signature = metadata.get('signature', func_name)
                    
#                     response += f"🔧 Function: `{signature}`\n"
#                     if class_name:
#                         response += f"📦 Class: `{class_name}`\n"
                        
#                 elif chunk_type == 'class':
#                     class_name = metadata.get('class_name', 'unknown')
#                     methods = metadata.get('methods', [])
#                     response += f"📦 Class: `{class_name}`\n"
#                     if methods:
#                         response += f"🔧 Methods: {', '.join(methods[:5])}\n"
                        
#                 elif chunk_type == 'file':
#                     language = metadata.get('language', 'unknown')
#                     total_lines = metadata.get('total_lines', 'unknown')
#                     response += f"📝 Language: {language}, Lines: {total_lines}\n"
                
#                 response += "---\n\n"
            
#             # Add repository overview
#             if len(results) > 5:
#                 response += f"... and {len(results) - 5} more results available.\n\n"
                
#             response += f"""💡 **Suggestions:**
# - Ask more specific questions about functions or classes
# - Query about code patterns: "Show me error handling code"  
# - Ask about structure: "What are the main components?"
# - Request examples: "How is authentication implemented?"
# """
            
#             return response
            
#         except Exception as e:
#             return f"❌ Error querying repository: {str(e)}"
    
    def get_processing_status(self):
        """Get current processing status"""
        return self.processing_status
    
    def get_repo_structure(self):
        """Get basic repository structure for display"""
        if not self.is_loaded or not self.repo_path:
            return "No repository loaded"
        
        try:
            structure = []
            for root, dirs, files in os.walk(self.repo_path):
                # Skip hidden directories and common non-code directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']]
                
                level = root.replace(self.repo_path, '').count(os.sep)
                indent = '  ' * level
                structure.append(f"{indent}{os.path.basename(root)}/")
                
                # Limit files shown per directory
                subindent = '  ' * (level + 1)
                for file in files[:10]:  # Show max 10 files per directory
                    if not file.startswith('.'):
                        structure.append(f"{subindent}{file}")
                
                if len(files) > 10:
                    structure.append(f"{subindent}... and {len(files) - 10} more files")
                
                # Limit depth to avoid too much output
                if level > 3:
                    dirs.clear()
            
            return '\n'.join(structure[:50])  # Limit total lines
            
        except Exception as e:
            return f"Error reading repository structure: {str(e)}"
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
                self.repo_path = None
                self.is_loaded = False
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

    def initialize_llm(self):
        """Initialize LLM model loading"""
        if not self.llm_loading_started:
            print("🔄 Starting LLM model loading...")
            self.llm.load_model_async()
            self.llm_loading_started = True
            return "🔄 LLM model loading started in background..."
        elif self.llm.is_model_ready():
            return "✅ LLM model is ready!"
        else:
            return "⏳ LLM model is still loading..."
        
    
    
    def _generate_basic_response(self, query_text: str, results: List[Dict[str, Any]]) -> str:
        """Generate basic response without LLM"""
        response = f"""🔍 **Search Results for:** "{query_text}"

📊 **Found {len(results)} relevant code sections:**

"""
        
        for i, result in enumerate(results[:5], 1):  # Show top 5 results
            metadata = result.get('metadata', {})
            score = result.get('score', 0)
            
            chunk_type = metadata.get('chunk_type', 'unknown')
            file_path = metadata.get('file_path', 'unknown')
            
            response += f"""**{i}. {chunk_type.title()} Match** (Similarity: {score:.2f})
📄 File: `{file_path}`
"""
            
            if chunk_type == 'function':
                func_name = metadata.get('function_name', 'unknown')
                class_name = metadata.get('class_name')
                signature = metadata.get('signature', func_name)
                
                response += f"🔧 Function: `{signature}`\n"
                if class_name:
                    response += f"📦 Class: `{class_name}`\n"
                    
            elif chunk_type == 'class':
                class_name = metadata.get('class_name', 'unknown')
                methods = metadata.get('methods', [])
                response += f"📦 Class: `{class_name}`\n"
                if methods:
                    response += f"🔧 Methods: {', '.join(methods[:5])}\n"
                    
            elif chunk_type == 'file':
                language = metadata.get('language', 'unknown')
                total_lines = metadata.get('total_lines', 'unknown')
                response += f"📝 Language: {language}, Lines: {total_lines}\n"
            
            response += "---\n\n"
        
        # Add suggestions
        if len(results) > 5:
            response += f"... and {len(results) - 5} more results available.\n\n"
            
        response += f"""💡 **Suggestions:**
- Ask more specific questions about functions or classes
- Query about code patterns: "Show me error handling code"  
- Ask about structure: "What are the main components?"
- Request examples: "How is authentication implemented?"
"""
        
        return response
