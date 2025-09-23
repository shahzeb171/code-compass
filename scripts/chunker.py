import ast
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import logging

logger = logging.getLogger("code_compass")
@dataclass
class CodeChunk:
    """Represents a hierarchical code chunk with metadata"""
    id: str
    content: str
    chunk_type: str  # 'file', 'class', 'function', 'block'
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary for storage"""
        return {
            'id': self.id,
            'content': self.content,
            'chunk_type': self.chunk_type,
            'metadata': self.metadata,
            'embedding': self.embedding
        }

class HierarchicalChunker:
    """
    Advanced hierarchical code chunker that creates multiple levels of chunks:
    Level 1: File-level context
    Level 2: Class-level chunks  
    Level 3: Function-level chunks
    Level 4: Code block chunks (for complex functions)
    """
    
    def __init__(self, complexity_threshold: int = 20):
        self.complexity_threshold = complexity_threshold
        self.supported_extensions = {
            '.py': self._parse_python,
            '.js': self._parse_javascript,
            '.ts': self._parse_typescript,
            '.java': self._parse_java,
            '.cpp': self._parse_cpp,
            '.c': self._parse_c,
            '.go': self._parse_go,
            '.rs': self._parse_rust,
            # Add more as needed
        }
        
    def chunk_repository(self, repo_path: str) -> List[CodeChunk]:
        """
        Main method to chunk entire repository hierarchically
        """
        chunks = []
        repo_name = os.path.basename(repo_path)
        
        logger.info(f"🔄 Starting hierarchical chunking of {repo_name}...")
        
        # Walk through repository
        for root, dirs, files in os.walk(repo_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and 
                      d not in ['node_modules', '__pycache__', 'venv', 'env', 'dist', 'build']]
            
            for file in files:
                if self._should_process_file(file):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    
                    try:
                        file_chunks = self._process_file(file_path, relative_path, repo_name)
                        logger.debug(f"File chunks: {[chunk.to_dict() for chunk in file_chunks]}")
                        chunks.extend(file_chunks)
                        logger.info(f"✅ Processed {relative_path} -> {len(file_chunks)} chunks")
                    except Exception as e:
                        logger.info(f"❌ Error processing {relative_path}: {str(e)}")
                        continue
        
        logger.info(f"🎉 Chunking complete! Generated {len(chunks)} total chunks")
        return chunks
    
    def _should_process_file(self, filename: str) -> bool:
        """Check if file should be processed for chunking"""
        ext = Path(filename).suffix.lower()
        
        # Skip files that are too large or unwanted
        unwanted_files = {
            'package-lock.json', 'yarn.lock', 'poetry.lock',
            'requirements.txt', '.gitignore', 'README.md',
            'LICENSE', 'CHANGELOG.md'
        }
        
        if filename in unwanted_files:
            return False
            
        # Process code files
        return ext in self.supported_extensions or ext in [
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.cs'
        ]
    
    def _process_file(self, file_path: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Process a single file and generate hierarchical chunks"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            logger.info(f"❌ Could not read {relative_path}: {e}")
            return chunks
        
        if not content.strip():
            return chunks
            
        file_ext = Path(file_path).suffix.lower()
        
        # Level 1: File-level chunk
        file_chunk = self._create_file_chunk(content, relative_path, repo_name)
        chunks.append(file_chunk)
        
        # Language-specific parsing for deeper levels
        if file_ext in self.supported_extensions:
            try:
                deeper_chunks = self.supported_extensions[file_ext](content, relative_path, repo_name)
                chunks.extend(deeper_chunks)
            except Exception as e:
                logger.info(f"⚠️  Advanced parsing failed for {relative_path}, using basic chunking: {e}")
                # Fallback to basic function extraction
                basic_chunks = self._basic_function_extraction(content, relative_path, repo_name)
                chunks.extend(basic_chunks)
        else:
            # For unsupported languages, do basic function/class detection
            basic_chunks = self._basic_function_extraction(content, relative_path, repo_name)
            chunks.extend(basic_chunks)
        
        return chunks
    
    def _create_file_chunk(self, content: str, relative_path: str, repo_name: str) -> CodeChunk:
        """Create Level 1: File-level context chunk"""
        
        # Extract file summary info
        lines = content.split('\n')
        total_lines = len(lines)
        
        # Get imports/includes
        imports = self._extract_imports(content, Path(relative_path).suffix)
        
        # Create condensed file overview
        file_summary = f"""File: {relative_path}
Lines: {total_lines}
Language: {Path(relative_path).suffix}

Imports/Dependencies:
{chr(10).join(imports[:10])}  # Show first 10 imports

File Purpose: {self._infer_file_purpose(relative_path, content)}

Main Components:
{self._extract_main_components_summary(content, Path(relative_path).suffix)}
"""
        
        chunk_id = self._generate_chunk_id(repo_name, relative_path, "file", "")
        
        metadata = {
            'repo_name': repo_name,
            'file_path': relative_path,
            'chunk_type': 'file',
            'level': 1,
            'language': Path(relative_path).suffix,
            'total_lines': total_lines,
            'imports': imports,
            'timestamp': datetime.now().isoformat()
        }
        
        return CodeChunk(
            id=chunk_id,
            content=file_summary,
            chunk_type='file',
            metadata=metadata
        )
    
    def _parse_python(self, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse Python files for classes and functions"""
        chunks = []
        
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.info(f"⚠️  Python syntax error in {relative_path}: {e}")
            return self._basic_function_extraction(content, relative_path, repo_name)
        
        # Level 2: Class chunks
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_chunk = self._create_class_chunk(node, content, relative_path, repo_name)
                chunks.append(class_chunk)
                
                # Level 3: Method chunks within class
                for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                    method_chunk = self._create_function_chunk(
                        method, content, relative_path, repo_name, 
                        parent_class=node.name
                    )
                    chunks.append(method_chunk)
                    
                    # Level 4: Complex method sub-chunks
                    if self._calculate_complexity(method) > self.complexity_threshold:
                        sub_chunks = self._create_sub_chunks(method, content, relative_path, repo_name)
                        chunks.extend(sub_chunks)
        
        # Level 3: Standalone functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip if it's inside a class (already handled above)
                parent_classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) 
                                if any(isinstance(child, ast.FunctionDef) and child.name == node.name 
                                      for child in ast.walk(n))]
                
                if not parent_classes:
                    func_chunk = self._create_function_chunk(node, content, relative_path, repo_name)
                    chunks.append(func_chunk)
                    
                    # Level 4: Complex function sub-chunks
                    if self._calculate_complexity(node) > self.complexity_threshold:
                        sub_chunks = self._create_sub_chunks(node, content, relative_path, repo_name)
                        chunks.extend(sub_chunks)
        
        return chunks
    
    def _create_class_chunk(self, class_node: ast.ClassDef, content: str, relative_path: str, repo_name: str) -> CodeChunk:
        """Create Level 2: Class-level chunk"""
        
        lines = content.split('\n')
        class_content = self._extract_node_content(class_node, lines)
        
        # Get class methods summary
        methods = [n.name for n in class_node.body if isinstance(n, ast.FunctionDef)]
        
        # Get docstring
        docstring = ast.get_docstring(class_node) or "No docstring available"
        
        # Get inheritance info
        bases = [self._get_node_name(base) for base in class_node.bases] if class_node.bases else []
        
        class_summary = f"""Class: {class_node.name}
File: {relative_path}
Inheritance: {' -> '.join(bases) if bases else 'No inheritance'}

Docstring:
{docstring[:300]}...

Methods ({len(methods)}):
{', '.join(methods)}

Full Class Definition:
{class_content[:1000]}...  # Truncated for embedding
"""
        
        chunk_id = self._generate_chunk_id(repo_name, relative_path, "class", class_node.name)
        
        metadata = {
            'repo_name': repo_name,
            'file_path': relative_path,
            'chunk_type': 'class',
            'level': 2,
            'class_name': class_node.name,
            'methods': methods,
            'inheritance': bases,
            'line_start': class_node.lineno,
            'line_end': getattr(class_node, 'end_lineno', class_node.lineno),
            'docstring': docstring,
            'timestamp': datetime.now().isoformat()
        }
        
        return CodeChunk(
            id=chunk_id,
            content=class_summary,
            chunk_type='class',
            metadata=metadata
        )
    
    def _create_function_chunk(self, func_node: ast.FunctionDef, content: str, relative_path: str, 
                             repo_name: str, parent_class: Optional[str] = None) -> CodeChunk:
        """Create Level 3: Function-level chunk"""
        
        lines = content.split('\n')
        func_content = self._extract_node_content(func_node, lines)
        
        # Get function signature
        args = [arg.arg for arg in func_node.args.args]
        signature = f"{func_node.name}({', '.join(args)})"
        
        # Get docstring
        docstring = ast.get_docstring(func_node) or "No docstring available"
        
        # Calculate complexity
        complexity = self._calculate_complexity(func_node)
        
        func_summary = f"""Function: {signature}
File: {relative_path}
Class: {parent_class or 'Standalone function'}
Complexity Score: {complexity}

Docstring:
{docstring[:200]}...

Function Implementation:
{func_content}
"""
        
        chunk_id = self._generate_chunk_id(
            repo_name, relative_path, "function", 
            f"{parent_class}.{func_node.name}" if parent_class else func_node.name
        )
        
        metadata = {
            'repo_name': repo_name,
            'file_path': relative_path,
            'chunk_type': 'function',
            'level': 3,
            'function_name': func_node.name,
            'class_name': parent_class,
            'signature': signature,
            'arguments': args,
            'complexity': complexity,
            'line_start': func_node.lineno,
            'line_end': getattr(func_node, 'end_lineno', func_node.lineno),
            'docstring': docstring,
            'timestamp': datetime.now().isoformat()
        }
        
        return CodeChunk(
            id=chunk_id,
            content=func_summary,
            chunk_type='function',
            metadata=metadata
        )
    
    def _create_sub_chunks(self, func_node: ast.FunctionDef, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Create Level 4: Sub-chunks for complex functions"""
        chunks = []
        
        # For now, create logical blocks based on control structures
        lines = content.split('\n')
        func_lines = lines[func_node.lineno-1:getattr(func_node, 'end_lineno', func_node.lineno)]
        
        # Simple block detection based on indentation and keywords
        blocks = self._detect_code_blocks(func_lines, func_node.name)
        
        for i, block in enumerate(blocks):
            if len(block['content']) > 50:  # Only create chunks for substantial blocks
                chunk_id = self._generate_chunk_id(
                    repo_name, relative_path, "block", f"{func_node.name}_block_{i}"
                )
                
                block_summary = f"""Code Block {i+1} in {func_node.name}()
Type: {block['type']}
Purpose: {block['purpose']}

Code:
{block['content']}
"""
                
                metadata = {
                    'repo_name': repo_name,
                    'file_path': relative_path,
                    'chunk_type': 'block',
                    'level': 4,
                    'function_name': func_node.name,
                    'block_index': i,
                    'block_type': block['type'],
                    'block_purpose': block['purpose'],
                    'timestamp': datetime.now().isoformat()
                }
                
                chunks.append(CodeChunk(
                    id=chunk_id,
                    content=block_summary,
                    chunk_type='block',
                    metadata=metadata
                ))
        
        return chunks
    
    # Helper methods
    def _extract_imports(self, content: str, file_ext: str) -> List[str]:
        """Extract import statements based on language"""
        imports = []
        lines = content.split('\n')
        
        if file_ext == '.py':
            for line in lines[:50]:  # Check first 50 lines
                stripped = line.strip()
                if stripped.startswith(('import ', 'from ')):
                    imports.append(stripped)
        elif file_ext in ['.js', '.ts']:
            for line in lines[:50]:
                stripped = line.strip()
                if stripped.startswith(('import ', 'const ', 'require(')):
                    imports.append(stripped)
        
        return imports
    
    def _infer_file_purpose(self, relative_path: str, content: str) -> str:
        """Infer the purpose of a file based on path and content"""
        filename = os.path.basename(relative_path).lower()
        
        if 'test' in filename:
            return "Test file"
        elif 'config' in filename:
            return "Configuration file"
        elif 'util' in filename or 'helper' in filename:
            return "Utility/Helper functions"
        elif '__init__' in filename:
            return "Package initialization"
        elif 'main' in filename:
            return "Main entry point"
        elif 'model' in filename:
            return "Data model/schema definition"
        elif 'view' in filename:
            return "View/UI component"
        elif 'controller' in filename:
            return "Controller/Logic handler"
        else:
            # Analyze content for clues
            if 'class ' in content and 'def __init__' in content:
                return "Class definition file"
            elif 'def ' in content:
                return "Function definitions"
            else:
                return "Code file"
    
    def _extract_main_components_summary(self, content: str, file_ext: str) -> str:
        """Extract summary of main components (classes, functions)"""
        if file_ext != '.py':
            return "Component analysis available for Python files only"
        
        try:
            tree = ast.parse(content)
            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            summary = ""
            if classes:
                summary += f"Classes: {', '.join(classes[:5])}\n"
            if functions:
                summary += f"Functions: {', '.join(functions[:10])}\n"
            
            return summary or "No major components detected"
        except:
            return "Could not analyze components"
    
    def _basic_function_extraction(self, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Fallback function extraction using regex patterns"""
        chunks = []
        # This is a simplified fallback - you can enhance with regex patterns
        # for different languages
        return chunks
    
    def _extract_node_content(self, node: ast.AST, lines: List[str]) -> str:
        """Extract the actual code content for an AST node"""
        start_line = node.lineno - 1
        end_line = getattr(node, 'end_lineno', node.lineno) - 1
        
        if end_line >= len(lines):
            end_line = len(lines) - 1
            
        return '\n'.join(lines[start_line:end_line + 1])
    
    def _get_node_name(self, node: ast.AST) -> str:
        """Get the name of an AST node"""
        if hasattr(node, 'id'):
            return node.id
        elif hasattr(node, 'attr'):
            return node.attr
        else:
            return str(node)
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
                
        return complexity
    
    def _detect_code_blocks(self, func_lines: List[str], func_name: str) -> List[Dict[str, str]]:
        """Detect logical code blocks within a function"""
        blocks = []
        current_block = []
        block_type = "sequential"
        
        for line in func_lines:
            stripped = line.strip()
            
            if any(keyword in stripped for keyword in ['if ', 'elif ', 'else:']):
                if current_block:
                    blocks.append({
                        'content': '\n'.join(current_block),
                        'type': block_type,
                        'purpose': f"Logic block in {func_name}"
                    })
                    current_block = []
                block_type = "conditional"
            elif any(keyword in stripped for keyword in ['for ', 'while ']):
                if current_block:
                    blocks.append({
                        'content': '\n'.join(current_block),
                        'type': block_type,
                        'purpose': f"Logic block in {func_name}"
                    })
                    current_block = []
                block_type = "loop"
            elif any(keyword in stripped for keyword in ['try:', 'except', 'finally:']):
                if current_block:
                    blocks.append({
                        'content': '\n'.join(current_block),
                        'type': block_type,
                        'purpose': f"Logic block in {func_name}"
                    })
                    current_block = []
                block_type = "exception_handling"
            
            current_block.append(line)
        
        if current_block:
            blocks.append({
                'content': '\n'.join(current_block),
                'type': block_type,
                'purpose': f"Final block in {func_name}"
            })
        
        return blocks
    
    def _generate_chunk_id(self, repo_name: str, file_path: str, chunk_type: str, identifier: str) -> str:
        """Generate unique chunk ID"""
        unique_string = f"{repo_name}:{file_path}:{chunk_type}:{identifier}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    # Placeholder methods for other languages
    def _parse_javascript(self, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse JavaScript files - placeholder for now"""
        return []
    
    def _parse_typescript(self, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse TypeScript files - placeholder for now"""
        return []
    
    def _parse_java(self, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse Java files - placeholder for now"""
        return []
    
    def _parse_cpp(self, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse C++ files - placeholder for now"""
        return []
    
    def _parse_c(self, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse C files - placeholder for now"""
        return []
    
    def _parse_go(self, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse Go files - placeholder for now"""
        return []
    
    def _parse_rust(self, content: str, relative_path: str, repo_name: str) -> List[CodeChunk]:
        """Parse Rust files - placeholder for now"""
        return []