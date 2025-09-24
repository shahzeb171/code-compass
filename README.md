# 🔍code-compass

An AI-powered tool for analyzing code repositories using hierarchical chunking and semantic search with Pinecone vector database.

## 🚀 Features

- **📥 Multiple Input Methods**: GitHub URLs or ZIP file uploads
- **🧠 Hierarchical Chunking**: Smart code parsing at multiple levels (file → class → function → block)
- **🔍 Semantic Search**: AI-powered natural language queries using Pinecone vector database
- **🤖 Intelligent Analysis**: Local LLM integration with Qwen2.5-Coder-7B-Instruct
- **💬 Conversation History**: Maintains context across multiple queries
- **📊 Repository Analytics**: Comprehensive statistics and structure analysis
- **🎯 Pinecone Integration**: Scalable vector database with automatic embedding generation
- **⚡ Optimized Performance**: Quantized models for efficient local inference

## 🛠️ Setup

### Prerequisites

1. **Python 3.8+**
2. **Pinecone Account**: Create a free account at [Pinecone.io](https://www.pinecone.io/)
3. **System Requirements** for LLM:
   - **RAM**: 8GB minimum (16GB+ recommended)
   - **Storage**: 5-8GB free space for model
   - **CPU**: Multi-core processor (supports GPU acceleration if available)

### Installation

1. **Clone or download this project**
   ```bash
   git clone <your-repo-url>
   cd code-repository-analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the LLM model**
   ```bash
   python download_model.py
   ```
   
   **Recommended**: Select option 1 (Q4_K_M) for the best balance of quality and performance.

4. **Set up Pinecone API Key**
   
   Option A - Environment Variable (Recommended):
   ```bash
   export PINECONE_API_KEY="your-pinecone-api-key-here"
   ```
   
   Option B - Create `.env` file:
   ```
   PINECONE_API_KEY=your-pinecone-api-key-here
   ```
   
   Option C - Enter in the web interface under "Advanced Options"

### Getting Your Pinecone API Key

1. Go to [Pinecone.io](https://www.pinecone.io/) and sign up for a free account
2. Navigate to the "API Keys" section in your dashboard
3. Create a new API key or copy an existing one
4. The free tier includes:
   - 1 index
   - 5M vector dimensions
   - Enough for most code analysis projects!

## 🚀 Usage

1. **Start the application**
   ```bash
   python main.py
   ```

2. **Open your browser** to `http://localhost:7860`

3. **Load a repository**
   - Enter a GitHub URL (e.g., `https://github.com/pallets/flask`)
   - Or upload a ZIP file of your code
   - Click "📁 Load Repository"

4. **Process the repository**
   - Click "🚀 Process Repository" to analyze and chunk your code
   - This creates hierarchical chunks and stores them in Pinecone with automatic embedding generation
   - Wait for processing to complete (may take 1-5 minutes depending on repo size)

5. **Initialize the AI model** (Optional but recommended)
   - Click "🚀 Initialize LLM" to start loading the local AI model
   - This will load Qwen2.5-Coder-7B-Instruct for intelligent code analysis
   - Initial loading takes 1-3 minutes

6. **Query your code**
   - Ask natural language questions like:
     - "What does this repository do?"
     - "Show me authentication functions"
     - "How is error handling implemented?"
     - "What are the main classes?"
   - Toggle "Use AI Analysis" for intelligent responses vs basic search results
   - The AI maintains conversation context for follow-up questions

## 📊 How It Works

### Hierarchical Chunking Strategy

The system creates multiple levels of code chunks:

**Level 1: File Context**
- Complete file overview with imports and purpose
- Metadata: file path, language, total lines

**Level 2: Class Chunks** 
- Full class definitions with inheritance and methods
- Metadata: class name, methods list, relationships

**Level 3: Function Chunks**
- Individual function implementations with signatures
- Metadata: function name, arguments, complexity score

**Level 4: Code Block Chunks**
- Sub-chunks for complex functions (loops, conditionals, error handling)
- Metadata: block type, purpose, parent function

### Vector Search Process

1. **Embedding Generation**: Code chunks are converted to vector embeddings using SentenceTransformers
2. **Vector Storage**: Embeddings stored in Pinecone with rich metadata
3. **Semantic Search**: User queries are embedded and matched against stored vectors
4. **Hybrid Filtering**: Results filtered by chunk type, file path, repository, etc.
5. **Ranked Results**: Most relevant code sections returned with similarity scores

## 🔧 Configuration Options

### Advanced Settings (in web interface)

- **Pinecone Environment**: Default is "us-west1-gcp-free" for free tier
- **Complexity Threshold**: Controls when functions are sub-chunked (default: 20)
- **Embedding Model**: Uses "all-MiniLM-L6-v2" for fast, accurate embeddings

### Supported Languages

Currently optimized for Python with basic support for:
- JavaScript/TypeScript
- Java
- C/C++
- Go
- Rust
- PHP
- Ruby

## 📝 Example Repositories

Try these public repositories:

- **Flask**: `https://github.com/pallets/flask` - Web framework
- **Requests**: `https://github.com/requests/requests` - HTTP library  
- **FastAPI**: `https://github.com/tiangolo/fastapi` - Modern web framework
- **Black**: `https://github.com/psf/black` - Code formatter

## 🔍 Example Queries

### General Repository Understanding
- "What is the main purpose of this repository?"
- "What are the core components and how do they interact?"
- "Show me the project architecture overview"

### Function & Class Discovery
- "What are the main classes and their responsibilities?"
- "Show me all authentication-related functions"
- "Find functions that handle file operations"
- "What utility functions are available?"

### Implementation Analysis  
- "How is error handling implemented?"
- "Show me configuration management code"
- "Find database-related functions"
- "How does logging work in this project?"

### Code Patterns
- "Show me decorator implementations"
- "Find async/await usage patterns"
- "What design patterns are used?"
- "How are tests structured?"

## 🛟 Troubleshooting

### Common Issues

**"Pinecone API key is required"**
- Make sure you've set the `PINECONE_API_KEY` environment variable
- Or enter it in the Advanced Options section

**"Error downloading repository"**
- Check that the GitHub URL is correct and public
- Ensure you have internet connection
- Large repositories may timeout - try smaller repos first

**"No chunks generated"**
- Make sure the repository contains supported code files
- Check that ZIP files aren't corrupted
- Python files work best currently

**"Vector store initialization failed"**
- Verify your Pinecone API key is valid
- Check your Pinecone account hasn't exceeded free tier limits
- Try a different environment region if needed

### Performance Tips

- Start with smaller repositories (< 100 files) to test
- Python repositories work best currently
- Processing time scales with repository size
- Queries are fast once processing is complete

## 🔮 Future Enhancements

- **More Language Support**: Better parsing for JavaScript, Java, etc.
- **Code Generation**: AI-powered code completion and generation
- **Diff Analysis**: Compare changes between repository versions
- **Team Collaboration**: Share analyzed repositories
- **Custom Embeddings**: Fine-tuned models for specific domains
- **API Integration**: REST API for programmatic access

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please open issues or submit pull requests.

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Open a GitHub issue with detailed error messages
3. Include your Python version and OS information