# RAG Model Comparison System

A demonstration application that allows you to compare different RAG (Retrieval Augmented Generation) models using Ollama for local inference and Hugging Face embeddings for document indexing.

## Features

- **PDF Document Processing**: Upload and analyze PDF documents for indexing.
- **Multiple Embedding Models**: Choose from various HuggingFace embedding models for indexing.
- **Various Ollama LLM Models**: Select from a variety of LLM models to generate responses.
- **Performance Metrics**: Compare the performance of different models (response time, memory usage, CPU usage).
- **Comparison Visualization**: Charts to compare the performance of different configurations.
- **Fully Local Processing**: No connection to external APIs required, all processing is done locally.

## Available Embedding Models

| Model | Description | Dimensions |
|--------|-------------|-------------|
| intfloat/e5-base-v2 | Base model with good performance for general text | 768 |
| BAAI/bge-m3 | Multilingual embedding model with high ranking | 1024 |
| mixedbread-ai/mxbai-embed-large-v1 | High performance for precise retrieval | 768 |
| nomic-ai/nomic-embed-text-v1.5 | Optimized for document-level retrieval | 768 |
| Snowflake/arctic-embed-s | Lightweight and fast, good for quick iterations | 384 |
| sentence-transformers/all-MiniLM-L6-v2 | Very small but effective for simple tasks | 384 |

## Available LLM Models

| Model | Size | Context | Description |
|--------|--------|----------|-------------|
| Neural-Chat 7B | 4.1 GB | 8192 | Optimized for conversational responses |
| DeepSeek-R1 1.5B | 1.1 GB | 8192 | Fastest model, good for quick iterations |
| DeepSeek-R1 7B | 4.7 GB | 8192 | Strong reasoning capabilities |
| Mistral 7B | 4.1 GB | 8192 | High quality responses |
| Llama3 8B | 4.7 GB | 4096 | Good all-round performance |
| Phi3.5 3.8B | 2.2 GB | 4096 | Lightweight model with great performance for its size |
| Gemma2 9B | 5.2 GB | 8192 | Google's high-quality model with good reasoning |
| DeepSeek-R1 8B | 4.8 GB | 8192 | Enhanced model with strong reasoning capabilities |
| Qwen2.5 7B | 4.1 GB | 32768 | Excellent multilingual capabilities with long context |
| OpenCoder 8B | 4.7 GB | 16384 | Specialized for code and technical documentation |

## System Requirements

- Python 3.9+ 
- CPU: Minimum 4 cores (8+ recommended)
- RAM: Minimum 8GB (16GB+ recommended)
- Disk space: Minimum 10GB for models
- Operating System: Windows, macOS, or Linux

## Installation

### 1. Install Ollama

First, you need to install Ollama, which is the engine for running LLMs locally:

#### For Windows:
Download and install Ollama from [https://ollama.com/download/windows](https://ollama.com/download/windows)

#### For macOS:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### For Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Set up Python Environment

Clone this repository and set up the environment:

```bash
# Clone the repository (if applicable)
git clone https://github.com/your-username/rag-model-comparison.git
cd rag-model-comparison

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### requirements.txt file
Make sure your `requirements.txt` file contains:

```
streamlit>=1.24.0
langchain>=0.1.0
langchain-community>=0.0.10
langchain-huggingface>=0.0.6
langchain-ollama>=0.0.3
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
pdfplumber>=0.10.2
numpy==1.24.3
torch>=2.0.0
protobuf>=4.25.1
scikit-learn==1.2.2
transformers>=4.30.0
psutil>=5.9.0
watchdog==6.0.0
matplotlib>=3.5.0
pandas>=1.3.0
```

### 3. Download Ollama Models

Before running the application, you can pre-download the models you want to use. This is optional, as the application will automatically download the necessary models, but doing so beforehand avoids waiting during use:

```bash
# Download basic models
ollama pull neural-chat:7b
ollama pull deepseek-r1:1.5b
ollama pull mistral:7b
ollama pull llama3:8b

# Download additional models you might need
ollama pull phi3.5:3.8b
ollama pull gemma2:9b
ollama pull qwen2.5:7b
ollama pull opencoder:8b
```

To see all available models:
```bash
ollama list
```

## Running the Application

Once all dependencies are installed and Ollama is running, you can start the application:

```bash
# Make sure the virtual environment is activated
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Start the application
streamlit run app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`.

## Using the Application

1. **Select Models**: In the sidebar, select the embedding model and LLM model you want to use.

2. **Load Document**: Upload a PDF file through the file upload button.

3. **Adjust Parameters**: Modify the advanced parameters according to your needs (temperature, number of chunks to retrieve, chunk size).

4. **Ask Questions**: Once the document is processed, ask questions about its content in the text box.

5. **Compare Performance**: After testing different models, observe the comparison metrics to identify which one best suits your needs.

## Troubleshooting

### Ollama Issues

- **Ollama won't start**: Check that the service is running. On Windows, check the task manager. On Linux/macOS, use `ps aux | grep ollama`.

- **Error downloading models**: Make sure you have enough disk space and a stable internet connection. Try running `ollama pull [model]` manually.

- **Slow performance**: Large models require good hardware. Try with smaller models like `deepseek-r1:1.5b` or `phi3.5:3.8b`.

### Embedding Issues

- **CUDA errors**: If you see CUDA-related errors, the embedding models may be trying to use GPU when it's not available. Make sure the application is configured to use CPU.

- **Insufficient memory**: If you receive memory errors, consider using smaller embedding models such as `Snowflake/arctic-embed-s` or `sentence-transformers/all-MiniLM-L6-v2`.

### General Issues

- **Long loading times**: The first time a model is used, it will be automatically downloaded, which can take time. Subsequent runs will be faster.

- **"Chunk size too large" error**: If you receive this error, reduce the chunk size in the advanced configuration.

## Contributions

Contributions are welcome. If you want to contribute:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Ollama](https://ollama.com/) for enabling local execution of LLMs
- [LangChain](https://langchain.org/) for providing the tools to build RAG applications
- [HuggingFace](https://huggingface.co/) for the embedding models
- [Streamlit](https://streamlit.io/) for the user interface framework