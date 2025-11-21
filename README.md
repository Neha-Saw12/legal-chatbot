# Legal AI Chatbot Platform

An advanced legal research and document drafting platform powered by Retrieval-Augmented Generation (RAG) and Large Language Models.

## Features

- **Legal Document Processing**: Process and analyze legal documents (PDF, DOCX, PPTX, TXT)
- **Intelligent Chat Interface**: Three distinct modes for different user needs:
  - Document Only (RAG): Answers based on uploaded legal documents
  - General Chat (No RAG): General legal knowledge responses
  - Layman Mode: Simplified explanations of complex legal concepts
- **Legal Document Drafting**: AI-powered generation of various legal documents
- **FastAPI Endpoints**: RESTful API for programmatic access to all features
- **Secure Architecture**: Built-in content filtering and PII detection

## Technology Stack

- **Frontend**: Streamlit for web interface
- **Backend**: Python with FastAPI
- **AI/ML**: LangChain, Hugging Face Transformers, Sentence Transformers
- **Vector Database**: Qdrant for document embeddings storage
- **LLM Providers**: Groq, OpenAI, and custom LLM support
- **Document Processing**: PyPDF, python-docx, python-pptx

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Neha-Saw12/legal-chatbot.git
   cd legal-chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   ```

## Configuration

Create a `.env` file with the following variables:
```env
# Qdrant Configuration
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here

# Groq API Configuration
GROQ_API_KEY=your_groq_api_key_here

# Hugging Face Configuration
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token_here

# OpenAI API Configuration (optional)
OPENAI_API_KEY=your_openai_api_key_here

# Custom LLM Configuration (optional)
CUSTOM_LLM_URL=your_custom_llm_url_here
CUSTOM_LLM_API_KEY=your_custom_llm_api_key_here
```

## Usage

### Web Interface
Run the Streamlit application:
```bash
streamlit run app.py
```

### API Server
Start the FastAPI server:
```bash
python api.py
```

API documentation is available at `http://localhost:8000/docs`

## Project Structure

```
legal-chatbot/
├── app.py                 # Main Streamlit application
├── api.py                 # FastAPI endpoints
├── chatbot.py             # Chatbot logic and management
├── vectors.py             # Document processing and embeddings
├── legal_drafting.py      # Legal document generation
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── .gitignore            # Git ignore rules
├── API_DOCUMENTATION.md  # API endpoint documentation
└── legal_documents/      # Sample legal documents
```

## API Endpoints

Key endpoints include:
- `POST /sessions/create` - Create a new session
- `POST /documents/process` - Process legal documents
- `POST /chat` - Chat with the AI assistant
- `POST /draft` - Generate legal documents

See `API_DOCUMENTATION.md` for detailed API documentation.

## Document Processing

Place legal documents in the `legal_documents/` folder. Supported formats:
- PDF (.pdf)
- Word Documents (.docx)
- PowerPoint Presentations (.pptx)
- Text Files (.txt)

## Legal Drafting

Generate various legal documents including:
- Contracts & Agreements
- Petitions & Applications
- Court Orders & Judgments
- Legal Briefs & Submissions
- Statutes & Regulations

## Security Features

- Content filtering for inappropriate content
- PII (Personally Identifiable Information) detection and redaction
- Input validation and sanitization
- Rate limiting to prevent abuse
- Secure API key management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on the GitHub repository.