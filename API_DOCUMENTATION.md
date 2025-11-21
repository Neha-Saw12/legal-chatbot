# Cognitbotz Legal AI Platform API

## Overview
This API provides RESTful endpoints for the Cognitbotz Legal AI Platform, enabling legal research and document drafting capabilities through programmatic access.

## Base URL
```
http://localhost:8000
```

## Endpoints

### Health Check
- **GET /** - Health check endpoint
- **GET /health** - Detailed health check

### Session Management
- **POST /sessions/create** - Create a new session
- **GET /sessions/{session_id}** - Get session information

### Document Processing
- **POST /documents/process** - Process legal documents and create embeddings

### Chat
- **POST /chat** - Chat with the legal AI assistant

### Legal Drafting
- **POST /draft** - Generate legal documents using AI
- **GET /document-types** - Get available document types
- **GET /drafting-styles** - Get available drafting styles
- **GET /available-clauses** - Get available legal clauses

## Usage Examples

### 1. Create a Session
```bash
curl -X POST "http://localhost:8000/sessions/create" -H "Content-Type: application/json"
```

### 2. Process Documents
```bash
curl -X POST "http://localhost:8000/documents/process" -H "Content-Type: application/json" -d '{
  "file_paths": ["legal_documents/COI.pdf", "legal_documents/BNS.pdf"],
  "session_id": "your-session-id"
}'
```

### 3. Chat with the AI
```bash
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{
  "message": "What are the fundamental rights mentioned in the Constitution of India?",
  "session_id": "your-session-id",
  "mode": "Document Only (RAG)"
}'
```

### 4. Generate Legal Documents
```bash
curl -X POST "http://localhost:8000/draft" -H "Content-Type: application/json" -d '{
  "doc_type": "Contracts & Agreements",
  "requirements": "Draft a service agreement between Company A and Company B for software development services",
  "style": "Formal Legal",
  "length": "Standard"
}'
```

## Running the API Server
```bash
python api.py
```

The API will be available at http://localhost:8000

## Dependencies
All dependencies are listed in requirements.txt and include:
- FastAPI
- Uvicorn
- Pydantic
- All existing project dependencies