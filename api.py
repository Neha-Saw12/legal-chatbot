"""
FastAPI endpoints for the Cognitbotz Legal AI Platform
Provides RESTful APIs for legal research and document drafting
"""

import os
import uuid
import logging
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules after setting up logging
try:
    from vectors import EmbeddingsManager, validate_file_security
    from chatbot import ChatbotManager
    from legal_drafting import LegalDraftingManager
    logger.info("Successfully imported required modules")
except ImportError as e:
    logger.error(f"Failed to import modules: {e}")
    # We'll handle this gracefully in the endpoints

# Initialize FastAPI app
app = FastAPI(
    title="Cognitbotz Legal AI Platform API",
    description="API for legal research and document drafting using AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for session management
active_sessions = {}
document_collections = {}

# Pydantic models for request/response validation
class ChatRequest(BaseModel):
    message: str
    session_id: str
    mode: str = "Document Only (RAG)"  # Options: "Document Only (RAG)", "General Chat (No RAG)", "Layman"
    enable_content_filter: bool = True
    enable_pii_detection: bool = True

class ChatResponse(BaseModel):
    session_id: str
    message: str
    response: str
    sources: List[Dict[str, Any]] = []
    tokens_used: int = 0
    is_flagged: bool = False
    flag_reason: Optional[str] = None
    response_type: str = "rag"

class DocumentProcessRequest(BaseModel):
    file_paths: List[str]
    session_id: Optional[str] = None

class DocumentProcessResponse(BaseModel):
    session_id: str
    collection_name: str
    processed_files: List[str]
    status: str
    message: str

class DraftRequest(BaseModel):
    doc_type: str
    requirements: str
    style: str = "Formal Legal"
    length: str = "Standard"
    clauses: Optional[List[str]] = None
    special_provisions: Optional[str] = None

class DraftResponse(BaseModel):
    document: str
    word_count: int
    tokens_used: int
    processing_time: float

class SessionInfo(BaseModel):
    session_id: str
    collection_name: Optional[str] = None
    documents_processed: List[str] = []
    created_at: str

# Health check endpoint
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Cognitbotz Legal AI Platform API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "legal-ai-platform"}

# Session management endpoints
@app.post("/sessions/create", response_model=SessionInfo)
async def create_session():
    """Create a new session"""
    session_id = str(uuid.uuid4())
    collection_name = f"legal_docs_{session_id[:8]}"
    
    active_sessions[session_id] = {
        "collection_name": collection_name,
        "documents_processed": [],
        "created_at": str(uuid.uuid1())
    }
    
    return SessionInfo(
        session_id=session_id,
        collection_name=collection_name,
        documents_processed=[],
        created_at=active_sessions[session_id]["created_at"]
    )

@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_info = active_sessions[session_id]
    return SessionInfo(
        session_id=session_id,
        collection_name=session_info.get("collection_name"),
        documents_processed=session_info.get("documents_processed", []),
        created_at=session_info.get("created_at")
    )

# Document processing endpoints
@app.post("/documents/process", response_model=DocumentProcessResponse)
async def process_documents(request: DocumentProcessRequest):
    """Process legal documents and create embeddings"""
    try:
        # Check if required modules are available
        if 'EmbeddingsManager' not in globals():
            raise HTTPException(status_code=500, detail="EmbeddingsManager not available")
            
        # Create session if not provided
        if not request.session_id:
            session_response = await create_session()
            session_id = session_response.session_id
        else:
            session_id = request.session_id
            
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        collection_name = active_sessions[session_id]["collection_name"]
        
        # Initialize Embeddings Manager
        embeddings_manager = EmbeddingsManager(
            model_name="BAAI/bge-small-en",
            device="cpu",
            encode_kwargs={"normalize_embeddings": True},
            qdrant_url=os.getenv('QDRANT_URL') or "https://default-qdrant-url.com",
            collection_name=collection_name,
            chunk_size=500,
            chunk_overlap=100,
            max_chunks=1000
        )
        
        processed_files = []
        
        # Process each file
        for file_path in request.file_paths:
            # Validate file
            is_valid, message = validate_file_security(file_path)
            if not is_valid:
                logger.warning(f"File validation failed for {file_path}: {message}")
                continue
                
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            # Process file
            try:
                result = embeddings_manager.create_embeddings(file_path)
                filename = os.path.basename(file_path)
                processed_files.append(filename)
                active_sessions[session_id]["documents_processed"].append(filename)
                logger.info(f"Successfully processed {filename}")
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        if not processed_files:
            raise HTTPException(status_code=400, detail="No documents were successfully processed")
            
        return DocumentProcessResponse(
            session_id=session_id,
            collection_name=collection_name,
            processed_files=processed_files,
            status="success",
            message=f"Successfully processed {len(processed_files)} documents"
        )
        
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")

# Chat endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the legal AI assistant"""
    try:
        # Check if required modules are available
        if 'ChatbotManager' not in globals():
            raise HTTPException(status_code=500, detail="ChatbotManager not available")
            
        # Validate session
        if request.session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
            
        collection_name = active_sessions[request.session_id]["collection_name"]
        
        # Initialize Chatbot Manager
        chatbot_manager = ChatbotManager(
            model_name="BAAI/bge-small-en",
            device="cpu",
            encode_kwargs={"normalize_embeddings": True},
            llm_model="llama-3.3-70b-versatile",
            llm_temperature=0.7,
            max_tokens=4000,
            qdrant_url=os.getenv('QDRANT_URL') or "https://default-qdrant-url.com",
            collection_name=collection_name,
            retrieval_k=10,
            score_threshold=0.2,
            use_custom_llm=False,
            custom_llm_url="",
            custom_llm_api_key="",
            custom_llm_model_name=""
        )
        
        # Determine RAG usage based on mode
        use_rag = True
        if request.mode == "General Chat (No RAG)":
            use_rag = False
        elif request.mode == "Document Only (RAG)":
            use_rag = True
        elif request.mode == "Layman":
            use_rag = len(active_sessions[request.session_id]["documents_processed"]) > 0
        
        # Determine if layman mode is enabled
        layman_mode = (request.mode == "Layman")
        
        # Get response from chatbot
        response_data = chatbot_manager.get_response(
            request.message,
            enable_content_filter=request.enable_content_filter,
            enable_pii_detection=request.enable_pii_detection,
            use_rag=use_rag,
            layman_mode=layman_mode
        )
        
        return ChatResponse(
            session_id=request.session_id,
            message=request.message,
            response=response_data.get('answer', 'No response generated'),
            sources=response_data.get('sources', []),
            tokens_used=response_data.get('tokens_used', 0),
            is_flagged=response_data.get('is_flagged', False),
            flag_reason=response_data.get('flag_reason'),
            response_type=response_data.get('response_type', 'rag')
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

# Legal drafting endpoints
@app.post("/draft", response_model=DraftResponse)
async def draft_legal_document(request: DraftRequest):
    """Generate legal documents using AI"""
    try:
        # Check if required modules are available
        if 'LegalDraftingManager' not in globals():
            raise HTTPException(status_code=500, detail="LegalDraftingManager not available")
            
        # Initialize Legal Drafting Manager
        drafting_manager = LegalDraftingManager(
            llm_model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=12000
        )
        
        # Generate document
        import time
        start_time = time.time()
        
        result = drafting_manager.generate_document(
            doc_type=request.doc_type,
            prompt=request.requirements,
            style=request.style,
            length=request.length,
            clauses=request.clauses if request.clauses else None,
            special_provisions=request.special_provisions if request.special_provisions else ""
        )
        
        processing_time = time.time() - start_time
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=f"Drafting failed: {result['error']}")
            
        return DraftResponse(
            document=result.get('document', ''),
            word_count=result.get('word_count', 0),
            tokens_used=result.get('tokens_used', 0),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Drafting error: {e}")
        raise HTTPException(status_code=500, detail=f"Drafting failed: {str(e)}")

# Document type information endpoints
@app.get("/document-types")
async def get_document_types():
    """Get available document types for drafting"""
    try:
        if 'LegalDraftingManager' not in globals():
            return {"error": "LegalDraftingManager not available"}
        drafting_manager = LegalDraftingManager()
        return drafting_manager.document_templates
    except Exception as e:
        logger.error(f"Error getting document types: {e}")
        return {"error": str(e)}

@app.get("/drafting-styles")
async def get_drafting_styles():
    """Get available drafting styles"""
    try:
        if 'LegalDraftingManager' not in globals():
            return {"error": "LegalDraftingManager not available"}
        drafting_manager = LegalDraftingManager()
        return drafting_manager.style_guides
    except Exception as e:
        logger.error(f"Error getting drafting styles: {e}")
        return {"error": str(e)}

@app.get("/available-clauses")
async def get_available_clauses():
    """Get available legal clauses"""
    try:
        if 'LegalDraftingManager' not in globals():
            return {"error": "LegalDraftingManager not available"}
        drafting_manager = LegalDraftingManager()
        return drafting_manager.clause_library
    except Exception as e:
        logger.error(f"Error getting available clauses: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)