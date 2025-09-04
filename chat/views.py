
# Create your views here.
import os
import uuid
import fitz  # PyMuPDF
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from django.core.cache import cache
from dotenv import load_dotenv
import openai
from chat.rag_utils import populate_vector_db, collection

load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class UploadDocView(APIView):
    """
    Handle PDF/TXT upload and ingest into ChromaDB.
    """

    def post(self, request):
        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"error": "No file uploaded"}, status=400)

        ext = os.path.splitext(uploaded_file.name)[1].lower()

        try:
            # Extract text
            if ext == ".pdf":
                text = self.extract_pdf_text(uploaded_file)
            elif ext == ".txt":
                text = uploaded_file.read().decode("utf-8")
            else:
                return Response({"error": "Unsupported file type"}, status=400)

            # Split into chunks
            chunks = self.chunk_text(text, chunk_size=500)

            # Create IDs and metadatas
            ids = [str(uuid.uuid4()) for _ in chunks]
            metadatas = [{"source": uploaded_file.name, "chunk": i} for i, _ in enumerate(chunks)]

            # Add to ChromaDB
            populate_vector_db(chunks, metadatas=metadatas, ids=ids)

            return Response({"message": f"✅ {len(chunks)} chunks ingested from {uploaded_file.name}"}, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)

    def extract_pdf_text(self, file):
        text = ""
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text("text") + "\n"
        return text

    def chunk_text(self, text, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap
        return chunks


class ChatView(APIView):
    """
    Handle chat queries with optional document upload and session context.
    """
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        question = request.data.get("question", "").strip()
        jurisdiction = request.data.get("jurisdiction", "IN")
        extra_context = request.data.get("extra_context", "")
        session_id = request.data.get("session_id", str(uuid.uuid4()))
        
        if not question:
            return Response({"error": "Question is required"}, status=400)

        try:
            # Get session context
            session_key = f"chat_session_{session_id}"
            chat_history = cache.get(session_key, [])
            
            # Extract text from uploaded file if provided
            uploaded_file = request.FILES.get("file")
            # uploaded_file = './data/test.pdf'
            document_text = ""
            if uploaded_file:
                document_text = self.extract_document_text(uploaded_file)
            elif extra_context:
                document_text = extra_context

            # Search relevant legal documents from vector DB
            relevant_docs = self.search_relevant_documents(question, n_results=3)
            
            # Build context for OpenAI
            context_parts = []
            if relevant_docs:
                context_parts.append("Relevant legal information:")
                for doc in relevant_docs:
                    context_parts.append(f"- {doc}")
            
            if document_text:
                context_parts.append(f"User provided document content: {document_text[:2000]}")
            
            # Create messages for OpenAI
            system_prompt = self.get_system_prompt(jurisdiction)
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add chat history
            messages.extend(chat_history[-10:])  # Keep last 10 messages
            
            # Add current query with context
            user_message = question
            if context_parts:
                user_message = f"{question}\n\nContext:\n" + "\n".join(context_parts)
            
            messages.append({"role": "user", "content": user_message})

            # Call OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Update session context
            chat_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
            cache.set(session_key, chat_history, timeout=3600)  # 1 hour timeout
            
            return Response({
                "answer": answer,
                "session_id": session_id,
                "citations": relevant_docs,
                "context_used": bool(context_parts)
            }, status=200)

        except Exception as e:
            return Response({"error": str(e)}, status=500)

    def extract_document_text(self, file):
        """Extract text from uploaded PDF or TXT file."""
        ext = os.path.splitext(file.name)[1].lower()
        
        if ext == ".pdf":
            text = ""
            file_bytes = file.read()
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                text += page.get_text("text") + "\n"
            doc.close()
            return text
        elif ext == ".txt":
            return file.read().decode("utf-8")
        else:
            return ""

    def search_relevant_documents(self, query, n_results=3):
        """Search for relevant documents in ChromaDB."""
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results['documents'][0] if results['documents'] else []
        except Exception:
            return []

    def get_system_prompt(self, jurisdiction):
        """Get system prompt based on jurisdiction."""
        base_prompt = """You are a legal advisor AI assistant specializing in {jurisdiction} law. 
Provide accurate, helpful legal information based on the context provided. 
Always include appropriate disclaimers about seeking professional legal advice.
Be concise but thorough in your responses."""
        
        jurisdiction_map = {
            "IN": "Indian",
            "US": "United States", 
            "UK": "United Kingdom"
        }
        
        jurisdiction_name = jurisdiction_map.get(jurisdiction, "Indian")
        return base_prompt.format(jurisdiction=jurisdiction_name)
