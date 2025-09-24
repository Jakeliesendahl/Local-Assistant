#!/usr/bin/env python3
"""
Example usage of the "Ask Your Files" functionality.
This demonstrates how to use RAG (Retrieval-Augmented Generation) to ask questions about your documents.
"""

import logging
from pathlib import Path
from core.rag import (
    ask_your_files,
    ask_your_files_simple,
    EmbeddingManager,
    ChromaVectorStore,
    DatabaseManager,
    process_directory_to_vector_store
)
from core.llm import LLMClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Demonstrate the Ask Your Files functionality."""
    
    print("🤖 Ask Your Files - RAG System Demo")
    print("=" * 50)
    
    # Check if we have any documents processed
    print("\n1️⃣ Checking if documents are available...")
    
    try:
        db_manager = DatabaseManager("data/db.sqlite")
        stats = db_manager.get_database_stats()
        
        print(f"   📊 Database contains {stats['document_count']} documents and {stats['chunk_count']} chunks")
        
        if stats['document_count'] == 0:
            print("   ⚠️  No documents found in database!")
            print("   💡 First, process some documents using:")
            print("      py example_usage.py")
            print("   💡 Or run: process_directory_to_vector_store('your_documents_folder')")
            return
        
        # Show available documents
        docs = db_manager.get_all_documents()
        print("   📁 Available documents:")
        for doc in docs[:5]:  # Show first 5
            print(f"      - {doc['file_name']} ({doc['file_type']}) - {doc['word_count']} words")
        
    except Exception as e:
        print(f"   ❌ Error checking database: {e}")
        return
    
    print("\n" + "=" * 50)
    
    # Example 1: Simple usage
    print("\n2️⃣ Example 1: Simple Ask Your Files")
    print("-" * 30)
    
    question1 = "What is this project about?"
    print(f"Question: {question1}")
    print("Processing...")
    
    try:
        # Using the simple function (auto-initializes everything)
        answer1 = ask_your_files_simple(question1)
        print(f"\nAnswer: {answer1}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("💡 Make sure Ollama is running with: ollama serve")
        print("💡 And that you have llama3 model: ollama pull llama3")
    
    print("\n" + "=" * 50)
    
    # Example 2: Advanced usage with detailed response
    print("\n3️⃣ Example 2: Advanced Ask Your Files")
    print("-" * 30)
    
    question2 = "How does the document processing pipeline work?"
    print(f"Question: {question2}")
    print("Processing...")
    
    try:
        # Initialize components manually for more control
        embedding_manager = EmbeddingManager('all-MiniLM-L6-v2')
        vector_store = ChromaVectorStore("data/chroma")
        db_manager = DatabaseManager("data/db.sqlite")
        llm_client = LLMClient(model="llama3")
        
        # Ask the question with detailed response
        result = ask_your_files(
            question=question2,
            embedding_manager=embedding_manager,
            vector_store=vector_store,
            db_manager=db_manager,
            llm_client=llm_client,
            k=3,  # Retrieve top 3 chunks
            include_metadata=True
        )
        
        print(f"\n📝 Answer:")
        print(result['answer'])
        
        print(f"\n📚 Sources used:")
        for i, source in enumerate(result['sources'][:3]):
            print(f"   {i+1}. {source['source_file']} (similarity: {source['similarity_score']:.3f})")
            if 'chunk_index' in source:
                print(f"      Chunk {source['chunk_index']+1}/{source['total_chunks']}")
        
        print(f"\n📊 Retrieval Info:")
        ri = result['retrieval_info']
        print(f"   - Chunks found: {ri['chunks_found']}")
        print(f"   - Chunks used: {ri['chunks_used']}")
        print(f"   - Context length: {ri['total_context_length']} characters")
        print(f"   - LLM used: {result['model_used'] if result['llm_used'] else 'None'}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 50)
    
    # Example 3: Interactive Q&A
    print("\n4️⃣ Example 3: Interactive Q&A")
    print("-" * 30)
    print("You can now ask questions about your documents!")
    print("Type 'quit' to exit")
    
    try:
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
            
            if not question:
                continue
            
            print("🤔 Thinking...")
            
            try:
                answer = ask_your_files_simple(question, k=3)
                print(f"\n🤖 Answer: {answer}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    print("\n" + "=" * 50)
    
    # Example 4: Different question types
    print("\n5️⃣ Example 4: Different Question Types")
    print("-" * 30)
    
    example_questions = [
        "What are the main features mentioned in the documents?",
        "How do I install the dependencies?",
        "What file formats are supported?",
        "Can you summarize the key points from the documents?",
        "What is the purpose of the database component?"
    ]
    
    print("Here are some example questions you could ask:")
    for i, q in enumerate(example_questions, 1):
        print(f"   {i}. {q}")
    
    print("\n💡 Tips for better results:")
    print("   - Be specific in your questions")
    print("   - Ask about topics that are likely in your documents")
    print("   - Try different phrasings if you don't get good results")
    print("   - The system works best with factual questions")
    
    print("\n🔧 Configuration options:")
    print("   - k: Number of document chunks to retrieve (default: 5)")
    print("   - max_context_length: Maximum characters sent to LLM (default: 4000)")
    print("   - llm_model: Ollama model to use (default: 'llama3')")
    print("   - include_metadata: Include database metadata (default: True)")

def demo_with_sample_document():
    """Create a sample document and demonstrate the system."""
    
    print("\n🎯 Creating sample document for demo...")
    
    # Create sample document
    sample_content = """# Local Assistant Project

## Overview
The Local Assistant is a comprehensive document processing and retrieval system that uses RAG (Retrieval-Augmented Generation) to enable intelligent question-answering over your document collection.

## Key Features
- **Document Ingestion**: Supports PDF, DOCX, and Markdown files
- **Text Chunking**: Intelligent text splitting with configurable chunk sizes and overlap
- **Embeddings**: Uses sentence-transformers (all-MiniLM-L6-v2) for semantic embeddings
- **Vector Storage**: Persistent storage using Chroma vector database
- **Metadata Database**: SQLite database for rich metadata tracking
- **Question Answering**: RAG-powered Q&A using local LLMs via Ollama

## Architecture
The system follows a modular architecture:
1. Document ingestion and preprocessing
2. Text chunking with smart boundary detection
3. Embedding generation using transformer models
4. Dual storage: vectors in Chroma, metadata in SQLite
5. Retrieval-augmented generation for question answering

## Installation
1. Install Python dependencies: `pip install -r requirements.txt`
2. Install Ollama and pull a model: `ollama pull llama3`
3. Process your documents using the provided scripts
4. Start asking questions about your documents!

## Benefits
- **Local Processing**: Everything runs locally, no data sent to external services
- **Rich Metadata**: Track document sources, chunk indices, and processing timestamps
- **Flexible Querying**: Both semantic search and traditional database queries
- **Source Attribution**: Always know which documents contributed to answers
"""
    
    sample_file = Path("sample_document.md")
    sample_file.write_text(sample_content, encoding='utf-8')
    
    print("   ✅ Created sample document")
    
    # Process the sample document
    print("   🔄 Processing sample document...")
    
    try:
        results = process_directory_to_vector_store(
            directory_path=".",  # Current directory
            file_types=['.md'],  # Only process our sample
            chunk_size=300,
            chunk_overlap=50
        )
        
        print(f"   ✅ Processed {results['processed_documents']} documents")
        print(f"   ✅ Created {results['total_chunks']} chunks")
        
        # Now demo the Q&A
        print("\n   🤖 Demo Q&A:")
        
        questions = [
            "What are the main features of the Local Assistant?",
            "How do I install the system?",
            "What file formats are supported?"
        ]
        
        for question in questions:
            print(f"\n   ❓ Q: {question}")
            try:
                answer = ask_your_files_simple(question, k=2)
                print(f"   🤖 A: {answer[:200]}...")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
    except Exception as e:
        print(f"   ❌ Error processing document: {e}")
    
    finally:
        # Clean up
        if sample_file.exists():
            sample_file.unlink()
            print("   🧹 Cleaned up sample document")

if __name__ == "__main__":
    try:
        main()
        
        # Offer to run demo if no documents are available
        db_manager = DatabaseManager("data/db.sqlite")
        stats = db_manager.get_database_stats()
        
        if stats['document_count'] == 0:
            response = input("\n💡 Would you like to run a demo with a sample document? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                demo_with_sample_document()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
