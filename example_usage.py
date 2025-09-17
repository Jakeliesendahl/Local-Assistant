#!/usr/bin/env python3
"""
Example usage of the document processing pipeline.
This script demonstrates how to process documents and store them in a Chroma vector database.
"""

import logging
from pathlib import Path
from core.rag import (
    process_directory_to_vector_store,
    EmbeddingManager,
    ChromaVectorStore,
    DatabaseManager,
    ingest_document,
    process_documents_to_vector_store,
    query_with_metadata,
    ask_your_files,
    ask_your_files_simple,
    ask_your_files_simple_with_citations
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Example usage of the document processing pipeline."""
    
    # Example 1: Process all documents in a directory
    print("=== Example 1: Processing documents from a directory ===")
    
    # Specify the directory containing your documents
    document_directory = "documents"  # Change this to your document directory
    
    # Check if directory exists
    if Path(document_directory).exists():
        try:
            results = process_directory_to_vector_store(
                directory_path=document_directory,
                persist_directory="data/chroma",  # Chroma database location
                db_path="data/db.sqlite",         # SQLite database location
                model_name='all-MiniLM-L6-v2',   # Embedding model
                chunk_size=500,                   # Characters per chunk
                chunk_overlap=50,                 # Overlap between chunks
                file_types=['.pdf', '.docx', '.md', '.markdown']
            )
            
            print(f"Processing completed!")
            print(f"- Processed documents: {results['processed_documents']}")
            print(f"- Total chunks created: {results['total_chunks']}")
            print(f"- Average chunk size: {results.get('average_chunk_size', 0):.1f} characters")
            print(f"- Vector store info: {results.get('vector_store_info', {})}")
            print(f"- Database stats: {results.get('database_stats', {})}")
            
        except Exception as e:
            print(f"Error processing directory: {e}")
    else:
        print(f"Directory '{document_directory}' not found. Please create it and add some documents.")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Process individual documents
    print("=== Example 2: Processing individual documents ===")
    
    # Initialize components manually for more control
    try:
        embedding_manager = EmbeddingManager(model_name='all-MiniLM-L6-v2')
        vector_store = ChromaVectorStore(persist_directory="data/chroma")
        db_manager = DatabaseManager(db_path="data/db.sqlite")
        
        # Process a single document (replace with actual file path)
        document_path = "README.md"  # Example file
        
        if Path(document_path).exists():
            # Ingest the document
            document_data = ingest_document(document_path)
            
            if document_data:
                # Process it through the pipeline
                results = process_documents_to_vector_store(
                    documents=[document_data],
                    vector_store=vector_store,
                    embedding_manager=embedding_manager,
                    db_manager=db_manager,  # Include database manager
                    chunk_size=300,  # Smaller chunks for this example
                    chunk_overlap=30
                )
                
                print(f"Single document processing completed!")
                print(f"- File: {document_data['file_name']}")
                print(f"- Chunks created: {results['total_chunks']}")
                print(f"- Average chunk size: {results.get('average_chunk_size', 0):.1f} characters")
                
                # Show database information
                db_stats = db_manager.get_database_stats()
                print(f"- Database documents: {db_stats['document_count']}")
                print(f"- Database chunks: {db_stats['chunk_count']}")
            else:
                print(f"Failed to ingest document: {document_path}")
        else:
            print(f"File '{document_path}' not found.")
            
        # Get vector store information
        info = vector_store.get_collection_info()
        print(f"\nVector Store Status:")
        print(f"- Collection: {info['name']}")
        print(f"- Total documents: {info['count']}")
        print(f"- Storage location: {info['persist_directory']}")
        
        # Show database status
        db_stats = db_manager.get_database_stats()
        print(f"\nDatabase Status:")
        print(f"- Database path: {db_stats['database_path']}")
        print(f"- Documents: {db_stats['document_count']}")
        print(f"- Chunks: {db_stats['chunk_count']}")
        print(f"- Size: {db_stats['database_size_mb']} MB")
        
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please install required packages:")
        print("py -m pip install sentence-transformers chromadb")
    except Exception as e:
        print(f"Error in individual document processing: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Query with metadata enrichment
    print("=== Example 3: Enhanced querying with metadata ===")
    
    try:
        embedding_manager = EmbeddingManager(model_name='all-MiniLM-L6-v2')
        vector_store = ChromaVectorStore(persist_directory="data/chroma")
        db_manager = DatabaseManager(db_path="data/db.sqlite")
        
        # Example query using the enhanced query function
        query_text = "What is this project about?"
        
        results = query_with_metadata(
            query_text=query_text,
            embedding_manager=embedding_manager,
            vector_store=vector_store,
            db_manager=db_manager,
            n_results=3
        )
        
        print(f"Query: '{results['query']}'")
        print(f"Found {results['total_results']} similar chunks:")
        
        for result in results['results']:
            print(f"\n{result['rank']}. Similarity: {result['similarity_score']:.3f}")
            
            # Show vector metadata
            vm = result['vector_metadata']
            print(f"   Source: {vm.get('source_file', 'Unknown')}")
            print(f"   Chunk {vm.get('chunk_index', 0)+1}/{vm.get('total_chunks', 0)}")
            
            # Show database metadata if available
            if 'database_metadata' in result:
                dm = result['database_metadata']
                print(f"   File Path: {dm['file_path']}")
                print(f"   File Type: {dm['file_type']}")
                print(f"   Added: {dm['created_at']}")
            
            print(f"   Preview: {result['chunk_preview']}")
        
        # Show system stats
        print(f"\nSystem Status:")
        print(f"- Vector store documents: {results['vector_store_info']['count']}")
        print(f"- Database documents: {results['database_stats']['document_count']}")
        print(f"- Database chunks: {results['database_stats']['chunk_count']}")
            
    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Please install required packages:")
        print("py -m pip install sentence-transformers chromadb")
    except Exception as e:
        print(f"Error querying with metadata: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 4: Database queries
    print("=== Example 4: Direct database queries ===")
    
    try:
        db_manager = DatabaseManager(db_path="data/db.sqlite")
        
        # Get all documents
        all_docs = db_manager.get_all_documents()
        print(f"Total documents in database: {len(all_docs)}")
        
        for doc in all_docs[:3]:  # Show first 3
            print(f"- {doc['file_name']} ({doc['file_type']}) - {doc['word_count']} words")
            
            # Get chunks for this document
            chunks = db_manager.get_chunks_by_document_id(doc['id'])
            print(f"  └─ {len(chunks)} chunks")
        
        # Search documents
        if all_docs:
            search_results = db_manager.search_documents("README")
            print(f"\nSearch results for 'README': {len(search_results)} documents")
            
        # Show database stats
        stats = db_manager.get_database_stats()
        print(f"\nDatabase Statistics:")
        print(f"- Path: {stats['database_path']}")
        print(f"- Documents: {stats['document_count']}")
        print(f"- Chunks: {stats['chunk_count']}")
        print(f"- Size: {stats['database_size_mb']} MB")
            
    except Exception as e:
        print(f"Error with database queries: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Example 5: Ask Your Files functionality
    print("=== Example 5: Ask Your Files (RAG) ===")
    
    try:
        db_manager = DatabaseManager(db_path="data/db.sqlite")
        stats = db_manager.get_database_stats()
        
        if stats['document_count'] == 0:
            print("No documents found in database. Please process some documents first.")
            print("Run examples 1 or 2 above to add documents to the system.")
        else:
            print(f"Found {stats['document_count']} documents and {stats['chunk_count']} chunks in database")
            
            # Example questions
            example_questions = [
                "What is this project about?",
                "What are the main features?",
                "How do I get started?"
            ]
            
            print("\nTesting Ask Your Files with example questions:")
            
            for i, question in enumerate(example_questions, 1):
                print(f"\n{i}. Question: {question}")
                try:
                    # Use enhanced function with citations for demo
                    answer = ask_your_files_simple_with_citations(question, k=3, show_detailed_citations=True)
                    
                    # Show first part of answer
                    if "📚 SOURCES:" in answer:
                        answer_part = answer.split("📚 SOURCES:")[0].strip()
                        print(f"   Answer: {answer_part[:150]}..." if len(answer_part) > 150 else f"   Answer: {answer_part}")
                        
                        # Show source count
                        sources_part = answer.split("📚 SOURCES:")[1]
                        source_count = sources_part.count("[")
                        print(f"   📚 Sources: {source_count} documents cited")
                    else:
                        print(f"   Answer: {answer[:200]}..." if len(answer) > 200 else f"   Answer: {answer}")
                    
                except Exception as e:
                    if "Ollama" in str(e):
                        print("   ⚠️  Ollama not available. Install and run:")
                        print("      1. Install Ollama from https://ollama.ai")
                        print("      2. Run: ollama serve")
                        print("      3. Pull a model: ollama pull llama3")
                        break
                    else:
                        print(f"   Error: {e}")
            
            print("\n💡 For interactive Q&A with citations:")
            print("   py ask_with_citations.py")
            print("   or: py ask_your_files_example.py")
            
    except Exception as e:
        print(f"Error testing Ask Your Files: {e}")

if __name__ == "__main__":
    main()
