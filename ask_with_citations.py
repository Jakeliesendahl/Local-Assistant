#!/usr/bin/env python3
"""
Command-line interface for Ask Your Files with detailed citations.
Usage: py ask_with_citations.py "Your question here"
"""

import sys
import argparse
from pathlib import Path
from core.rag import (
    ask_your_files_simple_with_citations,
    ask_your_files_with_citations,
    EmbeddingManager,
    ChromaVectorStore,
    DatabaseManager
)
from core.llm import LLMClient

def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about your documents with detailed citations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  py ask_with_citations.py "What is this project about?"
  py ask_with_citations.py "How do I install the system?" --simple
  py ask_with_citations.py "What are the main features?" --k 3 --no-metadata
        """
    )
    
    parser.add_argument(
        "question",
        help="The question to ask about your documents"
    )
    
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple citation format (less detailed)"
    )
    
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Don't show file metadata in citations"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of document chunks to retrieve (default: 5)"
    )
    
    parser.add_argument(
        "--model",
        default="llama3",
        help="LLM model to use (default: llama3)"
    )
    
    parser.add_argument(
        "--chroma-path",
        default="data/chroma",
        help="Path to Chroma database (default: data/chroma)"
    )
    
    parser.add_argument(
        "--db-path",
        default="data/db.sqlite",
        help="Path to SQLite database (default: data/db.sqlite)"
    )
    
    args = parser.parse_args()
    
    print("🤖 Ask Your Files - Enhanced Citations")
    print("=" * 60)
    print(f"Question: {args.question}")
    print("=" * 60)
    
    # Check if databases exist
    chroma_path = Path(args.chroma_path)
    db_path = Path(args.db_path)
    
    if not chroma_path.exists():
        print(f"❌ Chroma database not found at: {chroma_path}")
        print("💡 Run document processing first: py example_usage.py")
        return
    
    if not db_path.exists():
        print(f"❌ SQLite database not found at: {db_path}")
        print("💡 Run document processing first: py example_usage.py")
        return
    
    print("🔍 Searching documents...")
    
    try:
        if args.simple:
            # Use simple function
            answer = ask_your_files_simple_with_citations(
                question=args.question,
                persist_directory=args.chroma_path,
                db_path=args.db_path,
                llm_model=args.model,
                k=args.k,
                show_detailed_citations=False
            )
        else:
            # Use detailed function
            embedding_manager = EmbeddingManager()
            vector_store = ChromaVectorStore(args.chroma_path)
            db_manager = DatabaseManager(args.db_path)
            llm_client = LLMClient(model=args.model)
            
            answer = ask_your_files_with_citations(
                question=args.question,
                embedding_manager=embedding_manager,
                vector_store=vector_store,
                db_manager=db_manager,
                llm_client=llm_client,
                k=args.k,
                show_detailed_citations=True,
                show_metadata=not args.no_metadata
            )
        
        print("✅ Answer generated successfully!")
        print("\n" + "🤖 ANSWER:")
        print("-" * 60)
        print(answer)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        
        if "Ollama" in str(e):
            print("\n💡 Ollama Setup Required:")
            print("1. Install Ollama: https://ollama.ai")
            print("2. Start Ollama: ollama serve")
            print("3. Pull model: ollama pull llama3")
        elif "sentence-transformers" in str(e):
            print("\n💡 Install dependencies: py -m pip install sentence-transformers chromadb")
        else:
            print(f"\n💡 Make sure you have processed documents first:")
            print("   py example_usage.py")

def interactive_mode():
    """Interactive question-answering mode with citations."""
    
    print("🤖 Interactive Ask Your Files with Citations")
    print("=" * 60)
    print("Type your questions and get answers with source citations!")
    print("Commands: 'quit' to exit, 'help' for help")
    print("=" * 60)
    
    # Check if databases exist
    chroma_path = Path("data/chroma")
    db_path = Path("data/db.sqlite")
    
    if not chroma_path.exists() or not db_path.exists():
        print("❌ Databases not found. Please process documents first:")
        print("   py example_usage.py")
        return
    
    # Initialize components once
    try:
        print("🔧 Initializing components...")
        embedding_manager = EmbeddingManager()
        vector_store = ChromaVectorStore("data/chroma")
        db_manager = DatabaseManager("data/db.sqlite")
        llm_client = LLMClient(model="llama3")
        print("✅ Ready!")
        
        # Show available documents
        stats = db_manager.get_database_stats()
        docs = db_manager.get_all_documents()
        print(f"\n📚 Available: {stats['document_count']} documents, {stats['chunk_count']} chunks")
        if docs:
            print("📄 Documents:")
            for doc in docs[:5]:
                print(f"   • {doc['file_name']} ({doc['file_type']})")
            if len(docs) > 5:
                print(f"   ... and {len(docs) - 5} more")
        
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        return
    
    print("\n" + "="*60)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            elif question.lower() == 'help':
                print("\n💡 Help:")
                print("• Ask any question about your documents")
                print("• Examples:")
                print("  - 'What is this project about?'")
                print("  - 'How do I install the dependencies?'")
                print("  - 'What are the main features?'")
                print("• Type 'quit' to exit")
                continue
            elif not question:
                continue
            
            print("🤔 Thinking...")
            
            answer = ask_your_files_with_citations(
                question=question,
                embedding_manager=embedding_manager,
                vector_store=vector_store,
                db_manager=db_manager,
                llm_client=llm_client,
                k=5,
                show_detailed_citations=True,
                show_metadata=True
            )
            
            print("\n" + "🤖 ANSWER:")
            print("-" * 40)
            print(answer)
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, run interactive mode
        interactive_mode()
    else:
        # Arguments provided, run command-line mode
        main()
