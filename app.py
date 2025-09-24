# app.py
import streamlit as st
from pathlib import Path
from core.llm import LLMClient, LLMError
from core.rag import (
    ask_your_files_with_citations,
    EmbeddingManager,
    ChromaVectorStore,
    DatabaseManager,
    ingest_document,
    process_documents_to_vector_store,
    clear_all_chunks_and_documents
)
import tempfile
import time

def process_uploaded_files(uploaded_files, embedding_manager, vector_store, db_manager, chunk_size, chunk_overlap):
    """Process uploaded files with status updates."""
    
    # Create progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.container()
    
    total_files = len(uploaded_files)
    processed_docs = []
    failed_files = []
    
    # Process each file
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i) / total_files
            progress_bar.progress(progress)
            status_text.text(f"📄 Processing {uploaded_file.name}... ({i+1}/{total_files})")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                # Ingest the document
                doc_data = ingest_document(tmp_file_path)
                
                if doc_data:
                    # Update the file name to original name
                    doc_data['file_name'] = uploaded_file.name
                    doc_data['file_path'] = uploaded_file.name  # Use original name as path
                    processed_docs.append(doc_data)
                    
                    # Show individual file progress
                    with results_container:
                        st.success(f"✅ **{uploaded_file.name}** - {doc_data['word_count']} words, {doc_data['content_length']} characters")
                else:
                    failed_files.append((uploaded_file.name, "Failed to extract content"))
                    with results_container:
                        st.error(f"❌ **{uploaded_file.name}** - Failed to extract content")
                        
            finally:
                # Clean up temporary file
                Path(tmp_file_path).unlink(missing_ok=True)
                
        except Exception as e:
            failed_files.append((uploaded_file.name, str(e)))
            with results_container:
                st.error(f"❌ **{uploaded_file.name}** - Error: {e}")
    
    if processed_docs:
        # Update progress for vector processing
        progress_bar.progress(0.7)
        status_text.text(f"🧠 Generating embeddings and storing in vector database...")
        
        try:
            # Process documents through the pipeline
            results = process_documents_to_vector_store(
                documents=processed_docs,
                vector_store=vector_store,
                embedding_manager=embedding_manager,
                db_manager=db_manager,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            # Show final results
            st.success("🎉 **Upload and Processing Complete!**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Files Processed", results['processed_documents'])
            with col2:
                st.metric("Chunks Created", results['total_chunks'])
            with col3:
                st.metric("Avg Chunk Size", f"{results.get('average_chunk_size', 0):.0f} chars")
            
            # Show updated system stats
            if 'database_stats' in results:
                db_stats = results['database_stats']
                st.info(f"📊 **Updated System Status:**\n"
                       f"• Total documents: {db_stats['document_count']}\n"
                       f"• Total chunks: {db_stats['chunk_count']}\n"
                       f"• Database size: {db_stats['database_size_mb']} MB")
            
            # Show processing details
            with st.expander("📋 Processing Details", expanded=False):
                st.write("**Successfully processed files:**")
                for doc in processed_docs:
                    st.write(f"• {doc['file_name']}: {doc['word_count']} words → chunks created")
                
                if failed_files:
                    st.write("**Failed files:**")
                    for filename, error in failed_files:
                        st.write(f"• {filename}: {error}")
            
            # Suggest next steps
            st.info("💡 **Next Steps:**\n"
                   "• Switch to 'Ask Your Files' mode to query your documents\n"
                   "• Your uploaded documents are now searchable!")
            
        except Exception as e:
            progress_bar.progress(1.0)
            status_text.text("❌ Processing failed!")
            st.error(f"Error during vector processing: {e}")
            
    else:
        progress_bar.progress(1.0)
        status_text.text("❌ No files were successfully processed")
        st.error("No files were successfully processed. Please check file formats and try again.")
    
    # Clear progress after a delay
    time.sleep(2)
    progress_bar.empty()
    status_text.empty()

def display_source(source_info):
    """Display a single source with enhanced formatting."""
    with st.container():
        # Source header with number and name
        col1, col2 = st.columns([3, 1])
        
        with col1:
            source_title = f"**[{source_info['number']}] 📄 {source_info['name']}**"
            if 'section' in source_info:
                source_title += f" *(section {source_info['section']})*"
            st.markdown(source_title)
        
        with col2:
            # Extract relevance score if available
            relevance_line = next((line for line in source_info.get('details', []) if 'Relevance:' in line), None)
            if relevance_line:
                relevance = relevance_line.split('Relevance: ')[1].split('%')[0] + '%'
                st.metric("Relevance", relevance)
        
        # Display details in a nice format
        if source_info.get('details'):
            details_col1, details_col2 = st.columns(2)
            
            for i, detail in enumerate(source_info['details']):
                if 'Relevance:' not in detail:  # Skip relevance as it's shown as metric
                    with details_col1 if i % 2 == 0 else details_col2:
                        if '📂 Path:' in detail:
                            path = detail.replace('📂 Path: ', '').strip()
                            st.caption(f"📂 **Path:** `{path}`")
                        elif '📋 Type:' in detail:
                            file_type = detail.replace('📋 Type: ', '').strip()
                            st.caption(f"📋 **Type:** {file_type}")
                        elif '📅 Added:' in detail:
                            added = detail.replace('📅 Added: ', '').strip()
                            st.caption(f"📅 **Added:** {added}")
        
        # Add a subtle separator
        st.markdown('<hr style="margin: 10px 0; border: 1px solid #e0e0e0;">', unsafe_allow_html=True)

# Main Streamlit App
st.set_page_config(page_title="Local Assistant", page_icon="🤖", layout="wide")

# Sidebar for mode selection
st.sidebar.title("🤖 Local Assistant")
mode = st.sidebar.radio(
    "Choose Mode:",
    ["💬 Chat with LLM", "📚 Ask Your Files", "📁 Upload Documents"],
    index=1  # Default to Ask Your Files
)

if mode == "💬 Chat with LLM":
    st.title("💬 Chat with Local LLM")
    st.write("Direct conversation with your local language model")
    
    prompt = st.text_input("Prompt:", "Say exactly: Hello World")
    go = st.button("Send", key="chat_send")

    if go:
        try:
            llm = LLMClient(model="llama3")
            with st.spinner("Thinking locally..."):
                out = llm.chat(prompt, stream=False)
            st.success("Done!")
            st.text(out)
        except LLMError as e:
            st.error(str(e))

elif mode == "📚 Ask Your Files":
    st.title("📚 Ask Your Files")
    st.write("Ask questions about your documents with source citations")
    
    # Check if databases exist
    chroma_path = Path("data/chroma")
    db_path = Path("data/db.sqlite")
    
    if not chroma_path.exists() or not db_path.exists():
        st.error("📁 No documents found!")
        st.write("Please process your documents first:")
        st.code("py example_usage.py", language="bash")
        st.write("Or use the 'Upload Documents' tab to add files directly!")
        st.stop()
    
    # Initialize components (with caching)
    @st.cache_resource
    def init_components():
        try:
            embedding_manager = EmbeddingManager()
            vector_store = ChromaVectorStore("data/chroma")
            db_manager = DatabaseManager("data/db.sqlite")
            llm_client = LLMClient(model="llama3")
            return embedding_manager, vector_store, db_manager, llm_client
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            return None, None, None, None
    
    embedding_manager, vector_store, db_manager, llm_client = init_components()
    
    if embedding_manager is None:
        st.error("Failed to initialize system components")
        st.stop()
    
    # Show document statistics
    try:
        stats = db_manager.get_database_stats()
        docs = db_manager.get_all_documents()
        
        # Main statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Documents", stats['document_count'])
        with col2:
            st.metric("Chunks", stats['chunk_count'])
        with col3:
            st.metric("Database Size", f"{stats['database_size_mb']} MB")
        
        # Comprehensive file list
        st.subheader("📚 Available Documents")
        
        if not docs:
            st.info("No documents available. Upload some files using the 'Upload Documents' tab!")
        else:
            # File list controls
            col1, col2 = st.columns([3, 1])
            with col1:
                search_term = st.text_input("🔍 Search documents:", placeholder="Filter by filename...")
            with col2:
                show_details = st.checkbox("Show details", value=False)
            
            # Filter documents
            if search_term:
                filtered_docs = [doc for doc in docs if search_term.lower() in doc['file_name'].lower()]
            else:
                filtered_docs = docs
            
            st.write(f"Showing {len(filtered_docs)} of {len(docs)} documents:")
            
            # Display documents in a nice format
            for i, doc in enumerate(filtered_docs):
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        # File icon based on type
                        icon = {"pdf": "📄", "docx": "📝", ".pdf": "📄", ".docx": "📝", ".md": "📋", ".markdown": "📋"}.get(doc['file_type'], "📄")
                        st.write(f"{icon} **{doc['file_name']}**")
                        
                        if show_details:
                            # Get chunks info
                            chunks = db_manager.get_chunks_by_document_id(doc['id'])
                            st.caption(f"📂 Path: {doc.get('file_path', 'N/A')}")
                            st.caption(f"📊 {doc['word_count']} words, {len(chunks)} chunks")
                            st.caption(f"📅 Added: {doc['created_at'][:19] if doc.get('created_at') else 'Unknown'}")
                    
                    with col2:
                        st.metric("Words", f"{doc['word_count']:,}")
                    
                    with col3:
                        # Get chunk count
                        chunks = db_manager.get_chunks_by_document_id(doc['id'])
                        st.metric("Chunks", len(chunks))
                    
                    with col4:
                        st.write(f"**{doc['file_type']}**")
                        
                        # File actions (in a subtle way)
                        if show_details:
                            with st.popover("⚙️", help="File actions"):
                                st.write(f"**Actions for {doc['file_name']}**")
                                
                                if st.button("🗑️ Remove from database", key=f"delete_{doc['id']}", help="Remove this document and its chunks"):
                                    try:
                                        # Delete from database and vector store
                                        success = db_manager.delete_document(doc['id'], vector_store)
                                        if success:
                                            st.success(f"Removed {doc['file_name']} from database and vector store")
                                            st.rerun()
                                        else:
                                            st.error("Failed to remove document")
                                    except Exception as e:
                                        st.error(f"Error removing document: {e}")
                                
                                st.caption("⚠️ This will remove the document from the vector store and database.")
                
                # Add separator
                if i < len(filtered_docs) - 1:
                    st.divider()
            
            # Bulk actions
            if filtered_docs and show_details:
                st.markdown("---")
                st.write("**Bulk Actions:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Export Document List", help="Export list of documents as text"):
                        export_text = "Document List Export\n" + "="*50 + "\n\n"
                        for doc in filtered_docs:
                            chunks = db_manager.get_chunks_by_document_id(doc['id'])
                            export_text += f"• {doc['file_name']} ({doc['file_type']})\n"
                            export_text += f"  Words: {doc['word_count']:,}, Chunks: {len(chunks)}\n"
                            export_text += f"  Added: {doc['created_at'][:19] if doc.get('created_at') else 'Unknown'}\n\n"
                        
                        st.download_button(
                            "💾 Download List",
                            export_text,
                            file_name="document_list.txt",
                            mime="text/plain"
                        )
                
                with col2:
                    if st.button("🔄 Refresh Database Stats", help="Recalculate database statistics"):
                        st.rerun()
                
                with col3:
                    total_docs = len(docs)
                    if total_docs > 0:
                        st.metric("Total Storage", f"{stats['database_size_mb']} MB")
                        
                        # Initialize session state for clear confirmation
                        if 'show_clear_confirmation' not in st.session_state:
                            st.session_state.show_clear_confirmation = False
                        
                        # Clear All Data button
                        if not st.session_state.show_clear_confirmation:
                            if st.button("🗑️ Clear All Data", 
                                       help="Remove ALL documents from database and vector store", 
                                       type="secondary"):
                                st.session_state.show_clear_confirmation = True
                                st.rerun()
                        
                        # Show confirmation dialog
                        if st.session_state.show_clear_confirmation:
                            st.warning("⚠️ **DANGER: This will permanently delete ALL your documents!**")
                            st.write("This action will:")
                            st.write("• Remove all documents from the database")
                            st.write("• Clear the entire vector store") 
                            st.write("• Cannot be undone!")
                            
                            col_confirm1, col_confirm2 = st.columns(2)
                            with col_confirm1:
                                if st.button("❌ Cancel", key="cancel_clear"):
                                    st.session_state.show_clear_confirmation = False
                                    st.rerun()
                            
                            with col_confirm2:
                                if st.button("🗑️ Yes, Delete Everything", 
                                           key="confirm_clear", 
                                           type="primary"):
                                    try:
                                        with st.spinner("🗑️ Clearing all data..."):
                                            # Use the comprehensive clear function
                                            result = clear_all_chunks_and_documents(
                                                vector_store=vector_store,
                                                db_manager=db_manager
                                            )
                                            
                                            # Reset confirmation state
                                            st.session_state.show_clear_confirmation = False
                                            
                                            if result['success']:
                                                st.success("✅ **All data cleared successfully!**")
                                                
                                                # Show detailed results
                                                if result['stats_before']:
                                                    stats_before = result['stats_before']
                                                    cleared_docs = stats_before.get('document_count', 0)
                                                    cleared_chunks = stats_before.get('chunk_count', 0)
                                                    cleared_vector = stats_before.get('vector_store_count', 0)
                                                    
                                                    st.info(f"📊 **Cleanup Summary:**\n"
                                                           f"• Removed {cleared_docs} documents\n"
                                                           f"• Removed {cleared_chunks} chunks from database\n"
                                                           f"• Removed {cleared_vector} vectors from store")
                                                
                                                st.info("💡 You can now upload new documents using the 'Upload Documents' tab.")
                                                time.sleep(2)  # Brief pause to show success message
                                                st.rerun()
                                            else:
                                                st.error("❌ **Failed to clear all data**")
                                                
                                                # Show specific errors
                                                if result['errors']:
                                                    st.error("**Errors encountered:**")
                                                    for error in result['errors']:
                                                        st.error(f"• {error}")
                                                
                                                # Show partial success
                                                if result['database_cleared'] and not result['vector_store_cleared']:
                                                    st.warning("⚠️ Database cleared but vector store failed")
                                                elif result['vector_store_cleared'] and not result['database_cleared']:
                                                    st.warning("⚠️ Vector store cleared but database failed")
                                                    
                                    except Exception as e:
                                        st.session_state.show_clear_confirmation = False
                                        st.error(f"❌ Error clearing data: {e}")
                    
    except Exception as e:
        st.error(f"Error loading document stats: {e}")
        st.stop()
    
    # Configuration options
    with st.sidebar:
        st.subheader("⚙️ Settings")
        k = st.slider("Number of chunks to retrieve", 1, 10, 5)
        show_metadata = st.checkbox("Show file metadata", True)
        show_detailed = st.checkbox("Show detailed citations", True)
    
    # Enhanced Question Interface
    st.subheader("💬 Ask Your Documents")
    
    # Create two columns for the question interface
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Main question input
        question = st.text_input(
            "Enter your question:",
            placeholder="What would you like to know about your documents?",
            help="Ask specific questions about the content in your uploaded documents"
        )
    
    with col2:
        # Submit button
        st.write("")  # Add spacing
        ask_button = st.button("🔍 Submit", type="primary", use_container_width=True)
    
    
    
    if ask_button and question.strip():
        with st.spinner("🤔 Searching documents and generating answer..."):
            try:
                # Get answer with citations
                answer = ask_your_files_with_citations(
                    question=question,
                    embedding_manager=embedding_manager,
                    vector_store=vector_store,
                    db_manager=db_manager,
                    llm_client=llm_client,
                    k=k,
                    show_detailed_citations=show_detailed,
                    show_metadata=show_metadata
                )
                
                st.success("✅ Answer generated successfully!")
                
                # Enhanced Result Display with Clear Separation
                st.markdown("---")
                
                # Split answer from citations
                if "📚 SOURCES:" in answer:
                    answer_part, sources_part = answer.split("📚 SOURCES:", 1)
                    
                    # Display the answer in a prominent container
                    st.subheader("🤖 Answer")
                    with st.container():
                        st.markdown(f"""
                        <div style=padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                            {answer_part.strip()}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Clear visual separation
                    st.markdown("---")
                    
                    # Display sources with enhanced formatting
                    st.subheader("📚 Sources & Citations")
                    
                    # Parse and display sources in a structured way
                    sources_lines = sources_part.strip().split('\n')
                    current_source = {}
                    
                    for line in sources_lines:
                        line = line.strip()
                        if not line or line == '=' * 50:
                            continue
                            
                        if line.startswith('[') and ']' in line:
                            # New source entry
                            if current_source:
                                display_source(current_source)
                            
                            # Parse source header
                            source_num = line.split(']')[0][1:]
                            source_name = line.split('📄 ')[1].split(' (')[0] if '📄 ' in line else line.split('] ')[1]
                            
                            current_source = {
                                'number': source_num,
                                'name': source_name,
                                'details': []
                            }
                            
                            # Add section info if present
                            if ' (section ' in line:
                                section_info = line.split(' (section ')[1].split(')')[0]
                                current_source['section'] = section_info
                        
                        elif line.startswith('📊 Relevance:') or line.startswith('📂 Path:') or line.startswith('📋 Type:') or line.startswith('📅 Added:'):
                            if current_source:
                                current_source['details'].append(line)
                        
                        elif line.startswith('-' * 30):
                            # End of current source
                            if current_source:
                                display_source(current_source)
                                current_source = {}
                    
                    # Display last source if exists
                    if current_source:
                        display_source(current_source)
                        
                else:
                    # No sources section, display answer only
                    st.subheader("🤖 Answer")
                    with st.container():
                        st.markdown(f"""
                        <div style=padding: 20px; border-radius: 10px; border-left: 4px solid #1f77b4;">
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.info("ℹ️ No specific sources were cited for this response.")
                
            except Exception as e:
                st.error(f"❌ Error: {e}")
                
                if "Ollama" in str(e):
                    st.info("💡 **Ollama Setup Required:**")
                    st.code("""
# Install Ollama from https://ollama.ai
ollama serve
ollama pull llama3
                    """, language="bash")
                elif "sentence-transformers" in str(e):
                    st.info("💡 **Missing Dependencies:**")
                    st.code("py -m pip install sentence-transformers chromadb", language="bash")
    
    elif ask_button:
        st.warning("Please enter a question first!")
    
    # Tips section
    with st.sidebar:
        st.subheader("💡 Tips")
        st.write("""
        **For better results:**
        - Be specific in your questions
        - Ask about topics likely in your documents
        - Try different phrasings if needed
        - Use factual questions for best results
        """)
        
        st.subheader("🔧 System Info")
        try:
            vector_info = vector_store.get_collection_info()
            st.write(f"Vector store: {vector_info['count']} documents")
            st.write(f"Database: {stats['document_count']} documents")
        except:
            st.write("System status: Unknown")

elif mode == "📁 Upload Documents":
    st.title("📁 Upload Documents")
    st.write("Upload and process documents into the vector store")
    
    # Initialize components for ingestion
    @st.cache_resource
    def init_ingestion_components():
        try:
            embedding_manager = EmbeddingManager()
            vector_store = ChromaVectorStore("data/chroma")
            db_manager = DatabaseManager("data/db.sqlite")
            return embedding_manager, vector_store, db_manager
        except Exception as e:
            st.error(f"Error initializing components: {e}")
            return None, None, None
    
    embedding_manager, vector_store, db_manager = init_ingestion_components()
    
    if embedding_manager is None:
        st.error("Failed to initialize system components")
        st.stop()
    
    # Current system status
    try:
        stats = db_manager.get_database_stats()
        vector_info = vector_store.get_collection_info()
        
        st.subheader("📊 Current System Status")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Documents", stats['document_count'])
        with col2:
            st.metric("Chunks", stats['chunk_count'])
        with col3:
            st.metric("Vector Store", vector_info['count'])
        with col4:
            st.metric("DB Size", f"{stats['database_size_mb']} MB")
            
    except Exception as e:
        st.error(f"Error loading system status: {e}")
    
    # Upload configuration
    st.subheader("⚙️ Upload Settings")
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider("Chunk Size (characters)", 100, 1000, 500)
        chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 200, 50)
    with col2:
        st.info("💡 **Chunk Settings:**\n\n"
                "• **Chunk Size**: Larger chunks preserve context but may be less precise\n"
                "• **Overlap**: Helps maintain context across chunk boundaries")
    
    # File upload section
    st.subheader("📎 Select Files to Upload")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['pdf', 'docx', 'md', 'markdown'],
        help="Supported formats: PDF, DOCX, Markdown (.md)"
    )
    
    if uploaded_files:
        st.write(f"📁 Selected {len(uploaded_files)} file(s):")
        
        # Preview uploaded files
        for i, file in enumerate(uploaded_files, 1):
            file_size_mb = len(file.getvalue()) / (1024 * 1024)
            st.write(f"{i}. **{file.name}** ({file.type}) - {file_size_mb:.2f} MB")
        
        # Process files button
        process_button = st.button(
            f"🚀 Process {len(uploaded_files)} File(s)",
            type="primary",
            use_container_width=True
        )
        
        if process_button:
            process_uploaded_files(
                uploaded_files, 
                embedding_manager, 
                vector_store, 
                db_manager,
                chunk_size,
                chunk_overlap
            )
    else:
        st.info("👆 Upload files using the file picker above")
        
        # Show supported formats
        st.subheader("📋 Supported File Formats")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**📄 PDF Files**")
            st.write("• Research papers")
            st.write("• Reports")
            st.write("• Documentation")
        with col2:
            st.write("**📝 Word Documents**")
            st.write("• .docx files")
            st.write("• Meeting notes")
            st.write("• Proposals")
        with col3:
            st.write("**📋 Markdown Files**")
            st.write("• .md files")
            st.write("• README files")
            st.write("• Technical docs")