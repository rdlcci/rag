'''
'''

import streamlit as st
import pandas as pd
import numpy as np
# import json
import base64
# from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="RAG Builder",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 15px;
    }
    .stButton>button {
        width: 100%;
    }
    .header-container {
        display: flex;
        align-items: center;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for displaying info
def info_tooltip(text):
    return st.info(text)

# Title and description
st.title("RAG Builder")
st.markdown("Configure, build, and evaluate your Retrieval-Augmented Generation pipeline.")

# Sidebar for navigation
st.sidebar.title("Steps")
section = st.sidebar.radio(
    "Go to",
    ["1. Upload Data", "2. Ground truth", "3. Embedding", "4. Parsing", "5. Chunking", "6. Metadata", "6. Vector Database", 
     "7. Retrieval", "8. Search Setup", "9. LLM Parameters", "10. Evaluation Metrics", "11. Run Experiment", "12. History"]
)

# Sample data to show as example
sample_data = {
    "question": ["What is RAG?", "How does vector search work?", "What is the best chunking method?"],
    "ground_truth": ["RAG stands for Retrieval-Augmented Generation, a technique that enhances LLMs with external knowledge.", 
                    "Vector search works by converting documents into numerical vectors and finding the most similar vectors.",
                    "The best chunking method depends on your use case, but semantic chunking often performs well."]
}

# 0. data
if section == "1. Upload Data":
        # New: Document Upload Section for RAG
        st.markdown("---")
        st.header("1. Upload Source Documents for RAG")

        st.markdown(
            "Upload one or more files (PDF, Word, TXT, JSON, Excel, CSV) to use as the knowledge base for your RAG pipeline. "
            "These documents will be parsed, chunked, and indexed for retrieval."
        )

        uploaded_docs = st.file_uploader(
            "Select files to upload",
            type=["pdf", "docx", "txt", "json", "xlsx", "csv"],
            accept_multiple_files=True,
            key="rag_doc_upload"
        )

        if uploaded_docs:
            st.success(f"{len(uploaded_docs)} file(s) uploaded.")
            for doc in uploaded_docs:
                st.write(f"- {doc.name} ({doc.size // 1024} KB)")
        else:
            st.info("No files uploaded yet.")

        with st.expander("ℹ️ Upload Instructions"):
            st.markdown("""
            - You can upload multiple files at once.
            - Supported file types: **.pdf, .docx, .txt, .json, .xlsx, .csv**
            - Uploaded documents will be used as the source for retrieval and generation.
            - Large files may take longer to process.
            """)


# 1. Dataset Section
elif section == "2. Ground truth":
    st.header("2. Upload Ground truth Dataset for evaluating the performance")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Upload your QA pairs CSV")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("### CSV Format Requirements")
        st.markdown("Your CSV should have at least these columns:")
        st.code("question, ground_truth")
        
    with col2:
        with st.expander("Sample Data Format", expanded=True):
            st.dataframe(pd.DataFrame(sample_data))
            
            # Download sample csv
            sample_df = pd.DataFrame(sample_data)
            csv = sample_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="sample_qa_pairs.csv">Download sample CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with st.expander("ℹ️ About Dataset Format"):
        info_tooltip("""
        This f=dataset will act as a compass, to see if the RAG setup is able to find the right data from documents            
        The dataset should contain question-answer pairs where:
        - 'question' column: Contains the queries you want to test
        - 'ground_truth' column: Contains the expected answer
        
        Additional columns that can be useful:
        - 'document_id': If questions relate to specific documents
        - 'category': To group questions by topic
        - 'difficulty': To evaluate performance across different difficulty levels
        """)

# 2. Embedding Model Section
elif section == "3. Embedding":
    st.header("3. Choose Embedding Model, to convert your data into numbers that makes sense semantically")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        embedding_model = st.selectbox(
            "Select Embedding Model",
            ["PubMed BERT", "text-embedding-ada-003", "text-embedding-3-large-v2"]
        )
        
        # Display model properties based on selection
        if embedding_model == "PubMed BERT":
            model_dims = 768
            model_context = "Specialized for biomedical text"
        elif embedding_model == "text-embedding-ada-003":
            model_dims = 1536
            model_context = "OpenAI's Ada model, good general purpose"
        else:  # large-2
            model_dims = 3072
            model_context = "OpenAI's latest large embedding model"
            
        st.write(f"Default dimensions: {model_dims}")
        st.write(f"Specialized context: {model_context}")
        
        # Allow custom dimensions
        custom_dims = st.checkbox("Use custom dimensions")
        if custom_dims:
            embedding_dims = st.number_input("Embedding dimensions", min_value=128, max_value=4096, value=model_dims)
        else:
            embedding_dims = model_dims
            
    with col2:
        with st.expander("ℹ️ About Embedding Models", expanded=True):
            info_tooltip("""
            **PubMed BERT**: Optimized for medical and scientific text. Better performance for healthcare, pharma, and research use cases.
            
            **text-embedding-ada-003**: OpenAI's general-purpose embedding model. Good balance of quality and performance.
            
            **text-embedding-3-large-v2**: OpenAI's latest large embedding model with improved semantic understanding.
            
            **Custom dimensions**: Reducing dimensions can speed up retrieval but may reduce accuracy. Useful for:
            - Faster search in large document collections
            - Reducing storage requirements
            - Testing dimensionality impact on retrieval quality
            """)

# 3. Parser Section
elif section == "4. Parsing":
    st.header("4. Document Parsing to extarct text, images and other content from your document")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        parser_option = st.radio(
            "Select Parser",
            ["PyMuPDF + pdfplumber", "Unstructured.io"]
        )
        
        st.markdown("### Parser Configuration")
        
        if parser_option == "PyMuPDF + pdfplumber":
            table_format = st.selectbox("Table format", ["DataFrame", "JSON", "Markdown"])
            summarize_tables = st.checkbox("Summarize tables")
            extract_images = st.checkbox("Extract images", value=True)
            
        elif parser_option == "Unstructured.io":
            strategy = st.selectbox("Extraction strategy", ["Auto", "Fast", "Ocr", "Hi-res"])
            include_metadata = st.checkbox("Include element metadata", value=True)
            
    with col2:
        with st.expander("ℹ️ About Document Parsing", expanded=True):
            info_tooltip("""
            **PyMuPDF + pdfplumber**: Combines PyMuPDF for text and image extraction with pdfplumber for table extraction.
            - Good for documents with well-defined structure
            - Fast processing with good table recognition
            - Options for different table formats
            
            **Unstructured.io**: More sophisticated parsing that identifies document elements.
            - Better handling of complex layouts
            - Maintains hierarchical structure
            - More comprehensive metadata extraction
            
            **Table formats**:
            - DataFrame: Native pandas format, good for data analysis
            - JSON: Structured format for API responses and web applications
            - Markdown: Human-readable format suitable for documentation
            
            **Summarizing tables**: Uses LLM to create a concise description of table contents
            """)

# 4. Chunking Section
elif section == "5. Chunking":
    st.header("5. Document Chunking, to break your content into peices and arrange them to make search better")
    
    chunker_option = st.selectbox(
        "Select Chunking Strategy",
        ["Recursive", "Semantic", "SDPM", "Late", "Slumber", "Agentic", "Summary-based", "RAPTOR", "Contextualized"]
    )
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Chunker Configuration")
        
        if chunker_option == "Recursive":
            chunk_size = st.slider("Chunk size (characters)", 100, 2000, 512)
            overlap = st.slider("Overlap (characters)", 0, 200, 50)
            
        elif chunker_option == "Semantic":
            similarity_threshold = st.slider("Similarity threshold", 0.5, 0.99, 0.75, 0.01)
            min_chunk_size = st.number_input("Minimum chunk size", 20, 200, 50)
            max_chunk_size = st.number_input("Maximum chunk size", 500, 3000, 1000)
            
        elif chunker_option == "SDPM":
            chunk_size = st.slider("Initial chunk size", 100, 2000, 512)
            threshold = st.slider("Merge threshold", 0.5, 0.99, 0.75, 0.01)
            
        elif chunker_option == "Late":
            chunk_size = st.number_input("Token chunk size", 128, 2048, 512)
            
        elif chunker_option == "Slumber":
            initial_size = st.number_input("Initial window size", 500, 5000, 2000)
            overlap = st.slider("Window overlap", 50, 500, 200)
            
        elif chunker_option == "Agentic":
            max_chunks = st.number_input("Maximum chunks", 2, 20, 10)
            min_chunk_size = st.number_input("Minimum chunk size", 50, 500, 100)
            
        elif chunker_option == "Summary-based":
            chunk_size = st.slider("Base chunk size", 500, 5000, 2000)
            overlap = st.slider("Chunk overlap", 50, 500, 200)
            
        elif chunker_option == "RAPTOR":
            max_tokens = st.number_input("Max tokens per chunk", 128, 2048, 512)
            num_layers = st.number_input("Hierarchy levels", 1, 5, 2)
            top_k = st.number_input("Similar nodes for grouping", 1, 10, 3)
            
        elif chunker_option == "Contextualized":
            chunk_size = st.slider("Base chunk size", 500, 5000, 2000)
            overlap = st.slider("Chunk overlap", 50, 500, 200)
    
    with col2:
        with st.expander("ℹ️ About Chunking Strategies", expanded=True):
            info_tooltip("""
            **Recursive**: Splits text at natural boundaries (paragraphs, sentences) while respecting max size.
            - Simple and efficient
            - Preserves natural text boundaries
            
            **Semantic**: Groups similar sentences based on embedding similarity.
            - Creates more coherent chunks
            - Better for question-answering
            
            **SDPM**: Two-pass approach that merges similar adjacent chunks.
            - Balances size constraints with semantic coherence
            - Good for documents with varying section lengths
            
            **Late**: Fixed-size chunking preserving token-level embeddings.
            - Better for retrieval with attention to specific tokens
            - Good for technical documents with key terms
            
            **Slumber**: Uses LLM to find optimal split points between chunks.
            - Creates more meaningful chunk boundaries
            - Better for narrative text
            
            **Agentic**: LLM analyzes document structure to create logical chunks.
            - Adapts to document's inherent organization
            - Good for complex documents with varied sections
            
            **Summary-based**: Adds summaries to each chunk for enhanced context.
            - Improves retrieval with additional context
            - Good for long or complex documents
            
            **RAPTOR**: Hierarchical chunking with topic modeling and semantic relationships.
            - Creates topic-based structure
            - Tracks relationships between chunks
            - Best for large, diverse document collections
            
            **Contextualized**: Adds document-aware context to each chunk.
            - Enriches chunks with their document context
            - Good for documents where global context matters
            """)

# 5. Metadata Section
elif section == "6. Metadata":
    st.header("6. Document Metadata, to add more context to the broken pieces as they might not have all the context")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Select metadata to extract and include")
        
        metadata_options = {
            "Basic": ["filename", "file_type", "file_size", "creation_date", "last_modified"],
            "Content": ["title", "author", "page_count", "word_count", "language"],
            "Position": ["page_number", "start_char", "end_char", "section_number"],
            "Semantic": ["topics", "entities", "keywords", "summary", "category"]
        }
        
        selected_metadata = {}
        
        for category, options in metadata_options.items():
            st.subheader(category)
            selected = []
            for option in options:
                if st.checkbox(option, key=f"meta_{option}"):
                    selected.append(option)
            if selected:
                selected_metadata[category] = selected
        
        # Custom metadata fields
        st.subheader("Custom metadata")
        custom_field = st.text_input("Add custom field")
        if st.button("Add field") and custom_field:
            if "Custom" not in selected_metadata:
                selected_metadata["Custom"] = []
            selected_metadata["Custom"].append(custom_field)
            st.success(f"Added {custom_field}")
        
        if "Custom" in selected_metadata and selected_metadata["Custom"]:
            st.write("Custom fields:", ", ".join(selected_metadata["Custom"]))
    
    with col2:
        with st.expander("ℹ️ About Metadata", expanded=True):
            info_tooltip("""
            Metadata enriches your chunks with additional information that can improve retrieval and generation.
            
            **Basic metadata**: File properties and system information
            - Useful for filtering by source or recency
            
            **Content metadata**: Document-level properties
            - Helps understand document context and origin
            
            **Position metadata**: Location within document
            - Useful for ordering chunks and finding nearby information
            
            **Semantic metadata**: Content-based information
            - Improves retrieval by adding semantic signals
            - Can be used for filtering by topic or entity
            
            **Custom metadata**: Add domain-specific fields
            - Tailor metadata to your specific use case
            """)

# 6. Vector Database Section
elif section == "6. Vector Database":
    st.header("6. Vector Database, to store the vectorized form of your content and search better")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        vector_db = st.selectbox(
            "Select Vector Database",
            ["Milvus", "Qdrant", "Elasticsearch", "Weaviate"]
        )
        
        st.markdown("### Indexing Configuration")
        
        index_type = st.selectbox(
            "Indexing Type",
            ["HNSW", "PQ (Product Quantization)", "IVF (Inverted File Index)", "FLAT"]
        )
        
        if index_type == "HNSW":
            M = st.slider("M (max connections)", 4, 64, 16)
            ef_construction = st.slider("ef_construction", 50, 500, 200)
            ef_search = st.slider("ef_search", 50, 500, 100)
            
        elif index_type == "PQ (Product Quantization)":
            num_subvectors = st.slider("Number of subvectors", 4, 64, 16)
            bits_per_subvector = st.selectbox("Bits per subvector", [4, 8, 16], index=1)
            
        elif index_type == "IVF (Inverted File Index)":
            nlist = st.slider("Number of clusters (nlist)", 10, 1000, 100)
            nprobe = st.slider("Number to search (nprobe)", 1, 100, 10)
    
    with col2:
        st.markdown("### Database Settings")
        
        # Collection/Index name
        collection_name = st.text_input("Collection name", "rag_collection")
        
        # DB-specific settings
        if vector_db == "Milvus":
            consistency_level = st.selectbox("Consistency level", ["Strong", "Bounded", "Session", "Eventually"])
            metric_type = st.selectbox("Distance metric", ["L2", "IP", "Cosine"])
            
        elif vector_db == "Qdrant":
            distance = st.selectbox("Distance metric", ["Cosine", "Euclid", "Dot"])
            on_disk = st.checkbox("Store vectors on disk", value=False)
            
        elif vector_db == "Elasticsearch":
            similarity = st.selectbox("Similarity function", ["cosine", "dot_product", "l2_norm"])
            shards = st.slider("Number of shards", 1, 10, 1)
            
        elif vector_db == "Weaviate":
            vectorizer_type = st.selectbox("Vectorizer type", ["none", "text2vec"])
            tenant = st.text_input("Tenant (optional)")
    
    with col3:
        with st.expander("ℹ️ About Vector Databases", expanded=True):
            info_tooltip("""
            **Vector databases** store and efficiently search vector embeddings.
            
            **Database options**:
            - **Milvus**: High performance, highly scalable
            - **Qdrant**: Simple API, good for small-medium deployments
            - **Elasticsearch**: Full-text search + vector capabilities
            - **Weaviate**: Knowledge graph + vector search
            
            **Index types**:
            - **HNSW**: Fast, memory-intensive, high quality
            - **PQ**: Storage efficient, slight quality tradeoff
            - **IVF**: Balanced performance, works well with large datasets
            - **FLAT**: Highest quality, slowest performance
            
            **Key parameters**:
            - Higher M/ef values: Better accuracy, more memory, slower indexing
            - Lower M/ef values: Faster indexing, less memory, reduced accuracy
            - Increase nprobe/ef_search for better search quality
            """)
        
        with st.expander("📊 Performance Tradeoffs"):
            st.markdown("""
            | Index | Speed | Memory | Accuracy |
            |-------|-------|--------|----------|
            | HNSW | Fast | High | High |
            | PQ | Fast | Low | Medium |
            | IVF | Medium | Medium | Medium-High |
            | FLAT | Slow | Highest | Perfect |
            """)

# 7. Retrieval Section
elif section == "7. Retrieval":
    st.header("7. Retrieval Metrics, to see the performance of search")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Select metrics to evaluate retrieval quality")
        
        retrieval_metrics = {
            "MRR (Mean Reciprocal Rank)": st.checkbox("MRR", value=True),
            "EM (Exact Match)": st.checkbox("EM"),
            "Recall@1": st.checkbox("Recall@1", value=True),
            "Recall@10": st.checkbox("Recall@10", value=True),
            "Precision@1": st.checkbox("Precision@1"),
            "Precision@5": st.checkbox("Precision@5", value=True),
            "NDCG@5": st.checkbox("NDCG@5", value=True),
            "NDCG@10": st.checkbox("NDCG@10"),
            "Hit Rate@5": st.checkbox("Hit Rate@5", value=True),
            "Fuzzy Match": st.checkbox("Fuzzy Match"),
            "Coverage": st.checkbox("Coverage", value=True)
        }
        
        selected_metrics = [metric for metric, selected in retrieval_metrics.items() if selected]
        
        st.markdown("### Selected Metrics")
        if selected_metrics:
            st.write(", ".join(selected_metrics))
        else:
            st.warning("No metrics selected. Please select at least one metric.")
    
    with col2:
        with st.expander("ℹ️ About Retrieval Metrics", expanded=True):
            info_tooltip("""
            **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of relevant documents. Higher is better.
            
            **EM (Exact Match)**: Whether retrieved document exactly matches expected document. Binary metric.
            
            **Recall@k**: Portion of relevant documents retrieved in top k results. Measures if relevant content was found.
            
            **Precision@k**: Portion of top k results that are relevant. Measures quality of top results.
            
            **NDCG@k**: Normalized Discounted Cumulative Gain. Measures ranking quality considering relevance scores.
            
            **Hit Rate@k**: Whether any relevant document appears in the top k results. Binary per query.
            
            **Fuzzy Match**: Allows for partial matches using string similarity.
            
            **Coverage**: Percentage of queries for which at least one relevant document is retrieved.
            """)

# 8. Search Setup Section
elif section == "8. Search Setup":
    st.header("8. Search Configuration, to search using meaning or exact words")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Hybrid Search Setup")
        
        # Semantic vs Lexical slider
        semantic_weight = st.slider("Semantic vs Lexical Search Balance", 0, 100, 70, 
                                  help="Higher values favor semantic search, lower values favor keyword search")
        
        st.markdown(f"**Search weights**: {semantic_weight}% Semantic, {100-semantic_weight}% Lexical")
        
        # Reranking
        st.markdown("### Reranking")
        use_reranker = st.checkbox("Enable reranking")
        
        if use_reranker:
            reranker = st.selectbox("Reranker model", 
                               ["cohere-rerank", "bge-reranker-base", "bge-reranker-large", "Custom"])
            
            if reranker == "Custom":
                custom_reranker = st.text_input("Enter custom reranker name/path")
            
            top_k_to_rerank = st.slider("Number of results to rerank", 5, 100, 20)
        
        # Reciprocal Rank Fusion
        st.markdown("### Fusion Parameters")
        
        use_rrf = st.checkbox("Use Reciprocal Rank Fusion (RRF)", value=True)
        
        if use_rrf:
            rrf_k = st.number_input("RRF k value", 1, 100, 60, 
                                  help="Controls the impact of lower-ranked results")
    
    with col2:
        with st.expander("ℹ️ About Search Configuration", expanded=True):
            info_tooltip("""
            **Semantic vs Lexical Balance**: Controls the weight given to each search type.
            - Semantic search: Uses vector embeddings to find conceptually similar content
            - Lexical search: Uses keywords and exact matching
            
            Hybrid search often performs better than either method alone:
            - High semantic weight (80-100%): Better for complex questions and concept discovery
            - Balanced (40-60%): Good for general purpose applications
            - High lexical weight (0-30%): Better for specific term lookups or technical content
            
            **Reranking**: Applies a second scoring pass to improve result ranking
            - Cohere Rerank: State-of-the-art commercial reranking model
            - BGE Rerankers: Strong open-source alternatives
            - Improves precision by better aligning results with query intent
            
            **Reciprocal Rank Fusion (RRF)**: Method to combine multiple result lists
            - Combines semantic and lexical search results
            - The k parameter controls how quickly rank importance decays
            - Higher k: More emphasis on broader result set
            - Lower k: More emphasis on top results
            """)
        
        with st.expander("📊 Example Impact"):
            st.markdown("""
            | Search Balance | Good for |
            |--------------|----------|
            | 90% Semantic | Conceptual questions, inferential queries |
            | 70% Semantic | General knowledge questions |
            | 50% Balanced | Mixed content retrieval |
            | 30% Semantic | Technical document retrieval |
            | 10% Semantic | Specific term lookup |
            """)

# 9. LLM Parameters Section
elif section == "9. LLM Parameters":
    st.header("9. LLM Configuration, to control the output of llm ex: its creativity")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        llm_model = st.selectbox(
            "Select LLM",
            ["GPT-3.5-turbo", "GPT-4-turbo", "Claude 3 Sonnet", "Claude 3 Haiku", "Llama 3 70B", "Mistral Medium"]
        )
        
        st.markdown("### Generation Parameters")
        
        temp = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        top_p = st.slider("Top-p (nucleus sampling)", 0.0, 1.0, 0.9, 0.05)
        top_k = st.slider("Top-k", 1, 100, 40)
        
        repetition_penalty = st.slider("Repetition penalty", 1.0, 2.0, 1.1, 0.05)
        max_tokens = st.slider("Maximum output tokens", 100, 4000, 1000)
        
        include_logprobs = st.checkbox("Include logprobs")
        if include_logprobs:
            logprobs = st.slider("Number of logprobs", 1, 10, 5)
    
    with col2:
        with st.expander("ℹ️ About LLM Parameters", expanded=True):
            info_tooltip("""
            **Temperature**: Controls randomness in generation.
            - Lower (0.1-0.4): More deterministic, factual responses
            - Medium (0.5-0.7): Balanced creativity and coherence
            - Higher (0.8-1.0): More creative, varied outputs
            
            **Top-p (nucleus sampling)**: Controls diversity by limiting token selection to most likely tokens.
            - Lower values: More focused, conservative outputs
            - Higher values: More diverse outputs
            
            **Top-k**: Only samples from k most likely next tokens.
            - Lower k: More predictable text
            - Higher k: More diverse options
            
            **Repetition penalty**: Discourages repeating the same phrases.
            - Higher values reduce repetition but can impact fluency
            
            **Maximum tokens**: Limits response length.
            - Consider context length and typical answer length needed
            
            **Logprobs**: Returns probability details for generated tokens.
            - Useful for uncertainty estimation and analyzing model confidence
            """)
        
        with st.expander("📊 Parameter Recommendations"):
            st.markdown("""
            | Use Case | Temperature | Top-p | Top-k |
            |----------|------------|-------|-------|
            | Factual QA | 0.1-0.3 | 0.9 | 40 |
            | Summarization | 0.3-0.5 | 0.9 | 50 |
            | Explanation | 0.5-0.7 | 0.9 | 40 |
            | Creative | 0.7-1.0 | 0.95 | 60 |
            """)

# 10. Evaluation Metrics Section
elif section == "10. Evaluation Metrics":
    st.header("10. Generation Evaluation Metrics, to measure the performance of the generated output ex: hallucination")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("### Select metrics to evaluate generation quality")
        
        generation_metrics = {
            "BLEU": st.checkbox("BLEU", value=True),
            "ROUGE-1": st.checkbox("ROUGE-1", value=True),
            "ROUGE-2": st.checkbox("ROUGE-2"),
            "ROUGE-L": st.checkbox("ROUGE-L", value=True),
            "METEOR": st.checkbox("METEOR")
        }
        
        st.markdown("### RAGAS Metrics")
        ragas_metrics = {
            "Faithfulness": st.checkbox("Faithfulness", value=True),
            "Relevancy": st.checkbox("Relevancy", value=True),
            "Correctness": st.checkbox("Correctness", value=True),
            "Context Precision": st.checkbox("Context Precision"),
            "Context Recall": st.checkbox("Context Recall", value=True)
        }
        
        selected_gen_metrics = [metric for metric, selected in generation_metrics.items() if selected]
        selected_ragas_metrics = [metric for metric, selected in ragas_metrics.items() if selected]
        
        st.markdown("### Selected Metrics")
        if selected_gen_metrics or selected_ragas_metrics:
            if selected_gen_metrics:
                st.write("Generation metrics:", ", ".join(selected_gen_metrics))
            if selected_ragas_metrics:
                st.write("RAGAS metrics:", ", ".join(selected_ragas_metrics))
        else:
            st.warning("No metrics selected. Please select at least one metric.")
    
    with col2:
        with st.expander("ℹ️ About Generation Metrics", expanded=True):
            info_tooltip("""
            **BLEU**: Evaluates generated text against reference by n-gram precision.
            - Higher scores indicate better grammatical accuracy
            - Limited in semantic evaluation
            
            **ROUGE**: Family of metrics measuring overlap between generated and reference text.
            - ROUGE-1: Unigram overlap (word level matching)
            - ROUGE-2: Bigram overlap (phrase matching)
            - ROUGE-L: Longest common subsequence (fluency)
            
            **METEOR**: Evaluates considering synonyms, stemming, and word order.
            - Better than BLEU for semantic understanding
            - Correlates better with human judgments
            
            **RAGAS Metrics**:
            - **Faithfulness**: Measures if generated answer is factually consistent with retrieved context
            - **Relevancy**: Evaluates if retrieved context is relevant to the question
            - **Correctness**: Assesses factual accuracy of the answer
            - **Context Precision**: Measures how much of retrieved context was actually relevant
            - **Context Recall**: Evaluates whether all necessary information was retrieved
            """)
        
        with st.expander("📊 Metric Recommendations"):
            st.markdown("""
            | Use Case | Recommended Metrics |
            |----------|---------------------|
            | Factual QA | Faithfulness, Correctness, ROUGE-L |
            | Summarization | ROUGE scores, Context Precision |
            | Technical Support | Correctness, Faithfulness, Context Recall |
            | Creative Content | BLEU, ROUGE scores |
            """)

# 11. Run Experiment Section
elif section == "11. Run Experiment":
    st.header("11. Run Experiment, last step")
    
    # Display configuration summary
    st.markdown("### Configuration Summary")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Pipeline Components")
        st.markdown("- **Embedding**: text-embedding-3-large-v2")
        st.markdown("- **Parser**: PyMuPDF + pdfplumber")
        st.markdown("- **Chunker**: Semantic")
        st.markdown("- **Vector DB**: Milvus with HNSW indexing")
        
        st.subheader("Search Setup")
        st.markdown("- **Balance**: 70% Semantic, 30% Lexical")
        st.markdown("- **Reranker**: cohere-rerank")
        st.markdown("- **Fusion**: RRF (k=60)")
        
    with col2:
        st.subheader("LLM Configuration")
        st.markdown("- **Model**: GPT-4-turbo")
        st.markdown("- **Temperature**: 0.7")
        st.markdown("- **Max tokens**: 1000")
        
        st.subheader("Evaluation")
        st.markdown("- **Retrieval**: MRR, Recall@10, NDCG@5")
        st.markdown("- **Generation**: ROUGE-L, Faithfulness")

    # Run button
    if st.button("Run Experiment", type="primary"):
        with st.spinner("Running experiment..."):
            # Simulated progress
            progress_bar = st.progress(0)
            for i in range(100):
                # Update progress bar
                progress_bar.progress(i + 1)
                import time
                time.sleep(0.05)  # Simulate work being done
            
            st.success("Experiment completed successfully!")
            
            # Show sample results
            st.subheader("Results Preview")
            
            # Sample retrieval metrics
            retrieval_results = {
                "MRR": 0.83,
                "Recall@10": 0.91,
                "NDCG@5": 0.76,
                "Hit Rate@5": 0.88,
                "Coverage": 0.94
            }
            
            # Sample generation metrics
            generation_results = {
                "ROUGE-L": 0.72,
                "Faithfulness": 0.89,
                "Relevancy": 0.85,
                "Correctness": 0.79
            }
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### Retrieval Metrics")
                st.dataframe(pd.DataFrame({
                    "Metric": list(retrieval_results.keys()),
                    "Score": list(retrieval_results.values())
                }))
            
            with col2:
                st.markdown("#### Generation Metrics")
                st.dataframe(pd.DataFrame({
                    "Metric": list(generation_results.keys()),
                    "Score": list(generation_results.values())
                }))
                
            # Visual comparison
            st.subheader("Performance Visualization")
            chart_data = pd.DataFrame({
                "Metric": ["ROUGE-L", "Faithfulness", "Correctness", "Recall@10", "MRR"],
                "Score": [0.72, 0.89, 0.79, 0.91, 0.83]
            })
            
            st.bar_chart(chart_data.set_index("Metric"))
            
    else:
        st.info("Review your configuration and click 'Run Experiment' when ready.")


# Add new History section at the end
elif section == "12. History":
    st.header("Experiment History")
    st.markdown("Review and compare your previous RAG experiments and their performance metrics.")

    # In a real application, you would load this from a database
    # This is simulated data for display purposes
    history_data = [
        {
            "timestamp": "2025-05-28 09:15:22",
            "execution_time": "3m 42s",
            "config": {
                "embedding": "text-embedding-3-large-v2",
                "parser": "PyMuPDF + pdfplumber",
                "chunker": "Semantic (threshold=0.75)",
                "vector_db": "Milvus (HNSW)",
                "search_balance": "70% Semantic / 30% Lexical",
                "llm": "GPT-4-turbo (temp=0.7)"
            },
            "retrieval_metrics": {
                "MRR": 0.83,
                "Recall@10": 0.91,
                "NDCG@5": 0.76,
                "Hit Rate@5": 0.88,
                "Coverage": 0.94
            },
            "generation_metrics": {
                "ROUGE-L": 0.72,
                "Faithfulness": 0.89,
                "Relevancy": 0.85,
                "Correctness": 0.79
            }
        },
        {
            "timestamp": "2025-05-27 15:43:05",
            "execution_time": "4m 12s",
            "config": {
                "embedding": "text-embedding-ada-003",
                "parser": "Unstructured.io",
                "chunker": "RAPTOR (layers=2)",
                "vector_db": "Qdrant (HNSW)",
                "search_balance": "50% Semantic / 50% Lexical",
                "llm": "Claude 3 Sonnet (temp=0.5)"
            },
            "retrieval_metrics": {
                "MRR": 0.78,
                "Recall@10": 0.87,
                "NDCG@5": 0.72,
                "Hit Rate@5": 0.82,
                "Coverage": 0.91
            },
            "generation_metrics": {
                "ROUGE-L": 0.68,
                "Faithfulness": 0.92,
                "Relevancy": 0.81,
                "Correctness": 0.85
            }
        },
        {
            "timestamp": "2025-05-25 11:21:33",
            "execution_time": "2m 58s",
            "config": {
                "embedding": "PubMed BERT",
                "parser": "PyMuPDF + pdfplumber",
                "chunker": "Recursive (size=500)",
                "vector_db": "Elasticsearch (IVF)",
                "search_balance": "30% Semantic / 70% Lexical",
                "llm": "GPT-3.5-turbo (temp=0.3)"
            },
            "retrieval_metrics": {
                "MRR": 0.74,
                "Recall@10": 0.85,
                "NDCG@5": 0.69,
                "Hit Rate@5": 0.79,
                "Coverage": 0.88
            },
            "generation_metrics": {
                "ROUGE-L": 0.65,
                "Faithfulness": 0.86,
                "Relevancy": 0.78,
                "Correctness": 0.81
            }
        }
    ]

    # View options
    view_mode = st.radio("View mode", ["Summary", "Detailed"])
    
    if view_mode == "Summary":
        # Create a summary dataframe
        summary_data = []
        for entry in history_data:
            summary_data.append({
                "Date & Time": entry["timestamp"],
                "Execution Time": entry["execution_time"],
                "Config": f"{entry['config']['chunker']} + {entry['config']['llm']}",
                "MRR": entry["retrieval_metrics"]["MRR"],
                "Recall@10": entry["retrieval_metrics"]["Recall@10"],
                "ROUGE-L": entry["generation_metrics"]["ROUGE-L"],
                "Faithfulness": entry["generation_metrics"]["Faithfulness"]
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Visualize key metrics over time
        st.subheader("Key Metrics Over Time")
        
        metrics_to_plot = ["MRR", "Recall@10", "ROUGE-L", "Faithfulness"]
        chart_data = pd.DataFrame({
            "Date": [entry["timestamp"].split()[0] for entry in history_data],
            **{metric: [run["retrieval_metrics"].get(metric, run["generation_metrics"].get(metric, 0)) 
                    for run in history_data] for metric in metrics_to_plot}
        })
        
        st.line_chart(chart_data.set_index("Date"))
        
    else:  # Detailed view
        # For each historical run
        for i, entry in enumerate(history_data):
            with st.expander(f"Run {i+1}: {entry['timestamp']} ({entry['execution_time']})", expanded=(i==0)):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Configuration")
                    for param, value in entry["config"].items():
                        st.markdown(f"**{param.title()}**: {value}")
                
                with col2:
                    st.subheader("Performance")
                    
                    # Retrieval metrics
                    st.markdown("##### Retrieval Metrics")
                    ret_metrics_df = pd.DataFrame({
                        "Metric": list(entry["retrieval_metrics"].keys()),
                        "Score": list(entry["retrieval_metrics"].values())
                    })
                    st.dataframe(ret_metrics_df, hide_index=True)
                    
                    # Generation metrics
                    st.markdown("##### Generation Metrics")
                    gen_metrics_df = pd.DataFrame({
                        "Metric": list(entry["generation_metrics"].keys()),
                        "Score": list(entry["generation_metrics"].values())
                    })
                    st.dataframe(gen_metrics_df, hide_index=True)
                
                # Add a visualization for this specific run
                st.markdown("##### Performance Visualization")
                
                # Combine metrics for visualization
                all_metrics = {**entry["retrieval_metrics"], **entry["generation_metrics"]}
                chart_data = pd.DataFrame({
                    "Metric": list(all_metrics.keys()),
                    "Score": list(all_metrics.values())
                })
                
                st.bar_chart(chart_data.set_index("Metric"))
                
                # Add divider between entries
                if i < len(history_data) - 1:
                    st.markdown("---")
    
    # Add export functionality
    st.subheader("Export History")
    export_format = st.selectbox("Export format", ["CSV", "JSON", "Excel"])
    
    if st.button("Export Data"):
        # In a real app, you'd implement actual export functionality
        st.success(f"History data exported as {export_format}")
        
        if export_format == "CSV":
            # Create CSV for download
            summary_data = []
            for entry in history_data:
                summary_data.append({
                    "Date_Time": entry["timestamp"],
                    "Execution_Time": entry["execution_time"],
                    **entry["config"],
                    **{f"ret_{k}": v for k, v in entry["retrieval_metrics"].items()},
                    **{f"gen_{k}": v for k, v in entry["generation_metrics"].items()}
                })
            
            csv = pd.DataFrame(summary_data).to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="rag_experiment_history.csv">Download CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Add comparison feature
    st.subheader("Compare Experiments")
    
    # Let users select which experiments to compare
    options = [f"Run {i+1}: {entry['timestamp']}" for i, entry in enumerate(history_data)]
    selected_runs = st.multiselect("Select runs to compare", options, default=options[:2] if len(options) >= 2 else options)
    
    if selected_runs and st.button("Compare Selected"):
        # Get indices of selected runs
        indices = [int(run.split(":")[0].replace("Run ", "")) - 1 for run in selected_runs]
        
        # Create comparison data
        comparison_data = {
            "Config/Metric": ["Embedding", "Chunker", "Vector DB", "Search Balance", "LLM", 
                             "MRR", "Recall@10", "ROUGE-L", "Faithfulness"]
        }
        
        for i, idx in enumerate(indices):
            entry = history_data[idx]
            run_name = f"Run {idx+1}"
            comparison_data[run_name] = [
                entry["config"]["embedding"],
                entry["config"]["chunker"],
                entry["config"]["vector_db"],
                entry["config"]["search_balance"],
                entry["config"]["llm"],
                entry["retrieval_metrics"]["MRR"],
                entry["retrieval_metrics"]["Recall@10"],
                entry["generation_metrics"]["ROUGE-L"],
                entry["generation_metrics"]["Faithfulness"]
            ]
        
        # Display comparison table
        st.dataframe(pd.DataFrame(comparison_data).set_index("Config/Metric"), use_container_width=True)
        
        # Create comparative chart for metrics
        st.subheader("Metrics Comparison")
        
        chart_data = {
            "Metric": ["MRR", "Recall@10", "ROUGE-L", "Faithfulness"]
        }
        
        for i, idx in enumerate(indices):
            entry = history_data[idx]
            run_name = f"Run {idx+1}"
            chart_data[run_name] = [
                entry["retrieval_metrics"]["MRR"],
                entry["retrieval_metrics"]["Recall@10"],
                entry["generation_metrics"]["ROUGE-L"],
                entry["generation_metrics"]["Faithfulness"]
            ]
        
        comparison_chart = pd.DataFrame(chart_data).melt(
            id_vars=["Metric"], 
            var_name="Run", 
            value_name="Score"
        )
        
        # Plot the comparison as grouped bars (side by side for each metric)
        import altair as alt
        chart = alt.Chart(comparison_chart).mark_bar().encode(
            x=alt.X('Metric:N', title='Metric'),
            y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1]), title='Score'),
            color=alt.Color('Run:N', legend=alt.Legend(title="Run")),
            tooltip=['Run:N', 'Metric:N', 'Score:Q'],
            order=alt.Order('Run:N'),
        ).properties(width=120).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        )

        # Use a grouped bar chart (side-by-side bars for each metric)
        chart = chart.encode(
            x=alt.X('Metric:N', title='Metric', axis=alt.Axis(labelAngle=0)),
            xOffset='Run:N'
        )

        st.altair_chart(chart, use_container_width=True)