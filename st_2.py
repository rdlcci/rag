"""
RAG Builder: A comprehensive toolkit for building and optimizing Retrieval-Augmented Generation systems

Version: 0.2.0
Author: Rahul Damani
Date: May 28, 2025

Description:
    RAG Builder provides an interactive interface for designing, configuring, testing, and deploying
    Retrieval-Augmented Generation (RAG) systems. The application guides users through the entire RAG
    pipeline, from document processing to deployment and evaluation, with a focus on best practices
    and optimization at each step.

Modules:
    1. Document Processing: Configure document ingestion, parsing, OCR, and ground truth creation
    2. Embedding & Representation: Select and optimize embedding models and strategies
    3. Chunking & Indexing: Configure document splitting, cross-references, and metadata extraction
    4. Vector Database: Set up vector storage with advanced indexing and optimization
    5. Query Processing: Configure query transformation, planning, and routing
    6. Retrieval Strategy: Implement and optimize retrieval methods and hybrid search
    7. Reranking & Fusion: Configure advanced reranking and result fusion techniques
    8. LLM Integration: Set up LLM connectivity, prompting, and context handling
    9. Evaluation & Monitoring: Implement metrics, ground truth evaluation, and monitoring
    10. Deployment: Configure architecture, infrastructure, and security
    11. Run Experiment: Design and execute comparative experiments
    12. Results & History: Track results, configurations, and system performance over time

Dependencies:
    - streamlit: UI framework
    - pandas: Data handling and visualization
    - altair: Interactive charts
    - numpy: Numerical operations
    - datetime: Date and time utilities

Usage:
    Run the application with: streamlit run st_2.py

Changelog:
    - 0.2.0:
        - Added multi-modal document processing support
        - Enhanced embedding strategies with domain specialization
        - Improved chunking with adaptive and hierarchical options
        - Added advanced vector database optimizations
        - Implemented comprehensive query transformation pipeline
        - Enhanced retrieval with multi-stage and RAG-fusion capabilities
        - Added LLM-based evaluation and self-consistency techniques
        - Implemented full experiment framework with results tracking
        - Added comprehensive security and compliance options
        - Created knowledge base for best practices and troubleshooting

    - 0.1.0:
        - Initial version with basic RAG pipeline components
        - Basic embedding and retrieval options
        - Simple evaluation metrics
        - Limited deployment options

License: Proprietary (Eli Lilly and Company)

Features : 
Document Processing Enhancements: OCR capabilities, structured data handling, and ground truth are included
Advanced Embedding Strategies: Domain-specific embeddings, hierarchical embeddings, and fine-tuning options are present
Chunking Strategies: Adaptive chunking, hierarchical chunking, and cross-referencing are implemented
Vector Database Optimizations: Advanced filtering, performance optimizations, and various index types are covered
Retrieval Augmentations: Query transformation, multi-stage retrieval, and RAG-Fusion are included
LLM Integration: Prompt engineering, chain-of-thought, and context management are well-addressed
Evaluation and Monitoring: Comprehensive metrics, runtime monitoring, and continuous improvement are implemented
Deployment and Scaling: Architecture, infrastructure, and security considerations are covered
Experimentation: Robust experiment setup, execution, and analysis capabilities

TODO:
1. Embedding page has image missing
2. 
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
from PIL import Image
import altair as alt
import time
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Advanced RAG Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
st.markdown("""
<style>
    .main {padding: 2rem;}
    .info-box {background-color: #f0f2f6; border-radius: 8px; padding: 10px; margin-bottom: 15px;}
    .stButton>button {width: 100%;}
    .header-container {display: flex; align-items: center; gap: 10px;}
    .metric-card {background-color: #f9f9f9; border-radius: 5px; padding: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
    .feature-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 10px;}
    .feature-item {background: #f8f9fa; padding: 10px; border-radius: 4px; border-left: 4px solid #4CAF50;}
    .warning {color: #856404; background-color: #fff3cd; border-radius: 4px; padding: 10px;}
    .advanced {border-left: 4px solid #007bff; background: #f1f8ff;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {height: 40px; white-space: pre-wrap; border-radius: 4px;}
</style>
""", unsafe_allow_html=True)

# Helper function for tooltips
def info_tooltip(text):
    return st.info(text)

# App title
st.title("🧠 Advanced RAG Builder")
st.markdown("Configure, deploy, and optimize state-of-the-art RAG systems with advanced features")

# Create primary navigation
st.sidebar.title("Pipeline Configuration")
main_section = st.sidebar.radio(
    "Configuration Steps",
    ["1. Document Processing", 
     "2. Embedding & Representation", 
     "3. Chunking & Indexing", 
     "4. Vector Database", 
     "5. Query Processing",
     "6. Retrieval Strategy",
     "7. Reranking & Fusion",
     "8. LLM Integration",
     "9. Evaluation & Monitoring",
     "10. Deployment",
     "11. Run Experiment",
     "12. Results & History"]
)

# Create secondary navigation based on primary selection
if main_section in ["1. Document Processing", "2. Embedding & Representation", "3. Chunking & Indexing"]:
    advanced_mode = st.sidebar.checkbox("Advanced Mode", value=False, 
                                      help="Show advanced configuration options")

# Sample data
sample_data = {
    "question": ["What is RAG?", "How does vector search work?", "What is the best chunking method?"],
    "ground_truth": ["RAG stands for Retrieval-Augmented Generation, a technique that enhances LLMs with external knowledge.", 
                    "Vector search works by converting documents into numerical vectors and finding the most similar vectors.",
                    "The best chunking method depends on your use case, but semantic chunking often performs well."]
}


if 'metric_to_visualize' not in st.session_state:
    metric_to_visualize = "All Metrics"  # Default value
else:
    metric_to_visualize = st.session_state.metric_to_visualize


#########################
# 1. DOCUMENT PROCESSING
#########################
if main_section == "1. Document Processing":
    st.header("1. Document Processing")
    st.markdown("Configure how documents are processed, extracted, and enhanced before chunking and indexing.")
    
    # Create tabs for different aspects of document processing
    doc_tabs = st.tabs(["Document Upload", "Parsing Options", "OCR & Image Processing", "Structured Data", "Ground Truth"])
    
    # DOCUMENT UPLOAD TAB
    with doc_tabs[0]:
        st.subheader("Document Upload")
        st.markdown("Upload documents to be processed by the RAG pipeline.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Multi-file upload
            uploaded_files = st.file_uploader(
                "Upload Documents",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt", "csv", "json", "xlsx", "pptx", "md", "html", "xml"]
            )
            
            if uploaded_files:
                st.success(f"{len(uploaded_files)} documents uploaded")
                
                # Display uploaded files
                file_data = []
                for file in uploaded_files:
                    size_kb = file.size / 1024
                    file_data.append({
                        "Filename": file.name,
                        "Type": file.type,
                        "Size (KB)": f"{size_kb:.2f}"
                    })
                
                st.dataframe(pd.DataFrame(file_data))
            
        with col2:
            st.markdown("### Import Options")
            import_options = st.selectbox(
                "Additional Import Sources",
                ["None", "URL", "S3 Bucket", "Google Drive", "Database Connection"]
            )
            
            if import_options == "URL":
                url_input = st.text_input("Enter URL to scrape")
                if st.button("Import from URL"):
                    st.info("URL import simulation: Document would be fetched from the URL")
            
            elif import_options == "S3 Bucket":
                s3_path = st.text_input("Enter S3 path", "s3://bucket-name/path/")
                if st.button("Connect to S3"):
                    st.info("S3 import simulation: Documents would be fetched from S3")
            
            elif import_options == "Google Drive":
                st.info("Would connect to Google Drive API")
                
            elif import_options == "Database Connection":
                db_type = st.selectbox("Database Type", ["PostgreSQL", "MySQL", "MongoDB", "Elasticsearch"])
                conn_string = st.text_input("Connection String", "postgresql://user:password@localhost:5432/db")
                if st.button("Connect"):
                    st.info(f"Database import simulation: Would connect to {db_type}")
        
        # Document handling options
        with st.expander("Document Processing Options", expanded=False):
            st.checkbox("Remove duplicate documents", value=True)
            st.checkbox("Extract metadata from documents", value=True)
            st.checkbox("Maintain folder structure", value=False)
            st.checkbox("Version tracking", value=False)
        
        with st.expander("ℹ️ About Document Upload", expanded=True):
            info_tooltip("""
            **Supported File Types:**
            - **PDF**: Standard document format (.pdf)
            - **Word**: Microsoft Word documents (.docx)
            - **Text**: Plain text files (.txt)
            - **Spreadsheets**: Excel and CSV files (.xlsx, .csv)
            - **Presentations**: PowerPoint files (.pptx)
            - **Markup**: Markdown, HTML, and XML (.md, .html, .xml)
            
            **Import Sources:**
            - **URL**: Scrape content from web pages
            - **S3**: Import from AWS S3 storage
            - **Google Drive**: Import from Google Drive
            - **Database**: Extract text from database records
            
            **Processing Options:**
            - **Deduplication**: Remove identical or near-identical documents
            - **Metadata Extraction**: Pull author, date, title from document properties
            - **Folder Structure**: Maintain hierarchical relationships
            - **Version Tracking**: Track document versions and changes
            """)
    
    # PARSING OPTIONS TAB
    with doc_tabs[1]:
        st.subheader("Document Parsing")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            parser_option = st.selectbox(
                "Primary Parser",
                ["PyMuPDF + pdfplumber", "Unstructured.io", "Langchain Document Loaders", "Tika", "Custom Parser"]
            )
            
            st.markdown("### Parser Configuration")
            
            if parser_option == "PyMuPDF + pdfplumber":
                col1a, col1b = st.columns(2)
                with col1a:
                    table_format = st.selectbox("Table format", ["DataFrame", "JSON", "Markdown", "HTML"])
                    preserve_layout = st.checkbox("Preserve document layout", value=True)
                    
                with col1b:
                    extract_images = st.checkbox("Extract images", value=True)
                    extract_links = st.checkbox("Extract hyperlinks", value=True)
                    use_ocr_fallback = st.checkbox("Use OCR fallback for scanned pages", value=True)
                
                if extract_images:
                    image_format = st.selectbox("Image save format", ["PNG", "JPEG", "Original Format"])
                    image_dir = st.text_input("Image save directory", "images/")
                
            elif parser_option == "Unstructured.io":
                strategy = st.selectbox("Extraction strategy", ["Auto", "Fast", "OCR", "Hi-res"])
                include_metadata = st.checkbox("Include element metadata", value=True)
                hierarchy_mode = st.selectbox("Hierarchy extraction", ["Simple", "Full", "None"])
                
                st.markdown("### Element Filters")
                element_types = st.multiselect(
                    "Elements to extract",
                    ["Text", "Tables", "Images", "Lists", "Title", "Headers", "Footers", "Equations"],
                    default=["Text", "Tables", "Images", "Headers"]
                )
                
            elif parser_option == "Langchain Document Loaders":
                loader_type = st.selectbox(
                    "Document Loader Type", 
                    ["PDFLoader", "DocxLoader", "CSVLoader", "TextLoader", "JSONLoader", "UnstructuredLoader"]
                )
                
                recursive = st.checkbox("Process directories recursively", value=True)
                chunk_size = st.slider("Initial chunk size", 100, 2000, 500)
                
            elif parser_option == "Tika":
                tika_url = st.text_input("Apache Tika Server URL", "http://localhost:9998")
                extract_content = st.checkbox("Extract content", value=True)
                extract_metadata = st.checkbox("Extract metadata", value=True)
                
            elif parser_option == "Custom Parser":
                parser_script = st.text_area("Custom Parser Code (Python)", 
                """def parse_document(file_path):
    # Custom parsing logic here
    return {"text": content, "metadata": metadata}
                """)
                test_file = st.file_uploader("Test file for custom parser")
                if test_file and st.button("Test Parser"):
                    st.info("Custom parser would be executed on the test file")
        
        with col2:
            with st.expander("ℹ️ About Document Parsing", expanded=True):
                info_tooltip("""
                **Parser Options:**
                
                **PyMuPDF + pdfplumber**: Combines PyMuPDF for text and image extraction with pdfplumber for table extraction.
                - Good for documents with well-defined structure
                - Fast processing with good table recognition
                - Options for different table formats
                
                **Unstructured.io**: More sophisticated parsing that identifies document elements.
                - Better handling of complex layouts
                - Maintains hierarchical structure
                - More comprehensive metadata extraction
                - Good for mixed content types
                
                **Langchain Document Loaders**: Universal document processing system.
                - Wide variety of document formats
                - Integration with Langchain ecosystem
                - Simple API but less control over extraction
                
                **Tika**: Apache's document parsing framework.
                - Excellent format support (1000+ file types)
                - Language detection and metadata extraction
                - Requires running a Tika server
                
                **Custom Parser**: Write your own parsing logic.
                - Maximum flexibility
                - Can be tailored to specific document formats
                - Requires Python programming
                
                **Key Considerations:**
                - Layout preservation is important for documents where spatial arrangement matters
                - Table extraction is challenging - select the right format for your use case
                - Image extraction can provide additional context for multimodal models
                """)
            
            # Show an illustrative example
            if parser_option == "PyMuPDF + pdfplumber":
                st.markdown("#### Example Parser Output")
                st.code("""
# PyMuPDF + pdfplumber output example
{
    "page_content": "This is the extracted text from page 1...",
    "metadata": {
        "source": "document1.pdf",
        "page": 1,
        "tables": [
            {"rows": 5, "cols": 3, "data": [...]}
        ],
        "images": ["image_1.png", "image_2.png"],
        "layout": {
            "blocks": [
                {"type": "text", "bbox": [0.1, 0.2, 0.8, 0.3]},
                {"type": "image", "bbox": [0.5, 0.6, 0.7, 0.8]}
            ]
        }
    }
}
                """)
    
    # OCR & IMAGE PROCESSING TAB
    with doc_tabs[2]:
        st.subheader("OCR & Image Processing")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # OCR options
            ocr_engine = st.selectbox(
                "OCR Engine",
                ["None", "Tesseract", "Google Vision API", "Azure OCR", "AWS Textract", "PaddleOCR", "EasyOCR"]
            )
            
            if ocr_engine != "None":
                col_ocr1, col_ocr2 = st.columns(2)
                
                with col_ocr1:
                    languages = st.multiselect(
                        "OCR Languages",
                        ["English", "Spanish", "French", "German", "Chinese", "Japanese", "Arabic", "Russian"],
                        default=["English"]
                    )
                    
                    dpi = st.slider("Image DPI", 72, 600, 300)
                    
                with col_ocr2:
                    ocr_mode = st.selectbox("OCR Mode", ["Basic", "Advanced", "Document AI"])
                    
                    apply_ocr = st.radio(
                        "When to apply OCR",
                        ["Only on scanned documents", "On all documents", "Manual selection"]
                    )
                
                # Advanced OCR settings
                with st.expander("Advanced OCR Settings"):
                    st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
                    st.checkbox("Auto-rotate images", value=True)
                    st.checkbox("Use page segmentation", value=True)
                    st.checkbox("Apply image preprocessing", value=True)
                    st.selectbox("OCR Model Size", ["Small (Fast)", "Medium", "Large (Accurate)"])
            
            # Image processing options
            st.markdown("### Image Content Analysis")
            
            image_analysis = st.multiselect(
                "Image Analysis Features",
                ["None", "Caption Generation", "Object Detection", "Visual Question Answering", "Text from Charts"],
                default=["None"]
            )
            
            if "None" not in image_analysis:
                vision_model = st.selectbox(
                    "Vision-Language Model",
                    ["CLIP", "ViT-GPT2", "BLIP-2", "LLaVA", "GPT-4V", "Gemini Pro Vision"]
                )
                
                st.checkbox("Include image captions in document", value=True)
                st.checkbox("Extract data from charts and diagrams", value=True)
                st.checkbox("Store image embeddings", value=False)
        
        with col2:
            with st.expander("ℹ️ About OCR & Image Processing", expanded=True):
                info_tooltip("""
                **OCR (Optical Character Recognition)** converts images of text into machine-encoded text.
                
                **OCR Engines:**
                - **Tesseract**: Open-source engine, good for simple documents
                - **Google Vision**: Cloud-based with high accuracy, supports 200+ languages
                - **Azure OCR**: Microsoft's OCR with layout preservation
                - **AWS Textract**: Specialized for forms and tables
                - **PaddleOCR**: High-performance multilingual OCR
                - **EasyOCR**: Simple Python library with 80+ languages
                
                **OCR Parameters:**
                - **Languages**: Select all languages present in your documents
                - **DPI**: Higher values provide better quality but slower processing
                - **Confidence Threshold**: Minimum confidence for text recognition
                
                **Image Analysis:**
                - **Caption Generation**: Create descriptive captions for images
                - **Object Detection**: Identify objects within images
                - **Visual QA**: Answer questions about image content
                - **Text from Charts**: Extract data values from charts and graphs
                
                **Vision-Language Models:**
                - **CLIP**: OpenAI model for image-text matching
                - **BLIP-2**: State-of-the-art for image understanding
                - **GPT-4V**: Multimodal version of GPT-4
                - **Gemini Pro Vision**: Google's multimodal model
                """)
            
            # Example output
            if ocr_engine != "None" or "None" not in image_analysis:
                st.markdown("#### Example Multimodal Processing")
                
                if ocr_engine != "None":
                    st.markdown("**OCR Output Example:**")
                    st.code("""
# OCR result example
{
    "text": "The quarterly report shows a 12% increase in revenue...",
    "confidence": 0.98,
    "layout": {
        "paragraphs": [
            {"text": "The quarterly report...", "bbox": [0.1, 0.2, 0.8, 0.3]},
            {"text": "Financial highlights...", "bbox": [0.1, 0.4, 0.8, 0.6]}
        ],
        "text_orientation": "0°"
    }
}
                    """)
                
                if "None" not in image_analysis:
                    st.markdown("**Image Analysis Example:**")
                    st.code("""
# Image analysis result
{
    "caption": "A bar chart showing quarterly revenue growth across regions",
    "objects": ["bar chart", "legend", "x-axis", "y-axis"],
    "chart_data": {
        "type": "bar_chart",
        "x_axis": ["Q1", "Q2", "Q3", "Q4"],
        "series": [
            {"name": "North America", "values": [32, 35, 40, 45]},
            {"name": "Europe", "values": [28, 30, 33, 38]}
        ]
    }
}
                    """)
    
    # STRUCTURED DATA TAB
    with doc_tabs[3]:
        st.subheader("Structured Data Processing")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Table Extraction")
            table_extraction = st.selectbox(
                "Table Extraction Method",
                ["Basic", "Advanced (Transformer-based)", "Rule-based", "Custom"]
            )
            
            col_tab1, col_tab2 = st.columns(2)
            
            with col_tab1:
                table_output = st.selectbox(
                    "Table Output Format",
                    ["DataFrame", "JSON", "Markdown", "HTML", "CSV"]
                )
                
                max_tables = st.number_input("Maximum tables per document", 1, 100, 10)
                
            with col_tab2:
                table_confidence = st.slider("Table detection confidence", 0.0, 1.0, 0.7, 0.05)
                table_summary = st.checkbox("Generate table summaries", value=True)
            
            # Advanced table options
            with st.expander("Advanced Table Options"):
                st.checkbox("Preserve cell merges", value=True)
                st.checkbox("Detect headers", value=True)
                st.checkbox("Extract table captions", value=True)
                st.checkbox("Handle nested tables", value=False)
            
            # Structured data handling
            st.markdown("### Structured Data Handling")
            
            structured_types = st.multiselect(
                "Structured Data Types to Process",
                ["CSV", "JSON", "XML", "Database Records", "API Responses", "Excel Formulas"],
                default=["CSV", "JSON"]
            )
            
            if structured_types:
                structure_handling = st.radio(
                    "Structure Handling",
                    ["Preserve original structure", "Flatten into text", "Hybrid approach"]
                )
                
                if "JSON" in structured_types or "XML" in structured_types:
                    st.checkbox("Preserve nested relationships", value=True)
                
                if "CSV" in structured_types or "Database Records" in structured_types:
                    st.checkbox("Preserve row-column relationships", value=True)
        
        with col2:
            with st.expander("ℹ️ About Structured Data Processing", expanded=True):
                info_tooltip("""
                **Table Extraction Methods:**
                - **Basic**: Simple grid detection using heuristics
                - **Advanced (Transformer-based)**: ML models specialized for table detection
                - **Rule-based**: Custom rules for specific table formats
                - **Custom**: Write your own table extraction logic
                
                **Table Processing Features:**
                - **Confidence Threshold**: Minimum confidence for table detection
                - **Table Summaries**: LLM-generated descriptions of table contents
                - **Cell Merges**: Preserve merged cells in complex tables
                - **Nested Tables**: Handle tables within tables
                
                **Structured Data Handling:**
                - **Preserve Structure**: Keep the original format (best for specialized retrievers)
                - **Flatten**: Convert to plain text (simplest approach)
                - **Hybrid**: Combination of structured and text representations
                
                **Why This Matters:**
                - Tables and structured data often contain the most important information
                - Poor table extraction leads to lost information
                - Proper structure preservation improves retrieval for data-heavy queries
                
                **Example Use Cases:**
                - Financial reports with many numerical tables
                - Technical documentation with formatted data
                - Scientific papers with results in tabular form
                - API documentation with structured examples
                """)
            
            # Example structured data handling
            st.markdown("#### Example Table Processing")
            st.code("""
# Original Table in PDF
+-----------+--------+--------+--------+
| Product   | Q1     | Q2     | Q3     |
+-----------+--------+--------+--------+
| Widget A  | $1,200 | $1,350 | $1,450 |
| Widget B  | $950   | $1,000 | $1,100 |
+-----------+--------+--------+--------+

# Extracted as structured JSON
{
  "type": "table",
  "headers": ["Product", "Q1", "Q2", "Q3"],
  "rows": [
    ["Widget A", "$1,200", "$1,350", "$1,450"],
    ["Widget B", "$950", "$1,000", "$1,100"]
  ],
  "summary": "Quarterly sales figures for Widget A and B, showing consistent growth across three quarters."
}
            """)
    
    # GROUND TRUTH TAB
    with doc_tabs[4]:
        st.subheader("Ground Truth Dataset")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Upload Ground Truth for Evaluation")
            uploaded_gt = st.file_uploader("Choose a ground truth CSV file", type="csv")
            
            if uploaded_gt is not None:
                try:
                    df = pd.read_csv(uploaded_gt)
                    st.success("Ground truth file uploaded successfully!")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error: {e}")
            
            st.markdown("### Dataset Format")
            gt_format = st.selectbox(
                "Ground Truth Format",
                ["Question-Answer Pairs", "Question-Document Pairs", "Query-Passages", "Custom"]
            )
            
            if gt_format == "Question-Answer Pairs":
                st.markdown("Required columns: `question`, `ground_truth`")
                
            elif gt_format == "Question-Document Pairs":
                st.markdown("Required columns: `question`, `relevant_document_ids`")
                
            elif gt_format == "Query-Passages":
                st.markdown("Required columns: `query`, `relevant_passages`, `relevance_scores`")
                
            elif gt_format == "Custom":
                custom_columns = st.text_input("Custom columns (comma-separated)", "question,expected_answer,source,difficulty")
            
            # Additional ground truth options
            with st.expander("Additional Options"):
                st.checkbox("Split into train/test sets", value=False)
                st.checkbox("Generate synthetic test questions", value=False)
                st.checkbox("Calculate baseline metrics", value=True)
        
        with col2:
            with st.expander("ℹ️ About Ground Truth Dataset", expanded=True):
                info_tooltip("""
                **Ground Truth Dataset** provides a way to evaluate the performance of your RAG system.
                
                **Format Options:**
                - **Question-Answer Pairs**: Simple format with questions and expected answers
                - **Question-Document Pairs**: Links questions to relevant document IDs
                - **Query-Passages**: More detailed with specific passages and relevance scores
                - **Custom**: Define your own evaluation structure
                
                **Why It Matters:**
                - Provides objective measurement of RAG performance
                - Helps identify areas for improvement
                - Enables comparison between different configurations
                
                **Additional Features:**
                - **Train/Test Split**: Separate data for tuning vs. evaluation
                - **Synthetic Questions**: Generate additional test questions using LLMs
                - **Baseline Metrics**: Calculate initial performance benchmarks
                
                **Best Practices:**
                - Include a diverse set of question types
                - Cover different topics in your document collection
                - Include both simple and complex queries
                - Consider adding difficulty ratings to analyze performance by difficulty
                """)
            
            # Sample ground truth data
            st.markdown("#### Sample Ground Truth Dataset")
            st.dataframe(pd.DataFrame(sample_data))
            
            # Download template
            st.markdown("#### Download Template")
            csv = pd.DataFrame(sample_data).to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="ground_truth_template.csv">Download Template CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

#################################
# 2. EMBEDDING & REPRESENTATION
#################################
elif main_section == "2. Embedding & Representation":
    st.header("2. Embedding & Representation")
    st.markdown("Configure how document content is converted to vector representations")
    
    # Create tabs for different aspects of embeddings
    embedding_tabs = st.tabs(["Embedding Models", "Embedding Strategies", "Advanced Techniques", "Embedding Tuning"])
    
    # EMBEDDING MODELS TAB
    with embedding_tabs[0]:
        st.subheader("Embedding Models")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            embedding_provider = st.selectbox(
                "Embedding Provider",
                ["OpenAI", "Cohere", "Sentence Transformers", "Jina", "TensorFlow Hub", "Custom"]
            )
            
            # Different model options based on provider
            if embedding_provider == "OpenAI":
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
                )
                
                # Show model properties
                if embedding_model == "text-embedding-3-small":
                    dimensions = 1536
                    context_window = 8191
                    use_case = "General purpose, efficient"
                elif embedding_model == "text-embedding-3-large":
                    dimensions = 3072
                    context_window = 8191
                    use_case = "High accuracy, state-of-the-art"
                else:  # ada-002
                    dimensions = 1536
                    context_window = 8191
                    use_case = "Legacy model"
                
            elif embedding_provider == "Cohere":
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["embed-english-v3.0", "embed-multilingual-v3.0", "embed-english-light-v3.0"]
                )
                
                if "light" in embedding_model:
                    dimensions = 384
                else:
                    dimensions = 1024
                
                context_window = 512
                use_case = "Multilingual support" if "multilingual" in embedding_model else "English text"
                
            elif embedding_provider == "Sentence Transformers":
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["all-MiniLM-L6-v2", "multi-qa-mpnet-base-dot-v1", 
                     "all-mpnet-base-v2", "paraphrase-multilingual-mpnet-base-v2",
                     "gte-base", "gte-large", "sgpt-bloom-7b1-msmarco"]
                )
                
                # Model properties
                if embedding_model == "all-MiniLM-L6-v2":
                    dimensions = 384
                    context_window = 256
                    use_case = "Efficient general purpose"
                elif "mpnet" in embedding_model:
                    dimensions = 768
                    context_window = 512
                    use_case = "High quality, balanced performance"
                elif "gte" in embedding_model:
                    dimensions = 768 if "base" in embedding_model else 1024
                    context_window = 512
                    use_case = "General Text Embeddings, state-of-the-art"
                else:
                    dimensions = 768
                    context_window = 512
                    use_case = "Specialized model"
            
            elif embedding_provider == "Jina":
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["jina-embeddings-v2-base-en", "jina-embeddings-v2-small-en"]
                )
                dimensions = 768 if "base" in embedding_model else 512
                context_window = 8192
                use_case = "Long text, efficient"
                
            elif embedding_provider == "TensorFlow Hub":
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["Universal Sentence Encoder", "USE Multilingual", "BERT", "MuRIL (Indic Languages)"]
                )
                dimensions = 512 if "Universal" in embedding_model else 768
                context_window = 128 if "Universal" in embedding_model else 512
                use_case = "Multilingual" if "Multilingual" in embedding_model or "MuRIL" in embedding_model else "General"
                
            elif embedding_provider == "Custom":
                embedding_model = st.text_input("Model Name/Path", "path/to/local/model")
                dimensions = st.number_input("Embedding Dimensions", 32, 4096, 768)
                context_window = st.number_input("Context Window Size", 128, 32768, 512)
                use_case = st.text_input("Specialized Use Case", "Custom domain")
            
            # Display model properties
            st.markdown("### Model Properties")
            
            col_props1, col_props2 = st.columns(2)
            
            with col_props1:
                st.markdown(f"**Dimensions:** {dimensions}")
                st.markdown(f"**Context Window:** {context_window} tokens")
                
            with col_props2:
                st.markdown(f"**Optimized for:** {use_case}")
                
            # Dimension reduction option
            st.markdown("### Dimension Management")
            
            dim_reduction = st.checkbox("Enable dimension reduction")
            
            if dim_reduction:
                target_dims = st.slider("Target dimensions", 32, dimensions, dimensions // 2)
                reduction_method = st.selectbox(
                    "Reduction Method",
                    ["PCA", "UMAP", "t-SNE", "Random Projection", "Autoencoders"]
                )
            else:
                target_dims = dimensions
                
            # Model configuration
            st.markdown("### Embedding Configuration")
            
            batch_size = st.slider("Batch Size", 1, 256, 32, 
                                  help="Number of texts to embed in a single API call")
            
            normalize_embeddings = st.checkbox("Normalize embeddings (L2)", value=True,
                                            help="Make all embeddings have unit length")
            
            if embedding_provider in ["OpenAI", "Cohere"]:
                st.markdown("### API Settings")
                api_key = st.text_input(f"{embedding_provider} API Key", type="password", value="sk-...")
                api_base = st.text_input("API Base URL", value=f"https://api.{embedding_provider.lower()}.com/v1")
        
        with col2:
            with st.expander("ℹ️ About Embedding Models", expanded=True):
                info_tooltip("""
                **Embedding Models** convert text into numerical vectors that capture semantic meaning.
                
                **Provider Options:**
                - **OpenAI**: State-of-the-art models with excellent performance
                - **Cohere**: Strong multilingual support
                - **Sentence Transformers**: Open-source models with good balance of quality and speed
                - **Jina**: Optimized for long texts
                - **TensorFlow Hub**: Various models including multilingual options
                - **Custom**: Use your own models or fine-tuned versions
                
                **Key Parameters:**
                - **Dimensions**: Higher dimensions capture more information but use more storage
                - **Context Window**: Maximum text length the model can process
                - **Batch Size**: Process multiple texts at once for efficiency
                - **Normalization**: Makes vector similarity calculations more consistent
                
                **Dimension Reduction:**
                - **PCA**: Fast linear reduction, preserves global structure
                - **UMAP**: Better preserves local relationships, slower
                - **t-SNE**: Good for visualization, not recommended for retrieval
                - **Random Projection**: Very fast, works surprisingly well
                - **Autoencoders**: Neural network approach, can preserve nonlinear relationships
                
                **When to Use Dimension Reduction:**
                - Large document collections where storage is a concern
                - When faster similarity search is needed
                - If your vector database has dimension limitations
                """)
            
            # Visualization of embedding spaces
            st.markdown("#### Embedding Space Visualization")
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*_xCK0pJjByH_PcPoZ0pOKw.png", 
                     caption="Visualization of semantic document embeddings")
            
            # Use case recommendations
            st.markdown("#### Recommendations")
            if embedding_provider == "OpenAI" and embedding_model == "text-embedding-3-large":
                st.success("✅ Great choice for state-of-the-art performance")
            elif embedding_provider == "Sentence Transformers" and "gte" in embedding_model:
                st.success("✅ Excellent balance of performance and cost (free/self-hosted)")
            elif embedding_provider == "Sentence Transformers" and "all-MiniLM-L6-v2" in embedding_model:
                st.info("ℹ️ Good choice for efficiency, but consider larger models for better accuracy")
            elif embedding_provider == "Custom":
                st.warning("⚠️ Ensure custom model is properly trained for your domain")
    
    # EMBEDDING STRATEGIES TAB
    with embedding_tabs[1]:
        st.subheader("Embedding Strategies")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Embedding Levels")
            
            embedding_levels = st.multiselect(
                "Text Elements to Embed",
                ["Document", "Chunk", "Paragraph", "Sentence", "Entity"],
                default=["Chunk"]
            )
            
            # Hierarchical embedding options
            if len(embedding_levels) > 1:
                st.markdown("### Hierarchical Configuration")
                
                hierarchical_strategy = st.radio(
                    "Hierarchical Strategy",
                    ["Independent Embeddings", "Parent-Child Relationships", "Nested Representations"]
                )
                
                if hierarchical_strategy == "Parent-Child Relationships":
                    st.checkbox("Enable parent influence on child embeddings", value=True)
                    st.slider("Parent influence weight", 0.0, 1.0, 0.3, 0.05)
                
                if hierarchical_strategy == "Nested Representations":
                    st.checkbox("Enable cross-level attention", value=True)
                
            # Domain-specific embedding options
            st.markdown("### Domain Adaptation")
            
            domain_adaptation = st.checkbox("Enable domain-specific embeddings")
            
            if domain_adaptation:
                domain = st.selectbox(
                    "Domain",
                    ["Medical/Healthcare", "Legal", "Financial", "Technical/Scientific", "Academic", "Custom"]
                )
                
                if domain == "Medical/Healthcare":
                    domain_model = st.selectbox(
                        "Medical Embedding Model",
                        ["BiomedBERT", "PubMedBERT", "BioClinicalBERT", "BlueBERT"]
                    )
                
                elif domain == "Legal":
                    domain_model = st.selectbox(
                        "Legal Embedding Model",
                        ["Legal-BERT", "LexGLUE", "CaseLaw-BERT"]
                    )
                
                elif domain == "Financial":
                    domain_model = st.selectbox(
                        "Financial Embedding Model",
                        ["FinBERT", "BloombergGPT-Embeddings", "SEC-BERT"]
                    )
                
                elif domain == "Technical/Scientific":
                    domain_model = st.selectbox(
                        "Scientific Embedding Model",
                        ["SciBERT", "TechBERT", "ScholarBERT", "Specter"]
                    )
                
                elif domain == "Academic":
                    domain_model = st.selectbox(
                        "Academic Embedding Model",
                        ["Specter", "ScholarBERT", "AcademicBERT"]
                    )
                
                elif domain == "Custom":
                    domain_model = st.text_input("Custom Domain Model", "path/to/domain/model")
                    st.file_uploader("Upload domain-specific vocabulary", type=["txt", "json"])
            
            # Sparse-dense hybrid options
            st.markdown("### Hybrid Embeddings")
            
            hybrid_strategy = st.checkbox("Enable sparse-dense hybrid embeddings")
            
            if hybrid_strategy:
                sparse_method = st.selectbox(
                    "Sparse Representation Method",
                    ["BM25", "TF-IDF", "SPLADE", "DeepCT", "COIL"]
                )
                
                if sparse_method in ["SPLADE", "DeepCT", "COIL"]:
                    sparse_model = st.selectbox(
                        "Sparse Model",
                        ["SPLADE++", "distilSPLADE", "COIL-T5"]
                    )
                
                sparse_weight = st.slider("Sparse weight in hybrid retrieval", 0.0, 1.0, 0.3, 0.05)
        
        with col2:
            with st.expander("ℹ️ About Embedding Strategies", expanded=True):
                info_tooltip("""
                **Embedding Strategies** determine what text units are embedded and how they're represented.
                
                **Embedding Levels:**
                - **Document**: Entire document as one embedding (loses granularity)
                - **Chunk**: Standard approach for most RAG systems
                - **Paragraph**: Natural text divisions
                - **Sentence**: Fine-grained, good for precise QA
                - **Entity**: Special embeddings just for named entities
                
                **Hierarchical Embeddings:**
                - **Independent**: Each level embedded separately
                - **Parent-Child**: Child embeddings influenced by parents
                - **Nested**: Complex relationships between levels
                
                **Domain Adaptation:**
                - Uses specialized models trained on domain-specific texts
                - Dramatically improves performance for specialized content
                - Available for medical, legal, financial, and other domains
                
                **Hybrid Embeddings:**
                - **Dense**: Capture semantic meaning (concepts)
                - **Sparse**: Capture keywords and exact matches
                - **Hybrid**: Combine both for better retrieval
                
                **Sparse Methods:**
                - **BM25/TF-IDF**: Traditional keyword matching
                - **SPLADE**: Learned sparse representations
                - **DeepCT**: Context-aware term weighting
                - **COIL**: Contextualized interaction-based retrieval
                """)
            
            # Diagrams
            if len(embedding_levels) > 1:
                st.markdown("#### Hierarchical Embedding Example")
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*7__9XbxJ6tM28cqAGKkXOw.png", 
                         caption="Hierarchical document embeddings")
            
            if hybrid_strategy:
                st.markdown("#### Sparse-Dense Hybrid Retrieval")
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*wbvwQ_gBYoA82YlVrWUYKQ.png", 
                         caption="Hybrid retrieval combining dense and sparse representations")
            
            # Recommendations
            st.markdown("#### Strategy Recommendations")
            
            if "Chunk" in embedding_levels and not "Sentence" in embedding_levels:
                st.success("✅ Chunk-level embedding is a good default choice")
            
            if domain_adaptation:
                st.success(f"✅ Domain adaptation with {domain} models will improve accuracy significantly")
            
            if hybrid_strategy:
                st.success("✅ Hybrid sparse-dense embeddings will improve lexical matching")
            
            if len(embedding_levels) > 2:
                st.warning("⚠️ Using many embedding levels increases complexity and storage requirements")
    
    # ADVANCED TECHNIQUES TAB
    with embedding_tabs[2]:
        st.subheader("Advanced Embedding Techniques")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Representation Techniques")
            
            # Pooling strategies
            pooling_strategy = st.selectbox(
                "Pooling Strategy",
                ["Mean Pooling", "Max Pooling", "CLS Token", "Attention Pooling", "Weighted Mean"]
            )
            
            if pooling_strategy == "Weighted Mean":
                st.slider("Recency bias", 0.0, 1.0, 0.2, 0.05, 
                         help="Higher values give more weight to later tokens")
            
            if pooling_strategy == "Attention Pooling":
                st.selectbox("Attention Heads", [1, 2, 4, 8])
            
            # Special tokens handling
            st.markdown("### Special Token Handling")
            
            special_tokens = st.multiselect(
                "Special Token Treatment",
                ["Preserve code blocks", "Emphasize keywords", "Entity highlighting", "None"],
                default=["None"]
            )
            
            if "Preserve code blocks" in special_tokens:
                st.selectbox("Code block strategy", 
                           ["Separate embeddings", "Special tokens", "Enhanced weighting"])
                
            if "Emphasize keywords" in special_tokens:
                st.number_input("Keyword weight multiplier", 1.0, 5.0, 2.0, 0.1)
                
            if "Entity highlighting" in special_tokens:
                st.selectbox("Entity recognition model", 
                           ["spaCy", "NLTK", "Stanza", "Flair", "Custom"])
            
            # Cross-encoders
            st.markdown("### Cross-Encoder Integration")
            
            use_cross_encoders = st.checkbox("Use cross-encoders for reranking")
            
            if use_cross_encoders:
                cross_encoder_model = st.selectbox(
                    "Cross-Encoder Model",
                    ["ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-12-v2", "ms-marco-electra-base", "Custom"]
                )
                
                cross_encoder_batch = st.slider("Cross-encoder batch size", 4, 128, 16)
                
                st.checkbox("Cache cross-encoder results", value=True)
            
            # ColBERT-style late interaction
            st.markdown("### Late Interaction")
            
            late_interaction = st.checkbox("Enable ColBERT-style late interaction")
            
            if late_interaction:
                token_embeddings = st.slider("Max tokens to index", 32, 512, 128)
                interaction_method = st.selectbox(
                    "Token Interaction Method", 
                    ["MaxSim", "Sum MaxSim", "Attention Matrix"]
                )
        
        with col2:
            with st.expander("ℹ️ About Advanced Techniques", expanded=True):
                info_tooltip("""
                **Advanced Embedding Techniques** can significantly improve retrieval quality.
                
                **Pooling Strategies:**
                - **Mean Pooling**: Average of all token embeddings (balanced)
                - **Max Pooling**: Maximum value for each dimension (captures salient features)
                - **CLS Token**: Use the classification token (model-specific)
                - **Attention Pooling**: Learnable weighted average (highest quality)
                - **Weighted Mean**: Biases toward specific parts of text (e.g., beginning/end)
                
                **Special Token Handling:**
                - **Code Blocks**: Preserves programming syntax and structure
                - **Keywords**: Gives higher weight to important terms
                - **Entity Highlighting**: Special treatment for named entities
                
                **Cross-Encoders:**
                - Models that consider query and document together
                - Much more accurate than bi-encoders (standard embeddings)
                - Used for reranking due to computational cost
                - Can significantly boost retrieval precision
                
                **Late Interaction (ColBERT):**
                - Stores token-level embeddings instead of pooled embeddings
                - Enables fine-grained matching between query and document tokens
                - Provides better alignment for keyword matching
                - Higher storage requirements but superior retrieval quality
                
                **When to Use:**
                - Cross-encoders: When precision is critical and queries are limited
                - Late interaction: When fine-grained matching matters
                - Special token handling: For code documentation or technical content
                """)
            
            # Diagrams
            st.markdown("#### Late Interaction (ColBERT) Visualization")
            st.image("https://blog.vespa.ai/assets/2021-12-10-pretrained-transformer-language-models-for-search-part-4/colbert_model_architecture.png",
                    caption="ColBERT's token-level interactions between query and document")
            
            # Recommendations
            st.markdown("#### Technique Recommendations")
            
            if pooling_strategy == "Mean Pooling":
                st.info("ℹ️ Mean pooling is a good default, but consider attention pooling for better quality")
            
            if pooling_strategy == "Attention Pooling":
                st.success("✅ Attention pooling typically provides the best quality embeddings")
            
            if use_cross_encoders:
                st.success("✅ Cross-encoders will significantly improve ranking quality")
                st.warning("⚠️ But they increase computational cost - use for reranking only")
            
            if late_interaction:
                st.success("✅ Late interaction provides excellent retrieval quality")
                st.warning("⚠️ But requires more storage and computational resources")
    
    # EMBEDDING TUNING TAB
    with embedding_tabs[3]:
        st.subheader("Embedding Tuning & Optimization")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Fine-Tuning Options")
            
            enable_finetuning = st.checkbox("Enable embedding fine-tuning")
            
            if enable_finetuning:
                tuning_method = st.selectbox(
                    "Fine-Tuning Method",
                    ["Contrastive Learning", "Domain Adaptation", "Task-Specific Tuning", "Knowledge Distillation"]
                )
                
                if tuning_method == "Contrastive Learning":
                    loss_function = st.selectbox(
                        "Contrastive Loss Function",
                        ["InfoNCE", "Triplet Loss", "Multiple Negatives", "Hard Negatives"]
                    )
                    
                    if loss_function == "Hard Negatives":
                        st.slider("Hard negative mining ratio", 0.1, 0.9, 0.5, 0.05)
                    
                    st.checkbox("Use in-batch negatives", value=True)
                    
                elif tuning_method == "Domain Adaptation":
                    target_domain = st.text_input("Target Domain Description", "Medical research papers")
                    st.file_uploader("Upload domain documents", accept_multiple_files=True)
                    st.slider("Adaptation strength", 0.1, 1.0, 0.7, 0.05)
                
                elif tuning_method == "Task-Specific Tuning":
                    task = st.selectbox(
                        "Target Task",
                        ["Question Answering", "Document Similarity", "Passage Ranking", "Entity Search"]
                    )
                    st.file_uploader("Upload task examples", accept_multiple_files=True)
                
                elif tuning_method == "Knowledge Distillation":
                    teacher_model = st.selectbox(
                        "Teacher Model",
                        ["text-embedding-3-large", "cohere-embed-english-v3.0", "Custom"]
                    )
                    if teacher_model == "Custom":
                        st.text_input("Custom teacher model path")
            
            # Training parameters
            if enable_finetuning:
                st.markdown("### Training Configuration")
                
                col_train1, col_train2 = st.columns(2)
                
                with col_train1:
                    st.number_input("Epochs", 1, 100, 3)
                    st.number_input("Batch Size", 8, 512, 64)
                    
                with col_train2:
                    st.number_input("Learning Rate", 1e-6, 1e-2, 1e-5, format="%.6f")
                    st.selectbox("Optimizer", ["AdamW", "Adam", "SGD"])
            
            # Active learning
            st.markdown("### Active Learning")
            
            active_learning = st.checkbox("Enable active learning")
            
            if active_learning:
                feedback_method = st.selectbox(
                    "Feedback Collection Method",
                    ["Query-Document Relevance", "Query Reformulation", "Result Ranking", "Explicit Feedback"]
                )
                
                update_frequency = st.selectbox(
                    "Model Update Frequency",
                    ["Daily", "Weekly", "Monthly", "After N Feedbacks"]
                )
                
                if update_frequency == "After N Feedbacks":
                    st.number_input("N Feedbacks", 10, 1000, 100)
                
                st.checkbox("Enable online learning", value=False)
        
        with col2:
            with st.expander("ℹ️ About Embedding Tuning", expanded=True):
                info_tooltip("""
                **Embedding Tuning** adapts pretrained models to your specific domain or task.
                
                **Fine-Tuning Methods:**
                - **Contrastive Learning**: Teaches the model which texts are similar/dissimilar
                - **Domain Adaptation**: Specializes the model for your content domain
                - **Task-Specific**: Optimizes for specific tasks like QA or ranking
                - **Knowledge Distillation**: Transfers knowledge from larger models
                
                **Loss Functions:**
                - **InfoNCE**: Standard contrastive loss
                - **Triplet Loss**: Works with positive and negative examples
                - **Multiple Negatives**: Uses many negative examples for each positive
                - **Hard Negatives**: Focuses on challenging examples
                
                **Active Learning:**
                - Continuously improves embeddings based on user feedback
                - Prioritizes learning from difficult or ambiguous cases
                - Enables the system to adapt to user needs over time
                
                **Benefits:**
                - 10-30% improvement in retrieval quality
                - Better handling of domain-specific terminology
                - More accurate semantic understanding
                
                **When to Use:**
                - With specialized domain content (medical, legal, etc.)
                - When general embeddings underperform
                - For applications requiring high precision
                """)
            
            # Diagrams
            if enable_finetuning:
                st.markdown("#### Fine-Tuning Process")
                
                if tuning_method == "Contrastive Learning":
                    st.image("https://miro.medium.com/v2/resize:fit:1400/1*xwBWFLp8XPbDTxEJDXDWQQ.png",
                            caption="Contrastive learning with positive and negative pairs")
                
                elif tuning_method == "Knowledge Distillation":
                    st.image("https://production-media.paperswithcode.com/methods/Screen_Shot_2020-05-31_at_11.13.15_PM.png",
                            caption="Knowledge distillation from teacher to student model")
            
            # Recommendations
            st.markdown("#### Tuning Recommendations")
            
            if enable_finetuning:
                if tuning_method == "Contrastive Learning":
                    st.success("✅ Contrastive learning is excellent for improving similarity quality")
                
                if tuning_method == "Domain Adaptation" and loss_function == "Hard Negatives":
                    st.success("✅ Domain adaptation with hard negatives often gives the best results")
            
            if active_learning:
                st.success("✅ Active learning will help your system improve over time")
                st.info("ℹ️ Consider starting with weekly updates and adjusting based on feedback volume")

#############################
# 3. CHUNKING & INDEXING
#############################
elif main_section == "3. Chunking & Indexing":
    st.header("3. Chunking & Indexing")
    st.markdown("Configure how documents are split into retrievable chunks")
    
    # Create tabs for chunking options
    chunk_tabs = st.tabs(["Chunking Strategies", "Advanced Chunking", "Cross-References", "Metadata"])
    
    # CHUNKING STRATEGIES TAB
    with chunk_tabs[0]:
        st.subheader("Basic Chunking Strategies")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Primary chunking strategy
            chunker_option = st.selectbox(
                "Primary Chunking Strategy",
                ["Recursive", "Semantic", "Sliding Window", "SDPM", "Fixed Size", "Paragraph", 
                 "Sentence", "Hybrid", "Agentic", "RAPTOR", "Contextualized"]
            )
            
            # Different options based on chunking strategy
            if chunker_option == "Recursive":
                st.markdown("### Recursive Chunker Configuration")
                chunk_size = st.slider("Maximum chunk size (characters)", 100, 5000, 1000)
                chunk_overlap = st.slider("Chunk overlap (characters)", 0, 500, 100)
                
                st.markdown("### Splitting Preferences")
                split_level = st.multiselect(
                    "Preferred Split Boundaries",
                    ["Paragraph", "Sentence", "Newline", "Custom Delimiter"],
                    default=["Paragraph", "Sentence"]
                )
                
                if "Custom Delimiter" in split_level:
                    st.text_input("Custom delimiter", "---")
            
            elif chunker_option == "Semantic":
                st.markdown("### Semantic Chunker Configuration")
                similarity_threshold = st.slider("Similarity threshold", 0.5, 0.99, 0.75, 0.01)
                min_chunk_size = st.number_input("Minimum chunk size (characters)", 50, 1000, 200)
                max_chunk_size = st.number_input("Maximum chunk size (characters)", 500, 10000, 2000)
                
                st.markdown("### Semantic Model")
                semantic_model = st.selectbox(
                    "Semantic Similarity Model",
                    ["Same as Embedding Model", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "Custom"]
                )
                
                if semantic_model == "Custom":
                    st.text_input("Custom model path", "path/to/model")
            
            elif chunker_option == "Sliding Window":
                st.markdown("### Sliding Window Configuration")
                window_size = st.slider("Window size (characters)", 100, 5000, 1000)
                step_size = st.slider("Step size (characters)", 50, 2000, 500)
                
                st.markdown("### Window Anchoring")
                anchor_method = st.selectbox(
                    "Window Anchoring Method",
                    ["Regular Intervals", "Sentence Boundaries", "Paragraph Boundaries"]
                )
            
            elif chunker_option == "SDPM":
                st.markdown("### SDPM Configuration")
                initial_size = st.slider("Initial chunk size", 100, 5000, 1000)
                merge_threshold = st.slider("Merge similarity threshold", 0.5, 0.99, 0.75, 0.01)
                
                st.markdown("### Merging Strategy")
                merge_strategy = st.selectbox(
                    "Chunk Merging Strategy",
                    ["Pairwise", "Agglomerative", "Iterative"]
                )
                
                if merge_strategy == "Agglomerative":
                    st.selectbox("Linkage Method", ["Single", "Complete", "Average", "Ward"])
            
            elif chunker_option == "Fixed Size":
                st.markdown("### Fixed Size Configuration")
                size_unit = st.selectbox("Size Unit", ["Characters", "Words", "Tokens"])
                
                if size_unit == "Characters":
                    chunk_size = st.slider("Chunk size (characters)", 100, 5000, 1000)
                elif size_unit == "Words":
                    chunk_size = st.slider("Chunk size (words)", 20, 1000, 200)
                else:  # Tokens
                    chunk_size = st.slider("Chunk size (tokens)", 20, 2000, 512)
                    tokenizer = st.selectbox("Tokenizer", ["GPT", "BERT", "Claude", "Generic"])
                
                respect_boundaries = st.checkbox("Respect sentence boundaries", value=True)
            
            elif chunker_option == "Paragraph":
                st.markdown("### Paragraph Chunker Configuration")
                min_para_length = st.number_input("Minimum paragraph length (characters)", 10, 500, 100)
                max_para_length = st.number_input("Maximum paragraph length (characters)", 500, 10000, 3000)
                
                st.markdown("### Paragraph Detection")
                para_detection = st.selectbox(
                    "Paragraph Detection Method",
                    ["Double Newline", "Indentation", "Machine Learning", "Hybrid"]
                )
                
                merge_small_paras = st.checkbox("Merge small paragraphs", value=True)
            
            elif chunker_option == "Sentence":
                st.markdown("### Sentence Chunker Configuration")
                sent_batch_size = st.slider("Sentences per chunk", 1, 20, 5)
                
                st.markdown("### Sentence Detection")
                sent_detector = st.selectbox(
                    "Sentence Detection",
                    ["NLTK", "spaCy", "Stanza", "Custom Rules"]
                )
                
                if sent_detector == "Custom Rules":
                    st.text_area("Custom sentence break patterns (regex)", r"\. |\? |! |\.\n|\?\n|!\n")
            
            elif chunker_option == "Hybrid":
                st.markdown("### Hybrid Chunker Configuration")
                primary_strategy = st.selectbox(
                    "Primary Strategy",
                    ["Paragraph", "Fixed Size", "Semantic"]
                )
                secondary_strategy = st.selectbox(
                    "Secondary Strategy",
                    ["Sentence", "Sliding Window", "Recursive"]
                )
                
                st.markdown("### Hybrid Parameters")
                primary_threshold = st.slider("Primary threshold", 100, 5000, 1000, 
                                           help="Size threshold to trigger secondary strategy")
                
                st.checkbox("Apply secondary strategy to all chunks", value=False)
            
            elif chunker_option == "Agentic":
                st.markdown("### Agentic Chunker Configuration")
                agent_model = st.selectbox(
                    "LLM for Chunking",
                    ["gpt-4", "gpt-3.5-turbo", "claude-instant", "Claude 3 Haiku", "llama-3-8b"]
                )
                
                st.markdown("### Chunking Criteria")
                chunking_criteria = st.multiselect(
                    "Agent Chunking Criteria",
                    ["Topic Boundaries", "Semantic Coherence", "Information Density", 
                     "Logical Structure", "Question Relevance"],
                    default=["Topic Boundaries", "Semantic Coherence"]
                )
                
                st.number_input("Maximum chunks to create", 5, 100, 20)
                st.number_input("Minimum chunk size (characters)", 50, 1000, 200)
            
            elif chunker_option == "RAPTOR":
                st.markdown("### RAPTOR Chunker Configuration")
                max_tokens = st.number_input("Maximum tokens per chunk", 128, 2048, 512)
                num_layers = st.number_input("Hierarchy levels", 1, 5, 3)
                top_k = st.number_input("Similar nodes for grouping", 1, 10, 3)
                
                st.markdown("### Topic Modeling")
                topic_method = st.selectbox(
                    "Topic Identification Method",
                    ["LLM-based", "KeyBERT", "BERTopic", "LDA", "Word Frequency"]
                )
                
                if topic_method == "LLM-based":
                    st.selectbox("LLM for Topic Identification", 
                               ["gpt-3.5-turbo", "claude-instant", "LaMDA", "Custom"])
            
            elif chunker_option == "Contextualized":
                st.markdown("### Contextualized Chunker Configuration# filepath: st.py")
            elif chunker_option == "Contextualized":
                st.markdown("### Contextualized Chunker Configuration")
                base_chunker = st.selectbox(
                    "Base Chunking Method",
                    ["Recursive", "Paragraph", "Fixed Size", "Semantic"]
                )
                
                if base_chunker == "Recursive":
                    chunk_size = st.slider("Base chunk size", 500, 5000, 1500)
                    chunk_overlap = st.slider("Chunk overlap", 0, 500, 100)
                
                st.markdown("### Context Generation")
                context_method = st.selectbox(
                    "Context Generation Method",
                    ["LLM Summarization", "Parent-Child Context", "Document Structure", "Entity-Based"]
                )
                
                if context_method == "LLM Summarization":
                    context_model = st.selectbox(
                        "Context Generation LLM",
                        ["gpt-3.5-turbo", "claude-instant", "Claude 3 Haiku", "llama-3-8b"]
                    )
                    
                    st.slider("Context length (tokens)", 50, 500, 200)
                
                st.checkbox("Include document metadata in context", value=True)
                st.checkbox("Include position information", value=True)
        
        # Show chunking examples or recommendations
        with col2:
            with st.expander("ℹ️ About Chunking Strategies", expanded=True):
                info_tooltip("""
                **Chunking Strategies** determine how documents are split into retrievable units.
                
                **Strategy Options:**
                
                **Recursive**: Splits text at natural boundaries while respecting max size.
                - Simple and efficient
                - Preserves natural text boundaries
                - Good general-purpose approach
                
                **Semantic**: Groups similar sentences based on embedding similarity.
                - Creates more coherent chunks
                - Better for question-answering
                - Handles variable content density
                
                **Sliding Window**: Creates overlapping chunks of fixed size.
                - Simple implementation
                - Ensures context continuity
                - Good for dense, uniform content
                
                **SDPM**: Two-pass approach that merges similar adjacent chunks.
                - Balances size constraints with semantic coherence
                - Good for documents with varying section lengths
                
                **Fixed Size**: Strict size limits regardless of content.
                - Predictable chunk sizes
                - Easier for token limit management
                - Less semantic awareness
                
                **Paragraph**: Uses natural paragraph breaks.
                - Preserves author's intended structure
                - Variable chunk sizes
                - Simple implementation
                
                **Sentence**: Works at sentence level, grouping sentences.
                - Very fine-grained control
                - Good for precise QA
                - Can create many small chunks
                
                **Hybrid**: Combines multiple strategies.
                - Adaptable to document structure
                - Combines benefits of different approaches
                - More complex implementation
                
                **Agentic**: Uses LLM to determine logical chunks.
                - Adapts to document's inherent organization
                - Creates highly coherent chunks
                - Good for complex documents
                
                **RAPTOR**: Hierarchical chunking with topic modeling.
                - Creates topic-based structure
                - Tracks relationships between chunks
                - Best for large, diverse document collections
                
                **Contextualized**: Adds document-aware context to each chunk.
                - Enriches chunks with their document context
                - Good for documents where global context matters
                """)
                
            # Show example based on selected chunker
            st.markdown("#### Example Output")
            
            if chunker_option == "Recursive":
                st.code("""
# Original text
This is the first paragraph with important information.

This is the second paragraph with different content.
It continues for a bit with more details.

# Chunked output
[
  {
    "text": "This is the first paragraph with important information.",
    "metadata": {"start": 0, "end": 55}
  },
  {
    "text": "This is the second paragraph with different content.\\nIt continues for a bit with more details.",
    "metadata": {"start": 57, "end": 143}
  }
]
                """)
                
            elif chunker_option == "Semantic":
                st.code("""
# Chunked by semantic similarity
[
  {
    "text": "Machine learning is a subset of AI. Deep learning is a type of machine learning.",
    "metadata": {"similarity_group": "AI concepts", "avg_similarity": 0.87}
  },
  {
    "text": "Neural networks have neurons and layers. CNNs are used for image processing.",
    "metadata": {"similarity_group": "Neural networks", "avg_similarity": 0.92}
  }
]
                """)
                
            elif chunker_option in ["RAPTOR", "Agentic"]:
                st.code("""
# Hierarchical chunk with topics
{
  "text": "The treatment showed a 35% reduction in symptoms...",
  "metadata": {
    "chunk_type": "raptor",
    "topic": "Clinical efficacy results",
    "parent_id": "clinical_trial_results",
    "key_concepts": ["symptom reduction", "treatment efficacy", "statistical significance"],
    "tree_level": 2
  }
}
                """)
                
            elif chunker_option == "Contextualized":
                st.code("""
# Contextualized chunk
{
  "text": "The gradient descent algorithm updates weights iteratively...",
  "context": "This section explains optimization algorithms for neural networks. Located in Chapter 4: Training Deep Networks.",
  "metadata": {
    "chunk_type": "contextualized",
    "original_length": 120,
    "context_length": 95,
    "document": "Deep Learning Textbook.pdf",
    "section": "Optimization Methods"
  }
}
                """)
                
            # Show recommendations
            st.markdown("#### Recommendations")
            
            if chunker_option == "Recursive":
                st.success("✅ Good general-purpose choice - simple and effective")
                
            if chunker_option == "Semantic":
                st.success("✅ Excellent for maintaining conceptual coherence")
                st.info("ℹ️ Requires more processing time than simpler methods")
                
            if chunker_option in ["Agentic", "RAPTOR"]:
                st.success("✅ Advanced approach with superior chunk quality")
                st.warning("⚠️ More computationally expensive and complex")
                
            if chunker_option == "Fixed Size":
                st.info("ℹ️ Simple but may break content at awkward points")
                st.info("ℹ️ Consider Recursive instead for better chunk boundaries")
    
    # ADVANCED CHUNKING TAB
    with chunk_tabs[1]:
        st.subheader("Advanced Chunking Features")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Adaptive chunking
            st.markdown("### Adaptive Chunking")
            
            use_adaptive = st.checkbox("Enable adaptive chunking")
            
            if use_adaptive:
                adaptive_method = st.selectbox(
                    "Adaptation Method",
                    ["Content Density", "Semantic Complexity", "Information Entropy", "Hybrid Metrics"]
                )
                
                if adaptive_method == "Content Density":
                    st.slider("Density threshold", 0.1, 1.0, 0.5, 0.05)
                    
                elif adaptive_method == "Semantic Complexity":
                    st.slider("Complexity threshold", 0.1, 1.0, 0.5, 0.05)
                    st.selectbox("Complexity model", ["BERT", "RoBERTa", "Linguistic Features"])
                    
                elif adaptive_method == "Information Entropy":
                    st.slider("Entropy threshold", 0.1, 5.0, 2.5, 0.1)
                    
                st.checkbox("Apply adaptive refinement iteratively", value=True)
            
            # Discourse-aware chunking
            st.markdown("### Discourse-Aware Chunking")
            
            discourse_aware = st.checkbox("Enable discourse-aware chunking")
            
            if discourse_aware:
                discourse_method = st.selectbox(
                    "Discourse Analysis Method",
                    ["Rhetorical Structure", "Coreference Chains", "Topic Transitions", "Discourse Markers"]
                )
                
                if discourse_method == "Rhetorical Structure":
                    st.selectbox("RST Model", ["spaCy RST", "NLTK RST", "Custom"])
                    
                elif discourse_method == "Coreference Chains":
                    st.selectbox("Coreference Resolution", ["AllenNLP", "spaCy", "NeuralCoref"])
                    st.checkbox("Preserve complete coreference chains", value=True)
                    
                elif discourse_method == "Topic Transitions":
                    st.slider("Topic change threshold", 0.1, 1.0, 0.4, 0.05)
                
                st.checkbox("Visualize discourse structure", value=False)
            
            # Multi-scale chunking
            st.markdown("### Multi-Scale Chunking")
            
            multiscale = st.checkbox("Enable multi-scale chunking")
            
            if multiscale:
                scales = st.multiselect(
                    "Chunking Scales",
                    ["Document", "Section", "Paragraph", "Sentence", "Phrase"],
                    default=["Section", "Paragraph"]
                )
                
                st.markdown("### Scale Relationships")
                relationship_type = st.radio(
                    "Scale Relationship Type",
                    ["Hierarchical", "Independent", "Cross-Referenced"]
                )
                
                if relationship_type == "Hierarchical":
                    st.checkbox("Parent chunk influences child embeddings", value=True)
                    
                elif relationship_type == "Cross-Referenced":
                    st.checkbox("Add explicit cross-references between scales", value=True)
            
            # Content-type specific chunking
            st.markdown("### Content-Type Specific Chunking")
            
            content_specific = st.checkbox("Enable content-type specific chunking")
            
            if content_specific:
                content_types = st.multiselect(
                    "Content Types to Detect",
                    ["Code", "Tables", "Lists", "Equations", "Citations", "Headers", "Footnotes"],
                    default=["Code", "Tables"]
                )
                
                if "Code" in content_types:
                    code_handling = st.radio(
                        "Code Block Handling",
                        ["Preserve as Single Chunk", "Chunk by Function/Class", "Include in Surrounding Text"]
                    )
                    
                if "Tables" in content_types:
                    table_handling = st.radio(
                        "Table Handling",
                        ["Table as Single Chunk", "Row-Based Chunking", "Semantic Table Chunking"]
                    )
        
        with col2:
            with st.expander("ℹ️ About Advanced Chunking", expanded=True):
                info_tooltip("""
                **Advanced Chunking Features** provide more sophisticated ways to split documents based on content characteristics.
                
                **Adaptive Chunking:**
                - Adjusts chunk size based on content properties
                - Dense, complex content gets smaller chunks
                - Simple content gets larger chunks
                - Improves retrieval by balancing information density
                
                **Discourse-Aware Chunking:**
                - Preserves linguistic and logical relationships
                - Respects rhetorical structure of text
                - Maintains coreference chains (pronouns and their referents)
                - Creates more natural, coherent chunks
                
                **Multi-Scale Chunking:**
                - Creates chunks at multiple granularity levels
                - Enables "zooming in/out" during retrieval
                - Better handles nested information
                - Supports hierarchical navigation of content
                
                **Content-Type Specific Chunking:**
                - Applies different strategies to different content types
                - Special handling for code, tables, lists, equations
                - Preserves structural elements that need special treatment
                - Improves retrieval for specialized content
                
                **When to Use:**
                - Adaptive: For documents with varying information density
                - Discourse-aware: For narrative text with complex relationships
                - Multi-scale: For large, structured documents like textbooks
                - Content-specific: For technical documentation or mixed content
                """)
            
            # Example for the selected feature
            st.markdown("#### Example Application")
            
            if use_adaptive:
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*3q9_M12A-tDHTs0y9uJJdw.png",
                        caption="Adaptive chunking based on content density")
                
                st.code("""
# Adaptive chunking example
[
  {
    "text": "Simple introductory paragraph with basic concepts...",
    "metadata": {
      "complexity": "low",
      "chunk_size": 2000,  # Larger chunk for simple content
      "density_score": 0.31
    }
  },
  {
    "text": "Complex technical details with dense information...",
    "metadata": {
      "complexity": "high",
      "chunk_size": 800,   # Smaller chunk for complex content
      "density_score": 0.87
    }
  }
]
                """)
            
            elif discourse_aware:
                st.code("""
# Discourse-aware chunking preserves coreference
[
  {
    "text": "John Smith presented the new research findings. He emphasized the importance of methodology.",
    "metadata": {
      "coreference_chains": [{"John Smith": "He"}],
      "discourse_markers": ["emphasized"]
    }
  }
]
                """)
            
            elif multiscale:
                st.code("""
# Multi-scale chunking example
{
  "document": {
    "id": "doc_1",
    "title": "Introduction to Machine Learning",
    "embedding": [0.1, 0.2, ...]
  },
  "sections": [
    {
      "id": "sec_1",
      "title": "Supervised Learning",
      "embedding": [0.15, 0.25, ...],
      "parent_id": "doc_1",
      "paragraphs": [
        {
          "id": "para_1",
          "text": "Supervised learning uses labeled data...",
          "embedding": [0.17, 0.22, ...],
          "parent_id": "sec_1"
        }
      ]
    }
  ]
}
                """)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if use_adaptive:
                st.success("✅ Adaptive chunking works well for documents with varying complexity")
                st.info("ℹ️ Content Density is the simplest approach to implement")
                
            if discourse_aware:
                st.success("✅ Discourse-aware chunking improves coherence significantly")
                st.warning("⚠️ But adds computational overhead - use for narrative text")
                
            if multiscale:
                st.success("✅ Multi-scale chunking enables more flexible retrieval")
                st.info("ℹ️ Best paired with hierarchical embedding strategies")
                
            if content_specific:
                st.success("✅ Content-type specific chunking is valuable for mixed content")
                st.info("ℹ️ Particularly important for technical documentation")
    
    # CROSS-REFERENCES TAB
    with chunk_tabs[2]:
        st.subheader("Cross-References & Relationships")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable cross-references
            enable_xrefs = st.checkbox("Enable cross-references between chunks")
            
            if enable_xrefs:
                # Cross-reference types
                xref_types = st.multiselect(
                    "Cross-Reference Types",
                    ["Semantic Similarity", "Citation Links", "Entity References", "Sequential", "Hierarchical"],
                    default=["Semantic Similarity", "Sequential"]
                )
                
                if "Semantic Similarity" in xref_types:
                    similarity_threshold = st.slider("Similarity threshold", 0.5, 1.0, 0.8, 0.01)
                    max_similar_chunks = st.slider("Max similar chunks per reference", 1, 20, 5)
                    
                if "Citation Links" in xref_types:
                    citation_styles = st.multiselect(
                        "Citation Styles to Detect",
                        ["Academic ([Author, Year])", "Numbered ([1], [2])", "Footnotes", "URLs/Hyperlinks"],
                        default=["Academic ([Author, Year])", "URLs/Hyperlinks"]
                    )
                    
                if "Entity References" in xref_types:
                    entity_types = st.multiselect(
                        "Entity Types to Track",
                        ["Person", "Organization", "Location", "Date", "Technical Terms", "Custom Entities"],
                        default=["Person", "Organization", "Technical Terms"]
                    )
                    
                    if "Custom Entities" in entity_types:
                        st.text_area("Custom entity patterns (one per line)", "API_NAME\nFUNCTION_NAME")
            
            # Knowledge graph integration
            st.markdown("### Knowledge Graph")
            
            build_kg = st.checkbox("Build knowledge graph from chunks")
            
            if build_kg:
                kg_elements = st.multiselect(
                    "Knowledge Graph Elements",
                    ["Entities", "Concepts", "Events", "Relations", "Document Structure"],
                    default=["Entities", "Relations"]
                )
                
                kg_extraction = st.selectbox(
                    "Knowledge Extraction Method",
                    ["Rule-based", "NER + Relation Extraction", "Open IE", "LLM-based"]
                )
                
                if kg_extraction == "LLM-based":
                    kg_model = st.selectbox(
                        "Knowledge Extraction Model",
                        ["gpt-3.5-turbo", "gpt-4", "Claude 3 Opus", "Llama-3-70B"]
                    )
                
                st.checkbox("Visualize knowledge graph", value=True)
            
            # Citation tracking
            st.markdown("### Citation Tracking")
            
            track_citations = st.checkbox("Track citations and references")
            
            if track_citations:
                citation_method = st.selectbox(
                    "Citation Detection Method",
                    ["Pattern Matching", "Machine Learning", "Document Structure Analysis", "LLM-based"]
                )
                
                bibliography_handling = st.radio(
                    "Bibliography Handling",
                    ["Separate Index", "Include with Citing Chunks", "Both"]
                )
                
                st.checkbox("Resolve citation targets", value=True)
                st.checkbox("Link to external sources", value=False)
        
        with col2:
            with st.expander("ℹ️ About Cross-References", expanded=True):
                info_tooltip("""
                **Cross-References** create explicit connections between related chunks, enhancing retrieval.
                
                **Cross-Reference Types:**
                - **Semantic Similarity**: Links chunks with similar content
                - **Citation Links**: Connects text references to cited content
                - **Entity References**: Tracks mentions of the same entities
                - **Sequential**: Adjacent chunks in original document
                - **Hierarchical**: Parent-child relationships
                
                **Knowledge Graph Integration:**
                - Creates structured representation of document knowledge
                - Extracts entities, relationships, and concepts
                - Enables graph-based retrieval and navigation
                - Provides rich context for ambiguous queries
                
                **Citation Tracking:**
                - Identifies and resolves citations in the text
                - Connects citing text to referenced content
                - Improves evidence retrieval and attribution
                - Helps preserve source relationships
                
                **Benefits:**
                - More comprehensive retrieval (beyond vector similarity)
                - Better handling of complex relationships
                - Improved explanation and context
                - Support for multi-hop reasoning
                
                **When to Use:**
                - Academic or research content
                - Technical documentation with many references
                - Legal documents with citations
                - Any content where relationships matter
                """)
            
            # Example
            st.markdown("#### Example Cross-Referenced Chunks")
            
            if enable_xrefs:
                st.code("""
# Cross-referenced chunks example
{
  "chunks": [
    {
      "id": "chunk_1",
      "text": "Neural networks consist of layers of neurons...",
      "cross_references": [
        {"type": "semantic_similarity", "target_id": "chunk_7", "score": 0.92},
        {"type": "sequential", "target_id": "chunk_2", "direction": "next"},
        {"type": "entity", "entity": "neural network", "target_ids": ["chunk_7", "chunk_12"]}
      ]
    },
    {
      "id": "chunk_7",
      "text": "Deep learning models use multiple neural network layers...",
      "cross_references": [
        {"type": "semantic_similarity", "target_id": "chunk_1", "score": 0.92},
        {"type": "citation", "reference": "[Smith, 2019]", "target_id": "ref_15"}
      ]
    }
  ]
}
                """)
                
            if build_kg:
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*OOB6tYnjE6JucEXldWG2JQ.png",
                       caption="Knowledge graph extracted from document chunks")
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_xrefs:
                st.success("✅ Cross-references significantly improve retrieval quality")
                st.success("✅ Particularly valuable for complex, interconnected content")
                
            if "Entity References" in xref_types if enable_xrefs else []:
                st.success("✅ Entity tracking helps connect related information across documents")
                
            if build_kg:
                st.success("✅ Knowledge graphs enable more sophisticated retrieval patterns")
                st.warning("⚠️ But add significant processing overhead")
                
            if not enable_xrefs and not build_kg and not track_citations:
                st.info("ℹ️ Cross-references can significantly improve retrieval for complex documents")
                st.info("ℹ️ Consider enabling at least basic sequential references")
    
    # METADATA TAB
    with chunk_tabs[3]:
        st.subheader("Chunk Metadata")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("### Select metadata to extract and include")
            
            metadata_categories = {
                "Basic": {
                    "source_file": st.checkbox("Source filename", value=True),
                    "file_type": st.checkbox("File type", value=True),
                    "creation_date": st.checkbox("Creation date", value=True),
                    "last_modified": st.checkbox("Last modified date", value=True)
                },
                "Content": {
                    "title": st.checkbox("Document title", value=True),
                    "author": st.checkbox("Author", value=True),
                    "section": st.checkbox("Section title", value=True),
                    "page_count": st.checkbox("Page count", value=False)
                },
                "Position": {
                    "page_number": st.checkbox("Page number", value=True),
                    "start_char": st.checkbox("Start character position", value=True),
                    "end_char": st.checkbox("End character position", value=True),
                    "chunk_index": st.checkbox("Chunk index", value=True)
                },
                "Semantic": {
                    "topics": st.checkbox("Topics", value=True),
                    "entities": st.checkbox("Entities", value=True),
                    "keywords": st.checkbox("Keywords", value=True),
                    "summary": st.checkbox("Summary", value=False)
                }
            }
            
            # Custom metadata
            st.markdown("### Custom Metadata")
            
            custom_metadata = st.checkbox("Add custom metadata fields")
            
            if custom_metadata:
                custom_fields = st.text_area(
                    "Custom fields (one per line, format: field_name: extraction_method)",
                    "domain: regex(Domain: ([A-Za-z]+))\nimportance: llm(Rate importance 1-10)"
                )
                
                st.selectbox("Default extraction method for custom fields", 
                           ["Regular Expression", "LLM Extraction", "Rule-based", "Python Function"])
            
            # Metadata enrichment
            st.markdown("### Metadata Enrichment")
            
            enrich_metadata = st.checkbox("Enrich metadata with AI")
            
            if enrich_metadata:
                enrichment_types = st.multiselect(
                    "Enrichment Types",
                    ["Topic Classification", "Entity Recognition", "Sentiment Analysis", 
                     "Content Categorization", "Key Information Extraction"],
                    default=["Topic Classification", "Entity Recognition"]
                )
                
                if "Topic Classification" in enrichment_types:
                    topic_model = st.selectbox(
                        "Topic Classification Model",
                        ["LLM-based", "BERTopic", "KeyBERT", "LDA", "Custom Classifier"]
                    )
                    
                if "Entity Recognition" in enrichment_types:
                    ner_model = st.selectbox(
                        "Named Entity Recognition",
                        ["spaCy", "Flair", "LLM-based", "Stanza", "Custom NER"]
                    )
                    
                if "Sentiment Analysis" in enrichment_types:
                    sentiment_model = st.selectbox(
                        "Sentiment Analysis Model",
                        ["VADER", "RoBERTa-Sentiment", "LLM-based", "Custom"]
                    )
        
        with col2:
            with st.expander("ℹ️ About Chunk Metadata", expanded=True):
                info_tooltip("""
                **Chunk Metadata** provides additional context and filtering capabilities.
                
                **Metadata Categories:**
                - **Basic**: Source file information and system metadata
                - **Content**: Document-level properties
                - **Position**: Location information within source document
                - **Semantic**: Content-derived information
                
                **Custom Metadata:**
                - Extract domain-specific information
                - Use regex, rules, or LLMs for extraction
                - Add structured metadata for filtering
                
                **Metadata Enrichment:**
                - Use AI to add valuable information
                - Topic classification helps organize content
                - Entity recognition identifies key concepts
                - Sentiment analysis captures emotional tone
                
                **Benefits:**
                - Enables powerful filtering during retrieval
                - Adds context to improve relevance
                - Supports faceted search capabilities
                - Helps with result organization and presentation
                
                **Best Practices:**
                - Include position metadata for source attribution
                - Add semantic metadata for intelligent filtering
                - Use enrichment for content understanding
                - Balance metadata richness with processing cost
                """)
            
            # Example
            st.markdown("#### Example Chunk with Rich Metadata")
            
            metadata_example = {
                "basic": {},
                "content": {},
                "position": {},
                "semantic": {},
                "custom": {}
            }
            
            # Fill example based on selections
            for category, fields in metadata_categories.items():
                category_key = category.lower()
                for field_key, field_selected in fields.items():
                    if field_selected:
                        if field_key == "source_file":
                            metadata_example["basic"]["source_file"] = "annual_report_2024.pdf"
                        elif field_key == "file_type":
                            metadata_example["basic"]["file_type"] = "application/pdf"
                        elif field_key == "title":
                            metadata_example["content"]["title"] = "Annual Financial Report 2024"
                        elif field_key == "section":
                            metadata_example["content"]["section"] = "Financial Performance"
                        elif field_key == "page_number":
                            metadata_example["position"]["page_number"] = 27
                        elif field_key == "chunk_index":
                            metadata_example["position"]["chunk_index"] = 112
                        elif field_key == "topics":
                            metadata_example["semantic"]["topics"] = ["Revenue Growth", "Market Analysis"]
                        elif field_key == "entities":
                            metadata_example["semantic"]["entities"] = [
                                {"text": "Goldman Sachs", "type": "ORG"}, 
                                {"text": "Jane Smith", "type": "PERSON"},
                                {"text": "Q3 2024", "type": "DATE"}
                            ]
            
            if custom_metadata:
                metadata_example["custom"]["domain"] = "Finance"
                metadata_example["custom"]["importance"] = 8
            
            if enrich_metadata:
                if "Sentiment Analysis" in enrichment_types:
                    metadata_example["semantic"]["sentiment"] = {"score": 0.65, "label": "positive"}
                if "Content Categorization" in enrichment_types:
                    metadata_example["semantic"]["category"] = "Financial Report"
            
            # Clean up empty categories
            metadata_example = {k: v for k, v in metadata_example.items() if v}
            
            # Display example
            st.code(json.dumps(metadata_example, indent=2))
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            checked_count = sum(sum(field for field in fields.values()) 
                              for fields in metadata_categories.values())
            
            if checked_count >= 8:
                st.success("✅ Rich metadata will significantly improve filtering and context")
            elif checked_count >= 4:
                st.success("✅ Good baseline metadata selection")
            else:
                st.warning("⚠️ Consider adding more metadata fields for better retrieval")
                
            if enrich_metadata:
                st.success("✅ AI-enriched metadata will improve semantic understanding")
            else:
                st.info("ℹ️ Consider metadata enrichment for better content understanding")

###############################
# 4. VECTOR DATABASE
###############################
elif main_section == "4. Vector Database":
    st.header("4. Vector Database")
    st.markdown("Configure your vector database for efficient storage and retrieval")
    
    # Create tabs for database options
    db_tabs = st.tabs(["Database Selection", "Indexing", "Advanced Features", "Scaling & Optimization"])
    
    # DATABASE SELECTION TAB
    with db_tabs[0]:
        st.subheader("Vector Database Selection")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Database selection
            vector_db = st.selectbox(
                "Select Vector Database",
                ["Chroma", "Qdrant", "Weaviate", "Pinecone", "Milvus", "FAISS", "Elasticsearch", "Redis", "PostgreSQL/pgvector"]
            )
            
            # Database-specific configuration
            st.markdown("### Database Configuration")
            
            if vector_db == "Chroma":
                persistence_dir = st.text_input("Persistence Directory", "./chroma_db")
                collection_name = st.text_input("Collection Name", "document_collection")
                
                st.checkbox("Enable metadata filtering", value=True)
                st.checkbox("Use anonymized telemetry", value=False)
            
            elif vector_db == "Qdrant":
                qdrant_url = st.text_input("Qdrant URL", "http://localhost:6333")
                collection_name = st.text_input("Collection Name", "document_collection")
                
                st.checkbox("Use on-disk storage", value=True)
                st.checkbox("Enable payload compression", value=True)
                st.selectbox("Distance Metric", ["Cosine", "Dot Product", "Euclidean"])
            
            elif vector_db == "Weaviate":
                weaviate_url = st.text_input("Weaviate URL", "http://localhost:8080")
                class_name = st.text_input("Class Name", "Document")
                
                auth_type = st.selectbox("Authentication", ["None", "API Key", "OIDC"])
                if auth_type != "None":
                    st.text_input("Authentication Key", type="password")
                
                st.checkbox("Use batch import", value=True)
                st.checkbox("Enable cross-references", value=True)
                st.checkbox("Enable generative module", value=False)
            
            elif vector_db == "Pinecone":
                api_key = st.text_input("Pinecone API Key", type="password")
                environment = st.text_input("Environment", "us-west1-gcp")
                index_name = st.text_input("Index Name", "document-index")
                
                st.number_input("Dimension", 1, 4096, 768)
                st.selectbox("Metric", ["cosine", "dotproduct", "euclidean"])
                st.selectbox("Pod Type", ["p1", "s1", "p2", "starter"])
            
            elif vector_db == "Milvus":
                milvus_uri = st.text_input("Milvus URI", "http://localhost:19530")
                collection_name = st.text_input("Collection Name", "document_collection")
                
                st.number_input("Dimension", 1, 4096, 768)
                consistency_level = st.selectbox("Consistency Level", ["Strong", "Bounded", "Session", "Eventually"])
                metric_type = st.selectbox("Distance Metric", ["L2", "IP", "Cosine"])
            
            elif vector_db == "FAISS":
                index_path = st.text_input("Index File Path", "./faiss_index")
                
                st.selectbox("Index Type", ["Flat", "IVF", "HNSW", "PQ", "SQ", "Hybrid"])
                st.checkbox("Use GPU acceleration", value=False)
                st.checkbox("Save index to disk", value=True)
                
                st.info("Note: FAISS requires separate metadata storage")
            
            elif vector_db == "Elasticsearch":
                es_url = st.text_input("Elasticsearch URL", "http://localhost:9200")
                index_name = st.text_input("Index Name", "document_index")
                
                st.text_input("Username", "elastic")
                st.text_input("Password", type="password")
                
                st.number_input("Number of Shards", 1, 20, 1)
                st.number_input("Number of Replicas", 0, 3, 1)
                
                st.selectbox("Vector Similarity", ["cosine", "dot_product", "l2_norm"])
            
            elif vector_db == "Redis":
                redis_url = st.text_input("Redis URL", "redis://localhost:6379")
                index_name = st.text_input("Index Name", "document_idx")
                
                st.selectbox("Storage Type", ["HASH", "JSON"])
                st.checkbox("Use RediSearch", value=True)
                st.selectbox("Vector Similarity", ["COSINE", "IP", "L2"])
            
            elif vector_db == "PostgreSQL/pgvector":
                conn_string = st.text_input("Connection String", "postgresql://user:password@localhost:5432/vectors")
                table_name = st.text_input("Table Name", "document_embeddings")
                
                st.selectbox("Index Type", ["HNSW", "IVF", "None"])
                st.selectbox("Vector Similarity", ["cosine", "inner product", "l2"])
                st.checkbox("Use partitioning for large tables", value=False)
        
        with col2:
            with st.expander("ℹ️ About Vector Databases", expanded=True):
                info_tooltip("""
                **Vector Databases** store and efficiently search vector embeddings.
                
                **Database Options:**
                
                **Chroma**: Simple, easy to use
                - Open-source, Python-native
                - Great for getting started
                - Good for small-to-medium collections
                
                **Qdrant**: Fast, focus on filtering
                - Excellent metadata filtering
                - Good balance of features/simplicity
                - Works well for medium deployments
                
                **Weaviate**: Knowledge graph + vectors
                - Combines graph and vector capabilities
                - Good for relational data
                - Strong schema support
                
                **Pinecone**: Managed, serverless
                - Fully managed service
                - Simple scaling
                - Higher cost but minimal maintenance
                
                **Milvus**: High-scale, enterprise
                - Highly scalable, cloud-native
                - Rich feature set
                - Good for large-scale deployments
                
                **FAISS**: Specialized search library
                - Extremely fast for large datasets
                - More low-level, less metadata support
                - Great for pure vector search
                
                **Elasticsearch**: Full-text + vectors
                - Combines text and vector search
                - Rich ecosystem
                - Good if already using Elasticsearch
                
                **Redis**: In-memory, low latency
                - Very low latency
                - Good for real-time applications
                - Limited by memory
                
                **PostgreSQL/pgvector**: SQL + vectors
                - Familiar SQL interface
                - Good for hybrid structured/vector data
                - Easy integration with existing apps
                
                **Key Considerations:**
                - Scale requirements
                - Filtering needs
                - Hosting preferences
                - Integration with existing systems
                """)
            
            # Database comparison table
            st.markdown("#### Database Comparison")
            
            db_comparison = pd.DataFrame({
                "Database": ["Chroma", "Qdrant", "Weaviate", "Pinecone", "Milvus", "FAISS", "Elasticsearch", "Redis", "PostgreSQL"],
                "Scale": ["Small-Medium", "Medium", "Medium-Large", "Medium-Large", "Large", "Large", "Large", "Medium", "Medium"],
                "Hosting": ["Self", "Both", "Both", "Managed", "Both", "Self", "Both", "Both", "Both"],
                "Filtering": ["Good", "Excellent", "Excellent", "Basic", "Good", "Limited", "Excellent", "Basic", "Good"]
            })
            
            st.dataframe(db_comparison, hide_index=True)
            
            # Recommendations based on selection
            st.markdown("#### Recommendations")
            
            if vector_db == "Chroma":
                st.success("✅ Good choice for getting started quickly")
                st.info("ℹ️ Consider Qdrant or Weaviate as your needs grow")
            
            elif vector_db in ["Qdrant", "Weaviate"]:
                st.success("✅ Excellent choice balancing features and usability")
            
            elif vector_db == "FAISS":
                st.info("ℹ️ FAISS is powerful but requires more custom code")
                st.warning("⚠️ You'll need separate metadata storage")
            
            elif vector_db == "Pinecone":
                st.success("✅ Good choice for managed service with minimal maintenance")
                st.warning("⚠️ Higher cost compared to self-hosted options")
            
            elif vector_db == "Milvus":
                st.success("✅ Excellent for large-scale production deployments")
                st.warning("⚠️ More complex setup and operations")
    
    # INDEXING TAB
    with db_tabs[1]:
        st.subheader("Vector Indexing")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Indexing type
            index_type = st.selectbox(
                "Primary Indexing Type",
                ["HNSW", "IVF Flat", "IVF PQ", "Flat", "Scalar Quantization", "Product Quantization", "ANNOY", "ScaNN", "DiskANN", "LSH"]
            )
            
            # Index parameters based on type
            st.markdown("### Index Parameters")
            
            if index_type == "HNSW":
                st.slider("M (max connections per node)", 4, 128, 16, 
                        help="Higher values increase accuracy but use more memory")
                
                st.slider("ef_construction", 50, 1000, 200,
                        help="Higher values create more accurate indexes but slower builds")
                
                st.slider("ef_search", 10, 1000, 100,
                        help="Higher values increase search quality but reduce speed")
            
            elif index_type == "IVF Flat":
                st.slider("nlist (number of clusters)", 10, 10000, 1000,
                        help="More clusters: faster search, less accurate, larger index")
                
                st.slider("nprobe (clusters to search)", 1, 100, 10,
                        help="More probes: better accuracy, slower search")
            
            elif index_type == "IVF PQ":
                st.slider("nlist (number of clusters)", 10, 10000, 1000)
                st.slider("nprobe (clusters to search)", 1, 100, 10)
                
                st.slider("M (subvectors)", 1, 64, 8,
                        help="More subvectors: better accuracy, larger index")
                
                st.selectbox("nbits (bits per subvector)", [4, 8, 16], index=1,
                           help="More bits: better accuracy, larger index")
            
            elif index_type == "Product Quantization":
                st.slider("M (subvectors)", 1, 64, 8)
                st.selectbox("nbits (bits per subvector)", [4, 8, 16], index=1)
            
            elif index_type == "Scalar Quantization":
                st.selectbox("Quantization Type", ["QT_8bit", "QT_4bit", "QT_fp16"])
            
            elif index_type == "ANNOY":
                st.slider("n_trees", 10, 1000, 100,
                        help="More trees: higher accuracy, larger index")
                
                st.slider("search_k", 100, 10000, 1000,
                        help="More search_k: higher accuracy, slower search")
            
            elif index_type == "ScaNN":
                st.slider("num_leaves", 100, 10000, 1000)
                st.slider("avq_threshold", 0.0, 1.0, 0.2, 0.05)
                st.slider("dimensions_per_block", 1, 32, 2)
            
            elif index_type == "DiskANN":
                st.slider("graph_degree", 32, 512, 96)
                st.slider("search_list_size", 10, 500, 100)
                
            elif index_type == "LSH":
                st.slider("n_bits", 8, 256, 64)
                st.slider("n_tables", 1, 32, 4)
            
            # Hybrid indexing
            st.markdown("### Hybrid Indexing")
            
            use_hybrid = st.checkbox("Use hybrid indexing approach")
            
            if use_hybrid:
                secondary_index = st.selectbox(
                    "Secondary Index",
                    ["PQ", "SQ", "Graph-based", "Tree-based"]
                )
                
                st.selectbox("Hybrid Strategy", 
                           ["Coarse-then-fine", "Multi-index fusion", "Dynamic selection"])
            
            # Partitioning
            st.markdown("### Partitioning")
            
            use_partitioning = st.checkbox("Enable index partitioning")
            
            if use_partitioning:
                partition_strategy = st.selectbox(
                    "Partition Strategy",
                    ["Metadata-based", "Cluster-based", "Random", "Time-based"]
                )
                
                if partition_strategy == "Metadata-based":
                    partition_field = st.text_input("Partition Field", "category")
                
                st.number_input("Number of Partitions", 2, 1000, 10)
        
        with col2:
            with st.expander("ℹ️ About Vector Indexing", expanded=True):
                info_tooltip("""
                **Vector Indexing** dramatically speeds up similarity search by creating search-optimized structures.
                
                **Index Types:**
                
                **HNSW** (Hierarchical Navigable Small World)
                - Graph-based approach with hierarchical structure
                - Very high accuracy with reasonable memory usage
                - Fast search even for large collections
                - Memory-intensive but extremely effective
                
                **IVF** (Inverted File Index)
                - Clusters vectors and searches only relevant clusters
                - Good balance of speed, memory, and accuracy
                - Two main variants:
                  - IVF Flat: Stores full vectors (more accurate)
                  - IVF PQ: Uses product quantization (more compact)
                
                **Flat**
                - Brute-force exact search
                - 100% accurate but very slow for large collections
                - No approximation, good for small datasets only
                
                **Product Quantization (PQ)**
                - Compresses vectors by encoding subvectors
                - Massive memory savings
                - Trade-off between memory and accuracy
                
                **Scalar Quantization (SQ)**
                - Reduces precision of each value (e.g., float32 → int8)
                - Simple but effective compression
                - Less impact on accuracy than PQ
                
                **ANNOY**
                - Tree-based approach using random projections
                - Good for memory-mapping (disk-based search)
                - Less accurate than HNSW but uses less memory
                
                **ScaNN**
                - Google's vector search library
                - Excellent anisotropic quantization
                - Very high performance
                
                **DiskANN**
                - Microsoft's disk-based approximate search
                - Excellent for collections larger than memory
                - Near-SSD search speed with high recall
                
                **LSH** (Locality-Sensitive Hashing)
                - Hash-based approach for similarity search
                - Good for high-dimensional data
                - Less accurate than modern methods
                
                **Key Parameters:**
                - Higher values for construction parameters = better accuracy, slower indexing
                - Higher values for search parameters = better accuracy, slower search
                """)
            
            # Performance comparison
            st.markdown("#### Performance Comparison")
            
            index_comparison = pd.DataFrame({
                "Index Type": ["Flat", "HNSW", "IVF Flat", "IVF PQ", "PQ", "SQ", "ANNOY", "ScaNN", "DiskANN"],
                "Accuracy": ["Perfect", "Very High", "High", "Medium", "Low", "High", "Medium", "High", "High"],
                "Memory": ["Highest", "High", "Medium", "Low", "Lowest", "Medium", "Low", "Low", "Lowest"],
                "Build Speed": ["Fast", "Slow", "Medium", "Medium", "Fast", "Fast", "Medium", "Medium", "Slow"],
                "Query Speed": ["Very Slow", "Fast", "Fast", "Very Fast", "Very Fast", "Fast", "Fast", "Very Fast", "Fast"]
            })
            
            st.dataframe(index_comparison, hide_index=True)
            
            # Visualization
            st.markdown("#### Index Visualization")
            
            if index_type == "HNSW":
                st.image("https://milvus.io/static/2d8e440e32948834bf98d229d3dc8c3f/497c6/partition_graph.png", 
                         caption="HNSW graph structure visualization")
            elif "IVF" in index_type:
                st.image("https://milvus.io/static/1d38af354b99dbd51ffb94eb517a0bc3/497c6/ivf.png",
                         caption="IVF clustering visualization")
            elif index_type == "ANNOY" or index_type == "ScaNN":
                st.image("https://milvus.io/static/994607f0a3c9567926acd4378795f212/da2b1/annoy.png",
                         caption="Tree-based index visualization")
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if index_type == "HNSW":
                st.success("✅ HNSW provides excellent search quality")
                st.success("✅ Good choice for most production applications")
                st.info("ℹ️ Increasing M improves accuracy but uses more memory")
                
            elif index_type == "IVF Flat":
                st.success("✅ Good balance of accuracy and performance")
                st.info("ℹ️ Increase nprobe for better accuracy during search")
                
            elif index_type == "IVF PQ" or index_type == "Product Quantization":
                st.success("✅ Excellent for very large collections with memory constraints")
                st.warning("⚠️ Significant accuracy trade-off - test thoroughly")
                
            elif index_type == "Flat":
                st.warning("⚠️ Flat indexes don't scale beyond small collections")
                st.info("ℹ️ Consider HNSW for better performance with similar accuracy")
                
            elif index_type == "DiskANN":
                st.success("✅ Excellent for collections larger than available RAM")
                st.info("ℹ️ Requires SSD storage for good performance")
    
    # ADVANCED FEATURES TAB
    with db_tabs[2]:
        st.subheader("Advanced Database Features")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Metadata filtering
            st.markdown("### Metadata Filtering")
            
            enable_filtering = st.checkbox("Enable advanced metadata filtering", value=True)
            
            if enable_filtering:
                st.markdown("#### Metadata Index Configuration")
                
                metadata_fields = st.text_area(
                    "Metadata fields to index (one per line)",
                    "source_file\nauthor\ncategory\ncreation_date\ntopic"
                ).strip().split("\n")
                
                index_types = {}
                
                for field in metadata_fields:
                    if field:
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            st.markdown(f"**{field}**")
                        with col_b:
                            index_types[field] = st.selectbox(
                                f"Index type for {field}",
                                ["Keyword", "Text", "Integer", "Float", "Date", "Boolean", "Geo"],
                                key=f"idx_{field}"
                            )
                
                st.checkbox("Enable faceted search", value=True)
                st.checkbox("Enable range queries", value=True)
            
            # Hybrid search
            st.markdown("### Hybrid Search")
            
            enable_hybrid_search = st.checkbox("Enable hybrid search capabilities")
            
            if enable_hybrid_search:
                hybrid_methods = st.multiselect(
                    "Hybrid Search Methods",
                    ["BM25 + Vector", "Keyword Matching", "Sparse + Dense", "Vector + Filter"],
                    default=["BM25 + Vector"]
                )
                
                if "BM25 + Vector" in hybrid_methods:
                    st.slider("BM25 weight", 0.0, 1.0, 0.3, 0.05)
                
                if "Sparse + Dense" in hybrid_methods:
                    st.selectbox("Sparse Method", ["BM25", "TF-IDF", "SPLADE"])
                    st.slider("Sparse weight", 0.0, 1.0, 0.25, 0.05)
                
                st.selectbox("Result Fusion Method", 
                           ["Reciprocal Rank Fusion", "Linear Combination", "Max Score"])
            
            # Semantic caching
            st.markdown("### Semantic Caching")
            
            enable_caching = st.checkbox("Enable semantic caching")
            
            if enable_caching:
                cache_strategy = st.selectbox(
                    "Caching Strategy",
                    ["Query Embedding Similarity", "Result Set Caching", "Hybrid Approach"]
                )
                
                st.slider("Cache similarity threshold", 0.7, 1.0, 0.9, 0.01)
                st.slider("Cache TTL (minutes)", 1, 1440, 60)
                st.number_input("Max cache size (MB)", 10, 10000, 1000)
            
            # Multi-tenant support
            st.markdown("### Multi-tenancy")
            
            multi_tenant = st.checkbox("Enable multi-tenant support")
            
            if multi_tenant:
                tenant_isolation = st.selectbox(
                    "Tenant Isolation Strategy",
                    ["Separate Collections", "Filtered Queries", "Prefix Isolation", "Hybrid"]
                )
                
                st.checkbox("Enable tenant-specific configuration", value=True)
                st.checkbox("Tenant query rate limiting", value=True)
        
        with col2:
            with st.expander("ℹ️ About Advanced Features", expanded=True):
                info_tooltip("""
                **Advanced Database Features** enhance search capabilities beyond basic vector similarity.
                
                **Metadata Filtering:**
                - Combines vector search with structured filters
                - Examples: Filter by date, author, category, etc.
                - Dramatically improves search precision
                - Index types matter for query performance:
                  - Keyword: Exact match on whole values
                  - Text: Full-text search within field
                  - Numeric/Date: Range queries and sorting
                  - Geo: Location-based filtering
                
                **Hybrid Search:**
                - Combines multiple search modalities:
                  - Vector: Semantic understanding
                  - BM25/Keyword: Lexical matching
                  - Sparse: Keyword importance
                - Better results than either approach alone
                - Fusion methods determine how to combine scores
                
                **Semantic Caching:**
                - Caches search results based on query similarity
                - Improves performance for similar queries
                - Reduces API calls and latency
                - Particularly useful for common queries
                
                **Multi-tenancy:**
                - Supports multiple users/organizations
                - Ensures data isolation between tenants
                - Options include:
                  - Separate collections (stronger isolation)
                  - Query filtering (more efficient)
                
                **When to Use:**
                - Metadata filtering: Almost always valuable
                - Hybrid search: For improved retrieval quality
                - Semantic caching: High-volume applications
                - Multi-tenancy: Multi-user/org applications
                """)
            
            # Example queries
            st.markdown("#### Example Queries")
            
            if enable_filtering:
                st.markdown("**Metadata Filtering Query:**")
                st.code("""
# Example metadata filter query
results = db.query(
    vector=query_embedding,
    filter={
        "category": "Technical Documentation",
        "creation_date": {"$gte": "2023-01-01"},
        "author": {"$in": ["John Smith", "Jane Doe"]},
        "$or": [
            {"topic": "Machine Learning"},
            {"topic": "Neural Networks"}
        ]
    },
    top_k=5
)
                """)
            
            if enable_hybrid_search:
                st.markdown("**Hybrid Search Query:**")
                st.code("""
# Example hybrid search
results = db.hybrid_search(
    text_query="transformer architecture attention mechanism",
    vector_query=query_embedding,
    weights={"text": 0.3, "vector": 0.7},
    filter={"category": "Research Papers"},
    top_k=5
)
                """)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_filtering:
                st.success("✅ Metadata filtering greatly improves search precision")
                st.info("ℹ️ Consider carefully which fields need indexing")
            else:
                st.warning("⚠️ Metadata filtering is strongly recommended for most applications")
            
            if enable_hybrid_search:
                st.success("✅ Hybrid search combines the best of semantic and lexical search")
                
                if "BM25 + Vector" in hybrid_methods if enable_hybrid_search else []:
                    st.success("✅ BM25 + Vector is a good default hybrid approach")
            
            if enable_caching:
                st.success("✅ Semantic caching will improve performance for repeated queries")
                st.info("ℹ️ Tune the similarity threshold based on your application needs")
    
    # SCALING & OPTIMIZATION TAB
    with db_tabs[3]:
        st.subheader("Scaling & Optimization")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Deployment architecture
            st.markdown("### Deployment Architecture")
            
            deploy_mode = st.selectbox(
                "Deployment Mode",
                ["Single Node", "Distributed Cluster", "Serverless", "Hybrid"]
            )
            
            if deploy_mode == "Distributed Cluster":
                st.number_input("Number of nodes", 2, 100, 3)
                st.selectbox("Cluster Architecture", ["Sharded", "Replicated", "Hybrid"])
                st.checkbox("Enable automatic scaling", value=True)
            
            elif deploy_mode == "Serverless":
                st.slider("Min capacity units", 1, 20, 2)
                st.slider("Max capacity units", 2, 100, 10)
                st.checkbox("Auto-scaling", value=True)
            
            # Sharding strategy
            st.markdown("### Data Distribution")
            
            if deploy_mode in ["Distributed Cluster", "Hybrid"]:
                sharding_strategy = st.selectbox(
                    "Sharding Strategy",
                    ["Hash-based", "Range-based", "Metadata-based", "Consistent Hashing"]
                )
                
                if sharding_strategy == "Metadata-based":
                    shard_field = st.text_input("Sharding Field", "category")
                
                st.number_input("Replication Factor", 1, 5, 2)
                st.checkbox("Enable read from replicas", value=True)
            
            # Performance optimization
            st.markdown("### Performance Optimization")
            
            # Batch processing
            st.checkbox("Enable batch processing", value=True)
            batch_size = st.slider("Batch size", 10, 1000, 100)
            
            # Async operations
            st.checkbox("Enable asynchronous operations", value=True)
            
            # Caching strategies
            cache_strategy = st.multiselect(
                "Caching Strategies",
                ["Result Cache", "Hot Vectors", "Query Plan", "Filter Cache"],
                default=["Result Cache", "Hot Vectors"]
            )
            
            if "Result Cache" in cache_strategy:
                st.number_input("Result cache TTL (seconds)", 10, 86400, 600)
            
            if "Hot Vectors" in cache_strategy:
                st.number_input("Hot vectors cache size (MB)", 100, 10000, 1000)
            
            # Resource allocation
            st.markdown("### Resource Allocation")
            
            st.slider("Memory allocation (GB)", 1, 128, 8)
            st.slider("CPU cores", 1, 32, 4)
            
            st.checkbox("Enable GPU acceleration", value=False)
            if st.checkbox("Enable disk offloading for large indexes", value=False):
                st.selectbox("Storage Type", ["SSD", "HDD", "NVMe"])
        
        with col2:
            with st.expander("ℹ️ About Scaling & Optimization", expanded=True):
                info_tooltip("""
                **Scaling & Optimization** ensures your vector database performs well as data and traffic grow.
                
                **Deployment Architectures:**
                - **Single Node**: Simple setup, limited scalability
                - **Distributed Cluster**: Horizontal scaling across multiple nodes
                - **Serverless**: Managed auto-scaling with pay-per-use pricing
                - **Hybrid**: Combines multiple approaches
                
                **Sharding Strategies:**
                - **Hash-based**: Evenly distributes data
                - **Range-based**: Groups similar data together
                - **Metadata-based**: Shards by specific fields
                - **Consistent Hashing**: Minimizes redistribution when scaling
                
                **Performance Optimizations:**
                - **Batch Processing**: Improves throughput for bulk operations
                - **Async Operations**: Better concurrency and resource utilization
                - **Caching**: Reduces computation for frequent patterns
                  - Result Cache: Stores common query results
                  - Hot Vectors: Keeps popular vectors in memory
                  - Query Plan: Caches execution strategies
                  - Filter Cache: Speeds up common filters
                
                **Resource Allocation:**
                - RAM is critical for vector database performance
                - GPU acceleration helps with large batch processing
                - SSDs are strongly recommended for disk-based indexes
                
                **When to Scale:**
                - Collection size approaches 70-80% of available RAM
                - Query latency exceeds acceptable thresholds
                - Throughout requirements increase
                - System stability issues appear
                """)
            
            # Scaling metrics
            st.markdown("#### Performance Metrics")
            
            scaling_metrics = pd.DataFrame({
                "Metric": ["Query Latency", "Throughput", "Memory Usage", "Index Size"],
                "Small": ["< 10ms", "100 qps", "< 4GB", "< 1M vectors"],
                "Medium": ["10-50ms", "100-1K qps", "4-32GB", "1M-10M vectors"],
                "Large": ["50-200ms", "1K-10K qps", "32-256GB", "10M-100M vectors"],
                "X-Large": ["> 200ms", "> 10K qps", "> 256GB", "> 100M vectors"]
            })
            
            st.dataframe(scaling_metrics, hide_index=True)
            
            # Architecture diagram
            if deploy_mode == "Distributed Cluster":
                st.markdown("#### Distributed Architecture")
                st.image("https://milvus.io/static/a37b6a6a56649b2921a7d9f550bfb49b/497c6/distributed_milvus.png",
                        caption="Example distributed vector database architecture")
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if deploy_mode == "Single Node":
                st.info("ℹ️ Single node is good for development or small applications")
                st.warning("⚠️ Consider a distributed setup for production or larger datasets")
            
            elif deploy_mode == "Distributed Cluster":
                st.success("✅ Distributed setup provides good scalability and reliability")
                st.info("ℹ️ Start with 3 nodes for a good balance of reliability and cost")
            
            if "Result Cache" in cache_strategy and "Hot Vectors" in cache_strategy:
                st.success("✅ Your caching strategy will help with both repeated and similar queries")
            
            if deploy_mode in ["Distributed Cluster", "Hybrid"] and st.session_state.get("memory_allocation", 8) < 16:
                st.warning("⚠️ Consider increasing memory allocation for distributed deployments")

#########################
# 5. QUERY PROCESSING
#########################
elif main_section == "5. Query Processing":
    st.header("5. Query Processing")
    st.markdown("Configure how user queries are processed and enhanced before retrieval")
    
    # Create tabs for query processing options
    query_tabs = st.tabs(["Query Transformation", "Query Planning", "Expansion & Refinement", "Routing"])
    
    # QUERY TRANSFORMATION TAB
    with query_tabs[0]:
        st.subheader("Query Transformation")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Basic preprocessing
            st.markdown("### Basic Preprocessing")
            
            preprocessing_steps = st.multiselect(
                "Preprocessing Steps",
                ["Lowercasing", "Stopword Removal", "Stemming", "Lemmatization", "Special Character Handling", "Spell Correction"],
                default=["Lowercasing", "Special Character Handling"]
            )
            
            if "Special Character Handling" in preprocessing_steps:
                special_char_handling = st.selectbox(
                    "Special Character Strategy",
                    ["Remove All", "Keep Alphanumeric", "Replace with Space", "Keep Domain-Specific"]
                )
                
                if special_char_handling == "Keep Domain-Specific":
                    st.text_area("Domain-specific characters to keep", "+-*/=()[]{}<>$%#@&")
            
            if "Spell Correction" in preprocessing_steps:
                spell_correct_method = st.selectbox(
                    "Spell Correction Method",
                    ["Dictionary-based", "Statistical", "Neural", "Context-aware LLM"]
                )
            
            # Query rewriting
            st.markdown("### Query Rewriting")
            
            rewrite_methods = st.multiselect(
                "Query Rewriting Methods",
                ["Query Expansion", "Paraphrasing", "Format Standardization", "Entity Normalization", "Query Simplification", "LLM-based Reframing"],
                default=["Query Expansion"]
            )
            
            if "Query Expansion" in rewrite_methods:
                expansion_methods = st.multiselect(
                    "Expansion Methods",
                    ["Synonym Expansion", "Domain Ontology", "Statistical Associations", "Word Embeddings", "LLM Generation"],
                    default=["Synonym Expansion", "LLM Generation"]
                )
                
                if "LLM Generation" in expansion_methods:
                    expansion_llm = st.selectbox(
                        "Expansion LLM",
                        ["gpt-3.5-turbo", "Claude 3 Haiku", "Llama-3-8b", "Custom"]
                    )
                    
                    st.slider("Number of expansions", 1, 10, 3)
            
            if "Paraphrasing" in rewrite_methods:
                paraphrase_method = st.selectbox(
                    "Paraphrasing Method",
                    ["T5-based", "LLM-based", "Rule-based", "Statistical"]
                )
                
                if paraphrase_method in ["T5-based", "LLM-based"]:
                    st.slider("Diversity parameter", 0.0, 1.0, 0.7, 0.1)
                    st.slider("Number of paraphrases", 1, 5, 2)
            
            # Query normalization
            st.markdown("### Query Normalization")
            
            normalization_methods = st.multiselect(
                "Normalization Methods",
                ["Entity Linking", "Temporal Normalization", "Unit Conversion", "Domain-Specific Terms", "Acronym Expansion"],
                default=["Entity Linking", "Acronym Expansion"]
            )
            
            if "Entity Linking" in normalization_methods:
                entity_sources = st.multiselect(
                    "Entity Sources",
                    ["Wikidata", "Domain Ontology", "Custom KB", "Extracted Entities"],
                    default=["Domain Ontology", "Extracted Entities"]
                )
                
                if "Custom KB" in entity_sources:
                    st.file_uploader("Upload custom knowledge base", type=["csv", "json"])
        
        with col2:
            with st.expander("ℹ️ About Query Transformation", expanded=True):
                info_tooltip("""
                **Query Transformation** improves retrieval by modifying the original query.
                
                **Preprocessing:**
                - **Lowercasing**: Reduces case sensitivity issues
                - **Stopword Removal**: Eliminates common words with little semantic value
                - **Stemming/Lemmatization**: Normalizes word forms
                - **Special Character Handling**: Manages punctuation and symbols
                - **Spell Correction**: Fixes typos and misspellings
                
                **Query Rewriting:**
                - **Expansion**: Adds related terms to broaden retrieval
                - **Paraphrasing**: Reformulates query while preserving meaning
                - **Format Standardization**: Normalizes query structure
                - **Entity Normalization**: Standardizes named entities
                - **Query Simplification**: Removes complexity while preserving intent
                - **LLM-based Reframing**: Uses AI to rewrite for optimal retrieval
                
                **Query Normalization:**
                - **Entity Linking**: Maps mentions to knowledge base entities
                - **Temporal Normalization**: Standardizes date/time references
                - **Unit Conversion**: Normalizes measurements and units
                - **Acronym Expansion**: Expands abbreviations to full forms
                
                **Benefits:**
                - Improves recall by including synonyms and related concepts
                - Handles ambiguity and clarifies intent
                - Increases robustness to different query formulations
                - Bridges vocabulary gaps between queries and documents
                """)
            
            # Example of transformed query
            st.markdown("#### Example Transformations")
            
            original_query = "covid symptoms 2023"
            
            transformed_examples = []
            
            if "Lowercasing" in preprocessing_steps:
                transformed_examples.append("**Lowercasing:** `covid symptoms 2023`")
                
            if "Query Expansion" in rewrite_methods:
                transformed_examples.append("**Query Expansion:** `covid symptoms 2023 coronavirus sars-cov-2 signs clinical manifestations current year recent`")
                
            if "Entity Linking" in normalization_methods:
                transformed_examples.append("**Entity Linking:** `[COVID-19:Q84263196] symptoms [2023:Q192862]`")
                
            if "Paraphrasing" in rewrite_methods:
                transformed_examples.append("**Paraphrasing:** `What symptoms are associated with COVID-19 in 2023?`")
            
            st.markdown("**Original Query:**")
            st.code(original_query)
            
            st.markdown("**Transformed Queries:**")
            for example in transformed_examples:
                st.markdown(example)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if "Query Expansion" in rewrite_methods and "LLM Generation" in expansion_methods if "Query Expansion" in rewrite_methods else []:
                st.success("✅ LLM-based query expansion is highly effective for improving recall")
                
            if "Entity Linking" in normalization_methods:
                st.success("✅ Entity linking helps with precision for entity-centric queries")
                
            if "Paraphrasing" in rewrite_methods:
                st.success("✅ Paraphrasing helps overcome vocabulary mismatches")
                
            if not any(rewrite_methods):
                st.warning("⚠️ Query rewriting is recommended for most RAG systems")
    
    # QUERY PLANNING TAB
    with query_tabs[1]:
        st.subheader("Query Planning")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Query understanding
            st.markdown("### Query Understanding")
            
            understanding_methods = st.multiselect(
                "Understanding Methods",
                ["Intent Classification", "Query Type Detection", "Complexity Analysis", "Context Detection", "Language Detection"],
                default=["Intent Classification", "Query Type Detection"]
            )
            
            if "Intent Classification" in understanding_methods:
                intent_model = st.selectbox(
                    "Intent Classification Model",
                    ["Rule-based", "ML Classifier", "LLM-based", "Custom"]
                )
                
                intent_classes = st.multiselect(
                    "Intent Classes",
                    ["Factual", "Explanatory", "Comparative", "Procedural", "Opinion-seeking", "Definitional", "Hypothetical"],
                    default=["Factual", "Explanatory", "Procedural"]
                )
                
            if "Query Type Detection" in understanding_methods:
                query_types = st.multiselect(
                    "Query Types to Detect",
                    ["Question", "Command", "Keyword", "Conversational", "Boolean", "Natural Language"],
                    default=["Question", "Keyword", "Natural Language"]
                )
            
            # Query decomposition
            st.markdown("### Query Decomposition")
            
            enable_decomposition = st.checkbox("Enable query decomposition")
            
            if enable_decomposition:
                decomposition_method = st.selectbox(
                    "Decomposition Method",
                    ["Rule-based", "Parse Tree", "LLM-based", "Hybrid"]
                )
                
                if decomposition_method == "LLM-based":
                    decomp_llm = st.selectbox(
                        "Decomposition LLM",
                        ["gpt-4", "Claude 3 Sonnet", "Llama-3-70b", "Custom"]
                    )
                
                st.checkbox("Generate sub-queries automatically", value=True)
                st.checkbox("Enable multi-step reasoning", value=True)
                
                if st.checkbox("Custom decomposition rules", value=False):
                    st.text_area("Decomposition rules (one per line)", 
                                "Comparative -> Compare X and Y\nCause-effect -> First cause then effect")
            
            # Query optimization
            st.markdown("### Query Optimization")
            
            optimization_methods = st.multiselect(
                "Optimization Methods",
                ["Query Pruning", "Term Weighting", "Boolean Optimization", "Filter Planning", "Cost-Based Planning"],
                default=["Term Weighting", "Filter Planning"]
            )
            
            if "Term Weighting" in optimization_methods:
                weight_method = st.selectbox(
                    "Term Weighting Method",
                    ["TF-IDF", "BM25", "Neural", "Hybrid"]
                )
                
                st.checkbox("Use part-of-speech weighting", value=True)
        
        with col2:
            with st.expander("ℹ️ About Query Planning", expanded=True):
                info_tooltip("""
                **Query Planning** analyzes and structures queries for optimal retrieval performance.
                
                **Query Understanding:**
                - **Intent Classification**: Identifies the goal (factual answer, explanation, etc.)
                - **Query Type Detection**: Classifies query format and structure
                - **Complexity Analysis**: Assesses difficulty and requirements
                - **Context Detection**: Identifies domain context and assumptions
                
                **Query Decomposition:**
                - Breaks complex queries into simpler sub-queries
                - Enables handling multi-part questions
                - Facilitates multi-hop reasoning
                - Improves handling of compound questions
                
                **Query Optimization:**
                - **Query Pruning**: Removes non-essential terms
                - **Term Weighting**: Assigns importance to different query terms
                - **Boolean Optimization**: Restructures boolean logic for better performance
                - **Filter Planning**: Optimizes order and execution of filters
                
                **Benefits:**
                - Handles complex, multi-part questions
                - Improves precision for difficult queries
                - Enables step-by-step reasoning
                - Better utilization of retrieval capabilities
                
                **When to Use:**
                - For complex, multi-faceted questions
                - When supporting reasoning chains
                - For domain-specific applications
                - When precision is critical
                """)
            
            # Example of query planning
            st.markdown("#### Query Planning Example")
            
            complex_query = "Compare the side effects of aspirin and ibuprofen for treating headaches, and explain which is safer for people with stomach issues"
            
            st.markdown("**Original Complex Query:**")
            st.code(complex_query)
            
            if enable_decomposition:
                st.markdown("**Decomposed Sub-queries:**")
                st.code("""
1. "What are the side effects of aspirin for treating headaches?"
2. "What are the side effects of ibuprofen for treating headaches?"
3. "How do aspirin and ibuprofen affect people with stomach issues?"
4. "Which is safer between aspirin and ibuprofen for people with stomach issues?"
                """)
                
                st.markdown("**Execution Plan:**")
                st.code("""
1. Execute sub-queries 1 and 2 in parallel
2. Perform comparative analysis on results
3. Execute sub-query 3 with focused retrieval on stomach-related content
4. Synthesize final answer incorporating safety comparison
                """)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_decomposition:
                st.success("✅ Query decomposition significantly improves complex question handling")
                
                if decomposition_method == "LLM-based":
                    st.success("✅ LLM-based decomposition provides the most flexible approach")
                    
            else:
                st.warning("⚠️ Consider enabling query decomposition for complex questions")
                
            if "Intent Classification" in understanding_methods:
                st.info("ℹ️ Intent classification enables specialized handling of different query types")
    
    # EXPANSION & REFINEMENT TAB
    with query_tabs[2]:
        st.subheader("Query Expansion & Refinement")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Semantic expansion
            st.markdown("### Semantic Expansion")
            
            semantic_expansion = st.checkbox("Enable semantic expansion", value=True)
            
            if semantic_expansion:
                expansion_sources = st.multiselect(
                    "Expansion Sources",
                    ["Synonyms", "Hypernyms", "Hyponyms", "Related Concepts", "Word Embeddings", "Knowledge Base"],
                    default=["Synonyms", "Related Concepts"]
                )
                
                if "Knowledge Base" in expansion_sources:
                    kb_source = st.selectbox(
                        "Knowledge Base Source",
                        ["WordNet", "ConceptNet", "Wikidata", "Custom Domain KB", "Extracted Concepts"]
                    )
                    
                    if kb_source == "Custom Domain KB":
                        st.file_uploader("Upload domain knowledge base", type=["csv", "json", "txt"])
                
                expansion_strategy = st.selectbox(
                    "Expansion Strategy",
                    ["Static Expansion", "Dynamic (Query-time)", "Controlled Vocabulary", "LLM-guided"]
                )
                
                if expansion_strategy == "LLM-guided":
                    llm_model = st.selectbox(
                        "LLM Model",
                        ["gpt-3.5-turbo", "Claude 3 Haiku", "Llama-3-8b", "Custom"]
                    )
                
                st.slider("Maximum expansion terms", 1, 20, 5)
            
            # Query hypotheses
            st.markdown("### Query Hypotheses Generation")
            
            enable_hypotheses = st.checkbox("Enable query hypotheses", value=True)
            
            if enable_hypotheses:
                hypothesis_method = st.selectbox(
                    "Hypothesis Generation Method",
                    ["LLM Generation", "Template-based", "Statistical", "Hybrid"]
                )
                
                if hypothesis_method == "LLM Generation":
                    hypo_model = st.selectbox(
                        "Hypothesis LLM",
                        ["gpt-4", "Claude 3 Sonnet", "Llama-3-70b", "Custom"],
                        key="hypo_llm"
                    )
                
                st.number_input("Number of hypotheses", 1, 10, 3)
                st.checkbox("Diversify hypotheses", value=True)
                st.checkbox("Include original query", value=True)
            
            # RAG-fusion integration
            st.markdown("### RAG-Fusion")
            
            enable_rag_fusion = st.checkbox("Enable RAG-Fusion", value=False)
            
            if enable_rag_fusion:
                fusion_llm = st.selectbox(
                    "Query Generation LLM",
                    ["gpt-3.5-turbo", "Claude 3 Haiku", "Llama-3-8b", "Custom"]
                )
                
                st.slider("Number of generated queries", 1, 10, 5)
                st.slider("Temperature for generation", 0.0, 1.0, 0.7, 0.1)
                
                fusion_method = st.selectbox(
                    "Fusion Method",
                    ["Reciprocal Rank Fusion", "CombSUM", "CombMNZ", "Weighted Fusion"]
                )
                
                if fusion_method == "Reciprocal Rank Fusion":
                    st.slider("RRF constant (k)", 1, 100, 60)
                
                elif fusion_method == "Weighted Fusion":
                    st.slider("Original query weight", 0.1, 1.0, 0.5, 0.1)
        
        with col2:
            with st.expander("ℹ️ About Query Expansion & Refinement", expanded=True):
                info_tooltip("""
                **Query Expansion & Refinement** techniques improve recall by broadening search terms and generating alternatives.
                
                **Semantic Expansion:**
                - **Synonyms**: Add words with similar meaning
                - **Hypernyms**: Add more general terms (e.g., "vehicle" for "car")
                - **Hyponyms**: Add more specific terms (e.g., "sedan" for "car")
                - **Related Concepts**: Add conceptually related terms
                - **Word Embeddings**: Use vector similarity to find related terms
                - **Knowledge Base**: Use structured knowledge to expand
                
                **Query Hypotheses:**
                - Generate alternative formulations of the query
                - Create multiple versions to capture different aspects
                - Increases likelihood of matching relevant documents
                - Helps overcome vocabulary mismatch problems
                
                **RAG-Fusion:**
                - Uses LLM to generate multiple query variations
                - Retrieves results for each query variation
                - Combines results using rank fusion algorithms
                - Significantly improves recall without sacrificing precision
                - Advanced technique from recent RAG research
                
                **Benefits:**
                - Improves recall by capturing more relevant documents
                - Handles vocabulary mismatch problems
                - Makes retrieval more robust to query formulation
                - Particularly helpful for domain-specific terminology
                """)
            
            # Example of expansion and refinement
            st.markdown("#### Expansion Example")
            
            original_query = "cancer treatment advances"
            
            st.markdown("**Original Query:**")
            st.code(original_query)
            
            if semantic_expansion:
                st.markdown("**Semantically Expanded:**")
                st.code("""
"cancer treatment advances oncology therapy breakthroughs neoplasm medication 
innovations tumor remedies developments malignancy cure progress"
                """.strip())
            
            if enable_hypotheses:
                st.markdown("**Generated Hypotheses:**")
                st.code("""
1. "What are recent advances in cancer treatment?"
2. "Latest breakthroughs in oncology therapies"
3. "Innovative approaches to treating malignant tumors"
                """)
            
            if enable_rag_fusion:
                st.markdown("**RAG-Fusion Queries:**")
                st.code("""
1. "What recent advancements have occurred in cancer treatment?"
2. "Latest innovations in oncology therapeutic approaches"
3. "Breakthrough treatments for different types of cancer"
4. "Current research on improving cancer therapy outcomes"
5. "Emerging technologies for treating malignancies"
                """)
                
                st.markdown("**Fusion Process:**")
                fusion_diagram = """
1. Retrieve top-k results for each query
2. Assign scores using reciprocal rank formula: 1/(rank + k)
3. Sum scores across all queries for each document
4. Sort by combined score and return top results
                """
                st.code(fusion_diagram)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if semantic_expansion:
                st.success("✅ Semantic expansion will improve recall for most queries")
                
            if enable_hypotheses:
                st.success("✅ Query hypotheses help overcome vocabulary mismatches")
                
            if enable_rag_fusion:
                st.success("✅ RAG-fusion is a state-of-the-art technique for improving retrieval")
                st.info("ℹ️ May increase latency due to multiple retrievals but worth the cost")
    
    # ROUTING TAB
    with query_tabs[3]:
        st.subheader("Query Routing")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable query routing
            enable_routing = st.checkbox("Enable query routing", value=True)
            
            if enable_routing:
                # Query classification
                st.markdown("### Query Classification")
                
                classification_method = st.selectbox(
                    "Classification Method",
                    ["Rule-based", "ML Classifier", "LLM-based", "Hybrid"]
                )
                
                if classification_method == "ML Classifier":
                    classifier_model = st.selectbox(
                        "Classifier Model",
                        ["RandomForest", "SVM", "BERT", "Custom"]
                    )
                
                elif classification_method == "LLM-based":
                    llm_classifier = st.selectbox(
                        "LLM Classifier",
                        ["gpt-3.5-turbo", "Claude 3 Haiku", "Llama-3-8b", "Custom"]
                    )
                
                # Query categories
                st.markdown("### Query Categories")
                
                query_categories = st.multiselect(
                    "Categories to Route",
                    ["Factual", "Conceptual", "Procedural", "Temporal", "Entity-centric", "Numerical", "Comparative", "Domain-specific"],
                    default=["Factual", "Conceptual", "Procedural", "Entity-centric"]
                )
                
                # Show specialized retrievers section if routing enabled
                if query_categories:
                    st.markdown("### Specialized Retrievers")
                    
                    for category in query_categories:
                        col_a, col_b = st.columns([1, 2])
                        with col_a:
                            st.markdown(f"**{category}**")
                        with col_b:
                            st.selectbox(f"Retriever for {category}", 
                                       ["Vector Search", "BM25", "Hybrid", "Graph-based", "Custom"],
                                       key=f"retriever_{category}")
                
                # Routing strategy
                st.markdown("### Routing Strategy")
                
                routing_strategy = st.selectbox(
                    "Routing Strategy",
                    ["Single Best Path", "Multi-path with Fusion", "Confidence-based", "Adaptive"]
                )
                
                if routing_strategy == "Multi-path with Fusion":
                    st.selectbox("Fusion Method", 
                               ["Reciprocal Rank Fusion", "Linear Combination", "Weighted by Confidence"])
                
                elif routing_strategy == "Confidence-based":
                    st.slider("Confidence threshold", 0.0, 1.0, 0.7, 0.05)
                    st.checkbox("Use fallback for low confidence", value=True)
                
                # Fallback mechanism
                st.markdown("### Fallback Mechanism")
                
                enable_fallback = st.checkbox("Enable fallback mechanism", value=True)
                
                if enable_fallback:
                    fallback_strategy = st.multiselect(
                        "Fallback Strategies",
                        ["Default Retriever", "Query Simplification", "Broaden Search", "Multi-path Retrieval", "Direct LLM"],
                        default=["Default Retriever", "Query Simplification"]
                    )
                    
                    if "Query Simplification" in fallback_strategy:
                        st.checkbox("Use LLM for simplification", value=True)
        
        with col2:
            with st.expander("ℹ️ About Query Routing", expanded=True):
                info_tooltip("""
                **Query Routing** directs different types of queries to specialized retrieval paths.
                
                **Query Classification:**
                - Categorizes queries based on their characteristics
                - Enables specialized handling for different query types
                - Classification methods:
                  - **Rule-based**: Simple patterns and heuristics
                  - **ML Classifier**: Trained on labeled query data
                  - **LLM-based**: Uses AI to classify query intent and type
                
                **Specialized Retrievers:**
                - Different retrieval strategies optimized for specific query types:
                  - **Factual**: Precise fact retrieval with high confidence
                  - **Conceptual**: Broader semantic understanding
                  - **Procedural**: Step-by-step instructions
                  - **Entity-centric**: Entity-focused retrieval
                  - **Numerical**: Prioritizes numerical data
                  - **Temporal**: Time-sensitive information
                
                **Routing Strategies:**
                - **Single Best Path**: Use only the highest confidence route
                - **Multi-path**: Use multiple retrievers and combine results
                - **Confidence-based**: Route based on classification confidence
                - **Adaptive**: Learn and adjust based on feedback
                
                **Fallback Mechanism:**
                - Ensures system resilience when primary route fails
                - Provides alternative retrieval methods
                - Implements graceful degradation
                
                **Benefits:**
                - Significant precision and recall improvements
                - Better handling of diverse query types
                - More targeted and relevant responses
                - Improved system robustness
                """)
            
            # Example of query routing
            st.markdown("#### Query Routing Example")
            
            example_queries = {
                "Factual": "What is the boiling point of water?",
                "Conceptual": "How does quantum entanglement work?",
                "Procedural": "How do I reset my router?",
                "Entity-centric": "Tell me about Albert Einstein's contributions to physics."
            }
            
            routing_examples = []
            
            if enable_routing:
                for category, query in example_queries.items():
                    if category in query_categories:
                        retriever = st.session_state.get(f"retriever_{category}", "Vector Search")
                        routing_examples.append(f"**{query}** → Classified as **{category}** → Routed to **{retriever}**")
                
                for example in routing_examples:
                    st.markdown(example)
                    
                if routing_strategy == "Multi-path with Fusion":
                    st.markdown("**Multi-path Example:**")
                    st.code("""
Query: "How does aspirin reduce inflammation?"
- Classified as: Factual (0.7), Conceptual (0.6), Procedural (0.2)
- Primary path: Factual Retriever (Vector Search)
- Secondary path: Conceptual Retriever (Hybrid)
- Results fused using Reciprocal Rank Fusion
                    """)
                    
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_routing:
                st.success("✅ Query routing can significantly improve retrieval performance")
                
                if classification_method == "LLM-based":
                    st.success("✅ LLM-based classification provides the most flexible and accurate approach")
                
                if routing_strategy == "Multi-path with Fusion":
                    st.success("✅ Multi-path approach with fusion provides the best overall performance")
                
                if enable_fallback:
                    st.success("✅ Fallback mechanism ensures system robustness")
            else:
                st.info("ℹ️ Consider enabling query routing for improved performance")


#########################
# 6. RETRIEVAL STRATEGY
#########################
elif main_section == "6. Retrieval Strategy":
    st.header("6. Retrieval Strategy")
    st.markdown("Configure how information is retrieved from your vector database")
    
    # Create tabs for retrieval options
    retrieval_tabs = st.tabs(["Retrieval Methods", "Hybrid Search", "Multi-stage Retrieval", "Advanced Techniques"])
    
    # RETRIEVAL METHODS TAB
    with retrieval_tabs[0]:
        st.subheader("Core Retrieval Methods")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Semantic retrieval
            st.markdown("### Semantic Retrieval")
            
            vector_similarity = st.selectbox(
                "Vector Similarity Metric",
                ["Cosine", "Dot Product", "Euclidean", "Angular", "Manhattan"]
            )
            
            top_k = st.slider("Top K results", 1, 100, 20)
            
            semantic_threshold = st.slider(
                "Similarity threshold", 
                0.0, 1.0, 0.7, 0.01,
                help="Minimum similarity score to include in results"
            )
            
            embedding_config = st.selectbox(
                "Embedding Configuration",
                ["Use Embedding from Section 2", "Custom Embedding"]
            )
            
            if embedding_config == "Custom Embedding":
                embedding_model = st.selectbox(
                    "Embedding Model",
                    ["text-embedding-3-small", "text-embedding-3-large", "all-MiniLM-L6-v2"]
                )
            
            # Sparse retrieval
            st.markdown("### Sparse Retrieval")
            
            enable_sparse = st.checkbox("Enable sparse retrieval", value=True)
            
            if enable_sparse:
                sparse_method = st.selectbox(
                    "Sparse Method",
                    ["BM25", "TF-IDF", "SPLADE", "UniCOIL", "SPARC"]
                )
                
                if sparse_method in ["SPLADE", "UniCOIL", "SPARC"]:
                    sparse_model = st.selectbox(
                        "Neural Sparse Model",
                        ["SPLADE++", "distilSPLADE", "UniCOIL-base", "SPARC-base"],
                        help="Neural models for sparse representations"
                    )
                
                sparse_k = st.slider("Sparse top K", 1, 100, 20)
            
            # Exact match
            st.markdown("### Exact Match")
            
            enable_exact = st.checkbox("Enable exact match", value=True)
            
            if enable_exact:
                exact_method = st.selectbox(
                    "Exact Match Method",
                    ["Keyword", "Phrase", "Boolean", "Regex"]
                )
                
                exact_fields = st.multiselect(
                    "Fields to Search",
                    ["text", "title", "headings", "metadata.keywords", "metadata.entities"],
                    default=["text", "title"]
                )
                
                exact_boost = st.slider("Exact match boost factor", 1.0, 10.0, 1.5, 0.1)
                
            # Metadata filtering
            st.markdown("### Metadata Filtering")
            
            enable_filters = st.checkbox("Enable metadata filters", value=True)
            
            if enable_filters:
                filter_example = st.text_area(
                    "Example filter query (JSON)",
                    """{
  "$and": [
    {"category": {"$in": ["Technical", "Scientific"]}},
    {"creation_date": {"$gte": "2023-01-01"}},
    {"$or": [
      {"author": "John Smith"},
      {"department": "Research"}
    ]}
  ]
}"""
                )
                
                st.checkbox("Generate filters automatically from query", value=True)
                st.checkbox("Apply pre-filtering before vector search", value=True)
        
        with col2:
            with st.expander("ℹ️ About Retrieval Methods", expanded=True):
                info_tooltip("""
                **Retrieval Methods** determine how relevant information is found in your document collection.
                
                **Semantic Retrieval (Dense):**
                - Uses vector embeddings to capture meaning
                - Finds conceptually similar content even with different words
                - Similarity metrics:
                  - **Cosine**: Most common, angle between vectors, normalized for length
                  - **Dot Product**: Fast but sensitive to vector magnitude
                  - **Euclidean**: Actual distance in vector space
                
                **Sparse Retrieval:**
                - **BM25/TF-IDF**: Traditional keyword matching with term weighting
                - **SPLADE/UniCOIL/SPARC**: Neural sparse retrievers that combine benefits of keywords and semantic understanding
                - Better for exact term matches and rare terms
                
                **Exact Match:**
                - Ensures specific terms or phrases are present
                - Useful for technical terms, names, or specific identifiers
                - Can be boosted to prioritize exact matches
                
                **Metadata Filtering:**
                - Narrows results based on structured metadata
                - Examples: filter by date, author, category, etc.
                - Can dramatically improve precision
                - Should be applied early in retrieval for efficiency
                
                **When to Use Each:**
                - **Semantic**: Primary method for most queries
                - **Sparse**: To improve exact term matching
                - **Exact Match**: For specific terminology
                - **Filters**: When you need to narrow scope by metadata
                
                **Best Practice**: Combine multiple methods in a hybrid approach
                """)
            
            # Visual comparison
            st.markdown("#### Retrieval Method Comparison")
            
            comparison_data = pd.DataFrame({
                "Method": ["Semantic (Dense)", "Sparse", "Exact Match", "Metadata Filter"],
                "Strengths": [
                    "Conceptual understanding, handles synonyms",
                    "Term importance, rare word handling",
                    "Perfect precision for specific terms",
                    "Structured data filtering"
                ],
                "Weaknesses": [
                    "May miss exact terminology",
                    "Limited semantic understanding",
                    "No flexibility or fuzzy matching",
                    "Requires structured metadata"
                ]
            })
            
            st.dataframe(comparison_data, hide_index=True)
            
            # Example comparison
            st.markdown("#### Example Query Results")
            
            example_query = "adverse effects of metformin in elderly patients"
            
            st.markdown(f"**Query:** `{example_query}`")
            
            st.markdown("**Semantic (Dense) Results:**")
            semantic_results = """
1. "Side effects of metformin treatment in geriatric populations" (Score: 0.89)
2. "Safety profile of biguanides in older adults with diabetes" (Score: 0.82)
3. "Complications from oral hypoglycemics in senior patients" (Score: 0.78)
"""
            st.code(semantic_results)
            
            if enable_sparse:
                st.markdown("**Sparse Results:**")
                sparse_results = """
1. "Metformin adverse effects in elderly patients with renal impairment" (Score: 0.92)
2. "Common adverse effects of metformin: GI distress in elderly patients" (Score: 0.85)
3. "Patient guidelines for managing metformin side effects" (Score: 0.73)
"""
                st.code(sparse_results)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_sparse and enable_exact:
                st.success("✅ Using multiple retrieval methods will provide the best results")
                
            if vector_similarity == "Cosine":
                st.success("✅ Cosine similarity is a good default choice")
                
            if enable_filters:
                st.success("✅ Metadata filtering can dramatically improve precision")
                
            if sparse_method in ["SPLADE", "UniCOIL", "SPARC"] if enable_sparse else False:
                st.success("✅ Neural sparse methods offer better performance than traditional TF-IDF/BM25")
    
    # HYBRID SEARCH TAB
    with retrieval_tabs[1]:
        st.subheader("Hybrid Search Configuration")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable hybrid search
            enable_hybrid = st.checkbox("Enable hybrid search", value=True)
            
            if enable_hybrid:
                # Hybrid method
                hybrid_method = st.selectbox(
                    "Hybrid Method",
                    ["Linear Combination", "RRF (Reciprocal Rank Fusion)", "Learning to Rank", "Neural Ranker"]
                )
                
                if hybrid_method == "Linear Combination":
                    col_w1, col_w2 = st.columns(2)
                    with col_w1:
                        dense_weight = st.slider("Dense weight", 0.0, 1.0, 0.7, 0.05)
                    with col_w2:
                        sparse_weight = st.slider("Sparse weight", 0.0, 1.0, 1 - dense_weight, 0.05, disabled=True)
                    
                elif hybrid_method == "RRF":
                    rrf_k = st.slider("RRF constant (k)", 1, 100, 60)
                    
                elif hybrid_method == "Learning to Rank":
                    ranker_model = st.selectbox(
                        "LTR Model",
                        ["LambdaMART", "RankNet", "MART", "XGBoost Ranker"]
                    )
                    
                    st.checkbox("Use click data for training", value=True)
                    st.checkbox("Include contextual features", value=True)
                    
                elif hybrid_method == "Neural Ranker":
                    neural_ranker = st.selectbox(
                        "Neural Ranking Model",
                        ["MonoT5", "BERT Ranker", "ColBERTv2", "Custom"]
                    )
                    
                    st.checkbox("Pre-filter with BM25", value=True)
                
                # Components to combine
                st.markdown("### Components to Combine")
                
                components = st.multiselect(
                    "Retrieval Components",
                    ["Dense Vectors", "BM25", "TF-IDF", "SPLADE", "Exact Match", "Lexical Signals", "Neural Sparse"],
                    default=["Dense Vectors", "BM25"]
                )
                
                # Field weights
                st.markdown("### Field Weights")
                
                enable_field_weights = st.checkbox("Enable field-specific weights", value=True)
                
                if enable_field_weights:
                    col_f1, col_f2 = st.columns(2)
                    with col_f1:
                        st.slider("Title weight", 0.1, 5.0, 1.5, 0.1)
                        st.slider("Content weight", 0.1, 5.0, 1.0, 0.1)
                        st.slider("Headers weight", 0.1, 5.0, 1.2, 0.1)
                    with col_f2:
                        st.slider("Code weight", 0.1, 5.0, 0.8, 0.1)
                        st.slider("Tables weight", 0.1, 5.0, 0.8, 0.1)
                        st.slider("Lists weight", 0.1, 5.0, 0.9, 0.1)
                
                # Hybrid calibration
                st.markdown("### Score Calibration")
                
                calibration_method = st.selectbox(
                    "Score Calibration Method",
                    ["None", "Min-Max Normalization", "Z-Score", "Log Normalization", "Learned Calibration"]
                )
                
                if calibration_method == "Learned Calibration":
                    st.selectbox("Calibration Model", ["Isotonic Regression", "Platt Scaling", "Custom"])
        
        with col2:
            with st.expander("ℹ️ About Hybrid Search", expanded=True):
                info_tooltip("""
                **Hybrid Search** combines multiple retrieval methods for better performance.
                
                **Hybrid Methods:**
                - **Linear Combination**: Simple weighted sum of scores
                  - Easy to implement and tune
                  - Fast to execute
                  - Needs score normalization
                
                - **RRF** (Reciprocal Rank Fusion):
                  - Combines results based on rank rather than score
                  - Formula: score = 1/(rank + k)
                  - More robust to score differences
                  - Works well with diverse retrievers
                
                - **Learning to Rank**:
                  - Uses ML model trained on relevance data
                  - Can incorporate many features
                  - Requires training data
                  - Best performance but most complex
                
                - **Neural Ranker**:
                  - Uses deep learning for ranking
                  - Can process query-document pairs together
                  - Higher accuracy but more computational cost
                
                **Components:**
                - **Dense Vectors**: Semantic understanding
                - **BM25/TF-IDF**: Traditional lexical matching
                - **SPLADE/Neural Sparse**: Neural keyword matching
                - **Exact Match**: Boolean keyword presence
                - **Lexical Signals**: Special text features (e.g., title matches)
                
                **Field Weights:**
                - Assign different importance to different document fields
                - Title/headers often more important than body text
                - Essential for structured documents
                
                **Score Calibration:**
                - Ensures scores from different methods are comparable
                - Essential for linear combination methods
                - Less important for rank-based fusion
                
                **Recommendation:** Start with RRF (robust and simple), then experiment with Linear Combination or Learning to Rank as you gather more data.
                """)
            
            # Visual explanation
            st.markdown("#### Hybrid Search Process")
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*b4M7o7T3MLLF18Lmt3Ja1w.png",
                   caption="Hybrid search combining dense and sparse retrievers")
            
            # Example results
            st.markdown("#### Example Hybrid Results")
            
            if enable_hybrid:
                st.markdown("**Query:** `treatment options for type 2 diabetes`")
                
                hybrid_results = """
# Individual retrievers:

Dense Results:
1. "Current therapeutic approaches for managing type 2 diabetes" (0.89)
2. "Treatment strategies for adult-onset diabetes mellitus" (0.84)
3. "Pharmacological interventions for glycemic control" (0.81)

BM25 Results:
1. "Treatment options for type 2 diabetes: a comprehensive review" (12.7)
2. "Type 2 diabetes: comparing treatment modalities" (10.2)
3. "Surgical treatment options for type 2 diabetes patients" (9.4)

# Hybrid Results (after fusion):
1. "Treatment options for type 2 diabetes: a comprehensive review"
2. "Current therapeutic approaches for managing type 2 diabetes"
3. "Type 2 diabetes: comparing treatment modalities"
4. "First-line medications for treating type 2 diabetes"
5. "Lifestyle interventions as treatment for type 2 diabetes"
                """
                st.code(hybrid_results)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_hybrid:
                if hybrid_method == "RRF":
                    st.success("✅ RRF is an excellent choice for robust fusion")
                    
                if hybrid_method == "Linear Combination" and calibration_method == "None":
                    st.warning("⚠️ Linear combination works best with score calibration")
                    
                if len(components) >= 3:
                    st.success("✅ Using multiple components provides better coverage")
                    
                if enable_field_weights:
                    st.success("✅ Field weights can significantly improve relevance")
            else:
                st.warning("⚠️ Consider enabling hybrid search for better performance")
    
    # MULTI-STAGE RETRIEVAL TAB
    with retrieval_tabs[2]:
        st.subheader("Multi-Stage Retrieval")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable multi-stage
            enable_multistage = st.checkbox("Enable multi-stage retrieval", value=True)
            
            if enable_multistage:
                # Stage configuration
                st.markdown("### Retrieval Stages")
                
                num_stages = st.slider("Number of stages", 2, 5, 2)
                
                # Configure each stage
                for i in range(1, num_stages + 1):
                    with st.expander(f"Stage {i} Configuration", expanded=i==1):
                        stage_type = st.selectbox(
                            f"Stage {i} Type",
                            ["Retrieval", "Filtering", "Reranking", "Fusion", "Augmentation"],
                            index=0 if i==1 else 2 if i==2 else 0,
                            key=f"stage_{i}_type"
                        )
                        
                        if stage_type == "Retrieval":
                            st.selectbox(
                                f"Retrieval Method",
                                ["BM25", "Dense", "Hybrid", "Filter-first"],
                                key=f"stage_{i}_method"
                            )
                            
                            st.slider(f"Top K", 10, 1000, 100 if i==1 else 20,
                                    key=f"stage_{i}_k")
                            
                        elif stage_type == "Filtering":
                            st.multiselect(
                                f"Filter Types",
                                ["Metadata", "Score Threshold", "Semantic Similarity", "Content Type"],
                                default=["Metadata", "Score Threshold"],
                                key=f"stage_{i}_filters"
                            )
                            
                            st.slider(f"Filter Threshold", 0.0, 1.0, 0.5, 0.05,
                                    key=f"stage_{i}_threshold")
                            
                        elif stage_type == "Reranking":
                            st.selectbox(
                                f"Reranker Type",
                                ["Cross-Encoder", "Neural Ranker", "Feature-based", "LLM Reranker"],
                                key=f"stage_{i}_reranker"
                            )
                            
                            st.slider(f"Rerank Top K", 5, 100, 20,
                                    key=f"stage_{i}_rerank_k")
                            
                        elif stage_type == "Fusion":
                            st.selectbox(
                                f"Fusion Method",
                                ["RRF", "CombSUM", "CombMNZ", "Weighted Fusion"],
                                key=f"stage_{i}_fusion"
                            )
                            
                            st.multiselect(
                                f"Sources to Fuse",
                                ["Stage 1 Results", "Stage 2 Results", "Alternative Retrievers", "External Sources"],
                                default=["Stage 1 Results"],
                                key=f"stage_{i}_sources"
                            )
                            
                        elif stage_type == "Augmentation":
                            st.multiselect(
                                f"Augmentation Methods",
                                ["Related Entities", "Knowledge Graph", "Generated Content", "Cross References"],
                                default=["Related Entities"],
                                key=f"stage_{i}_augmentation"
                            )
                            
                            st.checkbox(f"Keep Original Results", value=True,
                                      key=f"stage_{i}_keep_original")
                
                # Stage interaction
                st.markdown("### Stage Interaction")
                
                interaction_mode = st.selectbox(
                    "Stage Interaction Mode",
                    ["Sequential", "Parallel with Fusion", "Conditional", "Hybrid"]
                )
                
                if interaction_mode == "Conditional":
                    st.text_area(
                        "Condition Rules (pseudo-code)",
                        """if first_stage_results.confidence < 0.7:
    run_alternative_retrieval()
else:
    proceed_to_reranking()"""
                    )
                    
                elif interaction_mode == "Parallel with Fusion":
                    st.selectbox("Fusion Method", ["RRF", "Weighted Combination", "Max Score"])
        
        with col2:
            with st.expander("ℹ️ About Multi-Stage Retrieval", expanded=True):
                info_tooltip("""
                **Multi-Stage Retrieval** uses a pipeline of progressive steps to improve results.
                
                **Common Stages:**
                
                1. **Initial Retrieval**:
                   - Fast, recall-oriented retrieval
                   - Usually BM25 or vector search
                   - Retrieves a larger candidate set (e.g., top-100)
                
                2. **Filtering**:
                   - Narrows down candidates
                   - Uses metadata or score thresholds
                   - Removes obviously irrelevant results
                
                3. **Reranking**:
                   - More expensive, precision-oriented
                   - Uses cross-encoders or neural rankers
                   - Reorders candidates for better precision
                
                4. **Fusion**:
                   - Combines results from multiple sources
                   - Integrates different retrieval strategies
                   - Improves coverage and diversity
                
                5. **Augmentation**:
                   - Adds related content not directly retrieved
                   - May use knowledge graphs or entity linking
                   - Enhances context for response generation
                
                **Stage Interaction Modes:**
                - **Sequential**: Each stage processes output from previous
                - **Parallel**: Multiple paths run independently then combine
                - **Conditional**: Dynamic path based on intermediate results
                - **Hybrid**: Combines aspects of multiple approaches
                
                **Benefits:**
                - Balances recall and precision
                - Enables use of expensive methods on smaller candidate sets
                - More flexible retrieval pipeline
                - Better handling of different query types
                """)
            
            # Visual representation
            st.markdown("#### Multi-Stage Process")
            st.image("https://miro.medium.com/v2/resize:fit:1400/1*sYqp0FQ6y0HeslTuuBJHew.png",
                   caption="Multi-stage retrieval pipeline")
            
            # Example results
            if enable_multistage:
                st.markdown("#### Multi-Stage Example")
                
                multistage_example = """
Query: "What are the environmental impacts of electric vehicles?"

Stage 1: Initial Retrieval (BM25)
- Retrieved 100 candidate documents
- Took 120ms
- Score range: 3.2 - 12.8

Stage 2: Cross-Encoder Reranking
- Reranked top 50 documents
- Took 350ms
- New top document: "Life-cycle assessment of electric vehicle environmental impact"
  (moved from position #8 to #1)
- Precision@5 improved from 0.6 to 0.9

Final Results:
1. "Life-cycle assessment of electric vehicle environmental impact" (0.95)
2. "Comparative environmental analysis of EVs vs. conventional vehicles" (0.92)
3. "Battery production and disposal: environmental challenges for EVs" (0.89)
4. "Carbon footprint reduction potential of electric transportation" (0.85)
5. "Environmental trade-offs of electric vs. combustion vehicles" (0.83)
                """
                st.code(multistage_example)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_multistage:
                if num_stages >= 2:
                    st.success("✅ Multi-stage retrieval can significantly improve relevance")
                    
                # Check if stage 1 is retrieval with higher K and stage 2 is reranking
                stage1_type = st.session_state.get('stage_1_type', '')
                stage2_type = st.session_state.get('stage_2_type', '')
                stage1_k = st.session_state.get('stage_1_k', 0)
                stage2_k = st.session_state.get('stage_2_k', 0)
                
                if stage1_type == "Retrieval" and stage2_type == "Reranking" and stage1_k > stage2_k:
                    st.success("✅ Good pattern: broad retrieval followed by focused reranking")
                
                if stage1_type == "Retrieval" and st.session_state.get('stage_1_method', '') == "BM25":
                    st.info("ℹ️ Consider using Hybrid retrieval for the first stage for better recall")
            else:
                st.info("ℹ️ Multi-stage retrieval is recommended for production systems")
    
    # ADVANCED TECHNIQUES TAB
    with retrieval_tabs[3]:
        st.subheader("Advanced Retrieval Techniques")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Iterative retrieval
            st.markdown("### Iterative Retrieval")
            
            enable_iterative = st.checkbox("Enable iterative retrieval", value=False)
            
            if enable_iterative:
                iteration_method = st.selectbox(
                    "Iteration Method",
                    ["Query Expansion", "Relevance Feedback", "HyDE (Hypothetical Document)", "Self-Query"]
                )
                
                if iteration_method == "Query Expansion":
                    st.checkbox("Use retrieved content for expansion", value=True)
                    st.slider("Maximum iterations", 1, 5, 2)
                    
                elif iteration_method == "Relevance Feedback":
                    st.selectbox("Feedback Collection", ["Implicit", "Explicit", "Pseudo-feedback"])
                    st.slider("Result refinement strength", 0.1, 1.0, 0.5, 0.05)
                    
                elif iteration_method == "HyDE":
                    hyde_model = st.selectbox(
                        "HyDE LLM",
                        ["gpt-3.5-turbo", "gpt-4", "Claude 3 Sonnet", "Llama-3-8b"]
                    )
                    
                    st.checkbox("Keep original results", value=True)
                    st.slider("HyDE weight", 0.1, 1.0, 0.7, 0.05)
                    
                elif iteration_method == "Self-Query":
                    st.selectbox(
                        "Self-Query LLM",
                        ["gpt-3.5-turbo", "gpt-4", "Claude 3 Sonnet", "Llama-3-8b"]
                    )
                    
                    st.checkbox("Enable filter extraction", value=True)
                    st.checkbox("Enable query refinement", value=True)
            
            # Document representation
            st.markdown("### Advanced Document Representation")
            
            doc_repr_methods = st.multiselect(
                "Document Representation Methods",
                ["Multi-vector", "ColBERT-style", "Hierarchical", "Graph-based", "Entity-centric"],
                default=[]
            )
            
            if "Multi-vector" in doc_repr_methods:
                multi_vector_method = st.selectbox(
                    "Multi-Vector Method",
                    ["Paragraph Vectors", "Sentence Vectors", "Dynamic Segmentation", "Section-based"]
                )
                
                st.slider("Vectors per document", 1, 50, 5)
                st.checkbox("Use aggregate document vector", value=True)
                
            if "ColBERT-style" in doc_repr_methods:
                st.number_input("Maximum tokens per document", 32, 512, 128)
                st.selectbox("Token Interaction", ["MaxSim", "Sum MaxSim", "Attention-based"])
                
            # Knowledge-enhanced retrieval
            st.markdown("### Knowledge-Enhanced Retrieval")
            
            enable_knowledge = st.checkbox("Enable knowledge-enhanced retrieval", value=False)
            
            if enable_knowledge:
                knowledge_sources = st.multiselect(
                    "Knowledge Sources",
                    ["Knowledge Graph", "Entity Database", "Structured Knowledge", "External APIs", "Domain Ontology"],
                    default=["Knowledge Graph", "Entity Database"]
                )
                
                if "Knowledge Graph" in knowledge_sources:
                    kg_source = st.selectbox(
                        "Knowledge Graph Source",
                        ["Built-in", "Wikidata", "Custom KG", "Neo4j", "Generated from Documents"]
                    )
                    
                    st.multiselect(
                        "Knowledge Graph Relations",
                        ["is_related_to", "is_a", "part_of", "causes", "treats", "located_in", "custom_relations"],
                        default=["is_related_to", "is_a"]
                    )
                
                knowledge_integration = st.selectbox(
                    "Knowledge Integration Method",
                    ["Query Augmentation", "Result Augmentation", "Guided Retrieval", "Entity-centric Search"]
                )
        
        with col2:
            with st.expander("ℹ️ About Advanced Retrieval Techniques", expanded=True):
                info_tooltip("""
                **Advanced Retrieval Techniques** incorporate more sophisticated approaches beyond basic search.
                
                **Iterative Retrieval:**
                - Performs multiple search rounds to refine results
                - Methods include:
                  - **Query Expansion**: Uses initial results to expand query
                  - **Relevance Feedback**: Uses feedback to improve results
                  - **HyDE**: Uses LLM to generate a hypothetical ideal document
                  - **Self-Query**: LLM reformulates query based on initial results
                
                **Advanced Document Representation:**
                - **Multi-vector**: Multiple embeddings per document
                  - Better captures different aspects or sections
                  - More granular similarity matching
                
                - **ColBERT-style**: Stores token-level embeddings
                  - Enables fine-grained matching between query and document tokens
                  - Better alignment but higher storage requirements
                
                - **Hierarchical**: Embeddings at different levels
                  - Document, section, paragraph, sentence
                  - Enables multi-level matching
                
                **Knowledge-Enhanced Retrieval:**
                - Incorporates structured knowledge into retrieval
                - Enables reasoning beyond pure text matching
                - Knowledge sources:
                  - **Knowledge Graphs**: Structured relationships between entities
                  - **Entity Databases**: Detailed information about entities
                  - **Domain Ontologies**: Formal representation of domain concepts
                
                **Benefits:**
                - Higher precision for complex queries
                - Better handling of ambiguous questions
                - Incorporation of external knowledge
                - Support for reasoning-based retrieval
                
                **Trade-offs:**
                - Increased complexity
                - Higher computational requirements
                - May need domain-specific knowledge
                """)
            
            # Examples of advanced techniques
            st.markdown("#### Advanced Technique Examples")
            
            if enable_iterative:
                st.markdown("**Iterative Retrieval Example:**")
                if iteration_method == "HyDE":
                    hyde_example = """
Query: "How do electric vehicles impact urban air quality?"

Step 1: Generate hypothetical ideal document with LLM:
"Electric vehicles have significant effects on urban air quality. 
Unlike conventional vehicles with internal combustion engines that 
emit pollutants like NOx, CO, and particulate matter, EVs produce 
zero tailpipe emissions. Studies in major cities have shown that 
increased EV adoption correlates with measurable reductions in 
ground-level pollution, especially in high-traffic areas..."

Step 2: Embed this hypothetical document

Step 3: Retrieve based on similarity to this ideal document

Result: More relevant documents that match the expected content, 
even if they use different terminology
                    """
                    st.code(hyde_example)
            
            if "ColBERT-style" in doc_repr_methods:
                st.markdown("**ColBERT Late Interaction Example:**")
                colbert_example = """
Document: "The treatment of diabetes involves insulin therapy..."

Traditional approach: Single embedding vector for entire document

ColBERT approach: Token-level embeddings:
- embedding("The")
- embedding("treatment")
- embedding("of")
- embedding("diabetes")
- embedding("involves")
- embedding("insulin")
- embedding("therapy")

Query: "insulin for diabetes"
Query tokens: embedding("insulin"), embedding("for"), embedding("diabetes")

Matching: Each query token finds best matching document token
- "insulin" → "insulin" (perfect match)
- "diabetes" → "diabetes" (perfect match)
- "for" → best available match

This approach captures token-level alignments that single-vector 
approaches might miss.
                """
                st.code(colbert_example)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_iterative and iteration_method == "HyDE":
                st.success("✅ HyDE is a powerful technique for complex queries")
                
            if "Multi-vector" in doc_repr_methods:
                st.success("✅ Multi-vector representation improves retrieval granularity")
                
            if "ColBERT-style" in doc_repr_methods:
                st.success("✅ ColBERT-style late interaction provides superior matching")
                st.warning("⚠️ But requires significantly more storage and computation")
                
            if enable_knowledge and "Knowledge Graph" in knowledge_sources:
                st.success("✅ Knowledge graph integration helps with entity-centric queries")

###############################
# 7. RERANKING & FUSION
###############################
elif main_section == "7. Reranking & Fusion":
    st.header("7. Reranking & Fusion")
    st.markdown("Configure advanced reranking and result fusion methods to improve retrieval precision")
    
    # Create tabs for reranking options
    reranking_tabs = st.tabs(["Rerankers", "Fusion Methods", "Ensemble Techniques", "Custom Scoring"])
    
    # RERANKERS TAB
    with reranking_tabs[0]:
        st.subheader("Reranking Models")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable reranking
            enable_reranking = st.checkbox("Enable reranking", value=True)
            
            if enable_reranking:
                # Reranker model selection
                reranker_type = st.selectbox(
                    "Reranker Type",
                    ["Cross-Encoder", "LLM-based", "Feature-based", "Multi-stage"]
                )
                
                if reranker_type == "Cross-Encoder":
                    cross_encoder = st.selectbox(
                        "Cross-Encoder Model",
                        ["ms-marco-MiniLM-L-6-v2", "ms-marco-MiniLM-L-12-v2", 
                         "cross-encoder/ms-marco-TinyBERT-L-2", "cross-encoder/ms-marco-electra-base",
                         "BAAI/bge-reranker-base", "BAAI/bge-reranker-large", "Custom"]
                    )
                    
                    # Cross-encoder parameters
                    col_ce1, col_ce2 = st.columns(2)
                    with col_ce1:
                        max_to_rerank = st.slider("Maximum documents to rerank", 10, 1000, 100)
                        batch_size = st.slider("Batch size", 4, 64, 16)
                    
                    with col_ce2:
                        ce_threshold = st.slider("Score threshold", 0.0, 1.0, 0.5, 0.05)
                        normalize_scores = st.checkbox("Normalize scores", value=True)
                    
                elif reranker_type == "LLM-based":
                    llm_reranker = st.selectbox(
                        "LLM Reranker",
                        ["gpt-3.5-turbo", "gpt-4", "Claude 3 Haiku", "Claude 3 Sonnet", "Llama-3-8b", "Custom"]
                    )
                    
                    # LLM reranker parameters
                    st.slider("Maximum documents to rerank", 3, 50, 10)
                    
                    reranker_prompt_template = st.text_area(
                        "Reranking Prompt Template",
                        """Rate the relevance of the following document to the query on a scale from 0 to 10.
                        
Query: {{query}}

Document: {{document}}

Relevance score (0-10):""",
                        height=150
                    )
                    
                    st.checkbox("Extract numerical score from response", value=True)
                    st.checkbox("Include reasoning in scoring", value=True)
                
                elif reranker_type == "Feature-based":
                    feature_based_model = st.selectbox(
                        "Feature-based Model",
                        ["LambdaMART", "RankNet", "XGBoost", "CatBoost", "Custom"]
                    )
                    
                    # Feature selection
                    st.multiselect(
                        "Ranking Features",
                        ["BM25 Score", "Vector Similarity", "Exact Match Count", "Position", 
                         "Document Length", "Query Term Density", "Term Frequency", 
                         "Proximity Features", "Document Freshness", "Click-through Rate"],
                        default=["BM25 Score", "Vector Similarity", "Exact Match Count"]
                    )
                    
                    st.number_input("Maximum documents to rerank", 10, 500, 100)
                    st.checkbox("Use trained model weights", value=True)
                
                elif reranker_type == "Multi-stage":
                    st.markdown("### Multi-stage Reranking Pipeline")
                    
                    # Configure reranking stages
                    stages = st.multiselect(
                        "Reranking Stages",
                        ["Coarse Filtering", "Cross-Encoder", "LLM Reranker", "Feature-based Scoring"],
                        default=["Coarse Filtering", "Cross-Encoder"]
                    )
                    
                    # Configure each stage
                    if "Coarse Filtering" in stages:
                        st.slider("Coarse filter: documents to keep", 50, 500, 100)
                    
                    if "Cross-Encoder" in stages:
                        st.selectbox(
                            "Cross-Encoder Model", 
                            ["ms-marco-MiniLM-L-6-v2", "BAAI/bge-reranker-base"], 
                            key="ms_cross_encoder"
                        )
                        st.slider("Cross-encoder: documents to keep", 10, 100, 20)
                    
                    if "LLM Reranker" in stages:
                        st.selectbox(
                            "LLM for final reranking", 
                            ["gpt-3.5-turbo", "Claude 3 Haiku"], 
                            key="ms_llm_reranker"
                        )
                        st.slider("LLM reranker: documents to keep", 3, 20, 5)
                
                # Integration settings
                st.markdown("### Reranker Integration")
                
                reranking_position = st.radio(
                    "When to Apply Reranking",
                    ["After Initial Retrieval", "After Fusion", "Before and After Fusion"]
                )
                
                # Performance optimization
                st.markdown("### Performance Optimization")
                
                st.checkbox("Cache reranker results", value=True)
                st.checkbox("Skip reranking for simple queries", value=False)
                st.checkbox("Apply reranking threshold filter", value=True)
        
        with col2:
            with st.expander("ℹ️ About Reranking", expanded=True):
                info_tooltip("""
                **Reranking** improves result precision by applying more sophisticated (and often more expensive) relevance models to reorder initial retrieval results.
                
                **Reranker Types:**
                
                **Cross-Encoder Models:**
                - Process query and document together (not separately)
                - Much higher accuracy than bi-encoders
                - Models like ms-marco-MiniLM specifically trained for ranking
                - Trade-off: Computationally expensive, can't be pre-computed
                
                **LLM-based Rerankers:**
                - Use LLMs to judge document relevance to query
                - Can provide reasoning and nuanced evaluation
                - Highly accurate but most expensive approach
                - Best for critical applications or final ranking stage
                
                **Feature-based Rerankers:**
                - Use machine learning with multiple ranking signals
                - Combine lexical, semantic, and metadata features
                - Can be trained on your specific data
                - Popular algorithms: LambdaMART, RankNet, etc.
                
                **Multi-stage Reranking:**
                - Apply progressively more expensive and accurate rankers
                - Filter down candidate set at each stage
                - Balances quality and performance
                
                **Best Practices:**
                - Apply reranking to small candidate sets (top-k results)
                - Consider cost vs. accuracy trade-offs
                - Use multi-stage for production systems
                - Cache reranking results when possible
                
                **When to Use:**
                - When precision is critical
                - For complex, ambiguous queries
                - When initial retrieval quality is insufficient
                - For systems where ranking quality directly impacts business outcomes
                """)
            
            # Example of reranking
            st.markdown("#### Reranking Example")
            
            if enable_reranking and reranker_type == "Cross-Encoder":
                example_query = "treatment options for type 2 diabetes"
                
                st.markdown("**Initial Retrieval Results:**")
                
                initial_results = """
1. "Type 2 diabetes: comparing treatment modalities" (0.82)
2. "Managing hyperglycemia in patients with metabolic disorders" (0.79)
3. "Treatment options for type 2 diabetes: a comprehensive review" (0.77)
4. "Insulin therapy protocols for diabetic patients" (0.75)
5. "Lifestyle interventions as treatment for type 2 diabetes" (0.72)
                """
                st.code(initial_results)
                
                st.markdown("**After Cross-Encoder Reranking:**")
                
                reranked_results = """
1. "Treatment options for type 2 diabetes: a comprehensive review" (0.94)
2. "Lifestyle interventions as treatment for type 2 diabetes" (0.89)
3. "Type 2 diabetes: comparing treatment modalities" (0.85)
4. "First-line medications for treating type 2 diabetes" (0.82)
5. "Insulin therapy protocols for diabetic patients" (0.76)
                """
                st.code(reranked_results)
                
                st.markdown("""
                **Note:** The cross-encoder moved the most relevant document from position #3 to #1, 
                and brought "Lifestyle interventions..." from #5 to #2 - significantly improving the ranking.
                """)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_reranking:
                if reranker_type == "Cross-Encoder":
                    st.success("✅ Cross-encoders provide excellent reranking at moderate cost")
                    
                elif reranker_type == "LLM-based":
                    st.success("✅ LLM reranking provides best quality but at higher cost")
                    st.info("ℹ️ Consider using in a multi-stage pipeline for cost efficiency")
                    
                elif reranker_type == "Multi-stage":
                    st.success("✅ Multi-stage approach balances quality and performance")
                    st.success("✅ Excellent choice for production systems")
                
                if max_to_rerank > 100 if reranker_type == "Cross-Encoder" else False:
                    st.warning("⚠️ Reranking large sets may impact performance - consider reducing")
            else:
                st.info("ℹ️ Reranking can significantly improve precision - consider enabling")
    
    # FUSION METHODS TAB
    with reranking_tabs[1]:
        st.subheader("Results Fusion Methods")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable fusion
            enable_fusion = st.checkbox("Enable result fusion", value=True)
            
            if enable_fusion:
                # Fusion method selection
                fusion_method = st.selectbox(
                    "Primary Fusion Method",
                    ["Reciprocal Rank Fusion (RRF)", "CombSUM", "CombMNZ", "Linear Interpolation", 
                     "Weighted Fusion", "Softmax Normalization", "Bayesian Fusion"]
                )
                
                if fusion_method == "Reciprocal Rank Fusion (RRF)":
                    rrf_k = st.slider("RRF constant (k)", 1, 100, 60)
                    st.info("Higher k values give more weight to lower-ranked results")
                    
                elif fusion_method == "Linear Interpolation":
                    st.slider("Dense retrieval weight", 0.0, 1.0, 0.7, 0.05)
                    st.slider("Sparse retrieval weight", 0.0, 1.0, 0.3, 0.05)
                
                elif fusion_method == "Weighted Fusion":
                    st.markdown("### Component Weights")
                    
                    col_w1, col_w2 = st.columns(2)
                    with col_w1:
                        st.slider("Dense vector weight", 0.0, 1.0, 0.6, 0.05)
                        st.slider("Lexical (BM25) weight", 0.0, 1.0, 0.3, 0.05)
                    
                    with col_w2:
                        st.slider("Exact match weight", 0.0, 1.0, 0.1, 0.05)
                        st.slider("Reranker weight", 0.0, 1.0, 0.0, 0.05)
                
                # Sources to fuse
                st.markdown("### Fusion Sources")
                
                fusion_sources = st.multiselect(
                    "Sources to Fuse",
                    ["Dense Vector Search", "BM25/Sparse Search", "Hybrid Search", 
                     "Keyword Search", "Knowledge Graph Results", "Query Variations"],
                    default=["Dense Vector Search", "BM25/Sparse Search"]
                )
                
                if "Query Variations" in fusion_sources:
                    st.number_input("Number of query variations", 2, 10, 3)
                
                # Score normalization
                st.markdown("### Score Normalization")
                
                normalization_method = st.selectbox(
                    "Score Normalization Method",
                    ["Min-Max Scaling", "Z-Score", "Log Normalization", "Softmax", "None"]
                )
                
                # Post-fusion processing
                st.markdown("### Post-Fusion Processing")
                
                post_fusion_steps = st.multiselect(
                    "Post-Fusion Steps",
                    ["Duplicate Removal", "Diversity Reranking", "Threshold Filtering", "Clustering"],
                    default=["Duplicate Removal"]
                )
                
                if "Diversity Reranking" in post_fusion_steps:
                    st.slider("Diversity factor", 0.0, 1.0, 0.3, 0.05)
                
                if "Clustering" in post_fusion_steps:
                    st.slider("Number of clusters", 2, 10, 3)
        
        with col2:
            with st.expander("ℹ️ About Result Fusion", expanded=True):
                info_tooltip("""
                **Result Fusion** combines multiple result sets or ranking signals to improve overall retrieval quality.
                
                **Fusion Methods:**
                
                **Reciprocal Rank Fusion (RRF):**
                - Scores based on rank position: score = 1/(rank + k)
                - Parameter k controls influence of lower ranks
                - Robust to score differences between systems
                - Excellent general-purpose fusion method
                
                **CombSUM:**
                - Simple sum of normalized scores
                - Requires good score normalization
                - Works well with similar retrieval methods
                
                **CombMNZ:**
                - CombSUM multiplied by number of systems that retrieved the document
                - Rewards documents found by multiple methods
                - Good for consensus building
                
                **Linear Interpolation:**
                - Weighted average of normalized scores
                - Simple and effective for two sources
                - Requires careful weight tuning
                
                **Weighted Fusion:**
                - More flexible weighting of multiple sources
                - Can incorporate many retrieval signals
                - Needs optimization for best results
                
                **Score Normalization:**
                - Essential for methods that combine raw scores
                - Makes scores comparable across different retrievers
                - Methods: min-max, z-score, log normalization, etc.
                
                **Benefits:**
                - Combines strengths of different retrieval approaches
                - More robust than any single retrieval method
                - Improves both recall and precision
                - Handles different types of queries better
                
                **When to Use:**
                - When combining different retrieval methods
                - For important applications requiring high quality
                - When different retrievers excel at different query types
                - To balance semantic vs. keyword search
                """)
                
            # Example of fusion
            st.markdown("#### Fusion Example")
            
            if enable_fusion:
                st.markdown("**Dense Vector Results:**")
                dense_results = """
1. "Diabetes treatment guidelines" (0.89)
2. "Managing blood glucose in type 2 diabetes" (0.85)
3. "Therapeutic approaches for hyperglycemia" (0.82)
                """
                st.code(dense_results)
                
                st.markdown("**BM25 Results:**")
                bm25_results = """
1. "Type 2 diabetes treatment review" (12.4)
2. "Novel treatments for type 2 diabetes" (10.8)
3. "Comparing diabetes treatment options" (9.5)
                """
                st.code(bm25_results)
                
                st.markdown("**After RRF Fusion:**")
                fusion_results = """
1. "Type 2 diabetes treatment review" (0.92)
2. "Diabetes treatment guidelines" (0.89)
3. "Novel treatments for type 2 diabetes" (0.85)
4. "Managing blood glucose in type 2 diabetes" (0.81)
5. "Therapeutic approaches for hyperglycemia" (0.78)
                """
                st.code(fusion_results)
                
                st.markdown("The fusion combined unique results from both sources, with better overall ranking.")
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_fusion:
                if fusion_method == "Reciprocal Rank Fusion (RRF)":
                    st.success("✅ RRF is an excellent general-purpose fusion method")
                
                if len(fusion_sources) >= 3:
                    st.success("✅ Combining multiple sources improves robustness")
                
                if normalization_method == "None" and fusion_method in ["CombSUM", "Linear Interpolation", "Weighted Fusion"]:
                    st.warning("⚠️ Score normalization is recommended for this fusion method")
                
            else:
                st.info("ℹ️ Result fusion can significantly improve retrieval quality")
    
    # ENSEMBLE TECHNIQUES TAB
    with reranking_tabs[2]:
        st.subheader("Ensemble Techniques")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable ensembles
            enable_ensemble = st.checkbox("Enable ensemble methods", value=False)
            
            if enable_ensemble:
                # Ensemble method
                ensemble_method = st.selectbox(
                    "Ensemble Method",
                    ["Stacking", "Bagging", "Boosting", "Voting", "Sequential"]
                )
                
                if ensemble_method == "Stacking":
                    st.markdown("### Stacking Configuration")
                    
                    base_models = st.multiselect(
                        "Base Models",
                        ["Dense Retrieval", "BM25", "SPLADE", "Cross-Encoder", "Feature-based Ranker"],
                        default=["Dense Retrieval", "BM25", "Cross-Encoder"]
                    )
                    
                    meta_model = st.selectbox(
                        "Meta Model",
                        ["LambdaMART", "XGBoost", "Neural Network", "Linear Model"]
                    )
                    
                    st.number_input("Number of training samples", 1000, 100000, 10000)
                
                elif ensemble_method == "Voting":
                    st.markdown("### Voting Configuration")
                    
                    voting_type = st.radio(
                        "Voting Type",
                        ["Majority Voting", "Weighted Voting", "Rank-based Voting"]
                    )
                    
                    voters = st.multiselect(
                        "Voting Models",
                        ["Dense Retriever", "BM25 Retriever", "Hybrid Retriever", 
                         "Cross-Encoder", "Feature Ranker", "LLM Ranker"],
                        default=["Dense Retriever", "BM25 Retriever", "Cross-Encoder"]
                    )
                    
                    if voting_type == "Weighted Voting":
                        st.text_area(
                            "Voter Weights (name: weight)",
                            """Dense Retriever: 0.4
BM25 Retriever: 0.3
Cross-Encoder: 0.3""",
                            height=100
                        )
                
                elif ensemble_method == "Sequential":
                    st.markdown("### Sequential Pipeline")
                    
                    pipeline_stages = []
                    max_stages = 5
                    
                    for i in range(max_stages):
                        stage = st.selectbox(
                            f"Stage {i+1}",
                            ["None", "Dense Retrieval", "Sparse Retrieval", "Cross-Encoder Reranking", 
                             "Feature Reranking", "LLM Reranking", "Result Filtering", "Fusion"],
                            index=0 if i > 2 else i+1,
                            key=f"seq_stage_{i}"
                        )
                        
                        if stage != "None":
                            pipeline_stages.append(stage)
                            
                            # Stage-specific parameters
                            if "Retrieval" in stage:
                                st.slider(f"Stage {i+1} top-k", 10, 1000, 100, key=f"stage_{i}_k")
                            elif "Reranking" in stage:
                                st.slider(f"Stage {i+1} top-k", 5, 200, 20, key=f"stage_{i}_k")
                    
                # Training and optimization
                st.markdown("### Training & Optimization")
                
                st.radio(
                    "Training Method",
                    ["Supervised Learning", "Online Learning", "Zero-shot", "Few-shot"]
                )
                
                training_data = st.selectbox(
                    "Training Data",
                    ["MS MARCO", "Custom Dataset", "Query Logs", "Synthetic Data", "None"]
                )
                
                st.checkbox("Enable hyperparameter optimization", value=True)
        
        with col2:
            with st.expander("ℹ️ About Ensemble Techniques", expanded=True):
                info_tooltip("""
                **Ensemble Techniques** combine multiple retrieval models to achieve better performance than any individual model.
                
                **Ensemble Methods:**
                
                **Stacking:**
                - Uses outputs from base models as features for a meta-model
                - Meta-model learns optimal combinations of base models
                - Requires training data but offers top performance
                - Can adapt to different query types and domains
                
                **Bagging:**
                - Uses multiple instances of the same model on different data subsets
                - Reduces variance and helps prevent overfitting
                - Good for stabilizing performance
                
                **Boosting:**
                - Sequentially trains models to focus on previous models' errors
                - Excellent for improving weak retrievers
                - Can lead to best overall performance
                
                **Voting:**
                - Combines models through voting mechanisms
                - Simple but effective approach
                - Types include majority, weighted, and rank-based voting
                
                **Sequential:**
                - Creates a pipeline of retrieval and reranking steps
                - Each stage narrows and refines the candidate set
                - Balances efficiency and quality
                
                **Benefits:**
                - Higher accuracy than individual models
                - More robust to different query types
                - Can balance precision and recall
                - Reduces variance in performance
                
                **When to Use:**
                - For mission-critical retrieval systems
                - When single models show inconsistent performance
                - When training data is available (for stacking/boosting)
                - For production systems where quality is paramount
                """)
            
            # Example of ensemble technique
            st.markdown("#### Ensemble Example")
            
            if enable_ensemble and ensemble_method == "Stacking":
                st.markdown("**Stacking Ensemble Process:**")
                
                stacking_process = """
1. Base Retrievers Generate Candidates:
   - Dense Retriever: 100 results
   - BM25 Retriever: 100 results
   - SPLADE Retriever: 100 results

2. Feature Extraction For Each Document:
   - Dense retrieval score
   - BM25 score
   - SPLADE score
   - Document length
   - Query-document term overlap
   - Entity match count

3. Meta-Ranker (XGBoost) Combines Signals:
   - Trained on labeled relevance data
   - Learns optimal weighting of features
   - Re-scores and ranks the candidates

4. Result: 
   - More accurate ranking that adapts to query type
   - ~25% better NDCG@10 than best individual retriever
                """
                st.code(stacking_process)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_ensemble:
                if ensemble_method == "Stacking":
                    st.success("✅ Stacking is powerful but requires training data")
                    
                elif ensemble_method == "Voting":
                    st.success("✅ Voting is simple yet effective")
                    
                elif ensemble_method == "Sequential":
                    st.success("✅ Sequential pipelines balance efficiency and quality")
                    
                if training_data == "None" and ensemble_method in ["Stacking", "Boosting"]:
                    st.warning("⚠️ This ensemble method works best with training data")
            else:
                st.info("ℹ️ Ensemble methods can significantly boost performance for critical applications")
    
    # CUSTOM SCORING TAB
    with reranking_tabs[3]:
        st.subheader("Custom Scoring Rules")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable custom scoring
            enable_custom_scoring = st.checkbox("Enable custom scoring rules", value=False)
            
            if enable_custom_scoring:
                # Scoring components
                st.markdown("### Scoring Components")
                
                scoring_components = st.multiselect(
                    "Components to Include in Custom Score",
                    ["Base Retrieval Score", "Semantic Similarity", "Term Frequency", "Metadata Relevance",
                     "Document Freshness", "Document Length", "Authority Score", "Popularity", "User Preference"],
                    default=["Base Retrieval Score", "Semantic Similarity", "Metadata Relevance"]
                )
                
                # Custom rules
                st.markdown("### Custom Rules")
                
                rule_types = st.multiselect(
                    "Rule Types",
                    ["Boosting", "Filtering", "Penalization", "Context-Aware"],
                    default=["Boosting"]
                )
                
                if "Boosting" in rule_types:
                    st.markdown("#### Boosting Rules")
                    
                    boost_rules = st.text_area(
                        "Boosting Rules (field: boost_factor)",
                        """title_match: 1.5
exact_phrase_match: 2.0
recency_boost: 1.2
trusted_source: 1.3""",
                        height=100
                    )
                
                if "Filtering" in rule_types:
                    st.markdown("#### Filtering Rules")
                    
                    filter_rules = st.text_area(
                        "Filtering Rules",
                        """min_score: 0.6
min_semantic_similarity: 0.7
exclude_outdated: true""",
                        height=100
                    )
                
                if "Penalization" in rule_types:
                    st.markdown("#### Penalization Rules")
                    
                    penalize_rules = st.text_area(
                        "Penalization Rules (condition: factor)",
                        """excessive_length: 0.9
poor_formatting: 0.85
low_specificity: 0.8""",
                        height=100
                    )
                
                # Combining method
                st.markdown("### Rule Combination")
                
                combination_method = st.selectbox(
                    "Rule Combination Method",
                    ["Multiplicative", "Additive", "Max Rule", "Custom Formula"]
                )
                
                if combination_method == "Custom Formula":
                    st.text_area(
                        "Custom Scoring Formula",
                        """final_score = base_score * 0.6 + 
             semantic_score * 0.3 + 
             (metadata_relevance * 0.1)
             
if exact_match:
    final_score *= 1.2
    
if document_age > 365:  # days
    final_score *= 0.9
""",
                        height=150
                    )
                
                # Score normalization
                st.checkbox("Normalize final scores", value=True)
        
        with col2:
            with st.expander("ℹ️ About Custom Scoring", expanded=True):
                info_tooltip("""
                **Custom Scoring Rules** allow you to incorporate domain knowledge and business logic into ranking decisions.
                
                **Scoring Components:**
                - **Base Retrieval Score**: The score from your primary retriever
                - **Semantic Similarity**: Vector similarity score
                - **Term Frequency**: Keyword matching signals (BM25, etc.)
                - **Metadata Relevance**: Scores based on document metadata
                - **Document Freshness**: Recency-based scoring
                - **Document Length**: Length normalization factors
                - **Authority Score**: Source credibility metrics
                - **Popularity**: Usage statistics and popularity
                - **User Preference**: Personalization factors
                
                **Rule Types:**
                - **Boosting**: Increase scores for documents with desired attributes
                - **Filtering**: Remove documents that don't meet criteria
                - **Penalization**: Reduce scores for documents with undesired attributes
                - **Context-Aware**: Rules that adapt based on query context
                
                **Combination Methods:**
                - **Multiplicative**: Multiply factors (more dramatic effect)
                - **Additive**: Add weighted components (more controlled)
                - **Max Rule**: Use maximum applicable rule (simple prioritization)
                - **Custom Formula**: Create complex scoring logic
                
                **Benefits:**
                - Incorporates business rules into ranking
                - Addresses domain-specific ranking needs
                - Can implement business priorities directly
                - Allows manual tuning and expert knowledge
                
                **Example Uses:**
                - Boosting recent content
                - Prioritizing trusted sources
                - Implementing content policies
                - Adapting to user context or preferences
                """)
            
            # Example of custom scoring
            st.markdown("#### Custom Scoring Example")
            
            if enable_custom_scoring:
                st.markdown("**Query:** `diabetes treatment guidelines`")
                
                scoring_example = """
Document: "ADA Guidelines for Type 2 Diabetes Management (2023)"

Scoring Process:
1. Base semantic similarity score: 0.85
2. Applied boosting rules:
   - Title match (+50%): 0.85 * 1.5 = 1.275
   - Trusted source (ADA) (+30%): 1.275 * 1.3 = 1.6575
   - Recent content (2023) (+20%): 1.6575 * 1.2 = 1.989
3. Score normalization: 0.95 (scaled 0-1)

Result: Document promoted to #1 position due to domain-specific factors
                """
                st.code(scoring_example)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_custom_scoring:
                st.success("✅ Custom scoring enables domain-specific optimizations")
                
                if "Boosting" in rule_types and "title_match" in boost_rules:
                    st.success("✅ Title matching is an effective boosting signal")
                
                if combination_method == "Multiplicative":
                    st.info("ℹ️ Multiplicative combination can create dramatic scoring changes")
                    st.info("ℹ️ Consider additive for more controlled effects")
            else:
                st.info("ℹ️ Custom scoring rules can incorporate domain knowledge")
                st.info("ℹ️ Particularly useful for specialized search applications")

###############################
# 8. LLM INTEGRATION
###############################
elif main_section == "8. LLM Integration":
    st.header("8. LLM Integration")
    st.markdown("Configure how the retrieved information is processed by language models to generate responses")
    
    # Create tabs for LLM options
    llm_tabs = st.tabs(["LLM Selection", "Prompt Engineering", "Context Management", "Advanced Techniques"])
    
    # LLM SELECTION TAB
    with llm_tabs[0]:
        st.subheader("Language Model Selection")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # LLM model selection
            llm_provider = st.selectbox(
                "LLM Provider",
                ["OpenAI", "Anthropic", "Meta AI", "Mistral AI", "Google", "Custom/Self-hosted"]
            )
            
            # Different model options based on provider
            if llm_provider == "OpenAI":
                llm_model = st.selectbox(
                    "OpenAI Model",
                    ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo", "gpt-4-32k"]
                )
                
                api_type = st.radio("API Type", ["OpenAI API", "Azure OpenAI"])
                
                if api_type == "Azure OpenAI":
                    azure_endpoint = st.text_input("Azure Endpoint", "https://example.openai.azure.com")
                    azure_deployment = st.text_input("Azure Deployment Name", "gpt-4")
                
            elif llm_provider == "Anthropic":
                llm_model = st.selectbox(
                    "Anthropic Model",
                    ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-2.1"]
                )
                
            elif llm_provider == "Meta AI":
                llm_model = st.selectbox(
                    "Meta AI Model",
                    ["Llama-3-70b", "Llama-3-8b", "Llama-2-70b", "Llama-2-13b", "Llama-2-7b"]
                )
                
                deployment_type = st.radio("Deployment Type", ["API", "Local", "Inference Endpoint"])
                
                if deployment_type == "Local":
                    quantization = st.selectbox("Quantization", ["None (float16)", "8-bit", "4-bit"])
                    
            elif llm_provider == "Mistral AI":
                llm_model = st.selectbox(
                    "Mistral AI Model",
                    ["Mistral Large", "Mistral Medium", "Mistral Small", "Mistral 7B Instruct"]
                )
                
            elif llm_provider == "Google":
                llm_model = st.selectbox(
                    "Google Model",
                    ["Gemini Ultra", "Gemini Pro", "Gemini Flash", "PaLM 2"]
                )
                
            elif llm_provider == "Custom/Self-hosted":
                llm_model = st.text_input("Model Name/Path", "mistralai/Mistral-7B-Instruct-v0.2")
                inference_server = st.selectbox(
                    "Inference Server",
                    ["vLLM", "TGI (Text Generation Inference)", "FastChat", "LMStudio", "Custom"]
                )
                server_url = st.text_input("Server URL", "http://localhost:8000")
            
            # General LLM settings
            st.markdown("### Model Parameters")
            
            col_params1, col_params2 = st.columns(2)
            
            with col_params1:
                temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
                top_p = st.slider("Top P (nucleus sampling)", 0.0, 1.0, 0.95, 0.05)
                
            with col_params2:
                max_tokens = st.slider("Max output tokens", 10, 4096, 1024)
                presence_penalty = st.slider("Presence penalty", -2.0, 2.0, 0.0, 0.1)
                frequency_penalty = st.slider("Frequency penalty", -2.0, 2.0, 0.0, 0.1)
            
            # Advanced model settings
            with st.expander("Advanced Model Settings"):
                st.checkbox("Stream response", value=True)
                st.checkbox("Include token probabilities", value=False)
                
                if st.checkbox("Custom stopping criteria", value=False):
                    st.text_area("Stop sequences (one per line)", "###\nAnswer:")
                
                if st.checkbox("Add custom model headers", value=False):
                    st.text_area("Custom headers (JSON format)", """{
    "x-custom-header": "value"
}""")
        
        with col2:
            with st.expander("ℹ️ About LLM Selection", expanded=True):
                info_tooltip("""
                **Language Model Selection** determines which AI model will generate responses using the retrieved information.
                
                **Provider Options:**
                
                **OpenAI Models:**
                - **GPT-4o**: Latest multimodal model with best performance
                - **GPT-4-turbo**: Excellent reasoning, factual accuracy
                - **GPT-3.5-turbo**: Good balance of performance and cost
                
                **Anthropic Models:**
                - **Claude-3-Opus**: Flagship model with superior reasoning
                - **Claude-3-Sonnet**: Balanced performance and efficiency
                - **Claude-3-Haiku**: Fastest and most economical option
                
                **Meta AI (Llama) Models:**
                - Open-source models with various sizes
                - Can be deployed locally or via API
                
                **Mistral AI Models:**
                - Strong performance/weight ratio
                - Good multilingual capabilities
                
                **Google Models:**
                - **Gemini Ultra**: Highest capabilities multimodal model
                - **Gemini Pro**: Strong performance for most use cases
                
                **Parameter Settings:**
                - **Temperature**: Controls randomness (0.0 = deterministic, higher = more creative)
                - **Top P**: Nucleus sampling threshold (smaller = more focused)
                - **Max tokens**: Maximum response length
                - **Presence/Frequency penalty**: Discourages repetition
                
                **Deployment Considerations:**
                - API-based: Simpler to implement, no infrastructure management
                - Self-hosted: More control, lower latency, data privacy
                - Quantization: Reduces memory requirements but may impact quality
                """)
            
            # LLM Comparison
            st.markdown("#### Model Comparison")
            
            model_comp = pd.DataFrame({
                "Model": ["GPT-4o", "Claude-3-Opus", "Llama-3-70B", "Mistral Large"],
                "Reasoning": ["Excellent", "Excellent", "Very Good", "Good"],
                "Factual": ["Very Good", "Excellent", "Good", "Good"],
                "Speed": ["Fast", "Moderate", "Varies", "Fast"],
                "Context": ["128K", "200K", "8K-128K", "32K"]
            })
            
            st.dataframe(model_comp, hide_index=True)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if llm_provider == "OpenAI" and llm_model == "gpt-4o":
                st.success("✅ GPT-4o offers excellent performance across most tasks")
                
            elif llm_provider == "Anthropic" and llm_model == "claude-3-opus":
                st.success("✅ Claude-3-Opus excels at factual accuracy and reasoning")
                
            elif llm_provider == "OpenAI" and llm_model == "gpt-3.5-turbo":
                st.info("ℹ️ Good balance of cost and performance for simpler use cases")
                
            if temperature < 0.3:
                st.success("✅ Low temperature is good for factual, consistent responses")
            elif temperature > 0.8:
                st.info("ℹ️ Higher temperature increases creativity but may reduce accuracy")
    
    # PROMPT ENGINEERING TAB
    with llm_tabs[1]:
        st.subheader("Prompt Engineering")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Prompt template approach
            prompt_approach = st.selectbox(
                "Prompt Template Approach",
                ["Standard Template", "Few-shot Learning", "Chain of Thought", "ReAct", "Custom Template"]
            )
            
            # Template configuration
            st.markdown("### Template Configuration")
            
            if prompt_approach == "Standard Template":
                st.text_area(
                    "System Message",
                    """You are a helpful assistant that answers questions based on the provided context.
Use only the information from the context to answer the question.
If the context doesn't contain the answer, say "I don't have enough information to answer this question."
Don't make up or infer information that isn't directly supported by the context.""",
                    height=150
                )
                
                prompt_template = st.text_area(
                    "Prompt Template",
                    """Context:
{context}

Question: {query}

Answer:""",
                    height=150
                )
                
            elif prompt_approach == "Few-shot Learning":
                st.text_area(
                    "System Message",
                    """You are a helpful assistant that answers questions based on the provided context.
Use the examples to understand the expected format and approach.""",
                    height=100
                )
                
                st.text_area(
                    "Few-shot Examples",
                    """Example 1:
Context: The first airplane flight was conducted by the Wright brothers, Orville and Wilbur, on December 17, 1903, at Kitty Hawk, North Carolina. The flight lasted 12 seconds and covered 120 feet.
Question: When did the Wright brothers first fly?
Answer: The Wright brothers conducted their first airplane flight on December 17, 1903.

Example 2:
Context: Photosynthesis is the process by which green plants and some other organisms use sunlight to synthesize nutrients from carbon dioxide and water. Photosynthesis occurs in the chloroplasts of plant cells.
Question: Where does photosynthesis occur in plants?
Answer: Photosynthesis occurs in the chloroplasts of plant cells.""",
                    height=200
                )
                
                prompt_template = st.text_area(
                    "Prompt Template",
                    """Context:
{context}

Question: {query}

Answer:""",
                    height=100,
                    key="few_shot_template"
                )
                
            elif prompt_approach == "Chain of Thought":
                st.text_area(
                    "System Message",
                    """You are a helpful assistant that answers questions based on the provided context.
Think step-by-step to answer the question accurately using only the information provided.""",
                    height=100
                )
                
                prompt_template = st.text_area(
                    "Prompt Template",
                    """Context:
{context}

Question: {query}

Let's think through this step by step to find the answer:
1. First, let's identify the key information we need.
2. Then, let's search for relevant details in the context.
3. Finally, let's formulate a complete answer.

Step-by-step thinking:""",
                    height=200
                )
                
            elif prompt_approach == "ReAct":
                st.text_area(
                    "System Message",
                    """You are a helpful assistant that answers questions based on the provided context.
Follow the format of Thought, Action, Observation, and Answer to solve problems step-by-step.""",
                    height=100
                )
                
                prompt_template = st.text_area(
                    "Prompt Template",
                    """Context:
{context}

Question: {query}

Thought: Let me analyze what information I need to answer this question.
Action: Search the context for relevant information.
Observation: [Identify key facts from the context]
Thought: Based on these observations, I can now formulate an answer.
Answer:""",
                    height=200
                )
                
            elif prompt_approach == "Custom Template":
                st.text_area(
                    "System Message",
                    """You are a helpful assistant with specific expertise.""",
                    height=100
                )
                
                prompt_template = st.text_area(
                    "Custom Prompt Template",
                    """[CONTEXT]
{context}
[/CONTEXT]

[QUERY]
{query}
[/QUERY]

[INSTRUCTIONS]
Provide a comprehensive answer based solely on the context.
Include relevant facts and figures if available.
Maintain a professional tone.
[/INSTRUCTIONS]

[ANSWER]""",
                    height=200
                )
            
            # Context formatting
            st.markdown("### Context Formatting")
            
            context_format = st.selectbox(
                "Context Format",
                ["Simple Concatenation", "Numbered Passages", "Relevance-Ordered", "Metadataß-Enriched"]
            )
            
            if context_format == "Numbered Passages":
                st.code("""
# Numbered passages format
Context [1]: The study found a 23% reduction in symptoms...
Context [2]: Patients reported improved quality of life...
Context [3]: Side effects were minimal according to...
                """)
                
            elif context_format == "Metadataß-Enriched":
                st.code("""
# Metadata-enriched format
[DOCUMENT: Clinical Trial Results, 2023]
The study found a 23% reduction in symptoms...

[DOCUMENT: Patient Survey, 2024]
Patients reported improved quality of life...
                """)
            
            # Advanced prompt features
            with st.expander("Advanced Prompt Features"):
                st.checkbox("Include source attribution guidance", value=True)
                st.checkbox("Add confidence estimation instructions", value=False)
                st.checkbox("Include forbidden response patterns", value=True)
                st.checkbox("Add output structure specification", value=False)
                
                if st.checkbox("Add query analysis", value=False):
                    st.text_area(
                        "Query Analysis Instructions",
                        "Before answering, analyze the query to identify key topics and requirements."
                    )
        
        with col2:
            with st.expander("ℹ️ About Prompt Engineering", expanded=True):
                info_tooltip("""
                **Prompt Engineering** designs effective instructions that guide the LLM to generate high-quality responses.
                
                **Prompt Approaches:**
                
                **Standard Template:**
                - Simple, direct instructions
                - System message + context + query
                - Good starting point for most applications
                
                **Few-shot Learning:**
                - Includes examples of desired behavior
                - Shows the model expected format and reasoning
                - Useful for consistent formatting or specialized tasks
                
                **Chain of Thought (CoT):**
                - Encourages step-by-step reasoning
                - Reduces logical errors and improves reasoning
                - Good for complex problems requiring multi-step thinking
                
                **ReAct (Reasoning + Acting):**
                - Combines reasoning with specific actions
                - Structures thinking into Thought→Action→Observation→Answer
                - Excellent for complex reasoning tasks
                
                **Context Formatting:**
                - **Simple Concatenation**: Straightforward but may lose structure
                - **Numbered Passages**: Clear references for attribution
                - **Relevance-Ordered**: Most relevant context first
                - **Metadata-Enriched**: Includes source information
                
                **Best Practices:**
                - Be specific about expected format
                - Include guidance on what to do when information is missing
                - Consider adding source attribution instructions
                - Structure prompts for clarity (context separate from instructions)
                """)
            
            # Example prompt
            st.markdown("#### Example Formatted Prompt")
            
            if prompt_approach == "Chain of Thought":
                example_prompt = """
System: You are a helpful assistant that answers questions based on the provided context.
Think step-by-step to answer the question accurately using only the information provided.

User: 
Context:
[1] A study published in JAMA (2023) found that patients taking metformin had a 23% reduction in HbA1c levels after 6 months of treatment compared to the placebo group.
[2] Common side effects of metformin include gastrointestinal issues such as nausea, diarrhea, and abdominal discomfort, which typically subside after a few weeks of treatment.
[3] The American Diabetes Association guidelines (2024) recommend metformin as the first-line medication for type 2 diabetes management due to its efficacy, safety profile, and low cost.

Question: What are the benefits and side effects of metformin according to recent studies?

Let's think through this step by step to find the answer:
1. First, let's identify the key information we need.
2. Then, let's search for relevant details in the context.
3. Finally, let's formulate a complete answer.

Step-by-step thinking:
"""
                st.code(example_prompt)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if prompt_approach == "Chain of Thought":
                st.success("✅ Chain of Thought is excellent for improving reasoning quality")
                
            elif prompt_approach == "Few-shot Learning":
                st.success("✅ Few-shot learning helps guide the model to follow specific patterns")
                
            elif prompt_approach == "ReAct":
                st.success("✅ ReAct provides the most structured approach to complex reasoning")
                
            if context_format == "Numbered Passages":
                st.success("✅ Numbered passages improve source attribution")
                
            elif context_format == "Relevance-Ordered":
                st.success("✅ Relevance ordering helps prioritize the most important information")
    
    # CONTEXT MANAGEMENT TAB
    with llm_tabs[2]:
        st.subheader("Context Management")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Context window utilization
            st.markdown("### Context Window Management")
            
            max_context_length = st.slider("Maximum context length (tokens)", 1000, 16000, 4000)
            
            context_strategy = st.selectbox(
                "Context Selection Strategy",
                ["Top K", "Semantic Search", "Hybrid (Top K + Semantic)", "Dynamic (Query-dependent)", "Map-Reduce"]
            )
            
            if context_strategy in ["Top K", "Hybrid (Top K + Semantic)"]:
                st.slider("Number of documents (K)", 1, 20, 5)
            
            if context_strategy in ["Semantic Search", "Hybrid (Top K + Semantic)"]:
                st.slider("Semantic similarity threshold", 0.5, 1.0, 0.75, 0.05)
            
            if context_strategy == "Map-Reduce":
                st.number_input("Number of map chunks", 2, 50, 10)
                
            # Context truncation
            st.markdown("### Context Truncation")
            
            truncation_method = st.selectbox(
                "Truncation Method",
                ["Simple Truncation", "Relevance-Based", "Recursive Summarization", "Token Management"]
            )
            
            if truncation_method == "Simple Truncation":
                trunc_direction = st.radio("Truncation Direction", ["End", "Beginning", "Both Ends"])
                
            elif truncation_method == "Relevance-Based":
                st.slider("Relevance threshold", 0.0, 1.0, 0.6, 0.05)
                st.checkbox("Maintain paragraph integrity", value=True)
                
            elif truncation_method == "Recursive Summarization":
                st.slider("Summary compression ratio", 0.1, 0.9, 0.5, 0.05)
                st.number_input("Max recursion depth", 1, 5, 2)
                
            # Context ordering
            st.markdown("### Context Ordering")
            
            ordering_method = st.selectbox(
                "Context Ordering Method",
                ["Relevance Score", "Chronological", "Document Order", "Hierarchical", "Custom Logic"]
            )
            
            if ordering_method == "Hierarchical":
                st.multiselect(
                    "Hierarchical Levels",
                    ["Document", "Section", "Paragraph", "Sentence"],
                    default=["Document", "Paragraph"]
                )
            
            elif ordering_method == "Custom Logic":
                st.text_area(
                    "Custom Ordering Logic (pseudocode)",
                    """# 1. Sort by relevance score
# 2. Ensure definitional content comes first
# 3. Place most recent information next
# 4. Put examples last"""
                )
        
        with col2:
            with st.expander("ℹ️ About Context Management", expanded=True):
                info_tooltip("""
                **Context Management** controls how retrieved information is selected, organized, and presented to the language model.
                
                **Context Selection Strategies:**
                - **Top K**: Simple selection of K most relevant documents
                - **Semantic Search**: Selection based on semantic similarity
                - **Hybrid**: Combines multiple selection criteria
                - **Dynamic**: Adjusts strategy based on query type
                - **Map-Reduce**: Processes multiple chunks and synthesizes results
                
                **Truncation Methods:**
                - **Simple**: Basic cutting off at token limit
                - **Relevance-Based**: Preserves most relevant content
                - **Recursive Summarization**: Summarizes less relevant parts
                - **Token Management**: Optimizes token allocation by section
                
                **Context Ordering:**
                - **Relevance Score**: Most relevant first (generally best)
                - **Chronological**: Time-based ordering
                - **Document Order**: Preserves original sequence
                - **Hierarchical**: Organized by structural levels
                
                **Why It Matters:**
                - LLMs are sensitive to context presentation
                - Most models pay more attention to beginning and end
                - Context window constraints require efficient use
                - Well-organized context improves factual accuracy
                
                **Best Practices:**
                - Place most important information first
                - Ensure complete coverage of relevant facts
                - Maintain document coherence when possible
                - Use metadata to help model understand sources
                """)
            
            # Context strategy examples
            st.markdown("#### Context Strategy Example")
            
            if context_strategy == "Map-Reduce":
                map_reduce_example = """
Query: "Summarize the latest treatment options for rheumatoid arthritis"

Map-Reduce Process:
1. Retrieve 10 relevant documents
2. Split into smaller chunks (Map step)
3. Process each chunk:
   - Chunk 1: "Biologics such as TNF inhibitors..."
   - Chunk 2: "JAK inhibitors have shown efficacy in..."
   - Chunk 3: "Non-pharmacological approaches include..."
4. Generate sub-summaries for each chunk
5. Combine sub-summaries into final answer (Reduce step)

This approach handles large volumes of context efficiently
by distributing processing across manageable chunks.
                """
                st.code(map_reduce_example)
                
            elif truncation_method == "Recursive Summarization":
                recursive_example = """
Original Context: 15,000 tokens (exceeds limit)

Level 1: Summarize less relevant sections
- Section A (highly relevant): 2,000 tokens → Keep original
- Section B (moderately relevant): 5,000 tokens → Summarize to 1,000 tokens
- Section C (low relevance): 8,000 tokens → Summarize to 1,000 tokens
Result: 4,000 tokens (within limit)

Final Context:
- Full text of Section A (most relevant)
- Summarized version of Section B
- Summarized version of Section C
                """
                st.code(recursive_example)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if context_strategy == "Hybrid (Top K + Semantic)":
                st.success("✅ Hybrid strategy provides a good balance of relevance and diversity")
                
            if truncation_method == "Relevance-Based":
                st.success("✅ Relevance-based truncation preserves the most important information")
                
            elif truncation_method == "Recursive Summarization":
                st.success("✅ Recursive summarization is excellent for handling large documents")
                st.info("ℹ️ But adds processing overhead and may lose details")
                
            if ordering_method == "Relevance Score":
                st.success("✅ Placing most relevant information first is generally optimal")
    
    # ADVANCED TECHNIQUES TAB
    with llm_tabs[3]:
        st.subheader("Advanced LLM Techniques")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Reliability enhancements
            st.markdown("### Reliability Enhancements")
            
            reliability_techniques = st.multiselect(
                "Reliability Techniques",
                ["Self-consistency", "Ensemble of Thought Paths", "Verification Steps", "Structured Output Constraints"],
                default=["Verification Steps"]
            )
            
            if "Self-consistency" in reliability_techniques:
                sc_samples = st.slider("Number of sample generations", 3, 10, 5)
                sc_method = st.selectbox("Consensus Method", ["Majority Voting", "Confidence Weighted", "LLM Judge"])
                
            if "Ensemble of Thought Paths" in reliability_techniques:
                st.number_input("Number of thought paths", 2, 10, 3)
                st.checkbox("Use different temperatures", value=True)
                
            if "Verification Steps" in reliability_techniques:
                verification_steps = st.multiselect(
                    "Verification Types",
                    ["Fact Checking", "Logical Consistency", "Source Validation", "Self-critique"],
                    default=["Fact Checking", "Self-critique"]
                )
                
                if "Fact Checking" in verification_steps:
                    st.text_area(
                        "Fact Checking Instructions",
                        """Before providing your final answer, verify each factual claim by checking if it's explicitly supported by the context. List any claims that cannot be verified."""
                    )
            
            # Response structuring
            st.markdown("### Response Structuring")
            
            structure_type = st.selectbox(
                "Response Structure",
                ["Free-form", "JSON", "Markdown", "Template-based", "Custom Schema"]
            )
            
            if structure_type == "JSON":
                st.text_area(
                    "JSON Schema",
                    """{
  "answer": "string",
  "sources": ["string"],
  "confidence": "number",
  "reasoning": "string"
}"""
                )
                
            elif structure_type == "Template-based":
                st.text_area(
                    "Response Template",
                    """## Answer
{answer}

## Sources
{sources}

## Confidence
{confidence}

## Reasoning
{reasoning}"""
                )
            
            # Advanced reasoning
            st.markdown("### Advanced Reasoning")
            
            reasoning_techniques = st.multiselect(
                "Advanced Reasoning Techniques",
                ["Multi-hop Reasoning", "Tool Use", "Retrieval Augmented Generation", "Self-reflection"],
                default=["Multi-hop Reasoning"]
            )
            
            if "Multi-hop Reasoning" in reasoning_techniques:
                st.slider("Maximum reasoning hops", 1, 10, 3)
                st.checkbox("Allow inter-hop retrieval", value=True)
                
            if "Tool Use" in reasoning_techniques:
                tools = st.multiselect(
                    "Available Tools",
                    ["Calculator", "Knowledge Base", "Search", "Code Execution", "Custom Tool"],
                    default=["Calculator"]
                )
                
                if "Custom Tool" in tools:
                    st.text_area(
                        "Custom Tool Definition",
                        """{
  "name": "query_database",
  "description": "Query the medical database for specific conditions",
  "parameters": {
    "condition": "string",
    "limit": "number"
  }
}"""
                    )
        
        with col2:
            with st.expander("ℹ️ About Advanced Techniques", expanded=True):
                info_tooltip("""
                **Advanced LLM Techniques** improve response quality, reliability, and capabilities beyond basic prompting.
                
                **Reliability Techniques:**
                
                **Self-consistency:**
                - Generate multiple answers with the same prompt
                - Select most consistent or use voting
                - Reduces randomness and improves accuracy
                - Especially useful for reasoning tasks
                
                **Ensemble of Thought Paths:**
                - Generate multiple reasoning paths
                - More comprehensive exploration of possibilities
                - Better for complex reasoning problems
                
                **Verification Steps:**
                - Add explicit fact-checking instructions
                - Prompt the model to verify its own outputs
                - Reduces hallucinations significantly
                
                **Response Structuring:**
                - **Free-form**: Natural language without constraints
                - **JSON**: Structured data format, good for APIs
                - **Markdown**: Readable format with structured elements
                - **Template-based**: Enforces specific sections
                
                **Advanced Reasoning:**
                - **Multi-hop Reasoning**: Sequential steps of inference
                - **Tool Use**: Augmenting LLM with external tools
                - **Retrieval Augmentation**: Dynamic context retrieval
                - **Self-reflection**: Critique and improve own responses
                
                **Benefits:**
                - Higher factual accuracy
                - Reduced hallucination
                - More robust reasoning
                - Better structured information
                """)
            
            # Advanced technique examples
            st.markdown("#### Advanced Technique Example")
            
            if "Self-consistency" in reliability_techniques:
                self_consistency_example = """
Query: "What is 24 × 37?"

Self-consistency approach:
1. Generate multiple answers with different sampling:
   - Sample 1: "24 × 37 = 24 × 30 + 24 × 7 = 720 + 168 = 888"
   - Sample 2: "24 × 37 = 20 × 37 + 4 × 37 = 740 + 148 = 888"
   - Sample 3: "24 × 37 = 24 × 40 - 24 × 3 = 960 - 72 = 888"
   - Sample 4: "24 × 37 = 888"
   - Sample 5: "24 × 37 = 24 × 35 + 24 × 2 = 840 + 48 = 888"

2. Compare answers and take majority: "888"

This approach significantly reduces calculation errors.
                """
                st.code(self_consistency_example)
                
            elif "Multi-hop Reasoning" in reasoning_techniques:
                multi_hop_example = """
Query: "Would metformin be suitable for a patient with kidney disease?"

Multi-hop reasoning:
1. First hop: What is metformin used for?
   → Metformin is an oral medication used to treat type 2 diabetes.

2. Second hop: How is metformin processed in the body?
   → Metformin is primarily eliminated unchanged by the kidneys.

3. Third hop: Are there contraindications for patients with kidney disease?
   → Metformin is contraindicated in patients with eGFR below 30 mL/min
     and requires dose adjustment for eGFR between 30-45 mL/min due to
     increased risk of lactic acidosis.

Final answer: "Metformin may not be suitable for patients with kidney
disease, particularly those with severe renal impairment (eGFR below 30
mL/min). For moderate kidney disease (eGFR 30-45 mL/min), dose adjustment
is required, and careful monitoring is necessary."
                """
                st.code(multi_hop_example)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if "Self-consistency" in reliability_techniques:
                st.success("✅ Self-consistency significantly improves factual accuracy")
                st.info("ℹ️ But increases computation cost with multiple generations")
                
            if "Verification Steps" in reliability_techniques:
                st.success("✅ Verification steps are a cost-effective way to reduce hallucination")
                
            if structure_type != "Free-form":
                st.success("✅ Structured output helps with parsing and integration")
                
            if "Tool Use" in reasoning_techniques:
                st.success("✅ Tool use enables capabilities beyond the LLM's knowledge")

###############################
# 9. EVALUATION & MONITORING
###############################
elif main_section == "9. Evaluation & Monitoring":
    st.header("9. Evaluation & Monitoring")
    st.markdown("Configure how to evaluate, monitor, and continuously improve your RAG system")
    
    # Create tabs for evaluation options
    eval_tabs = st.tabs(["Evaluation Metrics", "Ground Truth Evaluation", "Runtime Monitoring", "Continuous Improvement"])
    
    # EVALUATION METRICS TAB
    with eval_tabs[0]:
        st.subheader("Evaluation Metrics")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Categories of metrics
            st.markdown("### Metric Categories")
            
            retrieval_metrics = st.multiselect(
                "Retrieval Metrics",
                ["Recall@K", "Precision@K", "NDCG@K", "MRR", "MAP", "Hit Rate", "Relevance Score", "Custom Metric"],
                default=["Recall@K", "NDCG@K", "MRR"]
            )
            
            if "Recall@K" in retrieval_metrics or "Precision@K" in retrieval_metrics or "NDCG@K" in retrieval_metrics:
                k_values = st.multiselect("K values", [1, 3, 5, 10, 20, 50, 100], default=[5, 10])
            
            generation_metrics = st.multiselect(
                "Generation Metrics",
                ["ROUGE", "BLEU", "BERTScore", "Faithfulness", "Relevance", "Coherence", "Fluency", "Factuality", "Custom Metric"],
                default=["ROUGE", "Faithfulness", "Factuality"]
            )
            
            if "ROUGE" in generation_metrics:
                rouge_types = st.multiselect("ROUGE Types", ["ROUGE-1", "ROUGE-2", "ROUGE-L"], default=["ROUGE-L"])
            
            latency_metrics = st.multiselect(
                "Performance Metrics",
                ["Retrieval Latency", "Generation Latency", "End-to-end Latency", "Tokens per Second", "Custom Latency Metric"],
                default=["End-to-end Latency"]
            )
            
            # Custom metrics
            custom_metrics = []
            if "Custom Metric" in retrieval_metrics or "Custom Metric" in generation_metrics:
                st.markdown("### Custom Metrics")
                
                with st.expander("Define Custom Metrics"):
                    custom_metric_name = st.text_input("Custom Metric Name", "Domain Specificity")
                    custom_metric_type = st.selectbox("Metric Type", ["Retrieval", "Generation", "Performance"])
                    custom_metric_desc = st.text_area("Metric Description", "Measures how specific the answer is to the domain (scale 0-10)")
                    
                    if st.button("Add Custom Metric"):
                        custom_metrics.append({
                            "name": custom_metric_name,
                            "type": custom_metric_type,
                            "description": custom_metric_desc
                        })
                        st.success(f"Custom metric '{custom_metric_name}' added!")
            
            # LLM-based evaluation
            st.markdown("### LLM-based Evaluation")
            
            use_llm_eval = st.checkbox("Use LLM for evaluation", value=True)
            
            if use_llm_eval:
                llm_eval_model = st.selectbox(
                    "Evaluation LLM",
                    ["gpt-4", "gpt-3.5-turbo", "Claude 3 Opus", "Claude 3 Sonnet", "Custom"]
                )
                
                llm_metrics = st.multiselect(
                    "LLM Evaluation Criteria",
                    ["Answer Relevance", "Factual Correctness", "Citation Accuracy", "Answer Completeness", 
                     "Reasoning Quality", "Hallucination Detection", "Context Utilization"],
                    default=["Factual Correctness", "Citation Accuracy", "Hallucination Detection"]
                )
                
                st.checkbox("Use structured evaluation format (e.g., 1-5 scale)", value=True)
                st.checkbox("Require justification for scores", value=True)
                
                with st.expander("LLM Evaluation Prompt"):
                    st.text_area(
                        "Evaluation Prompt Template",
                        """You are an objective evaluator for question answering systems. Evaluate the following answer based on the provided context and question. 

Question: {{question}}

Context: {{context}}

Answer to evaluate: {{answer}}

Evaluate on the following criteria on a scale of 1-5 (1=poor, 5=excellent):
- Factual Correctness: Is the answer factually correct according to the context?
- Citation Accuracy: Does the answer correctly cite information from the context?
- Hallucination Detection: Does the answer contain information not present in the context?

For each criterion, provide:
1. Score (1-5)
2. Justification for the score
3. Specific examples from the answer

Finally, provide an overall assessment of the answer's quality.""",
                        height=250
                    )
        
        with col2:
            with st.expander("ℹ️ About Evaluation Metrics", expanded=True):
                info_tooltip("""
                **Evaluation Metrics** help measure and improve the performance of your RAG system.
                
                **Retrieval Metrics:**
                - **Recall@K**: Percentage of relevant documents retrieved in the top K results
                - **Precision@K**: Percentage of top K results that are relevant
                - **NDCG@K**: Normalized Discounted Cumulative Gain (considers rank order)
                - **MRR**: Mean Reciprocal Rank (position of first relevant result)
                - **MAP**: Mean Average Precision (average precision across queries)
                - **Hit Rate**: Whether relevant documents are retrieved at all
                
                **Generation Metrics:**
                - **ROUGE**: Measures overlap between generated text and reference
                - **BLEU**: Word overlap metric (originally for translation)
                - **BERTScore**: Semantic similarity using contextual embeddings
                - **Faithfulness**: Whether generated text is supported by retrieved content
                - **Factuality**: Factual accuracy of generated content
                - **Coherence**: How well the text flows and connects ideas
                
                **Performance Metrics:**
                - Measure system efficiency and resource usage
                - Important for production systems with latency requirements
                - Help identify bottlenecks in the pipeline
                
                **LLM-based Evaluation:**
                - Uses LLMs to assess quality across multiple dimensions
                - More nuanced than automated metrics
                - Can evaluate aspects like reasoning and factuality
                - Provides human-like assessment at scale
                
                **Best Practices:**
                - Use multiple complementary metrics
                - Include both automated and LLM-based evaluation
                - Focus on metrics aligned with your use case
                - Track metrics over time to measure improvement
                """)
            
            # Example metrics and charts
            st.markdown("#### Sample Evaluation Results")
            
            # Sample data for visualization
            metrics_sample = {
                "Retrieval": {
                    "Recall@5": 0.78,
                    "NDCG@10": 0.82,
                    "MRR": 0.67
                },
                "Generation": {
                    "ROUGE-L": 0.56,
                    "Faithfulness": 0.89,
                    "Factuality": 0.76
                }
            }
            
            # Create a combined metrics DataFrame for visualization
            metrics_df = pd.DataFrame({
                "Metric": list(metrics_sample["Retrieval"].keys()) + list(metrics_sample["Generation"].keys()),
                "Score": list(metrics_sample["Retrieval"].values()) + list(metrics_sample["Generation"].values()),
                "Category": ["Retrieval"] * len(metrics_sample["Retrieval"]) + ["Generation"] * len(metrics_sample["Generation"])
            })
            
            # Create a grouped bar chart
            chart = alt.Chart(metrics_df).mark_bar().encode(
                x=alt.X('Metric:N'),
                y=alt.Y('Score:Q', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('Category:N'),
                tooltip=['Metric', 'Score', 'Category']
            ).properties(
                height=200
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # LLM evaluation example
            if use_llm_eval:
                st.markdown("#### LLM Evaluation Example")
                
                llm_eval_example = """
**Question**: What are the side effects of metformin?

**LLM Evaluation Results**:

Factual Correctness: 4/5
- Justification: The answer correctly lists main side effects but misses lactic acidosis which is rare but serious.
- Example: "The answer correctly identifies GI issues as common side effects."

Citation Accuracy: 5/5
- Justification: All cited information can be found in the context documents.
- Example: "Side effects are properly attributed to document [3]."

Hallucination Detection: 4/5
- Justification: No major hallucinations, but slightly overstates frequency of vitamin B12 deficiency.
- Example: "The answer states B12 deficiency is 'common' when the context only mentions it as 'possible with long-term use'."

Overall Assessment: The answer is generally accurate and well-supported by the context. Minor improvements needed in precision of some statements and completeness of rare side effects.
                """
                st.code(llm_eval_example)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if retrieval_metrics and generation_metrics:
                st.success("✅ Good selection of both retrieval and generation metrics")
                
            if "Recall@K" in retrieval_metrics and "MRR" in retrieval_metrics:
                st.success("✅ Recall@K and MRR provide complementary retrieval insights")
                
            if use_llm_eval:
                st.success("✅ LLM-based evaluation provides nuanced quality assessment")
                
            if "Faithfulness" in generation_metrics:
                st.success("✅ Faithfulness is a critical metric for RAG systems")
                
            if not ("ROUGE" in generation_metrics or "BLEU" in generation_metrics or "BERTScore" in generation_metrics):
                st.info("ℹ️ Consider adding at least one automatic text similarity metric")
    
    # GROUND TRUTH EVALUATION TAB
    with eval_tabs[1]:
        st.subheader("Ground Truth Evaluation")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Test dataset
            st.markdown("### Test Dataset")
            
            dataset_source = st.selectbox(
                "Test Dataset Source",
                ["Upload Custom Dataset", "Use Ground Truth from Section 1", "Generate Synthetic Dataset", "Sample from Logs"]
            )
            
            if dataset_source == "Upload Custom Dataset":
                uploaded_test_data = st.file_uploader("Upload test dataset (CSV or JSON)", type=["csv", "json"])
                
                if uploaded_test_data:
                    st.success("Test dataset uploaded successfully")
                    
                    # Show sample if uploaded
                    st.markdown("#### Sample from dataset:")
                    test_data_sample = pd.DataFrame({
                        "question": ["What is RAG?", "How does vector search work?"],
                        "expected_answer": ["RAG stands for Retrieval-Augmented Generation...", "Vector search converts text to numerical vectors..."],
                        "relevant_docs": ["doc_123, doc_456", "doc_789, doc_101"]
                    })
                    st.dataframe(test_data_sample, use_container_width=True)
            
            elif dataset_source == "Generate Synthetic Dataset":
                st.markdown("### Synthetic Data Generation")
                
                synthetic_method = st.selectbox(
                    "Generation Method",
                    ["LLM-based", "Templates with Variations", "Document Extraction", "Hybrid"]
                )
                
                st.number_input("Number of test cases to generate", 10, 1000, 50)
                
                if synthetic_method == "LLM-based":
                    generation_model = st.selectbox(
                        "Generation Model",
                        ["gpt-4", "Claude 3 Opus", "PaLM", "Custom"]
                    )
                    
                    st.selectbox("Question Types", ["Diverse Mix", "Factoid", "Comparative", "Procedural", "Analytical"])
                    
                    with st.expander("Generation Prompt"):
                        st.text_area(
                            "Test Case Generation Prompt",
                            """Generate diverse, challenging test questions for a retrieval system about the following topics:
{{topics}}

For each question:
1. Create a realistic user query
2. Provide an ideal answer (what a perfect system should return)
3. List keywords that should be used for retrieval
4. Rate the difficulty (1-5)

Generate {{count}} questions, focusing on different types of queries including factoid, comparative, analytical, and procedural questions.""",
                            height=150
                        )
            
            # Evaluation process configuration
            st.markdown("### Evaluation Process")
            
            eval_process = st.radio(
                "Evaluation Approach",
                ["Automated Batch Evaluation", "Interactive Evaluation", "A/B Testing", "Continuous Monitoring"]
            )
            
            if eval_process == "Automated Batch Evaluation":
                st.number_input("Batch size", 10, 1000, 100)
                st.checkbox("Save evaluation results", value=True)
                st.checkbox("Generate evaluation report", value=True)
                
            elif eval_process == "Interactive Evaluation":
                st.checkbox("Show side-by-side with baseline", value=True)
                st.checkbox("Allow expert feedback collection", value=True)
                st.checkbox("Track evaluation time", value=False)
                
            elif eval_process == "A/B Testing":
                st.selectbox("A/B Test Type", ["50/50 Split", "Multi-armed Bandit", "Champion-Challenger"])
                st.number_input("Test duration (days)", 1, 30, 7)
                st.text_input("Metrics to compare", "relevance, user_satisfaction, click_through")
                
            # Evaluation customization
            st.markdown("### Evaluation Settings")
            
            with st.expander("Advanced Evaluation Settings"):
                st.checkbox("Run parallel evaluations", value=True)
                st.checkbox("Cache retrieval results to speed up evaluation", value=True)
                st.checkbox("Log all intermediate steps", value=False)
                st.slider("Evaluation timeout (seconds per query)", 5, 60, 30)
                
                st.selectbox(
                    "Failure Handling",
                    ["Skip and Log", "Retry (max 3 attempts)", "Default to Baseline", "Halt Evaluation"]
                )
        
        with col2:
            with st.expander("ℹ️ About Ground Truth Evaluation", expanded=True):
                info_tooltip("""
                **Ground Truth Evaluation** tests your RAG system against known correct answers to measure its performance.
                
                **Test Dataset Types:**
                - **Custom Dataset**: Manually created test cases with known answers
                - **Ground Truth from Section 1**: Uses data provided earlier in the workflow
                - **Synthetic Dataset**: Automatically generated test cases
                - **Sample from Logs**: Uses real user queries (if available)
                
                **Synthetic Data Generation:**
                - Creates diverse test cases to evaluate system across scenarios
                - LLM-based generation creates realistic, challenging questions
                - Can target specific domains or question types
                - Helps test edge cases and system limitations
                
                **Evaluation Approaches:**
                - **Automated Batch**: Process many queries at once
                - **Interactive**: Manual review and annotation
                - **A/B Testing**: Compare different configurations
                - **Continuous**: Ongoing monitoring in production
                
                **Benefits of Ground Truth Evaluation:**
                - Objective measurement of system quality
                - Consistent benchmarks for improvement
                - Identifies specific failure modes
                - Enables targeted optimization
                
                **Best Practices:**
                - Include diverse question types and difficulties
                - Test edge cases and challenging scenarios
                - Use multiple evaluation metrics
                - Compare against baseline and alternative systems
                - Regularly update test sets to prevent overfitting
                """)
            
            # Evaluation example
            st.markdown("#### Example Evaluation")
            
            if eval_process == "Automated Batch Evaluation":
                eval_results_example = pd.DataFrame({
                    "Question": [
                        "What is the mechanism of action for metformin?",
                        "What are common side effects of statins?",
                        "How do SGLT2 inhibitors work?",
                        "What is the recommended dosage of aspirin for prevention?"
                    ],
                    "Retrieval_Success": [True, True, True, False],
                    "Answer_Accuracy": [0.95, 0.87, 0.92, 0.45],
                    "Latency_ms": [234, 312, 198, 245]
                })
                
                st.dataframe(eval_results_example, use_container_width=True)
                
                st.markdown("#### Aggregate Results")
                
                col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                
                with col_metrics1:
                    st.metric("Retrieval Success", "75%")
                    
                with col_metrics2:
                    st.metric("Avg. Answer Accuracy", "0.80")
                    
                with col_metrics3:
                    st.metric("Avg. Latency", "247ms")
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if dataset_source == "Upload Custom Dataset":
                st.success("✅ Custom datasets provide the most reliable evaluation")
                
            elif dataset_source == "Generate Synthetic Dataset":
                st.info("ℹ️ Synthetic data is good for scale but verify with real queries")
                
            if eval_process == "A/B Testing":
                st.success("✅ A/B testing is excellent for comparing configurations")
                
            elif eval_process == "Automated Batch Evaluation":
                st.success("✅ Automated evaluation provides consistent benchmarks")
                
            if synthetic_method == "LLM-based" if dataset_source == "Generate Synthetic Dataset" else False:
                st.info("ℹ️ LLM-generated questions may have biases - review for quality")
    
    # RUNTIME MONITORING TAB
    with eval_tabs[2]:
        st.subheader("Runtime Monitoring")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable monitoring
            enable_monitoring = st.checkbox("Enable runtime monitoring", value=True)
            
            if enable_monitoring:
                # Monitoring categories
                st.markdown("### Monitoring Categories")
                
                performance_monitoring = st.multiselect(
                    "Performance Metrics",
                    ["Latency", "Throughput", "Error Rate", "Token Usage", "API Costs", "Resource Utilization", "Cache Hit Rate", "Custom"],
                    default=["Latency", "Error Rate", "Token Usage"]
                )
                
                quality_monitoring = st.multiselect(
                    "Quality Metrics",
                    ["Answer Relevance", "Source Quality", "User Feedback", "Hallucination Rate", "Query Success Rate", "Custom"],
                    default=["Answer Relevance", "User Feedback"]
                )
                
                system_monitoring = st.multiselect(
                    "System Health Metrics",
                    ["Component Status", "Database Health", "API Availability", "Resource Usage", "Queue Depth", "Custom"],
                    default=["Component Status", "API Availability"]
                )
                
                # Monitoring frequency
                st.markdown("### Monitoring Frequency")
                
                freq_options = {
                    "Per-Request": st.checkbox("Per-Request Logging", value=True),
                    "Real-time": st.checkbox("Real-time Dashboards", value=True),
                    "Periodic": st.checkbox("Periodic Reports", value=True)
                }
                
                if freq_options["Periodic"]:
                    report_frequency = st.selectbox(
                        "Report Frequency",
                        ["Hourly", "Daily", "Weekly", "Monthly"]
                    )
                
                # Alert configuration
                st.markdown("### Alert Configuration")
                
                enable_alerts = st.checkbox("Enable alerts", value=True)
                
                if enable_alerts:
                    alert_types = st.multiselect(
                        "Alert Types",
                        ["Latency Threshold", "Error Rate Spike", "Low Retrieval Quality", "Token Budget", "Component Failure", "Custom"],
                        default=["Error Rate Spike", "Component Failure"]
                    )
                    
                    for alert in alert_types:
                        with st.expander(f"{alert} Configuration"):
                            if alert == "Latency Threshold":
                                st.slider("Latency threshold (ms)", 100, 5000, 1000)
                                
                            elif alert == "Error Rate Spike":
                                st.slider("Error rate threshold (%)", 1, 20, 5)
                                st.slider("Detection window (minutes)", 1, 60, 5)
                                
                            elif alert == "Low Retrieval Quality":
                                st.slider("Minimum acceptable relevance score", 0.1, 1.0, 0.6, 0.05)
                                
                            elif alert == "Token Budget":
                                st.number_input("Daily token budget (thousands)", 10, 10000, 1000)
                                st.slider("Alert at usage percentage", 50, 100, 80)
                    
                    notification_channels = st.multiselect(
                        "Notification Channels",
                        ["Email", "Slack", "SMS", "Dashboard", "Webhook", "PagerDuty"],
                        default=["Email", "Dashboard"]
                    )
                
                # Dashboard configuration  
                st.markdown("### Dashboard Configuration")
                
                dashboard_type = st.selectbox(
                    "Dashboard Type",
                    ["Basic", "Comprehensive", "Executive", "Technical", "Custom"]
                )
                
                dashboard_components = st.multiselect(
                    "Dashboard Components",
                    ["Performance Charts", "Quality Metrics", "Error Logs", "User Feedback", "Cost Tracking", "System Health"],
                    default=["Performance Charts", "Quality Metrics", "User Feedback"]
                )
        
        with col2:
            with st.expander("ℹ️ About Runtime Monitoring", expanded=True):
                info_tooltip("""
                **Runtime Monitoring** tracks your RAG system's behavior, performance, and quality in production.
                
                **Monitoring Categories:**
                
                **Performance Metrics:**
                - **Latency**: Response time for queries
                - **Throughput**: Queries processed per time period
                - **Error Rate**: Percentage of failed requests
                - **Token Usage**: LLM token consumption
                - **API Costs**: Spending on external APIs
                - **Resource Utilization**: CPU, memory, storage usage
                
                **Quality Metrics:**
                - **Answer Relevance**: Quality of responses
                - **Source Quality**: Quality of retrieved documents
                - **User Feedback**: Explicit and implicit user ratings
                - **Hallucination Rate**: Detected incorrect generations
                - **Query Success Rate**: Successfully handled queries
                
                **System Health:**
                - Overall system status and component health
                - Infrastructure and dependency status
                - Helps prevent and quickly address outages
                
                **Monitoring Frequency:**
                - **Per-Request**: Detailed logging of each interaction
                - **Real-time**: Live dashboards for immediate visibility
                - **Periodic**: Scheduled reports and aggregations
                
                **Alert System:**
                - Proactively notifies about issues
                - Prevents degradation of user experience
                - Enables quick response to problems
                
                **Best Practices:**
                - Monitor across the full RAG pipeline
                - Balance between detail and information overload
                - Set actionable alerts with clear thresholds
                - Track trends over time, not just instantaneous values
                """)
            
            # Example dashboard
            st.markdown("#### Monitoring Dashboard Example")
            
            # Create sample monitoring data
            dates = pd.date_range(start="2025-05-20", end="2025-05-27", freq="D")
            latency_data = pd.DataFrame({
                "date": dates,
                "Retrieval Latency (ms)": [120, 125, 118, 145, 155, 130, 128, 122],
                "Generation Latency (ms)": [450, 460, 455, 490, 510, 470, 465, 455]
            })
            
            # Create a line chart for latency
            latency_chart = alt.Chart(latency_data.melt(id_vars=["date"], 
                                                      var_name="Metric", 
                                                      value_name="Milliseconds")).mark_line().encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Milliseconds:Q"),
                color="Metric:N",
                tooltip=["date:T", "Metric:N", "Milliseconds:Q"]
            ).properties(
                title="Latency Trends",
                height=200
            )
            
            st.altair_chart(latency_chart, use_container_width=True)
            
            # Sample quality metrics
            quality_data = pd.DataFrame({
                "Metric": ["Relevance Score", "Source Quality", "Hallucination Rate", "User Satisfaction"],
                "Current": [0.87, 0.92, 0.05, 4.2],
                "7-day Avg": [0.85, 0.90, 0.06, 4.1],
                "Trend": ["↑", "↑", "↓", "↑"]
            })
            
            st.dataframe(quality_data, use_container_width=True)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_monitoring:
                if "Latency" in performance_monitoring and "Error Rate" in performance_monitoring:
                    st.success("✅ Latency and error monitoring are essential for production systems")
                    
                if "User Feedback" in quality_monitoring:
                    st.success("✅ User feedback provides valuable real-world quality signals")
                    
                if enable_alerts:
                    st.success("✅ Alerts help prevent issues before they impact users")
                    
                missing_important = []
                if "Token Usage" not in performance_monitoring:
                    missing_important.append("Token Usage")
                if "API Availability" not in system_monitoring:
                    missing_important.append("API Availability")
                
                if missing_important:
                    st.info(f"ℹ️ Consider adding {', '.join(missing_important)} to your monitoring")
            else:
                st.warning("⚠️ Runtime monitoring is strongly recommended for production systems")
    
    # CONTINUOUS IMPROVEMENT TAB
    with eval_tabs[3]:
        st.subheader("Continuous Improvement")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Enable continuous improvement
            enable_ci = st.checkbox("Enable continuous improvement framework", value=True)
            
            if enable_ci:
                # Feedback collection
                st.markdown("### Feedback Collection")
                
                feedback_sources = st.multiselect(
                    "Feedback Sources",
                    ["User Ratings", "Thumbs Up/Down", "Follow-up Queries", "Time on Page", "Explicit Comments", 
                     "Support Tickets", "Monitoring Alerts", "Custom Sources"],
                    default=["User Ratings", "Thumbs Up/Down", "Explicit Comments"]
                )
                
                if "User Ratings" in feedback_sources:
                    rating_type = st.selectbox(
                        "Rating Type",
                        ["1-5 Stars", "NPS (0-10)", "Simple (Good/Bad)", "Multi-dimension"]
                    )
                    
                    if rating_type == "Multi-dimension":
                        st.multiselect(
                            "Rating Dimensions",
                            ["Relevance", "Accuracy", "Completeness", "Clarity", "Speed"],
                            default=["Relevance", "Accuracy", "Clarity"]
                        )
                
                # Feedback processing
                st.markdown("### Feedback Processing")
                
                feedback_processing = st.multiselect(
                    "Processing Methods",
                    ["Aggregate Analysis", "Pattern Detection", "Root Cause Analysis", "Failure Clustering", 
                     "User Segment Analysis", "LLM-based Analysis"],
                    default=["Aggregate Analysis", "Pattern Detection"]
                )
                
                if "LLM-based Analysis" in feedback_processing:
                    feedback_llm = st.selectbox(
                        "Feedback Analysis LLM",
                        ["gpt-4", "Claude 3 Sonnet", "Custom"]
                    )
                    
                    with st.expander("Feedback Analysis Prompt"):
                        st.text_area(
                            "Analysis Prompt",
                            """Analyze the following user feedback for our RAG system.
Identify patterns, common issues, and actionable insights.

User Feedback:
{{feedback}}

Please provide:
1. Top 3 issues identified
2. Root causes for each issue
3. Suggested improvements
4. Priority level (High/Medium/Low) for each suggestion""",
                            height=150
                        )
                
                # Improvement cycle
                st.markdown("### Improvement Cycle")
                
                improvement_cycle = st.selectbox(
                    "Improvement Approach",
                    ["Continuous Deployment", "Periodic Updates", "Manual Review & Update", "Hybrid"]
                )
                
                if improvement_cycle == "Continuous Deployment":
                    st.checkbox("Enable automatic A/B testing", value=True)
                    st.checkbox("Require human approval before deployment", value=True)
                    
                elif improvement_cycle == "Periodic Updates":
                    update_frequency = st.selectbox(
                        "Update Frequency",
                        ["Weekly", "Bi-weekly", "Monthly", "Quarterly"]
                    )
                    
                    st.checkbox("Include comprehensive evaluation before update", value=True)
                
                # Learning mechanisms
                st.markdown("### Learning Mechanisms")
                
                learning_mechanisms = st.multiselect(
                    "Learning Methods",
                    ["Data Augmentation", "Fine-tuning", "Rule Updates", "Prompt Engineering", 
                     "Retrieval Enhancement", "Document Refinement", "Custom Methods"],
                    default=["Data Augmentation", "Prompt Engineering", "Retrieval Enhancement"]
                )
                
                if "Fine-tuning" in learning_mechanisms:
                    ft_components = st.multiselect(
                        "Components to Fine-tune",
                        ["Embedding Model", "Reranker", "LLM", "Classification Models"],
                        default=["Reranker"]
                    )
                    
                    st.number_input("Minimum feedback samples before fine-tuning", 100, 10000, 1000)
        
        with col2:
            with st.expander("ℹ️ About Continuous Improvement", expanded=True):
                info_tooltip("""
                **Continuous Improvement** creates a systematic framework for enhancing RAG system quality over time.
                
                **Feedback Collection:**
                - Gathers signals about system performance
                - Both explicit (ratings) and implicit (behavior)
                - Critical for understanding real-world performance
                - Helps identify issues not caught in testing
                
                **Feedback Processing:**
                - Transforms raw feedback into actionable insights
                - Detects patterns across user interactions
                - Categorizes and prioritizes issues
                - Links feedback to specific system components
                
                **Improvement Cycle:**
                - **Continuous Deployment**: Automatic updates based on feedback
                - **Periodic Updates**: Scheduled improvement cycles
                - **Manual Review**: Human-in-the-loop improvements
                - **Hybrid**: Combination of automated and manual processes
                
                **Learning Mechanisms:**
                - **Data Augmentation**: Expand training/evaluation data
                - **Fine-tuning**: Adapt models based on feedback
                - **Rule Updates**: Refine system rules and heuristics
                - **Prompt Engineering**: Improve system prompts
                - **Retrieval Enhancement**: Update retrieval components
                - **Document Refinement**: Improve source content
                
                **Benefits:**
                - System improves rather than degrades over time
                - Adapts to changing content and user needs
                - Addresses edge cases and failure modes
                - Creates a virtuous feedback loop
                
                **Best Practices:**
                - Establish clear metrics for improvement
                - Create systematic processes for incorporating feedback
                - Balance automation with human oversight
                - Document changes and measure their impact
                """)
            
            # Example improvement cycle
            st.markdown("#### Continuous Improvement Process")
            
            ci_process = """
### Example Improvement Cycle

1. **Collect Feedback**:
   - User rates answer as 2/5 stars
   - Comment: "Missing key information about contraindications"

2. **Process & Analyze**:
   - Issue categorized as "Incomplete Answer"
   - Root cause: Related documents not retrieved
   - Priority: Medium

3. **Implement Improvements**:
   - Added keyword expansion to query processing
   - Updated chunking strategy to preserve medication sections
   - Added specific instructions in prompt about including contraindications

4. **Validate Changes**:
   - A/B test shows 18% improvement in completeness ratings
   - No negative impact on other metrics
   - Changes approved for production

5. **Monitor & Iterate**:
   - Continue monitoring completeness ratings
   - Add specific test cases for contraindication questions
   - Document learnings for future improvements
            """
            
            st.code(ci_process)
            
            # Example improvement trends
            st.markdown("#### Improvement Metrics Trend")
            
            # Sample improvement data
            improvement_dates = pd.date_range(start="2025-01-01", end="2025-05-01", freq="MS")
            improvement_data = pd.DataFrame({
                "date": improvement_dates,
                "Relevance Score": [0.72, 0.74, 0.78, 0.81, 0.85],
                "User Satisfaction": [3.8, 3.9, 4.0, 4.2, 4.3]
            })
            
            # Create a line chart for improvements
            improvement_chart = alt.Chart(improvement_data.melt(id_vars=["date"], 
                                                          var_name="Metric", 
                                                          value_name="Score")).mark_line().encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Score:Q"),
                color="Metric:N",
                tooltip=["date:T", "Metric:N", "Score:Q"]
            ).properties(
                title="Metrics Improvement Over Time",
                height=200
            )
            
            st.altair_chart(improvement_chart, use_container_width=True)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if enable_ci:
                if len(feedback_sources) >= 3:
                    st.success("✅ Multiple feedback sources provide more robust signals")
                    
                if "Pattern Detection" in feedback_processing:
                    st.success("✅ Pattern detection helps identify systematic issues")
                    
                if "LLM-based Analysis" in feedback_processing:
                    st.success("✅ LLM analysis can uncover insights from unstructured feedback")
                    
                if improvement_cycle == "Continuous Deployment":
                    st.success("✅ Continuous deployment enables rapid improvement")
                    st.info("ℹ️ Ensure proper safeguards and monitoring are in place")
                
                missing_recommended = []
                if "Data Augmentation" not in learning_mechanisms:
                    missing_recommended.append("Data Augmentation")
                if "Prompt Engineering" not in learning_mechanisms:
                    missing_recommended.append("Prompt Engineering")
                    
                if missing_recommended:
                    st.info(f"ℹ️ Consider including {', '.join(missing_recommended)} in your learning mechanisms")
            else:
                st.warning("⚠️ Without continuous improvement, system quality will plateau or degrade")

###############################
# 10. DEPLOYMENT
###############################
elif main_section == "10. Deployment":
    st.header("10. Deployment")
    st.markdown("Configure deployment options, infrastructure, and integration with existing systems")
    
    # Create tabs for deployment options
    deploy_tabs = st.tabs(["Deployment Architecture", "Infrastructure", "API & Integration", "Security & Compliance"])
    
    # DEPLOYMENT ARCHITECTURE TAB
    with deploy_tabs[0]:
        st.subheader("Deployment Architecture")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Deployment approach
            st.markdown("### Deployment Approach")
            
            deployment_model = st.selectbox(
                "Deployment Model",
                ["Serverless", "Container-based", "VM-based", "Hybrid", "Edge-Cloud Split"]
            )
            
            if deployment_model == "Serverless":
                serverless_provider = st.selectbox(
                    "Serverless Provider",
                    ["AWS Lambda", "Azure Functions", "Google Cloud Functions", "Vercel", "Netlify Functions", "Custom"]
                )
                
                if serverless_provider in ["AWS Lambda", "Azure Functions", "Google Cloud Functions"]:
                    st.number_input("Function Timeout (seconds)", 10, 900, 60)
                    st.number_input("Memory Allocation (MB)", 128, 10240, 2048)
                    st.checkbox("Enable Provisioned Concurrency", value=True)
            
            elif deployment_model == "Container-based":
                container_platform = st.selectbox(
                    "Container Platform",
                    ["Kubernetes", "Docker Compose", "AWS ECS", "Azure Container Apps", "Google Cloud Run", "Custom"]
                )
                
                if container_platform == "Kubernetes":
                    k8s_environment = st.selectbox(
                        "Kubernetes Environment",
                        ["EKS (AWS)", "AKS (Azure)", "GKE (Google)", "Self-managed", "OpenShift", "K3s"]
                    )
                    
                    st.number_input("Number of Replicas", 1, 100, 3)
                    st.checkbox("Enable Horizontal Pod Autoscaling", value=True)
                    st.checkbox("Use Helm Charts", value=True)
            
            elif deployment_model == "VM-based":
                vm_provider = st.selectbox(
                    "VM Provider",
                    ["AWS EC2", "Azure VMs", "Google Compute Engine", "On-premises", "Custom"]
                )
                
                st.number_input("Number of Instances", 1, 20, 2)
                st.selectbox("Load Balancing", ["Application LB", "Network LB", "HAProxy", "Nginx", "None"])
            
            # System architecture
            st.markdown("### System Architecture")
            
            architecture_type = st.selectbox(
                "Architecture Type",
                ["Microservices", "Monolithic", "Service-Oriented", "Event-Driven", "Hybrid"]
            )
            
            if architecture_type == "Microservices":
                core_services = st.multiselect(
                    "Core Services",
                    ["Document Processing Service", "Embedding Service", "Vector Store Service", 
                     "Retrieval Service", "LLM Service", "Orchestration Service", "API Gateway",
                     "Authentication Service", "Feedback Service", "Monitoring Service"],
                    default=["Document Processing Service", "Embedding Service", "Vector Store Service", 
                             "Retrieval Service", "LLM Service", "API Gateway"]
                )
                
                st.selectbox(
                    "Service Communication",
                    ["REST APIs", "gRPC", "Message Queue", "Event Bus", "Hybrid"]
                )
            
            # Pipeline configuration
            st.markdown("### Pipeline Configuration")
            
            pipeline_mode = st.selectbox(
                "Pipeline Mode",
                ["Synchronous", "Asynchronous", "Hybrid", "Batch Processing"]
            )
            
            if pipeline_mode in ["Asynchronous", "Hybrid"]:
                queue_system = st.selectbox(
                    "Queue System",
                    ["AWS SQS", "Azure Service Bus", "Google Pub/Sub", "RabbitMQ", "Kafka", "Redis Streams"]
                )
                
                st.checkbox("Enable Dead Letter Queue", value=True)
                st.checkbox("Enable Message Persistence", value=True)
            
            # Scaling configuration
            st.markdown("### Scaling Configuration")
            
            scaling_type = st.selectbox(
                "Scaling Type",
                ["Auto-scaling", "Manual Scaling", "Scheduled Scaling", "Predictive Scaling", "No Scaling"]
            )
            
            if scaling_type == "Auto-scaling":
                scaling_metrics = st.multiselect(
                    "Scaling Metrics",
                    ["CPU Utilization", "Memory Usage", "Request Count", "Queue Depth", "Latency", "Custom Metric"],
                    default=["CPU Utilization", "Request Count"]
                )
                
                st.slider("Scale-out Threshold", 50, 95, 70, 5, help="% utilization to trigger scaling out")
                st.slider("Scale-in Threshold", 10, 50, 30, 5, help="% utilization to trigger scaling in")
                st.number_input("Cooldown Period (seconds)", 30, 600, 120)
        
        with col2:
            with st.expander("ℹ️ About Deployment Architecture", expanded=True):
                info_tooltip("""
                **Deployment Architecture** defines how your RAG system will be deployed, scaled, and operated.
                
                **Deployment Models:**
                
                **Serverless:**
                - No infrastructure management
                - Pay-per-use pricing
                - Automatic scaling
                - Good for variable workloads
                - Limitations: cold starts, execution timeouts
                
                **Container-based:**
                - Consistent environment across deployment stages
                - Efficient resource utilization
                - Faster startup than VMs
                - Industry standard approach (especially Kubernetes)
                - More complex to set up and manage
                
                **VM-based:**
                - Dedicated resources
                - Full OS control
                - Simpler to set up than containers
                - Higher overhead and less efficient scaling
                
                **System Architectures:**
                
                **Microservices:**
                - Independent, loosely coupled services
                - Can scale and deploy components separately
                - Higher operational complexity
                - Better for large, complex systems
                
                **Monolithic:**
                - Single integrated application
                - Simpler to develop and deploy
                - Harder to scale individual components
                - Good for smaller applications
                
                **Pipeline Modes:**
                
                **Synchronous:**
                - Request-response pattern
                - Real-time responses
                - Limited by processing time
                
                **Asynchronous:**
                - Queue-based processing
                - Better handling of traffic spikes
                - Can process long-running tasks
                - More complex error handling
                
                **Scaling Considerations:**
                - Auto-scaling works well for variable workloads
                - Consider both scaling out (more instances) and scaling up (larger instances)
                - Set appropriate thresholds to avoid resource waste
                - Balance cost and performance requirements
                """)
            
            # Deployment architecture diagram
            st.markdown("#### Architecture Diagram")
            
            if architecture_type == "Microservices":
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*hkY_WaXg5ttY7AIRzfS7yg.png",
                       caption="Microservices architecture for RAG system")
                
            elif deployment_model == "Serverless":
                st.image("https://miro.medium.com/v2/resize:fit:1400/1*Kj5rUkTKACAWdK_XxrC5-g.png",
                       caption="Serverless architecture for RAG system")
            
            # Performance considerations
            st.markdown("#### Deployment Trade-offs")
            
            deployment_tradeoffs = pd.DataFrame({
                "Aspect": ["Scalability", "Cost", "Latency", "Management", "Flexibility"],
                "Serverless": ["Excellent", "Pay-per-use", "Cold starts", "Minimal", "Limited"],
                "Container": ["Very Good", "Resource-based", "Good", "Moderate", "Excellent"],
                "VM-based": ["Good", "Fixed+Variable", "Excellent", "High", "Very Good"]
            })
            
            st.dataframe(deployment_tradeoffs, hide_index=True)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if deployment_model == "Serverless":
                st.success("✅ Serverless is excellent for variable workloads and minimal management")
                
                if architecture_type == "Monolithic":
                    st.warning("⚠️ Consider microservices for better modularity with serverless")
                
                st.info("ℹ️ Be mindful of cold starts and timeout limits")
                
            elif deployment_model == "Container-based":
                st.success("✅ Container-based deployment offers great flexibility and scalability")
                
                if container_platform == "Kubernetes":
                    st.success("✅ Kubernetes is the industry standard for container orchestration")
                    
            if pipeline_mode == "Asynchronous" and architecture_type == "Microservices":
                st.success("✅ Asynchronous communication works well with microservices")
            
            if scaling_type == "Auto-scaling":
                st.success("✅ Auto-scaling will help optimize resource usage and costs")
                
    # INFRASTRUCTURE TAB
    with deploy_tabs[1]:
        st.subheader("Infrastructure Requirements")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Cloud provider
            st.markdown("### Cloud Provider")
            
            cloud_provider = st.selectbox(
                "Primary Cloud Provider",
                ["AWS", "Azure", "Google Cloud", "Multi-cloud", "On-premises", "Hybrid"]
            )
            
            if cloud_provider == "Multi-cloud":
                multi_cloud_providers = st.multiselect(
                    "Selected Providers",
                    ["AWS", "Azure", "Google Cloud", "IBM Cloud", "Oracle Cloud"],
                    default=["AWS", "Azure"]
                )
                
                st.selectbox(
                    "Multi-cloud Strategy",
                    ["Best-of-breed Services", "Redundancy", "Cost Optimization", "Avoiding Vendor Lock-in"]
                )
            
            elif cloud_provider == "Hybrid":
                st.multiselect(
                    "Hybrid Components",
                    ["Cloud Vector Database", "On-prem Document Storage", "Cloud LLM APIs", 
                     "On-prem Embedding Models", "Cloud Orchestration"],
                    default=["Cloud Vector Database", "Cloud LLM APIs"]
                )
            
            # Compute resources
            st.markdown("### Compute Resources")
            
            compute_type = st.selectbox(
                "Compute Type",
                ["CPU-optimized", "Memory-optimized", "GPU-enabled", "Balanced", "Custom"]
            )
            
            if compute_type == "CPU-optimized":
                st.slider("vCPUs", 2, 64, 8)
                st.slider("Memory (GB)", 4, 128, 16)
                
            elif compute_type == "Memory-optimized":
                st.slider("vCPUs", 2, 64, 4)
                st.slider("Memory (GB)", 8, 512, 32)
                
            elif compute_type == "GPU-enabled":
                st.slider("vCPUs", 4, 96, 8)
                st.slider("Memory (GB)", 16, 256, 32)
                st.selectbox(
                    "GPU Type",
                    ["NVIDIA T4", "NVIDIA A10G", "NVIDIA A100", "NVIDIA L4", "AMD Instinct"]
                )
                st.slider("Number of GPUs", 1, 8, 1)
            
            # Storage configuration
            st.markdown("### Storage Configuration")
            
            storage_types = st.multiselect(
                "Storage Types",
                ["Block Storage", "Object Storage", "File System", "Database Storage", "Cache"],
                default=["Block Storage", "Object Storage"]
            )
            
            if "Block Storage" in storage_types:
                st.slider("Block Storage Size (GB)", 10, 2000, 100)
                st.selectbox("Block Storage Type", ["SSD", "HDD", "NVMe", "Premium SSD"])
                
            if "Object Storage" in storage_types:
                st.text_input("Object Storage Bucket Name", "rag-documents-storage")
                st.checkbox("Enable Versioning", value=True)
                st.checkbox("Enable Object Lifecycle Policy", value=True)
            
            # Networking
            st.markdown("### Networking")
            
            networking_options = st.multiselect(
                "Networking Features",
                ["Load Balancer", "CDN", "VPC/VNET", "API Gateway", "VPN Connection", "Direct Connect"],
                default=["Load Balancer", "API Gateway"]
            )
            
            if "Load Balancer" in networking_options:
                lb_type = st.selectbox(
                    "Load Balancer Type",
                    ["Application LB", "Network LB", "Global LB", "Internal LB"]
                )
                
                st.checkbox("Enable SSL/TLS Termination", value=True)
                st.checkbox("Enable Session Affinity", value=False)
            
            # Database requirements
            st.markdown("### Database Requirements")
            
            db_options = st.multiselect(
                "Database Types",
                ["Vector Database", "Relational Database", "Document Store", "Key-Value Store", "Cache"],
                default=["Vector Database", "Relational Database"]
            )
            
            if "Vector Database" in db_options:
                vector_db_option = st.selectbox(
                    "Vector Database Deployment",
                    ["Self-hosted", "Managed Service", "Serverless", "Hybrid"]
                )
                
                if vector_db_option == "Self-hosted":
                    st.number_input("Number of Vector DB Nodes", 1, 10, 3)
                    st.checkbox("Configure High Availability", value=True)
        
        with col2:
            with st.expander("ℹ️ About Infrastructure Requirements", expanded=True):
                info_tooltip("""
                **Infrastructure Requirements** define the computing, storage, and networking resources needed to run your RAG system.
                
                **Cloud Provider Options:**
                
                **Single Cloud (AWS, Azure, Google Cloud):**
                - Simplified management with unified tooling
                - Integrated services and optimizations
                - Potential for vendor lock-in
                
                **Multi-cloud:**
                - Leverages best services from multiple providers
                - Provides redundancy and failover options
                - More complex to manage and integrate
                - Better negotiation leverage with providers
                
                **Hybrid:**
                - Combines cloud and on-premises resources
                - Good for data sovereignty or security requirements
                - More complex networking and integration
                
                **Compute Considerations:**
                - **CPU-optimized**: For embedding generation, reranking
                - **Memory-optimized**: For large models, vector operations
                - **GPU-enabled**: For local LLM hosting, batch embedding
                
                **Storage Requirements:**
                - **Block Storage**: For OS and applications
                - **Object Storage**: For documents and model weights
                - **File System**: For shared access to data
                - **Database Storage**: For vectors, metadata, user data
                
                **Networking Needs:**
                - **Load Balancer**: Distributes traffic across services
                - **CDN**: Caches responses for improved performance
                - **API Gateway**: Manages API traffic and security
                - **VPC/VNET**: Isolates and secures internal services
                
                **Database Considerations:**
                - Vector databases have specific performance requirements
                - Consider managed services for reduced operational overhead
                - High availability configurations for production systems
                - Proper backup and disaster recovery planning
                """)
            
            # Resource requirements table
            st.markdown("#### Resource Requirements by Component")
            
            component_resources = pd.DataFrame({
                "Component": ["Document Processing", "Embedding Generation", "Vector Search", "LLM Service", "API Layer"],
                "Compute": ["Medium CPU", "High CPU/GPU", "High Memory", "High CPU/GPU", "Low-Medium"],
                "Storage": ["High", "Low", "High", "Medium", "Low"],
                "Scaling Factor": ["Document volume", "Query volume", "Vector count", "Query complexity", "Request rate"]
            })
            
            st.dataframe(component_resources, hide_index=True)
            
            # Cost factors
            st.markdown("#### Key Cost Factors")
            
            cost_factors = [
                "**LLM API costs**: Based on input/output tokens",
                "**Vector DB costs**: Based on vector count and operations",
                "**Compute costs**: Based on instance types and running time",
                "**Storage costs**: Based on document volume and type",
                "**Data transfer**: Especially between cloud regions or providers"
            ]
            
            for factor in cost_factors:
                st.markdown(f"- {factor}")
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if cloud_provider == "Multi-cloud":
                st.info("ℹ️ Multi-cloud provides flexibility but increases complexity")
                st.info("ℹ️ Consider starting with a single cloud and expanding later")
                
            if compute_type == "GPU-enabled":
                st.success("✅ GPU resources enable local model deployment")
                st.warning("⚠️ GPUs significantly increase costs - ensure proper utilization")
                
            elif compute_type == "Memory-optimized":
                st.success("✅ Memory-optimized instances work well for vector operations")
                
            if "Vector Database" in db_options and vector_db_option == "Managed Service":
                st.success("✅ Managed vector database services reduce operational overhead")
            
            if "Block Storage" in storage_types and st.session_state.get("Block Storage Type") == "NVMe":
                st.success("✅ NVMe storage provides best performance for vector databases")
                
    # API & INTEGRATION TAB
    with deploy_tabs[2]:
        st.subheader("API & Integrations")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # API design
            st.markdown("### API Design")
            
            api_style = st.selectbox(
                "API Style",
                ["REST", "GraphQL", "gRPC", "WebSockets", "Hybrid"]
            )
            
            if api_style == "REST":
                st.markdown("#### REST API Endpoints")
                
                endpoints = [
                    {"name": "Query", "method": "POST", "path": "/api/v1/query", "description": "Submit a query and get response"},
                    {"name": "Document Upload", "method": "POST", "path": "/api/v1/documents", "description": "Upload documents to the system"},
                    {"name": "Feedback", "method": "POST", "path": "/api/v1/feedback", "description": "Submit user feedback on responses"},
                    {"name": "Health Check", "method": "GET", "path": "/api/v1/health", "description": "Check system health status"}
                ]
                
                endpoint_df = pd.DataFrame(endpoints)
                st.dataframe(endpoint_df, hide_index=True)
                
                with st.expander("Add Custom Endpoint"):
                    col_ep1, col_ep2 = st.columns(2)
                    with col_ep1:
                        ep_name = st.text_input("Endpoint Name", "Custom Endpoint")
                        ep_method = st.selectbox("HTTP Method", ["GET", "POST", "PUT", "DELETE"])
                    with col_ep2:
                        ep_path = st.text_input("Endpoint Path", "/api/v1/custom")
                        ep_desc = st.text_input("Description", "Custom endpoint description")
                    
                    if st.button("Add Endpoint"):
                        st.success(f"Endpoint {ep_name} added successfully!")
                
            elif api_style == "GraphQL":
                st.code("""
type Query {
  answer(question: String!): AnswerResponse!
  documents(filter: DocumentFilter): [Document!]!
  health: SystemStatus!
}

type AnswerResponse {
  answer: String!
  sources: [Source!]!
  confidence: Float!
}

type Source {
  documentId: ID!
  title: String!
  relevance: Float!
  snippet: String!
}

type Document {
  id: ID!
  title: String!
  content: String!
  metadata: JSONObject!
}

type SystemStatus {
  status: String!
  components: [ComponentStatus!]!
}

type ComponentStatus {
  name: String!
  status: String!
  latency: Int
}
                """, language="graphql")
                
            # Rate limiting and quotas
            st.markdown("### Rate Limiting & Quotas")
            
            enable_rate_limiting = st.checkbox("Enable Rate Limiting", value=True)
            
            if enable_rate_limiting:
                col_rate1, col_rate2 = st.columns(2)
                with col_rate1:
                    st.number_input("Requests per Minute (RPM)", 10, 10000, 60)
                    st.number_input("Burst Capacity", 5, 100, 20)
                with col_rate2:
                    st.selectbox("Rate Limit Algorithm", ["Fixed Window", "Sliding Window", "Token Bucket", "Leaky Bucket"])
                    st.checkbox("Use Different Limits per Endpoint", value=True)
                
                st.selectbox("Rate Limit Scope", ["Per API Key", "Per User", "Per IP", "Global"])
                
                st.text_input("Rate Limit Headers", "X-RateLimit-Limit, X-RateLimit-Remaining, X-RateLimit-Reset")
            
            # Authentication and authorization
            st.markdown("### Authentication & Authorization")
            
            auth_methods = st.multiselect(
                "Authentication Methods",
                ["API Key", "JWT", "OAuth 2.0", "OpenID Connect", "Basic Auth", "Custom"],
                default=["API Key", "OAuth 2.0"]
            )
            
            if "OAuth 2.0" in auth_methods:
                oauth_flows = st.multiselect(
                    "OAuth 2.0 Flows",
                    ["Authorization Code", "Client Credentials", "Implicit", "Password", "Device Code"],
                    default=["Authorization Code", "Client Credentials"]
                )
                
                st.checkbox("Enable PKCE for Authorization Code Flow", value=True)
            
            # Role-based access
            st.markdown("### Access Control")
            
            enable_rbac = st.checkbox("Enable Role-Based Access Control", value=True)
            
            if enable_rbac:
                roles = st.multiselect(
                    "Roles",
                    ["Admin", "User", "ReadOnly", "Editor", "System", "Custom"],
                    default=["Admin", "User", "ReadOnly"]
                )
                
                st.selectbox("Access Control Implementation", ["JWT Claims", "External Authorization Server", "Database-driven", "Custom Logic"])
            
            # Integration options
            st.markdown("### Integrations")
            
            integration_options = st.multiselect(
                "Integration Options",
                ["Webhooks", "SDK", "Direct API", "Message Queue", "SSE (Server-Sent Events)", "Custom Integration"],
                default=["SDK", "Direct API"]
            )
            
            if "Webhooks" in integration_options:
                st.text_input("Webhook URL Format", "https://your-domain.com/webhook")
                st.multiselect(
                    "Webhook Events",
                    ["Query Processed", "Document Indexed", "System Status Change", "Feedback Received", "Custom Event"],
                    default=["Query Processed", "Document Indexed"]
                )
                
            if "SDK" in integration_options:
                sdk_languages = st.multiselect(
                    "SDK Languages",
                    ["Python", "JavaScript/TypeScript", "Java", "C#", "Go", "Ruby", "PHP"],
                    default=["Python", "JavaScript/TypeScript"]
                )
        
        with col2:
            with st.expander("ℹ️ About API & Integrations", expanded=True):
                info_tooltip("""
                **API & Integrations** define how other systems interact with your RAG application.
                
                **API Styles:**
                
                **REST:**
                - Resource-oriented API design
                - HTTP methods (GET, POST, PUT, DELETE)
                - Simple to understand and implement
                - Widely supported by tools and frameworks
                - Stateless nature simplifies scaling
                
                **GraphQL:**
                - Query language for APIs
                - Request exactly what you need
                - Single endpoint for all operations
                - Introspection and strong typing
                - Reduces over-fetching and under-fetching
                
                **gRPC:**
                - High-performance RPC framework
                - Protocol Buffers for serialization
                - Excellent for microservice communication
                - Built-in code generation
                - Supports streaming
                
                **WebSockets:**
                - Full-duplex communication
                - Good for real-time updates
                - Maintains persistent connection
                - Useful for interactive applications
                
                **API Management:**
                - **Rate Limiting**: Prevents abuse and ensures fair usage
                - **Authentication**: Verifies identity of callers
                - **Authorization**: Controls access to resources
                - **Versioning**: Allows API evolution without breaking clients
                
                **Integration Options:**
                - **Webhooks**: Push notifications for events
                - **SDK**: Simplifies client-side integration
                - **Direct API**: Maximum flexibility
                - **Message Queue**: Asynchronous integration
                
                **Best Practices:**
                - Design APIs for backward compatibility
                - Use proper status codes and error responses
                - Implement comprehensive authentication
                - Consider rate limiting for stability
                - Document APIs thoroughly
                """)
            
            # Example API request
            st.markdown("#### Example API Request & Response")
            
            if api_style == "REST":
                st.markdown("**Query Endpoint Request:**")
                st.code("""
# POST /api/v1/query
{
  "query": "What are the side effects of metformin?",
  "max_results": 3,
  "similarity_threshold": 0.75,
  "include_sources": true
}
                """, language="json")
                
                st.markdown("**Response:**")
                st.code("""
{
  "answer": "The common side effects of metformin include gastrointestinal issues such as nausea, diarrhea, and abdominal discomfort. These typically subside after a few weeks of treatment. Long-term use may lead to vitamin B12 deficiency in some patients. Rarely, metformin can cause lactic acidosis, a serious but uncommon side effect.",
  "sources": [
    {
      "document_id": "doc_735",
      "title": "Metformin Side Effects Overview",
      "relevance": 0.92,
      "snippet": "Common side effects include gastrointestinal distress, nausea, and diarrhea, which typically improve over time."
    },
    {
      "document_id": "doc_841",
      "title": "Long-term Metformin Safety Profile",
      "relevance": 0.87,
      "snippet": "Extended metformin use has been associated with vitamin B12 deficiency in approximately 10-30% of patients."
    }
  ],
  "confidence": 0.89,
  "processing_time_ms": 245
}
                """, language="json")
            
            elif api_style == "GraphQL":
                st.markdown("**GraphQL Query:**")
                st.code("""
query {
  answer(question: "What are the side effects of metformin?") {
    answer
    sources {
      documentId
      title
      relevance
      snippet
    }
    confidence
  }
}
                """, language="graphql")
            
            # SDK example
            if "SDK" in integration_options:
                st.markdown("#### Example SDK Usage (Python)")
                st.code("""
from rag_client import RAGClient

# Initialize client
client = RAGClient(
    api_key="your_api_key",
    base_url="https://api.yourservice.com/v1"
)

# Query the RAG system
response = client.query(
    question="What are the side effects of metformin?",
    max_results=3,
    include_sources=True
)

# Process response
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")

# Access sources
for source in response.sources:
    print(f"- {source.title} (relevance: {source.relevance})")
                """, language="python")
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if api_style == "REST":
                st.success("✅ REST is a good default choice with wide support")
                
            elif api_style == "GraphQL":
                st.success("✅ GraphQL provides flexibility for complex data requirements")
                st.info("ℹ️ Consider providing a REST fallback for simpler integrations")
                
            if enable_rate_limiting:
                st.success("✅ Rate limiting is essential for production APIs")
                
            if "API Key" in auth_methods and "OAuth 2.0" in auth_methods:
                st.success("✅ Supporting multiple auth methods improves integration options")
                
            if "Webhooks" in integration_options and "SDK" in integration_options:
                st.success("✅ Combining push and pull integration methods provides flexibility")
    
    # SECURITY & COMPLIANCE TAB
    with deploy_tabs[3]:
        st.subheader("Security & Compliance")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Data security
            st.markdown("### Data Security")
            
            encryption_options = st.multiselect(
                "Encryption Options",
                ["Data at Rest", "Data in Transit", "End-to-End Encryption", "Field-level Encryption", "Key Rotation"],
                default=["Data at Rest", "Data in Transit"]
            )
            
            if "Data at Rest" in encryption_options:
                rest_encryption = st.selectbox(
                    "Data at Rest Encryption",
                    ["AES-256", "AWS KMS", "Azure Key Vault", "Google Cloud KMS", "HashiCorp Vault", "Custom"]
                )
                
                st.checkbox("Enable Customer-Managed Keys (BYOK)", value=False)
            
            # Network security
            st.markdown("### Network Security")
            
            network_security = st.multiselect(
                "Network Security Measures",
                ["TLS 1.3+", "IP Allowlisting", "WAF (Web Application Firewall)", "DDoS Protection", "API Gateway", "Network Isolation"],
                default=["TLS 1.3+", "WAF (Web Application Firewall)", "DDoS Protection"]
            )
            
            if "WAF" in network_security:
                waf_rules = st.multiselect(
                    "WAF Rules",
                    ["SQL Injection Protection", "XSS Protection", "OWASP Top 10", "Rate Limiting", "Bot Protection", "Custom Rules"],
                    default=["SQL Injection Protection", "XSS Protection", "OWASP Top 10"]
                )
            
            # Access controls
            st.markdown("### Access Controls")
            
            access_controls = st.multiselect(
                "Access Control Measures",
                ["Principle of Least Privilege", "Role-Based Access", "MFA", "SSO Integration", "Session Management", "IP-based Restrictions"],
                default=["Principle of Least Privilege", "Role-Based Access", "MFA"]
            )
            
            if "MFA" in access_controls:
                mfa_methods = st.multiselect(
                    "MFA Methods",
                    ["TOTP (Time-based OTP)", "SMS Codes", "Email Codes", "Hardware Keys", "Biometrics"],
                    default=["TOTP (Time-based OTP)", "Hardware Keys"]
                )
            
            # Compliance frameworks
            st.markdown("### Compliance & Governance")
            
            compliance_frameworks = st.multiselect(
                "Compliance Frameworks",
                ["GDPR", "CCPA/CPRA", "HIPAA", "SOC 2", "ISO 27001", "FedRAMP", "NIST", "Custom Framework"],
                default=[]
            )
            
            if len(compliance_frameworks) > 0:
                compliance_features = st.multiselect(
                    "Compliance Features",
                    ["Data Residency Controls", "Data Retention Policies", "Audit Logging", "User Consent Management", 
                     "Right to be Forgotten", "Data Processing Agreements", "Privacy Impact Assessment"],
                    default=["Data Residency Controls", "Audit Logging"]
                )
            
            # Auditing and logging
            st.markdown("### Auditing & Logging")
            
            enable_audit = st.checkbox("Enable Comprehensive Audit Logging", value=True)
            
            if enable_audit:
                audit_events = st.multiselect(
                    "Audit Events",
                    ["User Authentication", "Data Access", "Admin Actions", "Configuration Changes", 
                     "Query Submissions", "Document Processing", "Security Events", "Custom Events"],
                    default=["User Authentication", "Data Access", "Admin Actions", "Security Events"]
                )
                
                audit_retention = st.slider("Log Retention Period (days)", 30, 1095, 90)
                
                log_destinations = st.multiselect(
                    "Log Destinations",
                    ["CloudWatch Logs", "Elastic Stack", "Splunk", "DataDog", "Custom SIEM", "S3/Blob Storage"],
                    default=["CloudWatch Logs"]
                )
        
        with col2:
            with st.expander("ℹ️ About Security & Compliance", expanded=True):
                info_tooltip("""
                **Security & Compliance** ensures that your RAG system protects data, meets regulatory requirements, and follows security best practices.
                
                **Data Security:**
                
                **Encryption:**
                - **Data at Rest**: Protects stored data (documents, vectors, metadata)
                - **Data in Transit**: Protects data moving between components
                - **End-to-End**: Protects data throughout its lifecycle
                - **Field-level**: Encrypts specific sensitive fields
                - **Key Rotation**: Regularly updates encryption keys
                
                **Network Security:**
                - **TLS**: Secures communications between clients and servers
                - **WAF**: Protects against common web vulnerabilities (OWASP Top 10)
                - **IP Allowlisting**: Restricts access to approved IP addresses
                - **DDoS Protection**: Guards against denial of service attacks
                
                **Access Controls:**
                - **Least Privilege**: Users have only necessary permissions
                - **Role-Based**: Permissions assigned to roles, not users
                - **MFA**: Multiple factors required for authentication
                - **SSO**: Single sign-on for unified authentication
                
                **Compliance Considerations:**
                - **GDPR**: European data protection regulation
                - **CCPA/CPRA**: California privacy regulations
                - **HIPAA**: Healthcare data protection (US)
                - **SOC 2**: Trust services criteria for service organizations
                - **ISO 27001**: Information security standard
                
                **Audit Trail Requirements:**
                - Record who did what, when, and from where
                - Immutable logs for security and compliance
                - Comprehensive coverage of security-relevant events
                - Sufficient retention of historical data
                
                **Best Practices:**
                - Apply defense in depth (multiple security layers)
                - Regularly update and patch components
                - Conduct security assessments and penetration testing
                - Implement monitoring and alerting for security events
                """)
            
            # Security considerations for RAG
            st.markdown("#### RAG-specific Security Considerations")
            
            rag_security_considerations = [
                "**Prompt injection attacks**: Sanitize inputs and validate query structure",
                "**Training data poisoning**: Verify document sources before indexing",
                "**Sensitive data exposure**: Implement PII detection in documents",
                "**Inference attacks**: Control granularity of generated responses",
                "**Model theft**: Rate limit and monitor unusual access patterns",
                "**Data leakage via LLM**: Implement proper constraints on generated content"
            ]
            
            for consideration in rag_security_considerations:
                st.markdown(f"- {consideration}")
            
            # Compliance checklist example
            if len(compliance_frameworks) > 0:
                st.markdown("#### Sample Compliance Checklist")
                
                if "GDPR" in compliance_frameworks:
                    gdpr_items = [
                        {"Requirement": "Data Processing Records", "Status": "✅ Implemented"},
                        {"Requirement": "Lawful Basis for Processing", "Status": "⚠️ In Progress"},
                        {"Requirement": "Data Subject Rights", "Status": "✅ Implemented"},
                        {"Requirement": "Data Protection Impact Assessment", "Status": "❌ Not Started"},
                        {"Requirement": "Data Minimization", "Status": "✅ Implemented"}
                    ]
                    
                    st.dataframe(pd.DataFrame(gdpr_items), hide_index=True)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if "Data at Rest" in encryption_options and "Data in Transit" in encryption_options:
                st.success("✅ Comprehensive encryption strategy protects data throughout lifecycle")
                
            if "WAF" in network_security and "DDoS Protection" in network_security:
                st.success("✅ Multi-layered network security protects against common attacks")
                
            if len(compliance_frameworks) == 0:
                st.warning("⚠️ Consider which compliance frameworks may apply to your use case")
                
            if enable_audit:
                st.success("✅ Audit logging is essential for security monitoring and compliance")
                
                if audit_retention < 90:
                    st.warning("⚠️ Consider longer log retention for compliance requirements")
                    
            if "MFA" in access_controls:
                st.success("✅ Multi-factor authentication significantly improves security posture")

###############################
# 11. RUN EXPERIMENT
###############################
elif main_section == "11. Run Experiment":
    st.header("11. Run Experiment")
    st.markdown("Configure and execute experiments to test and compare different RAG configurations")
    
    # Create tabs for experiment options
    experiment_tabs = st.tabs(["Experiment Setup", "Run Configuration", "Results Analysis"])
    
    # EXPERIMENT SETUP TAB
    with experiment_tabs[0]:
        st.subheader("Experiment Setup")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Experiment naming and description
            st.markdown("### Experiment Details")
            
            experiment_name = st.text_input("Experiment Name", "RAG Configuration Comparison")
            experiment_desc = st.text_area("Experiment Description", 
                                          "Comparing different embedding models and retrieval strategies to optimize response quality.")
            
            # Define experiment variations
            st.markdown("### Experiment Variations")
            
            experiment_type = st.selectbox(
                "Experiment Type",
                ["A/B Test", "Ablation Study", "Parameter Sweep", "Multi-factorial", "Custom"]
            )
            
            # Different experiment configuration based on type
            if experiment_type == "A/B Test":
                st.markdown("#### A/B Test Configuration")
                
                variant_a = st.text_input("Variant A Name", "Baseline")
                variant_b = st.text_input("Variant B Name", "Experimental")
                
                st.markdown("##### Baseline Configuration (A)")
                baseline_model = st.selectbox("Baseline Embedding Model", ["text-embedding-3-small", "all-MiniLM-L6-v2"])
                baseline_retrieval = st.selectbox("Baseline Retrieval Strategy", ["BM25", "Vector Search", "Hybrid"])
                baseline_rerank = st.checkbox("Baseline Uses Reranking", value=False)
                
                st.markdown("##### Experimental Configuration (B)")
                exp_model = st.selectbox("Experimental Embedding Model", ["text-embedding-3-large", "all-mpnet-base-v2"])
                exp_retrieval = st.selectbox("Experimental Retrieval Strategy", ["Vector Search", "Hybrid", "RAG-Fusion"])
                exp_rerank = st.checkbox("Experimental Uses Reranking", value=True)
            
            elif experiment_type == "Ablation Study":
                st.markdown("#### Ablation Study Configuration")
                
                base_config = st.text_input("Base Configuration Name", "Full System")
                
                ablation_components = st.multiselect(
                    "Components to Ablate",
                    ["Query Expansion", "Document Reranking", "Self-consistency", "Structured Output", "Chain of Thought"],
                    default=["Query Expansion", "Document Reranking"]
                )
                
                for component in ablation_components:
                    st.checkbox(f"Include {component} variation", value=True)
                    
                st.number_input("Number of variations", 1, 10, len(ablation_components) + 1)
            
            elif experiment_type == "Parameter Sweep":
                st.markdown("#### Parameter Sweep Configuration")
                
                param_to_sweep = st.selectbox(
                    "Parameter to Sweep",
                    ["Embedding Model", "Chunk Size", "Retrieval Top-K", "Temperature", "Context Window Size", "Custom"]
                )
                
                if param_to_sweep == "Chunk Size":
                    min_chunk = st.number_input("Minimum Chunk Size", 100, 1000, 250)
                    max_chunk = st.number_input("Maximum Chunk Size", 1000, 10000, 2000)
                    step_size = st.number_input("Step Size", 50, 1000, 250)
                    
                elif param_to_sweep == "Retrieval Top-K":
                    min_k = st.number_input("Minimum K", 1, 10, 3)
                    max_k = st.number_input("Maximum K", 10, 100, 20)
                    k_step = st.number_input("Step Size", 1, 10, 3)
                    
                elif param_to_sweep == "Embedding Model":
                    embeddings_to_test = st.multiselect(
                        "Embedding Models to Test",
                        ["text-embedding-3-small", "text-embedding-3-large", "all-MiniLM-L6-v2", "all-mpnet-base-v2", "gte-base", "gte-large"],
                        default=["text-embedding-3-small", "text-embedding-3-large", "all-MiniLM-L6-v2"]
                    )
            
            elif experiment_type == "Multi-factorial":
                st.markdown("#### Multi-factorial Configuration")
                
                factors = st.multiselect(
                    "Experimental Factors",
                    ["Embedding Model", "Retrieval Strategy", "Chunk Size", "Reranking", "LLM Model", "Prompt Template"],
                    default=["Embedding Model", "Retrieval Strategy", "Reranking"]
                )
                
                # Configuration for each factor
                if "Embedding Model" in factors:
                    embedding_factors = st.multiselect(
                        "Embedding Models to Test",
                        ["text-embedding-3-small", "text-embedding-3-large", "all-MiniLM-L6-v2", "all-mpnet-base-v2"],
                        default=["text-embedding-3-small", "all-MiniLM-L6-v2"]
                    )
                    
                if "Retrieval Strategy" in factors:
                    retrieval_factors = st.multiselect(
                        "Retrieval Strategies to Test",
                        ["BM25", "Vector Search", "Hybrid", "Fusion"],
                        default=["BM25", "Vector Search"]
                    )
                    
                if "Chunk Size" in factors:
                    chunk_factors = st.multiselect(
                        "Chunk Sizes to Test",
                        [250, 500, 1000, 2000],
                        default=[500, 1000]
                    )
                    
                if "Reranking" in factors:
                    reranking_factors = st.multiselect(
                        "Reranking Options",
                        ["None", "Cross-Encoder", "LLM Reranking"],
                        default=["None", "Cross-Encoder"]
                    )
                
                # Calculate total variations
                total_variations = 1
                if "Embedding Model" in factors:
                    total_variations *= len(embedding_factors)
                if "Retrieval Strategy" in factors:
                    total_variations *= len(retrieval_factors)
                if "Chunk Size" in factors:
                    total_variations *= len(chunk_factors)
                if "Reranking" in factors:
                    total_variations *= len(reranking_factors)
                
                st.info(f"This will create {total_variations} different configurations to test")
            
            # Test dataset configuration
            st.markdown("### Test Dataset")
            
            test_dataset = st.selectbox(
                "Select Test Dataset",
                ["Use Ground Truth from Section 9", "Upload New Dataset", "Generate Synthetic Dataset"]
            )
            
            if test_dataset == "Upload New Dataset":
                uploaded_test_data = st.file_uploader("Upload test dataset (CSV or JSON)", type=["csv", "json"])
                
                if uploaded_test_data:
                    st.success("Test dataset uploaded successfully")
            
            elif test_dataset == "Generate Synthetic Dataset":
                st.number_input("Number of test cases to generate", 10, 1000, 50)
                
                query_types = st.multiselect(
                    "Query Types to Generate",
                    ["Factual", "Comparative", "Analytical", "Procedural", "Definitional", "Complex"],
                    default=["Factual", "Comparative", "Analytical"]
                )
                
            # Evaluation metrics configuration
            st.markdown("### Evaluation Metrics")
            
            eval_metrics = st.multiselect(
                "Evaluation Metrics",
                ["Answer Relevance", "Factual Accuracy", "Retrieval Precision", "Retrieval Recall", "ROUGE", "Latency", "Token Usage", "Custom"],
                default=["Answer Relevance", "Factual Accuracy", "Retrieval Precision", "Latency"]
            )
            
            if "Custom" in eval_metrics:
                custom_metric_name = st.text_input("Custom Metric Name", "Domain Specificity")
                custom_metric_desc = st.text_area("Custom Metric Description", "Measures how specific the answer is to the domain (scale 0-10)")
        
        with col2:
            with st.expander("ℹ️ About Experiment Setup", expanded=True):
                info_tooltip("""
                **Experiment Setup** defines what configurations you want to test and compare in your RAG system.
                
                **Experiment Types:**
                
                **A/B Test**:
                - Compares two configurations (baseline vs. experimental)
                - Simple and focused comparison
                - Good for testing a specific change
                
                **Ablation Study**:
                - Tests impact of removing specific components
                - Helps understand what features are most important
                - Identifies unnecessary complexity
                
                **Parameter Sweep**:
                - Tests a range of values for a specific parameter
                - Helps find optimal settings
                - Examples: chunk size, retrieval top-k, etc.
                
                **Multi-factorial**:
                - Tests combinations of different variables
                - Identifies interaction effects
                - More comprehensive but computationally expensive
                
                **Test Datasets:**
                - Should be representative of real queries
                - Include diverse query types
                - Contain ground truth for evaluation
                
                **Evaluation Metrics:**
                - Should align with your system goals
                - Include both quality and performance metrics
                - Allow for fair comparison across configurations
                
                **Best Practices:**
                - Start with focused experiments (A/B tests)
                - Control for confounding variables
                - Use statistical measures of significance
                - Document configurations thoroughly
                - Keep track of all results for comparison
                """)
            
            # Example experimental setup visualization
            st.markdown("#### Experiment Visualization")
            
            if experiment_type == "A/B Test":
                st.markdown("**A/B Test Configuration**")
                
                ab_config = pd.DataFrame({
                    "Component": ["Embedding Model", "Retrieval Strategy", "Reranking", "Context Size"],
                    "Baseline (A)": [baseline_model, baseline_retrieval, "No" if not baseline_rerank else "Yes", "4000 tokens"],
                    "Experimental (B)": [exp_model, exp_retrieval, "No" if not exp_rerank else "Yes", "4000 tokens"]
                })
                
                st.dataframe(ab_config, hide_index=True)
                
            elif experiment_type == "Parameter Sweep" and param_to_sweep == "Retrieval Top-K":
                st.markdown("**Parameter Sweep: Retrieval Top-K**")
                
                # Create sample data for a chart
                k_values = list(range(min_k, max_k + 1, k_step))
                
                # Sample data - this would be replaced by actual experiment results
                sample_accuracy = [0.72, 0.78, 0.82, 0.84, 0.85, 0.85, 0.84]
                sample_latency = [105, 120, 135, 152, 168, 185, 200]
                
                # Adjust based on actual parameter range
                k_values = k_values[:len(sample_accuracy)]
                
                df = pd.DataFrame({
                    "Top-K": k_values,
                    "Accuracy": sample_accuracy,
                    "Latency (ms)": sample_latency
                })
                
                # Create a dual-axis chart (mock)
                accuracy_chart = alt.Chart(df).mark_line(color='blue').encode(
                    x=alt.X('Top-K:Q'),
                    y=alt.Y('Accuracy:Q', scale=alt.Scale(domain=[0, 1])),
                    tooltip=['Top-K', 'Accuracy']
                ).properties(title="Effect of Top-K on Accuracy and Latency")
                
                latency_chart = alt.Chart(df).mark_line(color='red').encode(
                    x=alt.X('Top-K:Q'),
                    y=alt.Y('Latency (ms):Q', scale=alt.Scale(domain=[0, 250])),
                    tooltip=['Top-K', 'Latency (ms)']
                )
                
                st.altair_chart(accuracy_chart + latency_chart, use_container_width=True)
            
            elif experiment_type == "Multi-factorial":
                st.markdown("**Multi-factorial Design**")
                
                # Create a representation of the factorial design
                if "Embedding Model" in factors and "Retrieval Strategy" in factors:
                    # Create a sample grid visualization of combinations
                    grid_data = []
                    
                    for emb in embedding_factors:
                        for ret in retrieval_factors:
                            grid_data.append({
                                "Embedding": emb,
                                "Retrieval": ret,
                                "Configuration": f"{emb} + {ret}"
                            })
                    
                    grid_df = pd.DataFrame(grid_data)
                    st.dataframe(grid_df, hide_index=True)
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if experiment_type == "A/B Test":
                st.success("✅ A/B testing is a good approach for focused comparison")
                st.info("ℹ️ Consider running multiple test queries for statistical significance")
                
            elif experiment_type == "Parameter Sweep":
                st.success("✅ Parameter sweeping helps identify optimal settings")
                st.info("ℹ️ Looking for the 'elbow' in performance charts can identify best values")
                
            elif experiment_type == "Multi-factorial" and total_variations > 20 if 'total_variations' in locals() else False:
                st.warning("⚠️ Large factorial designs can be computationally expensive")
                st.info("ℹ️ Consider prioritizing the most important factors first")
                
            if len(eval_metrics) >= 3:
                st.success("✅ Using multiple metrics provides a more complete evaluation")
    
    # RUN CONFIGURATION TAB
    with experiment_tabs[1]:
        st.subheader("Run Configuration")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Execution settings
            st.markdown("### Execution Settings")
            
            parallel_execution = st.checkbox("Run configurations in parallel", value=True)
            
            if parallel_execution:
                max_workers = st.slider("Maximum parallel workers", 1, 10, 3)
                
            execution_mode = st.selectbox(
                "Execution Mode",
                ["Run All Configurations", "Sequential Elimination", "Early Stopping", "Custom"]
            )
            
            if execution_mode == "Sequential Elimination":
                elimination_metric = st.selectbox(
                    "Elimination Metric",
                    ["Answer Relevance", "Factual Accuracy", "Retrieval Precision", "Latency"]
                )
                elimination_threshold = st.slider("Elimination Threshold", 0.0, 1.0, 0.6)
                
            elif execution_mode == "Early Stopping":
                early_stop_condition = st.text_area(
                    "Early Stopping Condition",
                    "if any configuration achieves > 0.9 on Answer Relevance AND > 0.95 on Factual Accuracy"
                )
            
            # Resource allocation
            st.markdown("### Resource Allocation")
            
            resource_strategy = st.selectbox(
                "Resource Allocation Strategy",
                ["Equal Distribution", "Priority-based", "Dynamic Allocation"]
            )
            
            if resource_strategy == "Priority-based":
                st.text_area(
                    "Configuration Priorities (1-5)",
                    """Baseline: 3
Experimental: 5
Ablation_QueryExpansion: 4
Ablation_Reranking: 4"""
                )
            
            # Run settings
            st.markdown("### Run Settings")
            
            timeout_per_query = st.slider("Timeout per query (seconds)", 5, 300, 30)
            max_runtime = st.slider("Maximum total runtime (minutes)", 5, 240, 60)
            
            allow_caching = st.checkbox("Cache intermediate results", value=True)
            
            # Logging configuration
            st.markdown("### Logging Configuration")
            
            log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
            
            log_components = st.multiselect(
                "Components to Log",
                ["Retrieval Process", "Context Selection", "LLM Generation", "Evaluation", "System Metrics"],
                default=["Retrieval Process", "Evaluation", "System Metrics"]
            )
            
            save_results = st.checkbox("Save results to disk", value=True)
            
            if save_results:
                result_format = st.selectbox("Result Format", ["CSV", "JSON", "SQLite", "Parquet", "Excel"])
                
            # Execution controls
            st.markdown("### Execution Controls")
            
            st.warning("**Note:** Running experiments may consume significant API credits depending on the configuration.")
            
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                run_btn = st.button("Run Experiment", type="primary")
            with col_btn2:
                cancel_btn = st.button("Cancel Execution")
            
            if run_btn:
                st.info("Experiment queued for execution. Check the Results tab for progress updates.")
        
        with col2:
            with st.expander("ℹ️ About Run Configuration", expanded=True):
                info_tooltip("""
                **Run Configuration** defines how your experiment will be executed and managed.
                
                **Execution Settings:**
                - **Parallel Execution**: Run multiple configurations simultaneously
                - **Sequential Elimination**: Remove poor performers early
                - **Early Stopping**: Stop when success criteria are met
                
                **Resource Allocation:**
                - Determines how computing resources are distributed
                - Important for balancing speed vs. cost
                - Priority-based ensures key configurations get tested first
                
                **Run Settings:**
                - **Timeout**: Prevents hanging on problematic configurations
                - **Caching**: Reuse results when possible for efficiency
                - **Max Runtime**: Total time limit for the experiment
                
                **Logging Configuration:**
                - Controls what information is recorded
                - Useful for debugging and analysis
                - Higher detail levels produce more verbose logs
                
                **Best Practices:**
                - Start with small-scale tests before large experiments
                - Enable caching to avoid redundant computation
                - Set reasonable timeouts to handle edge cases
                - Use parallel execution where possible
                - Save detailed logs during development
                """)
            
            # Execution progress visualization
            st.markdown("#### Execution Flow")
            
            execution_flow = """
1. Initialization
   - Load test dataset
   - Prepare configurations
   - Initialize metrics collection

2. For each configuration:
   - Set up configuration parameters
   - For each test query:
     - Run retrieval process
     - Generate response
     - Calculate evaluation metrics
   - Compute aggregate metrics

3. Results aggregation
   - Compile metrics across configurations
   - Generate comparison visualizations
   - Export detailed results
            """
            
            st.code(execution_flow)
            
            # Mock progress visualization
            st.markdown("#### Execution Status")
            
            if run_btn:
                st.metric("Progress", "0/5 configurations")
                
                progress_bar = st.progress(0)
                
                st.markdown("**Current Configuration:** Setting up...")
                
                st.code("""
[INFO] Loading test dataset with 50 queries
[INFO] Initializing baseline configuration
[INFO] Setting up retrieval pipeline...
                """)
            else:
                st.info("Experiment not started. Click 'Run Experiment' to begin execution.")
            
            # Recommendations
            st.markdown("#### Recommendations")
            
            if parallel_execution:
                st.success("✅ Parallel execution will speed up the experiment")
                
            if allow_caching:
                st.success("✅ Caching intermediate results improves efficiency")
                
            if log_level == "DEBUG":
                st.info("ℹ️ Debug logging provides comprehensive information but may impact performance")
                
            if save_results and result_format in ["JSON", "Parquet"]:
                st.success("✅ Using structured formats facilitates deeper analysis")
    
    # RESULTS ANALYSIS TAB
    with experiment_tabs[2]:
        st.subheader("Results Analysis")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Load or generate sample results
            st.markdown("### Experiment Results")
            
            result_source = st.selectbox(
                "Result Source",
                ["Load from Previous Run", "Generate Sample Results", "No Results Available"]
            )
            
            if result_source == "Generate Sample Results":
                # Generate sample results for visualization
                variant_names = ["Baseline", "Experimental"]
                
                # Sample metrics
                accuracy_data = [0.78, 0.84]
                relevance_data = [0.72, 0.81]
                latency_data = [220, 280]
                precision_data = [0.68, 0.76]
                
                # Create dataframe
                results_df = pd.DataFrame({
                    "Variant": variant_names,
                    "Factual Accuracy": accuracy_data,
                    "Answer Relevance": relevance_data,
                    "Retrieval Precision": precision_data,
                    "Latency (ms)": latency_data
                })
                
                st.markdown("#### Overall Results")
                st.dataframe(results_df, hide_index=True)
                
                # Comparison charts
                st.markdown("#### Results Comparison")
                if metric_to_visualize == "All Metrics":
                    metric_to_visualize = st.selectbox(
                        "Select Metric to Visualize",
                        ["All Metrics", "Factual Accuracy", "Answer Relevance", "Retrieval Precision", "Latency"]
                    )
                
                if metric_to_visualize == "All Metrics":
                    # Reshape for grouped bar chart
                    chart_data = pd.melt(
                        results_df, 
                        id_vars=["Variant"],
                        value_vars=["Factual Accuracy", "Answer Relevance", "Retrieval Precision"],
                        var_name="Metric", 
                        value_name="Score"
                    )
                    
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x=alt.X("Variant:N"),
                        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 1])),
                        color="Metric:N",
                        column="Metric:N",
                        tooltip=["Variant", "Metric", "Score"]
                    ).properties(width=150)
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    # Latency comparison
                    latency_chart = alt.Chart(results_df).mark_bar().encode(
                        x="Variant:N",
                        y="Latency (ms):Q",
                        color="Variant:N",
                        tooltip=["Variant", "Latency (ms)"]
                    ).properties(
                        title="Latency Comparison",
                        width=300
                    )
                    
                    st.altair_chart(latency_chart, use_container_width=True)
                else:
                    # Single metric visualization
                    if metric_to_visualize != "Latency":
                        single_chart = alt.Chart(results_df).mark_bar().encode(
                            x="Variant:N",
                            y=alt.Y(f"{metric_to_visualize}:Q", scale=alt.Scale(domain=[0, 1])),
                            color="Variant:N",
                            tooltip=["Variant", metric_to_visualize]
                        ).properties(
                            title=f"{metric_to_visualize} by Variant",
                            width=500
                        )
                    else:
                        single_chart = alt.Chart(results_df).mark_bar().encode(
                            x="Variant:N",
                            y="Latency (ms):Q",
                            color="Variant:N",
                            tooltip=["Variant", "Latency (ms)"]
                        ).properties(
                            title="Latency by Variant",
                            width=500
                        )
                    
                    st.altair_chart(single_chart, use_container_width=True)
                
                # Statistical significance
                st.markdown("#### Statistical Analysis")
                
                significance_table = pd.DataFrame({
                    "Metric": ["Factual Accuracy", "Answer Relevance", "Retrieval Precision", "Latency"],
                    "Improvement": ["+7.7%", "+12.5%", "+11.8%", "-27.3%"],
                    "p-value": ["0.032", "0.018", "0.045", "0.027"],
                    "Significant": ["Yes", "Yes", "Yes", "Yes"]
                })
                
                st.dataframe(significance_table, hide_index=True)
                
                # Per-query analysis
                st.markdown("#### Per-Query Analysis")
                
                # Create some sample per-query data
                query_samples = [
                    "What are side effects of metformin?",
                    "How do SGLT2 inhibitors work?",
                    "Compare ACE inhibitors and ARBs",
                    "What is the recommended dosage of aspirin for prevention?",
                    "List contraindications for statins"
                ]
                
                baseline_scores = [0.92, 0.76, 0.65, 0.82, 0.73]
                experimental_scores = [0.94, 0.82, 0.85, 0.79, 0.80]
                
                query_results = pd.DataFrame({
                    "Query": query_samples,
                    "Baseline Score": baseline_scores,
                    "Experimental Score": experimental_scores,
                    "Difference": [exp - base for base, exp in zip(baseline_scores, experimental_scores)]
                })
                
                st.dataframe(query_results, hide_index=True)
                
                # Filter to show best/worst queries
                st.markdown("#### Significant Differences")
                
                diff_filter = st.selectbox(
                    "Show Queries With",
                    ["Largest Improvements", "Largest Regressions", "All Queries"]
                )
                
                # Sample query comparison
                st.markdown("#### Example Response Comparison")
                
                selected_query = st.selectbox(
                    "Select Query to Compare",
                    query_samples
                )
                
                col_q1, col_q2 = st.columns(2)
                
                with col_q1:
                    st.markdown("**Baseline Response:**")
                    st.markdown("""
                    SGLT2 inhibitors work by inhibiting sodium-glucose transport proteins in the kidneys, which prevents glucose reabsorption in the kidneys. This leads to increased glucose excretion in urine and lowers blood glucose levels.
                    """)
                
                with col_q2:
                    st.markdown("**Experimental Response:**")
                    st.markdown("""
                    SGLT2 inhibitors (sodium-glucose co-transporter-2 inhibitors) work by targeting the kidneys to reduce blood glucose levels. They specifically:
                    
                    1. Block SGLT2 proteins in the proximal tubule of the kidneys
                    2. Prevent reabsorption of filtered glucose
                    3. Increase glucose excretion in the urine (glucosuria)
                    4. Lower blood glucose levels independent of insulin
                    
                    This mechanism differs from other diabetes medications that target insulin production or sensitivity.
                    """)
            
            # Export results
            st.markdown("### Export Results")
            
            if result_source != "No Results Available":
                export_format = st.selectbox(
                    "Export Format",
                    ["CSV", "Excel", "JSON", "PDF Report"]
                )
                
                col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 1])
                
                with col_exp1:
                    st.button("Export Raw Data")
                
                with col_exp2:
                    st.button("Export Charts")
                
                with col_exp3:
                    st.button("Generate Report")
        
        with col2:
            with st.expander("ℹ️ About Results Analysis", expanded=True):
                info_tooltip("""
                **Results Analysis** helps you understand and compare the performance of different RAG configurations.
                
                **Key Analysis Components:**
                
                **Overall Metrics:**
                - Summary statistics across all test queries
                - Provides a high-level view of each configuration
                - Useful for quick comparisons
                
                **Comparative Visualization:**
                - Visual representation of results
                - Makes patterns easier to identify
                - Highlights relative strengths and weaknesses
                
                **Statistical Analysis:**
                - Determines if differences are statistically significant
                - Accounts for variance and sample size
                - Helps distinguish real improvements from noise
                
                **Per-Query Analysis:**
                - Examines performance on individual queries
                - Identifies strengths and weaknesses of each configuration
                - Helps understand where and why improvements occur
                
                **Response Comparison:**
                - Direct comparison of outputs between configurations
                - Qualitative assessment of differences
                - Reveals practical impact of configuration changes
                
                **Best Practices:**
                - Consider both statistical and practical significance
                - Look beyond averages to understand distribution
                - Analyze patterns in errors or improvements
                - Relate metrics to user experience goals
                - Use findings to iterate on your RAG system design
                """)
            
            # Decision support
            if result_source == "Generate Sample Results":
                st.markdown("#### Insights & Recommendations")
                
                insights = """
### Key Insights

1. **Experimental configuration outperforms baseline** across all quality metrics:
   - 7.7% improvement in Factual Accuracy
   - 12.5% improvement in Answer Relevance
   - 11.8% improvement in Retrieval Precision

2. **Response quality improvements are statistically significant** (p < 0.05)

3. **Latency increased by 27.3%** in the experimental configuration

4. **Largest improvements** observed for:
   - Complex comparative questions (+30.8%)
   - Queries requiring detailed explanations (+15.4%)

5. **Experimental configuration produces**:
   - More structured responses
   - More comprehensive answers
   - Better citations of source material
                """
                
                st.markdown(insights)
                
                recommendations = """
### Recommendations

✅ **Adopt the experimental configuration** for production use
   - Quality improvements outweigh the latency increase

⚠️ **Monitor and optimize latency**:
   - Consider caching common queries
   - Implement background processing for non-urgent queries

🔍 **Further investigate**:
   - Factual accuracy variations across different domains
   - Opportunities to optimize the reranking step

📈 **Next experiments to consider**:
   - Test additional embedding models
   - Evaluate impact of chunk size variations
   - Compare different reranking approaches
                """
                
                st.markdown(recommendations)
            
            # Recommendations
            st.markdown("#### Analysis Recommendations")
            
            if result_source != "No Results Available":
                st.success("✅ Compare metrics that align with your system's key objectives")
                st.success("✅ Look for patterns in where each configuration performs best")
                st.info("ℹ️ Balance metrics appropriately (e.g., quality vs. latency)")
                
                if metric_to_visualize == "All Metrics":
                    st.success("✅ Examining all metrics provides a comprehensive view")
            else:
                st.info("ℹ️ Run an experiment or generate sample results to see analysis")

###############################
# 12. RESULTS & HISTORY
###############################
elif main_section == "12. Results & History":
    st.header("12. Results & History")
    st.markdown("View previous experiment results, track configuration changes, and manage system history")
    
    # Create tabs for history options
    history_tabs = st.tabs(["Experiment Results", "Configuration History", "Metrics Dashboard", "Knowledge Base"])
    
    # EXPERIMENT RESULTS TAB
    with history_tabs[0]:
        st.subheader("Experiment Results Library")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Filters and search
            st.markdown("### Browse Experiments")
            
            search_query = st.text_input("Search experiments", placeholder="Search by name, description, or parameters...")
            
            col_filter1, col_filter2, col_filter3 = st.columns(3)
            with col_filter1:
                time_filter = st.selectbox(
                    "Time Period",
                    ["All Time", "Past Week", "Past Month", "Past Quarter", "Past Year"]
                )
            with col_filter2:
                type_filter = st.selectbox(
                    "Experiment Type",
                    ["All Types", "A/B Test", "Ablation Study", "Parameter Sweep", "Multi-factorial"]
                )
            with col_filter3:
                status_filter = st.selectbox(
                    "Status",
                    ["All", "Completed", "In Progress", "Failed", "Stopped"]
                )
            
            # Sample experiment list
            st.markdown("### Recent Experiments")
            
            # Mock data for experiments
            experiments = [
                {"id": "exp-001", "name": "Embedding Model Comparison", "type": "A/B Test", "date": "2025-05-25", "status": "Completed", 
                 "description": "Comparing text-embedding-3-large vs all-mpnet-base-v2"},
                {"id": "exp-002", "name": "Chunk Size Optimization", "type": "Parameter Sweep", "date": "2025-05-20", "status": "Completed",
                 "description": "Testing chunk sizes from 250 to 2000 characters"},
                {"id": "exp-003", "name": "Retriever Component Ablation", "type": "Ablation Study", "date": "2025-05-15", "status": "Completed",
                 "description": "Testing impact of removing query expansion and reranking"},
                {"id": "exp-004", "name": "RAG-fusion Implementation", "type": "A/B Test", "date": "2025-05-10", "status": "Completed",
                 "description": "Testing standard retrieval vs. RAG-fusion approach"},
                {"id": "exp-005", "name": "Multi-factor Optimization", "type": "Multi-factorial", "date": "2025-05-05", "status": "In Progress",
                 "description": "Testing combinations of embedding models, chunking strategies, and rerankers"}
            ]
            
            exp_df = pd.DataFrame(experiments)
            
            # Add view buttons to dataframe
            exp_df["action"] = "View"
            
            # Show the dataframe with a callback when view is clicked
            selected_exp = None
            for i, exp in enumerate(experiments):
                col_view1, col_view2, col_view3 = st.columns([3, 2, 1])
                with col_view1:
                    st.markdown(f"**{exp['name']}** ({exp['type']})")
                    st.caption(exp['description'])
                with col_view2:
                    st.markdown(f"**Date:** {exp['date']}")
                    st.markdown(f"**Status:** {exp['status']}")
                with col_view3:
                    if st.button("View Details", key=f"view_{exp['id']}"):
                        selected_exp = exp
                st.divider()
            
            # Experiment management
            st.markdown("### Experiment Management")
            col_mgmt1, col_mgmt2, col_mgmt3 = st.columns(3)
            with col_mgmt1:
                st.button("Export All Results", key="export_all")
            with col_mgmt2:
                st.button("Archive Selected", key="archive")
            with col_mgmt3:
                st.button("Delete Selected", key="delete")
        
        with col2:
            # Experiment details panel
            st.markdown("### Experiment Details")
            
            if selected_exp:
                st.markdown(f"## {selected_exp['name']}")
                st.markdown(f"**Type:** {selected_exp['type']}")
                st.markdown(f"**Run Date:** {selected_exp['date']}")
                st.markdown(f"**Status:** {selected_exp['status']}")
                st.markdown(f"**Description:** {selected_exp['description']}")
                
                # Show experiment results
                st.markdown("### Results Summary")
                
                if selected_exp['id'] == "exp-001":  # Embedding model comparison
                    results_df = pd.DataFrame({
                        "Metric": ["Relevance", "Accuracy", "Latency (ms)"],
                        "text-embedding-3-large": [0.84, 0.79, 250],
                        "all-mpnet-base-v2": [0.76, 0.72, 180]
                    })
                    st.dataframe(results_df, hide_index=True)
                    
                    improvement = pd.DataFrame({
                        "Metric": ["Relevance", "Accuracy", "Latency"],
                        "Improvement": ["+10.5%", "+9.7%", "-28%"],
                        "Significant": ["Yes", "Yes", "Yes"]
                    })
                    st.dataframe(improvement, hide_index=True)
                    
                    # Sample chart
                    chart_data = pd.DataFrame({
                        "Model": ["text-embedding-3-large", "text-embedding-3-large", "all-mpnet-base-v2", "all-mpnet-base-v2"],
                        "Metric": ["Relevance", "Accuracy", "Relevance", "Accuracy"],
                        "Score": [0.84, 0.79, 0.76, 0.72]
                    })
                    
                    chart = alt.Chart(chart_data).mark_bar().encode(
                        x='Model:N',
                        y='Score:Q',
                        color='Metric:N',
                        column='Metric:N'
                    ).properties(width=150)
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                # Actions for the selected experiment
                col_action1, col_action2, col_action3 = st.columns(3)
                with col_action1:
                    st.button("View Full Report", key="full_report")
                with col_action2:
                    st.button("Export Results", key="export_one")
                with col_action3:
                    st.button("Recreate Experiment", key="recreate")
                
                # Conclusions
                if selected_exp['id'] == "exp-001":
                    st.markdown("### Key Findings")
                    st.markdown("""
                    - text-embedding-3-large outperforms all-mpnet-base-v2 on all quality metrics
                    - Statistically significant improvements in both relevance (+10.5%) and accuracy (+9.7%)
                    - Trade-off: 28% increase in latency with text-embedding-3-large
                    - Recommendation: Use text-embedding-3-large for production
                    """)
            else:
                st.info("Select an experiment to view its details")
                
                # Show overall performance trends
                st.markdown("### System Performance Trends")
                
                # Sample data for a performance chart
                dates = pd.date_range(start='2025-01-01', end='2025-05-01', freq='MS')
                perf_data = pd.DataFrame({
                    'date': dates,
                    'Retrieval Precision': [0.65, 0.68, 0.72, 0.76, 0.82],
                    'Answer Relevance': [0.70, 0.73, 0.75, 0.78, 0.84]
                })
                
                # Create a line chart showing system improvement over time
                perf_chart = alt.Chart(perf_data.melt(
                    id_vars=['date'], 
                    value_vars=['Retrieval Precision', 'Answer Relevance'],
                    var_name='Metric',
                    value_name='Score'
                )).mark_line(point=True).encode(
                    x='date:T',
                    y=alt.Y('Score:Q', scale=alt.Scale(domain=[0.6, 0.9])),
                    color='Metric:N',
                    tooltip=['date', 'Metric', 'Score']
                ).properties(height=250)
                
                st.altair_chart(perf_chart, use_container_width=True)
    
    # CONFIGURATION HISTORY TAB
    with history_tabs[1]:
        st.subheader("Configuration History")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Configuration versions list
            st.markdown("### Configuration Versions")
            
            # Search and filter
        config_search = st.text_input("Search configurations", placeholder="Search by name, author, or component...")
        
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            config_time = st.selectbox(
                "Time Period",
                ["All Time", "Past Week", "Past Month", "Past Quarter", "Past Year"],
                key="config_history_time_period"  # Added unique key here
            )
        with col_cfg2:
            config_component = st.selectbox(
                "Component",
                ["All", "Embedding", "Chunking", "Retrieval", "LLM", "Full RAG Pipeline"],
                key="config_history_component"  # Added unique key here
            )
            
            # Sample configuration history
            configs = [
                {"id": "cfg-001", "name": "Initial Production Config", "date": "2025-01-15", "author": "John Smith", 
                 "component": "Full RAG Pipeline", "description": "First production configuration"},
                {"id": "cfg-002", "name": "Updated Embedding Model", "date": "2025-02-10", "author": "Jane Doe", 
                 "component": "Embedding", "description": "Switched to text-embedding-3-large"},
                {"id": "cfg-003", "name": "Optimized Chunking Strategy", "date": "2025-03-05", "author": "John Smith", 
                 "component": "Chunking", "description": "Implemented semantic chunking with 1000 char size"},
                {"id": "cfg-004", "name": "Added Cross-Encoder Reranking", "date": "2025-04-12", "author": "Sarah Johnson", 
                 "component": "Retrieval", "description": "Added BAAI/bge-reranker-large for precision improvement"},
                {"id": "cfg-005", "name": "Current Production Config", "date": "2025-05-01", "author": "Team Release", 
                 "component": "Full RAG Pipeline", "description": "Optimized full pipeline with latest improvements"}
            ]
            
            # Show configurations with view/compare buttons
            selected_config = None
            for i, cfg in enumerate(configs):
                col_cfg1, col_cfg2, col_cfg3 = st.columns([3, 2, 1])
                with col_cfg1:
                    st.markdown(f"**{cfg['name']}**")
                    st.caption(f"{cfg['component']} | {cfg['description']}")
                with col_cfg2:
                    st.markdown(f"**Date:** {cfg['date']}")
                    st.markdown(f"**Author:** {cfg['author']}")
                with col_cfg3:
                    if st.button("View", key=f"view_cfg_{cfg['id']}"):
                        selected_config = cfg
                st.divider()
            
            # Configuration comparison
            st.markdown("### Compare Configurations")
        
            col_comp1, col_comp2, col_comp3 = st.columns(3)
            with col_comp1:
                base_config = st.selectbox(
                    "Base Configuration",
                    [f"{c['name']} ({c['date']})" for c in configs],
                    key="compare_base_config"  # Added unique key here
                )
            with col_comp2:
                target_config = st.selectbox(
                    "Target Configuration",
                    [f"{c['name']} ({c['date']})" for c in configs],
                    key="compare_target_config"  # Added unique key here
                )
            with col_comp3:
                st.button("Compare Configurations", key="compare_configs")
        
        with col2:
            # Configuration details or comparison
            if selected_config:
                st.markdown(f"### Configuration: {selected_config['name']}")
                st.markdown(f"**Date:** {selected_config['date']}")
                st.markdown(f"**Author:** {selected_config['author']}")
                st.markdown(f"**Component:** {selected_config['component']}")
                st.markdown(f"**Description:** {selected_config['description']}")
                
                # Show configuration details - will depend on the selected config
                st.markdown("### Configuration Details")
                
                if selected_config['id'] == "cfg-005":  # Current production config
                    st.code("""
{
  "embedding": {
    "model": "text-embedding-3-large",
    "dimensions": 1536,
    "normalize": true
  },
  "chunking": {
    "strategy": "semantic",
    "size": 1000,
    "overlap": 100
  },
  "retrieval": {
    "strategy": "hybrid",
    "vector_weight": 0.7,
    "keyword_weight": 0.3,
    "top_k": 10,
    "reranker": {
      "model": "BAAI/bge-reranker-large",
      "threshold": 0.6
    }
  },
  "llm": {
    "model": "gpt-4-turbo",
    "temperature": 0.7,
    "max_tokens": 1000,
    "system_prompt": "You are a helpful assistant..."
  }
}
                    """, language="json")
                    
                # Actions
                col_cfg_action1, col_cfg_action2, col_cfg_action3 = st.columns(3)
                with col_cfg_action1:
                    st.button("Restore This Config", key="restore_config")
                with col_cfg_action2:
                    st.button("Export Config", key="export_config")
                with col_cfg_action3:
                    st.button("View Performance", key="view_performance")
            else:
                st.info("Select a configuration to view its details")
                
                # Configuration change visualization
                st.markdown("### Configuration Evolution")
                
                # Sample data for tracking component changes
                component_changes = pd.DataFrame({
                    'date': pd.date_range(start='2025-01-01', end='2025-05-01', freq='MS'),
                    'component': ['Initial Setup', 'Embedding', 'Chunking', 'Retrieval', 'Full Pipeline'],
                    'change': ['Initial deployment', 'Model upgrade', 'Strategy change', 'Added reranking', 'Optimization']
                })
                
                # Create a timeline chart
                timeline = alt.Chart(component_changes).mark_circle(size=100).encode(
                    x='date:T',
                    y='component:N',
                    tooltip=['date', 'component', 'change'],
                    color='component:N'
                ).properties(height=200)
                
                timeline_line = alt.Chart(component_changes).mark_line().encode(
                    x='date:T',
                    y='component:N',
                    color='component:N'
                )
                
                st.altair_chart(timeline + timeline_line, use_container_width=True)
                
                # Show active configuration
                st.markdown("### Current Active Configuration")
                st.info("**Current Production:** Current Production Config (2025-05-01)")
                
                col_curr1, col_curr2 = st.columns(2)
                with col_curr1:
                    st.button("View Current Config", key="view_current")
                with col_curr2:
                    st.button("Create New Config", key="create_config")
    
    # METRICS DASHBOARD TAB
    with history_tabs[2]:
        st.subheader("System Metrics Dashboard")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Dashboard filters
            st.markdown("### Metrics Configuration")
            
            time_range = st.selectbox(
                "Time Range",
                ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last Quarter", "Last Year", "Custom Range"]
            )
            
            if time_range == "Custom Range":
                col_date1, col_date2 = st.columns(2)
                with col_date1:
                    start_date = st.date_input("Start Date", datetime.now() - pd.Timedelta(days=30))
                with col_date2:
                    end_date = st.date_input("End Date", datetime.now())
            
            # Metrics selection
            st.markdown("### Select Metrics")
            
            performance_metrics = st.multiselect(
                "Performance Metrics",
                ["Latency (ms)", "Throughput (QPS)", "Error Rate (%)", "Token Usage"],
                default=["Latency (ms)", "Error Rate (%)"]
            )
            
            quality_metrics = st.multiselect(
                "Quality Metrics",
                ["Relevance Score", "User Satisfaction", "Retrieval Precision", "Answer Accuracy"],
                default=["Relevance Score", "User Satisfaction"]
            )
            
            system_metrics = st.multiselect(
                "System Metrics",
                ["CPU Usage", "Memory Usage", "GPU Utilization", "Cache Hit Rate", "DB Query Time"],
                default=["CPU Usage", "Memory Usage"]
            )
            
            # Visualization options
            st.markdown("### Visualization Options")
            
            chart_type = st.selectbox(
                "Chart Type",
                ["Line Chart", "Bar Chart", "Area Chart", "Scatter Plot", "Heatmap"]
            )
            
            aggregation = st.selectbox(
                "Aggregation",
                ["Hourly", "Daily", "Weekly", "Monthly"]
            )
            
            # Generate dashboard
            st.button("Update Dashboard", type="primary", key="update_dashboard")
        
        with col2:
            # Main dashboard area
            st.markdown("### System Performance Dashboard")
            
            # Create tabs for different metric categories
            metric_tabs = st.tabs(["Performance", "Quality", "System Resources"])
            
            # Performance metrics tab
            with metric_tabs[0]:
                st.markdown("#### Performance Metrics")
                
                # Sample performance data
                dates = pd.date_range(start='2025-05-01', periods=30, freq='D')
                latency = np.random.normal(250, 25, 30) 
                error_rate = np.random.normal(2, 0.5, 30)
                
                perf_df = pd.DataFrame({
                    'date': dates,
                    'Latency (ms)': latency,
                    'Error Rate (%)': error_rate
                })
                
                # Create line charts
                latency_chart = alt.Chart(perf_df).mark_line(point=True, color='blue').encode(
                    x='date:T',
                    y=alt.Y('Latency (ms):Q', scale=alt.Scale(zero=False)),
                    tooltip=['date', 'Latency (ms)']
                ).properties(height=200, title="Average Response Latency")
                
                error_chart = alt.Chart(perf_df).mark_line(point=True, color='red').encode(
                    x='date:T',
                    y=alt.Y('Error Rate (%):Q', scale=alt.Scale(zero=False)),
                    tooltip=['date', 'Error Rate (%)']
                ).properties(height=200, title="System Error Rate")
                
                st.altair_chart(latency_chart, use_container_width=True)
                st.altair_chart(error_chart, use_container_width=True)
                
                # Key metrics
                col_pm1, col_pm2, col_pm3 = st.columns(3)
                with col_pm1:
                    st.metric("Avg Latency", f"{latency.mean():.1f} ms", f"{-5.2:.1f}%")
                with col_pm2:
                    st.metric("Error Rate", f"{error_rate.mean():.2f}%", f"{-0.5:.1f}%") 
                with col_pm3:
                    st.metric("99th Percentile", f"{np.percentile(latency, 99):.1f} ms", f"{2.3:.1f}%")
            
            # Quality metrics tab
            with metric_tabs[1]:
                st.markdown("#### Quality Metrics")
                
                # Sample quality data
                dates = pd.date_range(start='2025-05-01', periods=30, freq='D')
                relevance = np.clip(np.random.normal(0.8, 0.05, 30), 0, 1)
                satisfaction = np.clip(np.random.normal(4.2, 0.3, 30), 1, 5)
                
                quality_df = pd.DataFrame({
                    'date': dates,
                    'Relevance Score': relevance,
                    'User Satisfaction': satisfaction
                })
                
                # Create charts
                relevance_chart = alt.Chart(quality_df).mark_line(point=True, color='green').encode(
                    x='date:T',
                    y=alt.Y('Relevance Score:Q', scale=alt.Scale(domain=[0, 1])),
                    tooltip=['date', 'Relevance Score']
                ).properties(height=200, title="Answer Relevance Score")
                
                satisfaction_chart = alt.Chart(quality_df).mark_line(point=True, color='purple').encode(
                    x='date:T',
                    y=alt.Y('User Satisfaction:Q', scale=alt.Scale(domain=[1, 5])),
                    tooltip=['date', 'User Satisfaction']
                ).properties(height=200, title="User Satisfaction Rating (1-5)")
                
                st.altair_chart(relevance_chart, use_container_width=True)
                st.altair_chart(satisfaction_chart, use_container_width=True)
                
                # Key metrics
                col_qm1, col_qm2, col_qm3 = st.columns(3)
                with col_qm1:
                    st.metric("Avg Relevance", f"{relevance.mean():.2f}", f"{0.03:.2f}")
                with col_qm2:
                    st.metric("Satisfaction", f"{satisfaction.mean():.1f}/5", f"{0.2:.1f}")
                with col_qm3:
                    st.metric("Low Quality %", f"{(relevance < 0.7).mean()*100:.1f}%", f"{-2.5:.1f}%")
            
            # System resources tab
            with metric_tabs[2]:
                st.markdown("#### System Resource Utilization")
                
                # Sample resource data
                dates = pd.date_range(start='2025-05-01', periods=24, freq='H')
                cpu = np.clip(np.random.normal(45, 15, 24), 0, 100)
                memory = np.clip(np.random.normal(60, 10, 24), 0, 100)
                
                resource_df = pd.DataFrame({
                    'time': dates,
                    'CPU Usage (%)': cpu,
                    'Memory Usage (%)': memory
                })
                
                # Create area charts
                resource_chart = alt.Chart(resource_df.melt(
                    id_vars=['time'], 
                    value_vars=['CPU Usage (%)', 'Memory Usage (%)'],
                    var_name='Resource',
                    value_name='Usage'
                )).mark_area(opacity=0.5).encode(
                    x='time:T',
                    y=alt.Y('Usage:Q', scale=alt.Scale(domain=[0, 100])),
                    color='Resource:N',
                    tooltip=['time', 'Resource', 'Usage']
                ).properties(height=300, title="System Resource Utilization")
                
                st.altair_chart(resource_chart, use_container_width=True)
                
                # Current resource usage
                col_rm1, col_rm2, col_rm3 = st.columns(3)
                with col_rm1:
                    st.metric("Current CPU", f"{cpu[-1]:.1f}%", f"{cpu[-1] - cpu[-2]:.1f}%")
                with col_rm2:
                    st.metric("Current Memory", f"{memory[-1]:.1f}%", f"{memory[-1] - memory[-2]:.1f}%")
                with col_rm3:
                    st.metric("Peak Usage", f"{max(cpu.max(), memory.max()):.1f}%")
            
            # Export options
            col_exp1, col_exp2 = st.columns(2)
            with col_exp1:
                st.button("Export Dashboard as PDF", key="export_dashboard")
            with col_exp2:
                st.button("Schedule Regular Reports", key="schedule_reports")
    
    # KNOWLEDGE BASE TAB
    with history_tabs[3]:
        st.subheader("RAG Knowledge Base")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Knowledge search
            st.markdown("### Search Knowledge Base")
            
            kb_search = st.text_input("Search for articles, guides, and best practices", placeholder="Search for RAG concepts, tutorials, examples...")
            
            # Knowledge categories
            kb_categories = st.multiselect(
                "Filter by Category",
                ["Tutorials", "Best Practices", "Troubleshooting", "Case Studies", "Research Papers", "Tools & Libraries"],
                default=[]
            )
            
            # Sample knowledge articles
            kb_articles = [
                {"id": "kb-001", "title": "Optimizing Retrieval Parameters for Better Precision", 
                 "category": "Best Practices", "author": "John Smith", "date": "2025-04-15",
                 "snippet": "Learn how to fine-tune your retrieval parameters for different types of queries..."},
                {"id": "kb-002", "title": "Troubleshooting High Latency in RAG Systems", 
                 "category": "Troubleshooting", "author": "Jane Doe", "date": "2025-03-22",
                 "snippet": "Diagnose and fix common causes of high latency in production RAG deployments..."},
                {"id": "kb-003", "title": "Case Study: Implementing RAG for Medical Knowledge", 
                 "category": "Case Studies", "author": "Dr. Sarah Johnson", "date": "2025-02-18",
                 "snippet": "How we built a specialized RAG system for accessing medical knowledge with high precision..."},
                {"id": "kb-004", "title": "Advanced Prompt Engineering for RAG", 
                 "category": "Tutorials", "author": "Michael Zhang", "date": "2025-01-30",
                 "snippet": "Step-by-step guide to crafting effective prompts that maximize RAG system performance..."},
                {"id": "kb-005", "title": "Latest Research in Retrieval Techniques", 
                 "category": "Research Papers", "author": "Research Team", "date": "2025-05-02",
                 "snippet": "Summary of recent academic papers advancing the state of retrieval techniques..."}
            ]
            
            # Display articles with filtering based on search/categories
            st.markdown("### Knowledge Articles")
            
            selected_article = None
            for article in kb_articles:
                if kb_categories and article["category"] not in kb_categories:
                    continue
                    
                st.markdown(f"#### {article['title']}")
                st.markdown(f"**Category:** {article['category']} | **Author:** {article['author']} | **Date:** {article['date']}")
                st.markdown(article['snippet'])
                
                col_art1, col_art2, col_art3 = st.columns([1, 1, 3])
                with col_art1:
                    if st.button("Read Full Article", key=f"read_{article['id']}"):
                        selected_article = article
                with col_art2:
                    st.button("Save for Later", key=f"save_{article['id']}")
                st.divider()
            
            # Knowledge base management
            st.markdown("### Knowledge Management")
            
            col_km1, col_km2 = st.columns(2)
            with col_km1:
                st.button("Submit New Article", key="new_article")
            with col_km2:
                st.button("View Saved Articles", key="saved_articles")
        
        with col2:
            # Article viewer or knowledge summary
            if selected_article:
                st.markdown(f"## {selected_article['title']}")
                st.markdown(f"**Category:** {selected_article['category']} | **Author:** {selected_article['author']} | **Date:** {selected_article['date']}")
                
                # Display full article content - this would be the actual article in a real app
                if selected_article['id'] == "kb-004":  # Advanced Prompt Engineering
                    st.markdown("""
                    ## Advanced Prompt Engineering for RAG Systems
                    
                    Effective prompt engineering is critical for maximizing the performance of RAG systems. This article covers advanced techniques for crafting prompts that help retrieve relevant information and generate accurate, helpful responses.
                    
                    ### Key Components of RAG Prompts
                    
                    1. **System Instructions**
                       - Set clear expectations for the model's role
                       - Provide guidance on response format and style
                       - Specify how to handle uncertainty or missing information
                    
                    2. **Context Utilization**
                       - Direct the model how to use retrieved context
                       - Provide instructions for handling conflicting information
                       - Guide attribution and source citation
                    
                    3. **Response Formatting**
                       - Specify structure for complex responses
                       - Include instructions for lists, tables, or special formatting
                       - Control verbosity and detail level
                    
                    ### Example Templates
                    
                    #### Basic RAG Prompt Template:
                    ```
                    You are a helpful assistant that answers questions based on the provided context.
                    Use only information from the context to answer the question.
                    If the information is not in the context, say "I don't have enough information to answer this question."
                    
                    Context:
                    {{context}}
                    
                    Question: {{query}}
                    
                    Answer:
                    ```
                    
                    #### Advanced RAG Prompt Template:
                    ```
                    You are an expert assistant that provides accurate, helpful answers using the provided context.
                    
                    When answering:
                    1. Use only information present in the context
                    2. Cite sources using [doc1], [doc2] notation
                    3. If the context has insufficient information, explain what's missing
                    4. Present information in a structured, easy-to-understand format
                    5. Consider the confidence level of your answer based on context quality
                    
                    Context:
                    {{context}}
                    
                    Question: {{query}}
                    
                    First, analyze what information you need to answer this question.
                    Then, check if this information exists in the context.
                    Finally, provide a comprehensive answer with appropriate citations.
                    ```
                    
                    ### Testing and Optimization
                    
                    To optimize your prompts:
                    
                    1. **A/B Test Different Variants**
                       - Compare performance across metrics
                       - Test with diverse query types
                    
                    2. **Iterate Based on Errors**
                       - Analyze failure cases
                       - Add specific instructions to address common errors
                    
                    3. **Balance Precision and Flexibility**
                       - Very rigid prompts may limit adaptability
                       - Too flexible prompts may allow hallucinations
                    
                    4. **Consider Using Chain of Thought**
                       - Encourage step-by-step reasoning
                       - Helps with complex reasoning tasks
                    
                    ### Advanced Techniques
                    
                    - **Query-dependent Prompting**: Adjust prompt based on query type
                    - **Few-shot Examples**: Include examples for format guidance
                    - **Self-consistency**: Generate multiple responses and find consensus
                    - **Self-critique**: Have the model evaluate its own response
                    
                    ### Tools and Resources
                    
                    - Prompt engineering libraries
                    - Templates repository
                    - Evaluation frameworks for prompt quality
                    
                    By applying these advanced prompt engineering techniques, you can significantly improve the performance of your RAG system across various metrics including relevance, accuracy, and helpfulness.
                    """)
                    
                    # Article actions
                    col_aa1, col_aa2, col_aa3 = st.columns(3)
                    with col_aa1:
                        st.button("Print Article", key="print_article")
                    with col_aa2:
                        st.button("Share Article", key="share_article")
                    with col_aa3:
                        st.button("Save as PDF", key="save_pdf")
                    
                    # Related articles
                    st.markdown("### Related Articles")
                    
                    related = ["Optimizing Retrieval Parameters for Better Precision", 
                              "Chain of Thought Prompting Techniques",
                              "Evaluating RAG Response Quality"]
                    
                    for r in related:
                        st.markdown(f"- [{r}](#)")
                else:
                    st.info("Full article content would be displayed here")
            else:
                # Knowledge base summary and stats
                st.markdown("### Knowledge Base Overview")
                
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                with col_stats1:
                    st.metric("Total Articles", "128")
                with col_stats2:
                    st.metric("Categories", "6")
                with col_stats3:
                    st.metric("Contributors", "24")
                
                # Popular topics chart
                st.markdown("### Popular Topics")
                
                topics = ["Retrieval", "Embeddings", "Prompting", "Evaluation", "Deployment", "Optimization"]
                topic_counts = [42, 35, 28, 25, 18, 12]
                
                topic_df = pd.DataFrame({
                    "Topic": topics,
                    "Article Count": topic_counts
                })
                
                topic_chart = alt.Chart(topic_df).mark_bar().encode(
                    x=alt.X("Article Count:Q"),
                    y=alt.Y("Topic:N", sort="-x"),
                    color=alt.Color("Topic:N", legend=None),
                    tooltip=["Topic", "Article Count"]
                ).properties(height=250)
                
                st.altair_chart(topic_chart, use_container_width=True)
                
                # Recent activity
                st.markdown("### Recent Activity")
                
                activity = [
                    "New article: 'Latest Research in Retrieval Techniques' (2 days ago)",
                    "Updated: 'Optimizing Retrieval Parameters' (5 days ago)",
                    "New comments on 'Troubleshooting High Latency' (1 week ago)",
                    "New tutorial series on RAG evaluation (2 weeks ago)"
                ]
                
                for item in activity:
                    st.markdown(f"- {item}")