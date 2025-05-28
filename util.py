"""
Utility Functions for Text Processing, Document Parsing and Chunking
====================================================================

This module provides a comprehensive set of utilities for document processing,
text parsing, and advanced chunking strategies for Retrieval-Augmented Generation
(RAG) pipelines.

The module contains two main classes:

1. Parsers: Tools for extracting text, images, tables, and metadata from documents
   - PDF parsing with PyMuPDF and pdfplumber
   - Structured document parsing with unstructured.io
   - Support for table extraction in various formats (DataFrame, JSON, Markdown)
   - Image extraction and saving

2. Chunkers: A variety of chunking strategies for optimizing retrieval
   - Recursive: Splits text into fixed-size chunks with natural boundaries
   - Semantic: Groups semantically similar sentences based on embeddings
   - SDPM: Two-pass approach with semantic merging of adjacent chunks
   - Late: Fixed-size chunking preserving token-level embeddings
   - Slumber: LLM-guided optimal splitting points
   - Agentic: LLM analysis for logical document structure chunking
   - Summary-based: Adds summaries to chunks for enhanced context
   - RAPTOR: Hierarchical chunking with topic modeling and semantic relationships
   - Contextualized: Adds document-aware context to each chunk

Dependencies:
------------
- pandas, numpy, scipy: Data manipulation and mathematical operations
- PyMuPDF (fitz), pdfplumber: PDF parsing
- sentence-transformers, transformers: Embedding models
- unstructured: Structured document parsing
- tiktoken: Token counting for LLM context management
- re: Regular expressions for text processing

Example Usage:
-------------
# Parse a PDF document
result = Parsers.parse_pdf("document.pdf", table_format="json")

# Create chunks using different strategies
chunks = Chunkers.recursive(result["text"], chunk_size=500)
semantic_chunks = Chunkers.semantic(result["text"], similarity_threshold=0.8)
summarized_chunks = Chunkers.summary_based(result["text"])

Author: Rahul Damani
Version: 0.1.0
Last Updated: May 28, 2025
"""

import pandas as pd
import os
import re
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from llm_call import ask_llm

MODELS = {
    "url"  : "https://api.dev.cortex.lilly.com",
    "model"  : "text2sql"
}

class Parsers:
    # parse pdf
    @staticmethod
    def parse_pdf(file_path : str, table_format : str ="dataframe", summarize_tables : bool=False):
        import fitz  # PyMuPDF
        import pdfplumber

        # Metadata extraction
        doc = fitz.open(file_path)
        metadata = doc.metadata

        # Extract text and images
        text_content = ""
        images = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text_content += page.get_text()
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = f"{os.path.splitext(file_path)[0]}_page{page_num+1}_img{xref}.{image_ext}"
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)
                images.append(image_filename)

        # Extract tables
        tables = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    if summarize_tables:
                        # Placeholder for LLM summarization - replace with actual implementation
                        summary = f"Table with {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns)}"
                        tables.append(summary)
                    elif table_format == "json":
                        tables.append(df.to_json(orient="records"))
                    elif table_format == "markdown":
                        tables.append(df.to_markdown(index=False))
                    else:
                        tables.append(df)

        # Combine all metadata
        result = {
            "metadata": metadata,
            "text": text_content,
            "images": images,
            "tables": tables
        }
        return result

    @staticmethod
    def parse_pdf_with_unstructured(file_path):
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(filename=file_path)
        results = []
        for el in elements:
            results.append({
                "type": el.category,
                "text": el.text,
                "metadata": el.metadata.to_dict() if el.metadata else {}
            })
        return results


class Chunkers:
    @staticmethod
    def recursive(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Recursively splits text into chunks of specified size with overlap.
        
        Args:
            text: The text to split into chunks
            chunk_size: Maximum size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks = []
        start = 0
        
        # Try to split on paragraph or sentence boundaries when possible
        paragraph_splits = [m.start() for m in re.finditer(r'\n\s*\n', text)]
        sentence_splits = [m.start() for m in re.finditer(r'[.!?]\s+', text)]
        
        while start < len(text):
            # Find the ideal end point with chunk_size
            ideal_end = start + chunk_size
            
            # If we're at the end of the text, just take the rest
            if ideal_end >= len(text):
                chunks.append({
                    "text": text[start:],
                    "metadata": {
                        "start_char": start,
                        "end_char": len(text),
                        "chunk_type": "recursive"
                    }
                })
                break
                
            # Try to find a paragraph break near the ideal end
            paragraph_break = next((p for p in paragraph_splits if p > start and p <= ideal_end), None)
            sentence_break = next((s for s in sentence_splits if s > start and s <= ideal_end), None)
            
            # Use the closest natural break point, or the exact ideal_end if none are found
            if paragraph_break:
                end = paragraph_break
            elif sentence_break:
                end = sentence_break + 1  # Include the punctuation
            else:
                # No natural break found, find the last space before ideal_end
                space_before = text.rfind(' ', start, ideal_end)
                end = space_before if space_before > start else ideal_end
                
            # Create the chunk
            chunks.append({
                "text": text[start:end],
                "metadata": {
                    "start_char": start,
                    "end_char": end,
                    "chunk_type": "recursive"
                }
            })
            
            # Move to next chunk position with overlap
            start = max(start + 1, end - overlap)
            
        return chunks

    @staticmethod
    def semantic(
        text: str, 
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        similarity_threshold: float = 0.75, 
        min_chunk_size: int = 50,
        max_chunk_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Splits text into semantically coherent chunks using embeddings.
        
        Args:
            text: Text to split into chunks
            embedding_model: Model to use for creating embeddings
            similarity_threshold: Threshold for semantic similarity (0-1)
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Load embedding model
        model = SentenceTransformer(embedding_model)
        
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return []
            
        # Get embeddings for each sentence
        sentence_embeddings = model.encode(sentences)
        
        # Initialize chunks
        chunks = []
        current_chunk = [sentences[0]]
        current_embedding = sentence_embeddings[0].reshape(1, -1)
        current_start = 0
        
        # Process sentences to form chunks
        for i in range(1, len(sentences)):
            # Get embedding for current sentence
            sentence_embedding = sentence_embeddings[i].reshape(1, -1)
            
            # Calculate cosine similarity with current chunk embedding
            similarity = np.dot(current_embedding, sentence_embedding.T) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(sentence_embedding)
            )
            
            # Calculate current chunk length
            current_length = sum(len(s) for s in current_chunk)
            next_length = current_length + len(sentences[i])
            
            # Determine if we should add to current chunk or start a new one
            if (similarity > similarity_threshold and next_length < max_chunk_size) or current_length < min_chunk_size:
                # Add to current chunk
                current_chunk.append(sentences[i])
                # Update embedding as average of all sentences in chunk
                current_embedding = np.mean(
                    np.vstack([current_embedding, sentence_embedding]), axis=0
                ).reshape(1, -1)
            else:
                # Finalize current chunk
                chunk_text = ' '.join(current_chunk)
                end = current_start + len(chunk_text)
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "start_char": current_start,
                        "end_char": end,
                        "chunk_type": "semantic",
                        "sentences": len(current_chunk)
                    }
                })
                
                # Start new chunk
                current_chunk = [sentences[i]]
                current_embedding = sentence_embedding
                current_start = end + 1  # +1 for the space
        
        # Add the last chunk if it exists
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            end = current_start + len(chunk_text)
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "start_char": current_start,
                    "end_char": end,
                    "chunk_type": "semantic",
                    "sentences": len(current_chunk)
                }
            })
        
        return chunks

    @staticmethod
    def sdpm(
        text: str,
        chunk_size: int = 512,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.75
    ) -> List[Dict[str, Any]]:
        """
        SDPM (Semantic Double-Pass Merging) chunking.
        
        First splits text into small chunks, then merges similar adjacent chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Initial chunk size for first pass
            embedding_model: Model to use for embeddings
            threshold: Similarity threshold for merging
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # First pass: split into small chunks
        small_chunks = Chunkers.recursive(text, chunk_size=chunk_size//2)
        
        # Load embedding model
        model = SentenceTransformer(embedding_model)
        
        # Get embeddings for each chunk
        chunk_texts = [chunk["text"] for chunk in small_chunks]
        chunk_embeddings = model.encode(chunk_texts)
        
        # Second pass: merge similar chunks
        merged_chunks = []
        current_merged = [small_chunks[0]]
        current_embedding = chunk_embeddings[0].reshape(1, -1)
        
        for i in range(1, len(small_chunks)):
            # Calculate similarity with current merged chunk
            similarity = np.dot(current_embedding, chunk_embeddings[i]) / (
                np.linalg.norm(current_embedding) * np.linalg.norm(chunk_embeddings[i])
            )
            
            # If similar enough, merge
            if similarity > threshold:
                current_merged.append(small_chunks[i])
                # Update embedding as weighted average
                total_len = sum(len(c["text"]) for c in current_merged)
                weights = np.array([len(c["text"])/total_len for c in current_merged])
                embeddings_to_merge = np.vstack([current_embedding] + [chunk_embeddings[i]])
                current_embedding = np.average(embeddings_to_merge, axis=0, weights=weights).reshape(1, -1)
            else:
                # Finalize the current merged chunk
                start_char = current_merged[0]["metadata"]["start_char"]
                end_char = current_merged[-1]["metadata"]["end_char"]
                merged_text = text[start_char:end_char]
                
                merged_chunks.append({
                    "text": merged_text,
                    "metadata": {
                        "start_char": start_char,
                        "end_char": end_char,
                        "chunk_type": "sdpm",
                        "merged_count": len(current_merged)
                    }
                })
                
                # Start a new merged chunk
                current_merged = [small_chunks[i]]
                current_embedding = chunk_embeddings[i].reshape(1, -1)
        
        # Add the last merged chunk
        if current_merged:
            start_char = current_merged[0]["metadata"]["start_char"]
            end_char = current_merged[-1]["metadata"]["end_char"]
            merged_text = text[start_char:end_char]
            
            merged_chunks.append({
                "text": merged_text,
                "metadata": {
                    "start_char": start_char,
                    "end_char": end_char,
                    "chunk_type": "sdpm",
                    "merged_count": len(current_merged)
                }
            })
        
        return merged_chunks

    @staticmethod
    def late(
        text: str,
        chunk_size: int = 512,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ) -> List[Dict[str, Any]]:
        """
        Late interaction chunking.
        
        Splits text into fixed-size chunks and stores token-level embeddings for late interaction.
        
        Args:
            text: Text to chunk
            chunk_size: Chunk size in tokens
            embedding_model: Model for embeddings
            
        Returns:
            List of chunk dictionaries with text, token embeddings and metadata
        """
        # Load embedding model and tokenizer
        model = SentenceTransformer(embedding_model)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
        # Tokenize text
        tokens = tokenizer.tokenize(text)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # Split into chunks by token count
        chunks = []
        for i in range(0, len(token_ids), chunk_size):
            chunk_token_ids = token_ids[i:i+chunk_size]
            chunk_text = tokenizer.decode(chunk_token_ids)
            
            # Get token-level embeddings for each chunk
            token_embeddings = model.encode([tokenizer.decode([tid]) for tid in chunk_token_ids], 
                                           convert_to_numpy=True)
            
            # Add chunk to results
            start_char = text.find(chunk_text)
            end_char = start_char + len(chunk_text)
            
            chunks.append({
                "text": chunk_text,
                "token_embeddings": token_embeddings,
                "metadata": {
                    "start_char": start_char,
                    "end_char": end_char,
                    "chunk_type": "late",
                    "tokens": len(chunk_token_ids)
                }
            })
        
        return chunks

    # For the Slumber Chunker
    @staticmethod
    def slumber(
        text: str,
        model_config: Dict = MODELS,
        initial_chunk_size: int = 2000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Slumber chunking using an LLM to determine optimal splitting points.
        
        Args:
            text: Text to chunk
            model_config: Configuration for the LLM
            initial_chunk_size: Initial size of text windows to analyze
            overlap: Overlap between initial chunks
            
        Returns:
            List of chunks with metadata
        """
        # Create initial overlapping chunks for analysis
        initial_chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + initial_chunk_size, len(text))
            initial_chunks.append(text[start:end])
            start += initial_chunk_size - overlap
        
        # If we only have one chunk, just return it
        if len(initial_chunks) <= 1:
            return [{
                "text": text,
                "metadata": {
                    "start_char": 0,
                    "end_char": len(text),
                    "chunk_type": "slumber"
                }
            }]
            
        # Process chunk pairs to find optimal split points
        final_chunks = []
        for i in range(len(initial_chunks) - 1):
            # Create context from the two adjacent chunks
            context = initial_chunks[i] + initial_chunks[i+1]
            
            # Only process if context is under 8000 characters (LLM context limit)
            if len(context) > 8000:
                context = context[:8000]
            
            # Ask LLM to identify optimal split point
            prompt = f"""
            I have two sections of text that overlap. Please identify where the optimal splitting point 
            would be to create semantically coherent chunks. Return only the line or sentence where the 
            split should happen. Be concise and only return the split text.
            
            Text:
            {context}
            """
            
            try:
                # Use ask_llm to get the split point
                split_text = ask_llm(model_config, prompt)
                
                # Find position of suggested split point
                split_pos = context.find(split_text)
                if split_pos > 0:
                    # Create chunk up to the split point
                    chunk_text = context[:split_pos + len(split_text)]
                    
                    # Calculate absolute position in original text
                    abs_start = max(0, (i * (initial_chunk_size - overlap)))
                    abs_end = abs_start + len(chunk_text)
                    
                    final_chunks.append({
                        "text": chunk_text,
                        "metadata": {
                            "start_char": abs_start,
                            "end_char": abs_end,
                            "chunk_type": "slumber",
                            "split_reason": "LLM determined"
                        }
                    })
                else:
                    # Fallback: just use the first initial chunk
                    abs_start = max(0, (i * (initial_chunk_size - overlap)))
                    abs_end = abs_start + len(initial_chunks[i])
                    
                    final_chunks.append({
                        "text": initial_chunks[i],
                        "metadata": {
                            "start_char": abs_start,
                            "end_char": abs_end,
                            "chunk_type": "slumber",
                            "split_reason": "fallback - split point not found"
                        }
                    })
            except Exception as e:
                # Fallback if LLM fails
                abs_start = max(0, (i * (initial_chunk_size - overlap)))
                abs_end = abs_start + len(initial_chunks[i])
                
                final_chunks.append({
                    "text": initial_chunks[i],
                    "metadata": {
                        "start_char": abs_start,
                        "end_char": abs_end,
                        "chunk_type": "slumber",
                        "split_reason": f"error: {str(e)}"
                    }
                })
        
        # Add the last chunk
        if initial_chunks:
            last_chunk = initial_chunks[-1]
            abs_start = max(0, ((len(initial_chunks) - 1) * (initial_chunk_size - overlap)))
            abs_end = abs_start + len(last_chunk)
            
            final_chunks.append({
                "text": last_chunk,
                "metadata": {
                    "start_char": abs_start,
                    "end_char": abs_end,
                    "chunk_type": "slumber",
                    "split_reason": "last chunk"
                }
            })
        
        return final_chunks

    # For the Agentic Chunker
    @staticmethod
    def agentic(
        text: str,
        model_config: Dict = MODELS,
        max_chunks: int = 10,
        min_chunk_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Agentic chunking using an LLM to adaptively determine how to chunk the text.
        
        Args:
            text: Text to chunk
            model_config: Configuration for the LLM
            max_chunks: Maximum number of chunks to create
            min_chunk_size: Minimum size of each chunk
            
        Returns:
            List of chunks with metadata
        """
        try:
            # For very long texts, use recursive chunking first to get manageable parts
            if len(text) > 8000:
                initial_chunks = Chunkers.recursive(text, chunk_size=7000, overlap=0)
                all_chunks = []
                
                for chunk in initial_chunks:
                    part_chunks = Chunkers.agentic(chunk["text"], model_config, max_chunks, min_chunk_size)
                    all_chunks.extend(part_chunks)
                
                return all_chunks
            
            # Use LLM to analyze and propose chunk boundaries
            prompt = f"""
            Analyze the following text and identify {min(max_chunks, 5)} optimal places to split it into logical chunks.
            Each chunk should be at least {min_chunk_size} characters long.
            
            Return your answer as a JSON list of objects, each with 'start_position' (character position to start chunk) 
            and 'reason' (why split here). The first chunk should always start at position 0.
            
            Example format:
            [
                {{"start_position": 0, "reason": "Introduction section"}},
                {{"start_position": 2045, "reason": "Methods section"}}
            ]
            
            Text to analyze:
            {text[:8000]}  # Limit to 8000 chars for LLM context
            """
            
            # Call LLM
            response = ask_llm(model_config, prompt)
            
            # Parse response - handle potential formatting issues
            try:
                import json
                chunk_boundaries = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response if it's not pure JSON
                import re
                json_match = re.search(r'\[\s*{.*}\s*\]', response.replace('\n', ' '), re.DOTALL)
                if json_match:
                    chunk_boundaries = json.loads(json_match.group(0))
                else:
                    # Fallback to basic chunking
                    raise ValueError("Could not parse LLM response as JSON")
            
            # Create chunks based on boundaries
            chunks = []
            for i in range(len(chunk_boundaries)):
                start = chunk_boundaries[i]["start_position"]
                
                # Determine end position (either next boundary or end of text)
                if i < len(chunk_boundaries) - 1:
                    end = chunk_boundaries[i + 1]["start_position"]
                else:
                    end = len(text)
                
                # Extract chunk text
                chunk_text = text[start:end]
                
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "start_char": start,
                        "end_char": end,
                        "chunk_type": "agentic",
                        "chunk_index": i,
                        "split_reason": chunk_boundaries[i]["reason"]
                    }
                })
            
            return chunks
            
        except Exception as e:
            # Fallback to recursive chunking if any part fails
            print(f"Agentic chunking failed: {e}. Falling back to recursive chunking.")
            return Chunkers.recursive(text, chunk_size=1000)

    # For the Summary-based Chunker
    @staticmethod
    def summary_based(
        text: str,
        model_config: Dict = MODELS,
        chunk_size: int = 2000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Summary-based chunking that adds summaries to the chunked text using LLM.
        
        Args:
            text: Text to chunk
            model_config: Configuration for the LLM
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunks with summary metadata
        """
        # First create base chunks using recursive chunking
        base_chunks = Chunkers.recursive(text, chunk_size=chunk_size, overlap=overlap)
        
        # Enhance with summaries
        summary_chunks = []
        for chunk in base_chunks:
            chunk_text = chunk["text"]
            
            try:
                # Generate summary for the chunk using LLM
                if len(chunk_text) > 8000:
                    # For very long chunks, just use the first part
                    summarization_text = chunk_text[:8000]
                else:
                    summarization_text = chunk_text
                    
                prompt = f"""
                Please provide a concise summary of the following text in 2-3 sentences:
                
                {summarization_text}
                """
                
                summary = ask_llm(model_config, prompt)
                
                # Add summary to metadata
                enhanced_chunk = chunk.copy()
                enhanced_chunk["metadata"] = enhanced_chunk.get("metadata", {})
                enhanced_chunk["metadata"].update({
                    "chunk_type": "summary_based",
                    "summary": summary
                })
                
                summary_chunks.append(enhanced_chunk)
                
            except Exception as e:
                # If summarization fails, use the original chunk
                chunk["metadata"] = chunk.get("metadata", {})
                chunk["metadata"]["chunk_type"] = "summary_based"
                chunk["metadata"]["summary_error"] = str(e)
                summary_chunks.append(chunk)
        
        return summary_chunks

    # For the RAPTOR Chunker (still a simulation but more LLM-based)
    @staticmethod
    def raptor(
        text: str,
        model_config: Dict = MODELS,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens_per_chunk: int = 512,
        num_layers: int = 2,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        RAPTOR (Recursive Abstractive Processing & Topical Organization for Retrieval) chunking.
        
        This implementation includes core RAPTOR functionality including:
        1. Hierarchical tree structure
        2. Semantic similarity clustering
        3. Topic extraction
        
        Args:
            text: Text to chunk
            model_config: Configuration for the LLM for topic enhancement
            embedding_model: Model for embeddings
            max_tokens_per_chunk: Maximum tokens per chunk
            num_layers: Number of hierarchical layers to build
            top_k: Number of similar nodes to group together
            
        Returns:
            List of chunks with rich metadata including topics and semantic relationships
        """
        import tiktoken
        import numpy as np
        from scipy import spatial
        from sentence_transformers import SentenceTransformer
        import re
        from collections import defaultdict
        
        # Simple Node class to replace the imported one
        class Node:
            def __init__(self, text, index, children=None, embedding=None):
                self.text = text
                self.index = index
                self.children = children or set()
                self.embedding = embedding
                self.parent = None
                self.level = 0
                self.summary = None

        # Helper functions
        def split_text_into_chunks(text, max_tokens):
            """Split text into chunks of approximately max_tokens"""
            tokenizer = tiktoken.get_encoding("cl100k_base")
            
            # Split by paragraphs first
            paragraphs = text.split('\n\n')
            chunks = []
            current_chunk = []
            current_length = 0
            
            for para in paragraphs:
                # Skip empty paragraphs
                if not para.strip():
                    continue
                    
                para_tokens = len(tokenizer.encode(para))
                
                # If paragraph is too long, split into sentences
                if para_tokens > max_tokens:
                    sentences = re.split(r'(?<=[.!?])\s+', para)
                    for sentence in sentences:
                        sentence_tokens = len(tokenizer.encode(sentence))
                        if current_length + sentence_tokens > max_tokens:
                            if current_chunk:
                                chunks.append(' '.join(current_chunk))
                                current_chunk = []
                                current_length = 0
                        current_chunk.append(sentence)
                        current_length += sentence_tokens
                else:
                    # If adding paragraph exceeds max tokens, create a new chunk
                    if current_length + para_tokens > max_tokens:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    current_chunk.append(para)
                    current_length += para_tokens
            
            # Add the last chunk if it exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                
            return chunks
        
        # Function to calculate cosine distance between embeddings
        def calculate_similarity(emb1, emb2):
            return 1 - spatial.distance.cosine(emb1, emb2)
        
        # Function to get the most similar nodes
        def get_similar_nodes(node, nodes, top_k):
            similarities = []
            for other_node in nodes:
                if other_node.index != node.index:
                    similarity = calculate_similarity(node.embedding, other_node.embedding)
                    similarities.append((other_node, similarity))
            
            # Sort by similarity in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        
        # Initialize embedding model
        embedder = SentenceTransformer(embedding_model)
        
        # Step 1: Split text into chunks and create leaf nodes
        chunks = split_text_into_chunks(text, max_tokens_per_chunk)
        leaf_nodes = []
        
        for i, chunk_text in enumerate(chunks):
            # Create embedding for the chunk
            embedding = embedder.encode(chunk_text)
            
            # Create leaf node
            node = Node(text=chunk_text, index=i, embedding=embedding)
            leaf_nodes.append(node)
        
        # Step 2: Build hierarchical structure
        all_nodes = leaf_nodes.copy()
        current_layer = leaf_nodes
        layer_to_nodes = {0: leaf_nodes}
        
        # Track parent-child relationships
        node_to_parent = {}
        
        # Build layers from bottom up
        for layer in range(1, num_layers + 1):
            next_layer = []
            processed_nodes = set()
            
            # Group similar nodes together
            for node in current_layer:
                if node.index in processed_nodes:
                    continue
                    
                # Find similar nodes
                similar_nodes = get_similar_nodes(node, current_layer, top_k)
                group_nodes = [node] + [n for n, _ in similar_nodes]
                
                # Mark these nodes as processed
                for n in group_nodes:
                    processed_nodes.add(n.index)
                
                # Create a parent node with the group's text
                group_text = "\n\n".join([n.text for n in group_nodes])
                
                # Try to summarize using LLM if text isn't too long
                summary = ""
                if len(group_text) < 4000:  # Assuming LLM has 4K token limit
                    try:
                        prompt = f"Summarize the following text in 1-2 sentences, preserving key information:\n\n{group_text}"
                        summary = ask_llm(model_config, prompt)
                    except Exception as e:
                        # Fallback to first sentence of first chunk
                        summary = re.split(r'(?<=[.!?])\s+', group_nodes[0].text)[0]
                else:
                    # Text too long, use first sentence of each chunk
                    first_sentences = []
                    for n in group_nodes[:3]:  # Limit to first 3 nodes
                        sentences = re.split(r'(?<=[.!?])\s+', n.text)
                        if sentences:
                            first_sentences.append(sentences[0])
                    summary = " ".join(first_sentences)
                
                # Create parent node
                parent_idx = len(all_nodes)
                parent_node = Node(
                    text=summary, 
                    index=parent_idx,
                    children={n.index for n in group_nodes},
                    embedding=embedder.encode(summary)
                )
                parent_node.level = layer
                
                # Set relationships
                for child in group_nodes:
                    child.parent = parent_node
                    node_to_parent[child.index] = parent_idx
                
                all_nodes.append(parent_node)
                next_layer.append(parent_node)
            
            # Update current layer and layer mapping
            current_layer = next_layer
            layer_to_nodes[layer] = next_layer
        
        # Step 3: Convert to our standardized chunk format with rich metadata
        result_chunks = []
        
        # Process leaf nodes first (the actual text chunks)
        for node in leaf_nodes:
            # Get start and end positions (approximate)
            start_pos = text.find(node.text[:50])
            if start_pos == -1:
                start_pos = 0
            end_pos = start_pos + len(node.text)
            
            # Get topic and key concepts using LLM
            topic = ""
            key_concepts = []
            
            if len(node.text) > 100:  # Only process substantial chunks
                try:
                    # Limit text length for LLM processing
                    topic_text = node.text[:3000] if len(node.text) > 3000 else node.text
                    
                    prompt = f"""
                    Given this text, identify:
                    1. A specific topic title (3-5 words)
                    2. Three key concepts mentioned
                    
                    Format your response as JSON with keys "topic" and "key_concepts" (an array).
                    
                    Text:
                    {topic_text}
                    """
                    
                    response = ask_llm(model_config, prompt)
                    
                    # Try to parse JSON response
                    try:
                        import json
                        topic_data = json.loads(response)
                        topic = topic_data.get("topic", "")
                        key_concepts = topic_data.get("key_concepts", [])
                    except:
                        # Direct use if not valid JSON
                        topic = response[:50].strip()
                except Exception as e:
                    # Fallback - use first sentence
                    sentences = re.split(r'(?<=[.!?])\s+', node.text)
                    if sentences:
                        topic = sentences[0][:50]
            
            # Find related chunks based on embedding similarity
            related_chunks = []
            for other_node in leaf_nodes:
                if other_node.index != node.index:
                    sim = calculate_similarity(node.embedding, other_node.embedding)
                    if sim > 0.5:  # Only include if similarity is significant
                        related_chunks.append({
                            "chunk_index": other_node.index,
                            "similarity": float(sim)
                        })
            # Sort by similarity and take top 3
            related_chunks.sort(key=lambda x: x["similarity"], reverse=True)
            related_chunks = related_chunks[:3]
            
            # Create the standardized chunk
            chunk = {
                "text": node.text,
                "metadata": {
                    "start_char": start_pos,
                    "end_char": end_pos,
                    "chunk_type": "raptor",
                    "chunk_index": node.index,
                    "topic": topic,
                    "key_concepts": key_concepts,
                    "tree_level": 0,  # Leaf nodes are at level 0
                    "related_chunks": related_chunks
                }
            }
            
            # Add hierarchical information
            if node.index in node_to_parent:
                parent_idx = node_to_parent[node.index]
                chunk["metadata"]["parent_node"] = parent_idx
                
                # Find siblings (nodes with same parent)
                siblings = []
                for idx, parent in node_to_parent.items():
                    if parent == parent_idx and idx != node.index:
                        siblings.append(idx)
                if siblings:
                    chunk["metadata"]["sibling_nodes"] = siblings
            
            result_chunks.append(chunk)
        
        return result_chunks
    
    @staticmethod
    def contextualized(
        text: str,
        model_config: Dict = MODELS,
        chunk_size: int = 2000,
        overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Contextualized chunking that adds LLM-generated context to each chunk.
        
        This method:
        1. Splits the text into chunks using recursive chunking
        2. For each chunk, generates additional context information using LLM
        3. Appends the context to the chunk text
        
        Args:
            text: Text to chunk
            model_config: Configuration for the LLM
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
            
        Returns:
            List of contextualized chunks with metadata
        """
        # First create base chunks using recursive chunking
        base_chunks = Chunkers.recursive(text, chunk_size=chunk_size, overlap=overlap)
        
        # Generate contextual information for each chunk
        contextualized_chunks = []
        for i, chunk in enumerate(base_chunks):
            chunk_text = chunk["text"]
            
            try:
                # Generate context for the chunk using LLM
                context = Chunkers._generate_context(text, chunk_text, i, len(base_chunks), model_config)
                
                # Add context to chunk text
                contextualized_text = f"{chunk_text}\n\nContext: {context}"
                
                # Create enhanced chunk with context
                contextualized_chunk = chunk.copy()
                contextualized_chunk["text"] = contextualized_text
                contextualized_chunk["metadata"] = contextualized_chunk.get("metadata", {})
                contextualized_chunk["metadata"].update({
                    "chunk_type": "contextualized",
                    "original_length": len(chunk_text),
                    "context_length": len(context)
                })
                
                contextualized_chunks.append(contextualized_chunk)
                
            except Exception as e:
                # If context generation fails, use the original chunk
                chunk["metadata"] = chunk.get("metadata", {})
                chunk["metadata"]["chunk_type"] = "contextualized"
                chunk["metadata"]["context_error"] = str(e)
                contextualized_chunks.append(chunk)
        
        return contextualized_chunks

    @staticmethod
    def _generate_context(
        document: str, 
        chunk_text: str, 
        chunk_index: int, 
        total_chunks: int,
        model_config: Dict = MODELS
    ) -> str:
        """
        Generate contextual information for a chunk using LLM.
        
        Args:
            document: Full document text
            chunk_text: The text of the current chunk
            chunk_index: Index of the current chunk
            total_chunks: Total number of chunks
            model_config: Configuration for the LLM
            
        Returns:
            Generated context as a string
        """
        # For very long documents, we'll use a summarized version
        doc_for_context = document
        if len(document) > 8000:
            # Get first and last parts
            first_part = document[:3000]
            last_part = document[-3000:]
            middle_summary = "... [middle content omitted for length] ..."
            doc_for_context = f"{first_part}\n{middle_summary}\n{last_part}"
        
        # Craft prompt for context generation
        prompt = f"""
        I need context for a chunk of text that will be used in retrieval.
        
        The chunk is #{chunk_index+1} out of {total_chunks} total chunks from a longer document.
        
        Full document (possibly truncated):
        ```
        {doc_for_context}
        ```
        
        Current chunk:
        ```
        {chunk_text}
        ```
        
        Please provide context about:
        1. Where this chunk fits in the overall document
        2. Key topics or entities mentioned in this chunk
        3. Relevant connections to other parts of the document
        
        Keep the context concise (3-5 sentences) and focus on information that would help understand this chunk in isolation.
        """
        
        # Use ask_llm to generate context
        context = ask_llm(model_config, prompt)
        
        return context