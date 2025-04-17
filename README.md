# project-new
import os
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import logging
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import plotly.express as px
import plotly.graph_objects as go
import chromadb
from chromadb.utils import embedding_functions
import uuid
from typing import List, Dict, Any, Tuple
from pathlib import Path

# -----------------------------------------------------------------------------
# Setup Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(_name_)

# -----------------------------------------------------------------------------
# Constants and Configuration
# -----------------------------------------------------------------------------
CHROMA_PATH = Path("chroma_db")
SQLITE_DB_PATH = Path("semantic_search.db")
MODEL_NAME = 'all-MiniLM-L6-v2'
COLLECTION_NAME = "comments_collection"

# Initialize paths
CHROMA_PATH.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# State Management
# -----------------------------------------------------------------------------
@st.cache_resource
def initialize_resources():
    """Initialize and cache resource-heavy components."""
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    return {
        'model': SentenceTransformer(MODEL_NAME),
        'lemmatizer': WordNetLemmatizer(),
        'stop_words': set(stopwords.words('english')),
        'chroma_client': chromadb.PersistentClient(path=str(CHROMA_PATH))
    }

# -----------------------------------------------------------------------------
# Database Operations
# -----------------------------------------------------------------------------
def init_sqlite_db():
    """Initialize SQLite database with error handling."""
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            c = conn.cursor()
            c.execute('''
                CREATE TABLE IF NOT EXISTS comments (
                    comment_id TEXT PRIMARY KEY,
                    original_text TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"SQLite initialization error: {e}")
        st.error(f"Database initialization failed: {e}")
        raise

def get_collection(chroma_client):
    """Get or create ChromaDB collection with error handling."""
    try:
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=MODEL_NAME
        )
        return chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
    except Exception as e:
        logger.error(f"ChromaDB collection error: {e}")
        st.error(f"Failed to initialize collection: {e}")
        raise

# -----------------------------------------------------------------------------
# Text Processing
# -----------------------------------------------------------------------------
def clean_text(text: str, resources: dict) -> str:
    """Clean and normalize input text."""
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if not text:
        return ""
        
    words = text.split()
    return ' '.join([
        resources['lemmatizer'].lemmatize(word) 
        for word in words 
        if word not in resources['stop_words']
    ])

# -----------------------------------------------------------------------------
# Data Processing
# -----------------------------------------------------------------------------
def process_csv_data(
    df: pd.DataFrame,
    text_column: str,
    resources: dict
) -> Tuple[List[str], List[str]]:
    """Process CSV data with progress tracking and error handling."""
    ids = []
    cleaned_texts = []
    original_texts = []
    
    total_rows = len(df)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        with sqlite3.connect(SQLITE_DB_PATH) as conn:
            cursor = conn.cursor()
            
            for idx, row in df.iterrows():
                try:
                    original_text = str(row[text_column])
                    cleaned_text = clean_text(original_text, resources)
                    
                    if cleaned_text.strip():
                        comment_id = str(uuid.uuid4())
                        embedding = resources['model'].encode([cleaned_text])[0].tolist()
                        
                        cursor.execute(
                            """INSERT INTO comments 
                               (comment_id, original_text, cleaned_text, embedding) 
                               VALUES (?, ?, ?, ?)""",
                            (comment_id, original_text, cleaned_text, str(embedding))
                        )
                        
                        ids.append(comment_id)
                        cleaned_texts.append(cleaned_text)
                        original_texts.append(original_text)
                    
                    # Update progress
                    progress = (idx + 1) / total_rows
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {progress:.1%} complete")
                    
                except Exception as e:
                    logger.warning(f"Error processing row {idx}: {e}")
                    continue
            
            conn.commit()
            
    except Exception as e:
        logger.error(f"Database error during processing: {e}")
        st.error(f"Error processing data: {e}")
        raise
    finally:
        progress_bar.empty()
        status_text.empty()
    
    return ids, cleaned_texts, original_texts

# -----------------------------------------------------------------------------
# Search and Visualization
# -----------------------------------------------------------------------------
def semantic_search(
    query: str,
    collection,
    resources: dict,
    n_results: int = 10
) -> List[Dict[str, Any]]:
    """Perform semantic search with error handling."""
    try:
        cleaned_query = clean_text(query, resources)
        if not cleaned_query:
            st.warning("Please enter a valid search query.")
            return []
            
        query_embedding = resources['model'].encode([cleaned_query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            include=["embeddings", "distances", "documents"]
        )
        
        if not results["documents"][0]:
            return []
            
        return [
            {
                'id': results["ids"][0][i],
                'text': results["documents"][0][i],
                'distance': results["distances"][0][i],
                'embedding': results["embeddings"][0][i]
            }
            for i in range(len(results["documents"][0]))
        ]
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        st.error(f"Search failed: {e}")
        return []

def create_visualization(
    results: List[Dict[str, Any]],
    query_embedding: List[float],
    title: str
) -> go.Figure:
    """Create interactive visualization with error handling."""
    try:
        embeddings = np.array([r['embedding'] for r in results])
        embeddings = np.vstack([embeddings, query_embedding])
        
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(embeddings)
        
        plot_df = pd.DataFrame({
            'PCA1': points_2d[:-1, 0],
            'PCA2': points_2d[:-1, 1],
            'Distance': [r['distance'] for r in results],
            'Text': [r['text'] for r in results]
        })
        
        fig = px.scatter(
            plot_df,
            x='PCA1',
            y='PCA2',
            color='Distance',
            color_continuous_scale='Viridis',
            hover_data=['Text', 'Distance'],
            title=title
        )
        
        fig.add_trace(
            go.Scatter(
                x=[points_2d[-1, 0]],
                y=[points_2d[-1, 1]],
                mode='markers',
                marker=dict(size=15, symbol='star', color='red'),
                name='Query',
                hoverinfo='name'
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Visualization error: {e}")
        st.error(f"Failed to create visualization: {e}")
        return None

# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Semantic Search & Visualization",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Semantic Search & Visualization")
    
    try:
        # Initialize resources
        resources = initialize_resources()
        
        # Initialize databases
        init_sqlite_db()
        collection = get_collection(resources['chroma_client'])
        
        # Sidebar
        with st.sidebar:
            st.header("Upload Data")
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                text_column = st.selectbox(
                    "Select text column:",
                    df.columns.tolist()
                )
                
                if st.button("Process Data"):
                    with st.spinner("Processing data..."):
                        ids, cleaned_texts, original_texts = process_csv_data(
                            df, text_column, resources
                        )
                        
                        if ids:
                            collection.add(
                                documents=cleaned_texts,
                                ids=ids,
                                metadatas=[{"original": text} for text in original_texts]
                            )
                            st.success(f"Processed {len(ids)} documents")
                        else:
                            st.warning("No valid data to process")
            
            st.header("Database Info")
            st.write(f"Total Documents: {collection.count()}")
        
        # Main content
        tab1, tab2 = st.tabs(["Search", "Visualize"])
        
        with tab1:
            col1, col2 = st.columns([3, 1])
            with col1:
                query = st.text_input("Enter search query:")
            with col2:
                n_results = st.slider(
                    "Results:",
                    1,
                    min(50, collection.count()),
                    10
                )
            
            if st.button("Search", type="primary"):
                if query and collection.count() > 0:
                    results = semantic_search(
                        query, collection, resources, n_results
                    )
                    
                    if results:
                        query_embedding = resources['model'].encode([query]).tolist()
                        fig = create_visualization(
                            results,
                            query_embedding,
                            "Search Results (PCA)"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        st.write("### Results")
                        for i, res in enumerate(results, 1):
                            with st.expander(f"Result {i} (Distance: {res['distance']:.4f})"):
                                st.write(res['text'])
                    else:
                        st.info("No results found")
                else:
                    st.warning("Please enter a query and ensure data is loaded")
        
        with tab2:
            if collection.count() > 0:
                if st.button("Generate Visualization"):
                    with st.spinner("Creating visualization..."):
                        data = collection.get(
                            include=["embeddings", "documents"],
                            limit=collection.count()
                        )
                        
                        embeddings = np.array(data["embeddings"])
                        pca = PCA(n_components=2)
                        points_2d = pca.fit_transform(embeddings)
                        
                        df = pd.DataFrame({
                            'PCA1': points_2d[:, 0],
                            'PCA2': points_2d[:, 1],
                            'Text': data["documents"]
                        })
                        
                        fig = go.Figure()
                        fig.add_trace(
                            go.Histogram2dContour(
                                x=df['PCA1'],
                                y=df['PCA2'],
                                colorscale='Viridis',
                                showscale=True,
                                name='Density'
                            )
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=df['PCA1'],
                                y=df['PCA2'],
                                mode='markers',
                                marker=dict(
                                    color='rgba(255, 255, 255, 0.6)',
                                    size=8
                                ),
                                hovertext=df['Text'],
                                name='Documents'
                            )
                        )
                        
                        fig.update_layout(
                            title="Document Embedding Distribution",
                            xaxis_title="PCA Component 1",
                            yaxis_title="PCA Component 2",
                            hovermode='closest'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please upload and process data first")
                
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"An error occurred: {e}")

if _name_ == "_main_":
    main()
