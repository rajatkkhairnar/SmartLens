# --- PATCH: Fix SQLite for ChromaDB on Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import logging
import gc

# --- PATCH: Settings to prevent crashes ---
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger('transformers').setLevel(logging.ERROR)

from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb

# --- CONFIGURATION ---
IMAGE_FOLDER = './images'
DB_PATH = './chroma_db'
COLLECTION_NAME = 'personal_photos'

if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_model():
    # FIX: Use the full Hugging Face repository name
    return SentenceTransformer('sentence-transformers/clip-ViT-B-32')

# --- 2. DATABASE SETUP ---
def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# --- 3. INDEXING LOGIC (Memory Safe) ---
def index_images(model, collection):
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.warning(f"No images found in {IMAGE_FOLDER}.")
        return

    # Check existing files to avoid re-work
    existing_ids = set(collection.get()['ids']) # Use set for faster lookup
    new_files = [f for f in image_files if f not in existing_ids]

    if not new_files:
        st.info("No new images to index.")
        return

    # Progress bar for user feedback
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # BATCH SIZE 5 (Very small to prevent crashing)
    batch_size = 5
    total_batches = (len(new_files) + batch_size - 1) // batch_size

    for i in range(0, len(new_files), batch_size):
        try:
            # 1. Clear memory before starting a batch
            gc.collect() 
            
            batch_files = new_files[i : i + batch_size]
            batch_images = []
            valid_ids = []

            # 2. Load images one by one
            for file_name in batch_files:
                try:
                    img_path = os.path.join(IMAGE_FOLDER, file_name)
                    # Open and convert immediately to save RAM
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    valid_ids.append(file_name)
                except Exception as e:
                    print(f"Skipping {file_name}: {e}")

            if batch_images:
                # 3. Generate Embeddings
                embeddings = model.encode(batch_images)
                
                # 4. Save to ChromaDB
                collection.add(
                    embeddings=embeddings.tolist(),
                    ids=valid_ids,
                    metadatas=[{"filename": f} for f in valid_ids]
                )
            
            # Update progress
            current_progress = (i + batch_size) / len(new_files)
            progress_bar.progress(min(current_progress, 1.0))
            status_text.text(f"Indexed {min(i + batch_size, len(new_files))}/{len(new_files)} images...")
            
        except Exception as e:
            st.error(f"Crash at batch {i}: {e}")
            break
            
    st.success(f"Finished! Indexed {len(new_files)} new images.")

# --- 4. MAIN UI ---
def main():
    st.title("📸 SmartLens Search")
    st.write("Search your local photos using **natural language**.")

    # Load resources
    try:
        model = load_model()
        collection = get_chroma_collection()
    except Exception as e:
        st.error(f"Error loading model or DB: {e}")
        return
    
    #Sidebar for controls
    with st.sidebar:
        st.header("Controls")
        if st.button("Scan & Index Images"):
            index_images(model, collection)
        
        # Check if folder exists, otherwise show 0
        if os.path.exists(IMAGE_FOLDER):
            count = len(os.listdir(IMAGE_FOLDER))
        else:
            count = 0
        st.write(f"Images in folder: {count}")
    
    # Search Interface
    query = st.text_input("Search:", placeholder="e.g., 'Mountains' or 'Buildings")

    if query:
        # 1. Embed the query text
        query_embedding = model.encode([query])
        # 2. Query ChromaDB
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3 # Return top 3 matches
        )
        
        # 3. Display Results
        if results['ids'] and results['ids'][0]:
            st.write("### Top Matches")

            # Safely get distances; default to an empty list if None
            # Safely get distances; default to an empty list if None
            distances = results.get('distances', [[]])[0] if results.get('distances') else []

            cols = st.columns(3)
            for idx, file_id in enumerate(results['ids'][0]):
                image_path = os.path.join(IMAGE_FOLDER, file_id)
                # Ensure we have a distance for this specific index
                distance = distances[idx] if idx < len(distances) else 0.0
                
                if os.path.exists(image_path):
                    with cols[idx]:
                        st.image(image_path, caption=file_id)
        else:
            st.info("No matches found.")

if __name__ == "__main__":
    main()