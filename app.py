# --- PATCH: Fix SQLite for ChromaDB on Streamlit Cloud ---
# This MUST be the very first code in the file
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import logging

# --- PATCH: Fix "Connection Reset" / Crashes ---
# 1. Disable Tokenizer Parallelism (prevents thread deadlocks/crashes)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 2. Silence Warnings
logging.getLogger('transformers').setLevel(logging.ERROR)

from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
IMAGE_FOLDER = './images'
DB_PATH = './chroma_db'
COLLECTION_NAME = 'photos'

# Ensure image directory exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# --- 1. LOAD MODEL (Cached) ---
@st.cache_resource
def load_model():
    # Removed use_fast=True to avoid the warning/error
    return SentenceTransformer('clip-ViT-B-32')

# --- 2. DATABASE SETUP ---
def get_chroma_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection

# --- 3. INDEXING LOGIC ---
def index_images(model, collection):
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.warning(f"No images found in {IMAGE_FOLDER}.")
        return

    existing_ids = collection.get()['ids']
    new_images = []
    new_ids = []
    
    for img_file in image_files:
        if img_file not in existing_ids:
            try:
                image_path = os.path.join(IMAGE_FOLDER, img_file)
                image = Image.open(image_path)
                new_images.append(image.convert("RGB")) 
                new_ids.append(img_file)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")

    if new_images:
        with st.spinner(f"Indexing {len(new_images)} new images..."):
            # Batch processing to prevent Memory Crashes
            batch_size = 50
            for i in range(0, len(new_images), batch_size):
                batch_imgs = new_images[i : i + batch_size]
                batch_ids = new_ids[i : i + batch_size]
                
                embeddings = model.encode(batch_imgs)
                collection.add(
                    embeddings=embeddings.tolist(),
                    ids=batch_ids,
                    metadatas=[{"filename": f} for f in batch_ids]
                )
        st.success(f"Indexed {len(new_images)} new images!")
    else:
        st.info("No new images to index.")

# --- 4. MAIN UI ---
def main():
    st.title("📸 SmartLens Search")

    model = load_model()
    collection = get_chroma_collection()

    with st.sidebar:
        st.header("Controls")
        # MOVED indexing behind this button to stop startup crashes
        if st.button("Scan & Index Images"):
            index_images(model, collection)
        
        count = len(os.listdir(IMAGE_FOLDER))
        st.write(f"Images in folder: {count}")

    # Search Interface
    query = st.text_input("Search:", placeholder="e.g., 'cat sleeping'")

    if query:
        query_embedding = model.encode([query])
        results = collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=3
        )
        
        if results['ids'] and results['ids'][0]:
            cols = st.columns(3)
            for idx, file_id in enumerate(results['ids'][0]):
                image_path = os.path.join(IMAGE_FOLDER, file_id)
                if os.path.exists(image_path):
                    with cols[idx]:
                        st.image(image_path, caption=file_id)
        else:
            st.info("No matches found. Did you click 'Scan & Index'?")

if __name__ == "__main__":
    main()