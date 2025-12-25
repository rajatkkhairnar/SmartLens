import streamlit as st
import os
from PIL import Image
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# --- CONFIGURATION ---
IMAGE_FOLDER = './images'
DB_PATH = './chroma_db'
COLLECTION_NAME = 'personal_photos'

# Ensure image directory exists
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# --- RECRUITER NOTES: CONCEPTS ---
# 1. Vector Embeddings: 
#    Computers don't understand images or text directly; they understand numbers.
#    An "embedding" is a list of floating-point numbers (a vector) that represents 
#    the semantic meaning of the data. Similar images will have mathematically 
#    similar lists of numbers.
#
# 2. Why CLIP (Contrastive Language-Image Pre-Training)?
#    Standard models work on either Text OR Images. CLIP is "Multimodal."
#    It maps both text and images into the SAME vector space. This allows us 
#    to compare a text query ("dog on beach") directly with an image 
#    using cosine similarity.

# --- 1. LOAD MODEL (Cached) ---
@st.cache_resource
def load_model():
    """
    Loads the CLIP model. Cached so we don't reload it 
    every time the user interacts with the UI.
    """
    return SentenceTransformer('clip-ViT-B-32')

# --- 2. DATABASE SETUP ---
def get_chroma_collection():
    """
    Initializes a persistent ChromaDB client.
    """
    client = chromadb.PersistentClient(path=DB_PATH)
    # Get or create the collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # We use the default L2 (Euclidean) distance, but Cosine is also popular for CLIP
        metadata={"hnsw:space": "cosine"} 
    )
    return collection

# --- 3. INDEXING LOGIC ---
def index_images(model, collection):
    """
    Scans the ./images folder, generates embeddings, and saves them to ChromaDB.
    """
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.warning(f"No images found in {IMAGE_FOLDER}. Please add some photos!")
        return

    # Check which images are already indexed to avoid re-processing
    existing_ids = collection.get()['ids']
    
    new_images = []
    new_ids = []
    
    for img_file in image_files:
        if img_file not in existing_ids:
            try:
                # Open image
                image_path = os.path.join(IMAGE_FOLDER, img_file)
                image = Image.open(image_path)
                
                # Convert to RGB to avoid alpha channel issues
                new_images.append(image.convert("RGB")) 
                new_ids.append(img_file) # Use filename as ID
            except Exception as e:
                print(f"Error loading {img_file}: {e}")

    if new_images:
        with st.spinner(f"Indexing {len(new_images)} new images..."):
            # Generate Embeddings (The "Magic" part)
            embeddings = model.encode(new_images)
            
            # Add to ChromaDB
            collection.add(
                embeddings=embeddings.tolist(),
                ids=new_ids,
                metadatas=[{"filename": f} for f in new_ids]
            )
        st.success(f"Successfully indexed {len(new_images)} new images!")
    else:
        pass # No new images to index

# --- 4. MAIN UI ---
def main():
    st.title("📸 SmartLens: Semantic Image Search")
    st.markdown("Search your local photos using **natural language**.")

    # Load resources
    model = load_model()
    collection = get_chroma_collection()

    # Sidebar for controls
    with st.sidebar:
        st.header("Settings")
        if st.button("Re-Index Image Folder"):
            index_images(model, collection)
        st.write(f"Images in folder: {len(os.listdir(IMAGE_FOLDER))}")

    # Run indexing on startup (optional, keeps DB fresh)
    index_images(model, collection)

    # Search Interface
    query = st.text_input("What are you looking for?", placeholder="e.g., 'receipts for coffee' or 'cat sleeping'")

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
            
            cols = st.columns(3)
            for idx, file_id in enumerate(results['ids'][0]):
                image_path = os.path.join(IMAGE_FOLDER, file_id)
                distance = results['distances'][0][idx]
                
                if os.path.exists(image_path):
                    with cols[idx]:
                        st.image(image_path, caption=f"{file_id}\n(Score: {distance:.4f})")
                else:
                    st.error(f"Image {file_id} not found on disk.")
        else:
            st.info("No matching images found.")

if __name__ == "__main__":
    main()