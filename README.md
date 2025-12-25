# SmartLens: Semantic Image Search 📸

Hey there! Welcome to **SmartLens**, a fun little project I built to make searching through personal photos way easier. Ever wished you could just type "that time we went hiking" and instantly find the right pictures? Well, now you can! This app uses AI to understand natural language queries and match them to your images.

## What's This All About?

SmartLens is a Streamlit-based web app that lets you search your local photo collection using plain English descriptions. It combines the power of CLIP (a multimodal AI model) with ChromaDB (a vector database) to turn your text queries into image matches. No cloud uploads, no privacy worries – everything runs locally on your machine.

## Key Features

- **Natural Language Search**: Describe what you're looking for in everyday words, like "family picnic" or "sunset over mountains."
- **AI-Powered Matching**: Uses CLIP to create "embeddings" (think of them as smart fingerprints) for both images and text, then finds the best matches.
- **Local and Private**: Your photos never leave your computer. The database is stored locally too.
- **Easy Setup**: Just drop your images in a folder and run the app.
- **Fast and Efficient**: Indexes images once, then searches in seconds.

## How It Works (The Techy Bits)

1. **Indexing**: The app scans your `images/` folder, uses CLIP to generate vector embeddings for each photo, and stores them in ChromaDB.
2. **Searching**: When you type a query, it embeds your text and compares it to the image embeddings using cosine similarity.
3. **Results**: It shows the top matches with similarity scores.

It's like having a super-smart photo assistant!

## Prerequisites

- Python 3.8 or higher
- A bunch of images (PNG, JPG, or JPEG) in the `images/` folder
- That's it! No fancy hardware needed, though a GPU would make things faster if you have one and you've more than 100 images.

## Installation

1. **Clone or Download**: Grab this repo and unzip it somewhere convenient.
2. **Set Up a Virtual Environment** (recommended to keep things tidy):
   - `python -m venv venv`
   - Activate it: `venv\Scripts\activate` on Windows, or `source venv/bin/activate` on Mac/Linux.
3. **Install Dependencies**:
   - `pip install -r requirements.txt`
   - This will grab Streamlit, PIL, SentenceTransformers, and ChromaDB.
4. **Add Your Photos**: Toss some images into the `images/` folder. The app will handle the rest.

## Usage

1. **Run the App**:
   - `streamlit run app.py`
   - This should open your browser to `http://localhost:8501`.
2. **Index Your Images**: The app does this automatically on startup, but you can hit "Re-Index Image Folder" in the sidebar if you add new pics.
3. **Search Away**: Type what you're looking for in the search box. Try things like "dog playing in snow" or "birthday cake."
4. **View Results**: The top 3 matches appear with thumbnails and similarity scores.

Pro Tip: The more descriptive your query, the better the matches!

## Troubleshooting

- **No Images Found?** Double-check the `images/` folder has some photos and they're in supported formats.
- **App Won't Start?** Make sure your virtual environment is activated and all dependencies are installed.
- **Slow Indexing?** That's normal for the first run – CLIP is doing heavy lifting. Subsequent searches are quick.
- **Errors in Console?** Check for missing packages or file permission issues. Feel free to open an issue if you're stuck!

## Contributing

This was a personal project, but if you spot a bug or have an idea to make it better, hit me up! Pull requests welcome.

## License

MIT License – do whatever you want with it, just don't blame me if your cat photos get mixed up. 😄

---

Built with ❤️ using Streamlit, CLIP, and ChromaDB. Hope it brings some joy to your photo hunting!