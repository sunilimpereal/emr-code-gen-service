import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

# Check if your MPS backend is available (should output True)
print("MPS available:", torch.backends.mps.is_available())

# Set directories for CSVs and embeddings
csv_folder = "csvs"
embedding_folder = "embedding"

# Create the embeddings directory if it doesn't exist
if not os.path.exists(embedding_folder):
    os.makedirs(embedding_folder)

#tried models
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
# model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Load the embedding model - you can change the model name as desired
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Iterate over all CSV files in the csv_folder
for filename in os.listdir(csv_folder):
    if filename.lower().endswith('.csv'):
        csv_path = os.path.join(csv_folder, filename)
        print(f"Processing file: {csv_path}")
        
        # Load CSV and ensure 'description' column exists and is string type
        df = pd.read_csv(csv_path)
        if 'description' not in df.columns:
            print(f"Skipping {filename}: no 'description' column found.")
            continue
        
        df['description'] = df['description'].astype(str)
        
        # Generate embeddings for the description field
        embeddings = model.encode(df['description'].tolist(), convert_to_numpy=True)
        
        # Save the embeddings to .npy file in the embedding folder using a modified filename
        embedding_filename = os.path.splitext(filename)[0] + '_embeddings.npy'
        embedding_path = os.path.join(embedding_folder, embedding_filename)
        np.save(embedding_path, embeddings)
        print(f"Embeddings saved to: {embedding_path}")
        
        # Optional: If you want to save the CSV with embeddings appended or other modifications,
        # you can do so here (uncomment the following line to save a version of the CSV).
        # df.to_csv(os.path.join(embedding_folder, os.path.splitext(filename)[0] + '_embedded.csv'), index=False)
