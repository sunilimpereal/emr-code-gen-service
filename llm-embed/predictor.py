import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time

# Mapping for each section to its corresponding CSV and embedding file
section_config = {
    "Complaints": {
        "field": "complaint",
        "csv_file": "csvs/Complaint_Master_DDCOMPLAINT_202406251104.csv",
        "embedding_file": "embedding/Complaint_Master_DDCOMPLAINT_202406251104_embeddings.npy"
    },
    "investigations": {
        "field": "investigation",
        "csv_file": "csvs/Investigation_Master_LABTESTMAST_202406251106.csv",
        "embedding_file": "embedding/Investigation_Master_LABTESTMAST_202406251106_embeddings.npy"
    },
    "MedicalAdvice": {
        "field": "medication_advice",
        "csv_file": "csvs/Drug_Material_Master_PHMTRLMST_202406251103.csv",
        "embedding_file": "embedding/Drug_Material_Master_PHMTRLMST_202406251103_embeddings.npy"
    }
}

json_file = "data.json"

start_total = time.time()

# Load model
start = time.time()
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print(f"Loaded model in {time.time() - start:.2f} seconds")

# Load input JSON
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Function to predict code using FAISS
def predict_code(input_text, df, index):
    input_embedding = model.encode(input_text, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(input_embedding.reshape(1, -1))
    distances, indices = index.search(input_embedding.reshape(1, -1), 1)
    best_idx = indices[0][0]
    return {
        "predicted_code": df.iloc[best_idx]['code'],
        "matched_description": df.iloc[best_idx]['description'],
        "similarity_score": distances[0][0]
    }

# Process each section
for section, config in section_config.items():
    print(f"\n--- Processing: {section} ---")
    field = config["field"]

    # Load section-specific CSV and embedding
    start = time.time()
    df = pd.read_csv(config["csv_file"])
    embeddings = np.load(config["embedding_file"]).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print(f"Loaded and indexed {section} in {time.time() - start:.2f} seconds")

    items = data.get(section, [])
    for i, item in enumerate(items):
        text = item.get(field, "").strip()
        if text:
            result = predict_code(text, df, index)
            print(f"\nItem {i+1}: {text}")
            print(f"Predicted Code: {result['predicted_code']}")
            print(f"Matched Description: {result['matched_description']}")
            print(f"Similarity Score: {result['similarity_score']:.4f}")
        else:
            print(f"\nItem {i+1}: No input found.")

print(f"\n✅ Total processing time: {time.time() - start_total:.2f} seconds")
