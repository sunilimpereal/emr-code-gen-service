import os
import time
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from config import section_config

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def load_embeddings_and_index(embedding_file):
    embeddings = np.load(embedding_file).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return embeddings, index

def predict_code(input_text, df, index):
    input_embedding = model.encode(input_text, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(input_embedding.reshape(1, -1))
    distances, indices = index.search(input_embedding.reshape(1, -1), 1)
    best_idx = int(indices[0][0])  # Cast to native Python int
    return {
        "predicted_code": str(df.iloc[best_idx]['code']),  # Ensure it's string or cast to int if needed
        "matched_description": str(df.iloc[best_idx]['description']),
        "similarity_score": float(distances[0][0])  # Cast to native Python float
    }
    

def process_input(data: dict):
    result = {}

    for section, config in section_config.items():
        print(f"\n--- Processing: {section} ---")
        field = config["field"]
        section_results = []

        # Load section-specific CSV and embeddings
        start = time.time()
        df = pd.read_csv(config["csv_file"])
        embeddings = np.load(config["embedding_file"]).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        print(f"Loaded and indexed {section} in {time.time() - start:.2f} seconds")

        items = data.get(section, [])
        for i, item in enumerate(items):
            input_text = item.get(field, "").strip()
            updated_item = item.copy()  # start with a copy of the original item

            if input_text:
                prediction = predict_code(input_text, df, index)

                # Add predictions to the item
                updated_item["code"] = prediction["predicted_code"]
                updated_item["matched_description"] = prediction["matched_description"]
                updated_item["similarity_score"] = prediction["similarity_score"]
                updated_item["input"] = input_text

                print(f"\nItem {i+1}: {input_text}")
                print(f"Predicted Code: {prediction['predicted_code']}")
                print(f"Matched Description: {prediction['matched_description']}")
                print(f"Similarity Score: {prediction['similarity_score']:.4f}")
            else:
                updated_item["error"] = f"No input found for field '{field}'"
                updated_item["input"] = ""

                print(f"\nItem {i+1}: No input found.")

            section_results.append(updated_item)

        result[section] = section_results

    return result
