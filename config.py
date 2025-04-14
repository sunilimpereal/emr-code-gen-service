# Common paths
base_csv_path = "llm-embed/csvs/"
base_embedding_path = "llm-embed/embedding/"

# Section-specific configuration
section_config = {
    "Complaints": {
        "field": "complaint",
        "csv_file": f"{base_csv_path}Complaint_Master_DDCOMPLAINT_202406251104.csv",
        "embedding_file": f"{base_embedding_path}Complaint_Master_DDCOMPLAINT_202406251104_embeddings.npy"
    },
    "investigations": {
        "field": "investigation",
        "csv_file": f"{base_csv_path}Investigation_Master_LABTESTMAST_202406251106.csv",
        "embedding_file": f"{base_embedding_path}Investigation_Master_LABTESTMAST_202406251106_embeddings.npy"
    },
    "MedicalAdvice": {
        "field": "medication_advice",
        "csv_file": f"{base_csv_path}Drug_Material_Master_PHMTRLMST_202406251103.csv",
        "embedding_file": f"{base_embedding_path}Drug_Material_Master_PHMTRLMST_202406251103_embeddings.npy"
    },
     "medical_history": {
        "field": "medical_history",
        "csv_file": f"{base_csv_path}Drug_Material_Master_PHMTRLMST_202406251103.csv",
        "embedding_file": f"{base_embedding_path}Drug_Material_Master_PHMTRLMST_202406251103_embeddings.npy"
    }
    
}
