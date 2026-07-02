from sentence_transformers import SentenceTransformer

def download_model():
    model_name = "all-MiniLM-L6-v2"
    save_path = "./local_model"
    print(f"Downloading {model_name} to {save_path}...")
    model = SentenceTransformer(model_name)
    model.save(save_path)
    print("Model downloaded and saved successfully!")

if __name__ == "__main__":
    download_model()
