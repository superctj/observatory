import pickle


def save_embeddings_to_pickle(file_path: str, embeddings):
    with open(file_path, "wb") as f:
        pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("=" * 50)
    print(f"Saved embeddings to: {file_path}")