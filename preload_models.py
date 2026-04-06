import os


def main() -> None:
    model_name = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return

    try:
        SentenceTransformer(model_name, cache_folder=os.getenv("SENTENCE_TRANSFORMERS_HOME"))
        print(f"Preloaded embedding model: {model_name}")
    except Exception as exc:
        print(f"Skipping model preload for {model_name}: {exc}")


if __name__ == "__main__":
    main()
