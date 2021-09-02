import sys
import joblib

import pandas as pd
from sentence_transformers import SentenceTransformer, util

SIMILARITY_MATRIX_SAVE_PATH = "colab-files/embeddings-allenai-specter.pkl"
SAVED_DF_PATH = "models/df.json"
TOP_K = 15
MODEL_NAME = "allenai/specter"


def read_data(data_file_path):
  return pd.read_json(data_file_path, orient="index")


def get_title_abstracts(df):
  return [paper["title"] + "[SEP]" + paper["abstract"] for idx, paper in df.iterrows()]


def init_model():
  return SentenceTransformer(MODEL_NAME)


def compute_sentence_embeddings(model, sentences):
  return model.encode(sentences, convert_to_tensor=True)


def get_most_similar_docs(df, model, embeddings, abstract, top_k=10):
  query_embedding = model.encode(abstract, convert_to_tensor=True)
  search_hits = util.semantic_search(query_embedding, embeddings, top_k=top_k)[0]
  papers = []

  for hit in search_hits:
    paper = df.iloc[hit["corpus_id"], :]
    papers.append({
      "score": hit["score"],
      "id": paper["id"],
      "title": paper["title"],
      "abstract": paper["abstract"],
      # "general_category": paper["general_category"
    })

  return papers


if __name__ == "__main__":
  args = sys.argv

  operation = "process" if len(args) < 2 else "most_similar"
  df_path = SAVED_DF_PATH if len(args) < 3 else args[2]

  if operation == "process":
    print(f"[INFO] reading data from {df_path}")
    df = read_data(df_path)
    title_abstracts = get_title_abstracts(df)
    model = init_model()

    print("[INFO] computing embeddings...")
    sentence_embeddings = compute_sentence_embeddings(model, title_abstracts)

    print("[INFO] saving sentence embeddings...")
    joblib.dump(sentence_embeddings, SIMILARITY_MATRIX_SAVE_PATH)

  elif operation == "most_similar":
    print(f"[INFO] reading data from {df_path}")

    document_id = 0 if len(args) < 4 else int(args[3])

    print(f"[INFO] processing document #{document_id}")
    df = read_data(df_path)
    model = init_model()

    document = df[df.index == document_id]
    abstract = document["abstract"][document_id]

    sentence_embeddings = joblib.load(SIMILARITY_MATRIX_SAVE_PATH)
    similar_papers = get_most_similar_docs(df, model, sentence_embeddings, abstract, top_k=TOP_K)

    print(similar_papers)
