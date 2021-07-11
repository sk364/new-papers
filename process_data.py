import sys
import numpy as np
import pandas as pd
import joblib

from simpletransformers.language_representation import RepresentationModel
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances


SIMILARITY_MATRIX_SAVE_PATH = "models/similarities.pkl"
SAVED_DF_PATH = "models/df.json"


def read_data(data_file_path):
  return pd.read_json(data_file_path, orient="index")


def get_abstracts(df):
  return df['abstract']


def init_model():
  return RepresentationModel(
    model_type="bert",
    model_name="bert-base-uncased",
    use_cuda=False
  )


def compute_sentence_embeddings(model, sentences):
  return model.encode_sentences(sentences, combine_strategy="mean")


def most_similar(doc_id, similarity_matrix):
  return np.argsort(similarity_matrix[doc_id])[::-1]


if __name__ == "__main__":
  args = sys.argv

  operation = "process" if len(args) < 2 else "most_similar"
  df_path = SAVED_DF_PATH if len(args) < 3 else args[2]


  if operation == "process":
    print(f"[INFO] reading data from {df_path}")
    df = read_data(df_path)
    abstracts = get_abstracts(df)
    model = init_model()

    print("[INFO] computing embeddings...")
    sentence_embeddings = compute_sentence_embeddings(model, abstracts)

    print("[INFO] computing similarities...")
    similarities = cosine_similarity(sentence_embeddings)

    print("[INFO] saving similarity matrix...")
    joblib.dump(similarities, SIMILARITY_MATRIX_SAVE_PATH)
  elif operation == "most_similar":
    document_id = 0 if len(args) < 4 else int(args[3])
    print(f"[INFO] processing document #{document_id}")

    print(f"[INFO] reading data from {df_path}")
    df = read_data(df_path)
    model = init_model()
    document = df[df.index == document_id]
    abstract = document['abstract'][0]

    abstract_embedding = compute_sentence_embeddings(model, [abstract])[0]

    print("[INFO] finding similar documents...")
    similarity_matrix = joblib.load(SIMILARITY_MATRIX_SAVE_PATH)
    most_similar_documents = most_similar(document_id, similarity_matrix)

    print(f'[INFO] found {len(most_similar_documents)} documents:')
    print(", ".join([str(doc_id) for doc_id in most_similar_documents]))
