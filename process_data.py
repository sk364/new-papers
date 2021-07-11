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


def get_most_similar_docs(abstract):
  """
  Returns the most similar documents using cosine similarity of the saved abstract embeddings
  and the given `abstract`

  Args:
    - abstract: `str`

  Returns:
    - data frame of most similar documents: `pd.DataFrame`
  """

  df = read_data(SAVED_DF_PATH)
  model = init_model()
  abstract_embedding = compute_sentence_embeddings(model, [abstract])[0]
  sentence_embeddings = joblib.load(SIMILARITY_MATRIX_SAVE_PATH)
  similarity_matrix = cosine_similarity(sentence_embeddings, [abstract_embedding])
  most_similar_documents = np.argsort(similarity_matrix.reshape(df.shape[0]))[::-1]

  return df[df.index.isin(most_similar_documents)]


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

    print("[INFO] saving sentence embeddings...")
    joblib.dump(sentence_embeddings, SIMILARITY_MATRIX_SAVE_PATH)
  elif operation == "most_similar":
    document_id = 0 if len(args) < 4 else int(args[3])
    print(f"[INFO] processing document #{document_id}")

    print(f"[INFO] reading data from {df_path}")
    df = read_data(df_path)
    model = init_model()
    document = df[df.index == document_id]
    abstract = document['abstract'][0]

    abstract_embedding = compute_sentence_embeddings(model, [abstract])[0]
    sentence_embeddings = joblib.load(SIMILARITY_MATRIX_SAVE_PATH)
    similarity_matrix = cosine_similarity(sentence_embeddings, [abstract_embedding])

    print("[INFO] finding similar documents...")
    most_similar_documents = np.argsort(similarity_matrix)[::-1]

    print(f'[INFO] found {len(most_similar_documents)} documents:')
    print(", ".join([str(doc_id) for doc_id in most_similar_documents]))
