import pickle 
from tqdm import tqdm
import sys
import numpy as np
from sentence_transformers import SentenceTransformer


def get_null_documents(documents_tokenized_sentence):
    null_documents_idx = []
    for idx, doc in enumerate(documents_tokenized_sentence):
        if not doc:
            null_documents_idx.append(idx)
    return null_documents_idx

def remove_null_documents(null_documents_idx, documents_tokenized_sentence):
    for i, null_idx in enumerate(null_documents_idx):
        documents_tokenized_sentence.pop(null_idx - i)

def bi_encoder_on_documents(model, documents_tokenized_sentence):
    sentence_embeddings = []

    print("----- Train bi-encoder")
    for doc_id, document in enumerate(tqdm(documents_tokenized_sentence)):
        batch_embeddings = model.encode(document)
        document_embedding = np.mean(batch_embeddings, axis=0)
        sentence_embeddings.append(document_embedding)

    return sentence_embeddings

def get_sentence_embeddings():
    with open('data/sentence_tokenized_final.pkl', 'rb') as file:
        documents_tokenized_sentence = pickle.load(file)

    null_documents_idx = get_null_documents(documents_tokenized_sentence)
    remove_null_documents(null_documents_idx, documents_tokenized_sentence)

    bi_encoder_model = SentenceTransformer('bkai-foundation-models/vietnamese-bi-encoder')
    sentence_embeddings = bi_encoder_on_documents(bi_encoder_model, documents_tokenized_sentence)
    return sentence_embeddings