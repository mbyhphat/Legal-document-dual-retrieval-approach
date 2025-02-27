import pandas as pd
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from underthesea import word_tokenize
import torch
from tqdm import tqdm
import sys
import pickle
from bi_encoder import  get_null_documents, get_sentence_embeddings

def read_data(corpus_path, train_path):
    corpus = pd.read_csv(corpus_path)
    train = pd.read_csv(train_path, on_bad_lines='skip')

    # Replace nan qid = 0
    nan_qid = train[train['qid'].isna() == True].index
    train.loc[nan_qid[0], 'qid'] = 0
    train['qid'] = train['qid'].astype(int)

    # Convert cid to list format for iterable
    train['list_cid'] = train['cid'].apply(lambda x: x.strip('[]').split()).tolist()
    train['list_cid'] = train['list_cid'].apply(lambda x: [int(i) for i in x])

    # Extract necessary data
    documents_dir = dict(zip(corpus['cid'], corpus['text']))
    queries_dir = dict(zip(train['qid'], train['question']))
    queries_doc = dict(zip(train['qid'], train['list_cid']))
    queries_map = dict(zip(train['question'], train['qid']))

    documents_id = list(documents_dir.keys())
    documents_list = list(documents_dir.values())
    queries_id = list(queries_dir.keys())
    queries_list = list(queries_dir.values())

    return documents_dir, queries_dir, queries_doc, queries_map, documents_id, documents_list, queries_id, queries_list

def remove_null_elements(queries_dir, queries_doc, documents_id, documents_list, queries_id, queries_list, documents_tokenized_sentence):
    null_documents_idx = get_null_documents(documents_tokenized_sentence)
    for i, null_idx in enumerate(null_documents_idx):
        documents_list.pop(null_idx - i)
        documents_id.pop(null_idx - i)
    
    null_queries_id = []
    null_queries_idx = []
    for idx, (query, doc) in enumerate(queries_doc.items()):
        if any(item in null_documents_idx for item in doc):
            null_queries_id.append(query)
            null_queries_idx.append(idx)

    for i, null_query_idx in enumerate(null_queries_idx):
        queries_id.pop(null_query_idx - i)
        queries_list.pop(null_query_idx - i)

    for null_query_id in null_queries_id:
        queries_doc.pop(null_query_id)
        queries_dir.pop(null_query_id)
    
    return queries_dir, queries_doc, documents_id, documents_list, queries_id, queries_list

def transform_tokenized_to_vncorenlp_form(tokenized_string):
    transformed_list = [token.replace(" ", "_") for token in tokenized_string]
    final_string = " ".join(transformed_list)
    return final_string

def cross_encoder_on_documents(bi_encoder_model, rerank_model, queries_list, documents_tokenized_sentence, documents_id, queries_doc, queries_map):
    sentence_embeddings = get_sentence_embeddings()
    batch_size = 32
    query_batches = [queries_list[i:i + batch_size] for i in range(0, len(queries_list), batch_size)]
    top_k = 50

    true_position_all_queries = []

    print("----- Cross-encoder ------")

    for query_batch in tqdm(query_batches):
        tokenized_queries = [word_tokenize(query) for query in query_batch]
        transformed_queries = [transform_tokenized_to_vncorenlp_form(tokenized_query) for tokenized_query in tokenized_queries]
        query_embeddings = bi_encoder_model.encode(transformed_queries, batch_size=batch_size)

        cosine_scores_batch = util.cos_sim(query_embeddings, sentence_embeddings)

        for idx, query in enumerate(query_batch):
            cosine_scores = cosine_scores_batch[idx]
            top_results = torch.topk(cosine_scores, k=top_k)

            retrieved_candidates = {}
            for score, index in zip(top_results.values, top_results.indices):
                doc = documents_tokenized_sentence[index]
                retrieved_candidates[index] = [([query], doc)]

            tokenized_pairs = {}
            for i, candidates in retrieved_candidates.items():
                tokenized_pairs[i] = [
                [transformed_queries[idx], sentence]
                for pair in candidates
                for sentence in pair[1]
                ]

            final_scores = {}
            for idx, tokenized_pair in tokenized_pairs.items():
                rerank_model.model.half()
                scores = rerank_model.predict(tokenized_pair)
                final_scores[idx.item()] = np.mean(scores)

            sorted_scores_with_indices = sorted(enumerate(final_scores.items()), reverse=True, key=lambda x: x[1][1])
            sorted_indices = [index for index, value in sorted_scores_with_indices]

            true_position_per_query = []
            for i in range(10):
                document_idx = sorted_scores_with_indices[i][1][0]
                score = sorted_scores_with_indices[i][1][1]

                if documents_id[document_idx] in queries_doc[queries_map[query]]:
                    true_position_per_query.append(i + 1)

            true_position_all_queries.append(true_position_per_query)

    return true_position_all_queries

def calculate_mrr_at_10(true_position_all_queries, k=10):
    """
    Tính toán MRR@10 cho một tập các truy vấn.

    Args:
    - true_relevant_positions (list): Danh sách các vị trí của tài liệu đúng trong top-k kết quả.
      Mỗi phần tử trong danh sách là vị trí của kết quả đúng (bắt đầu từ 1).
    - k (int): Số lượng kết quả cần đánh giá, mặc định là 10.

    Returns:
    - float: Giá trị MRR@10.
    """
    num_queries = len(true_position_all_queries)
    mrr_total = 0.0
    for true_position_per_query in true_position_all_queries:
        num_true_queries = len(true_position_per_query)
        mrr_query = 0.0
        for position in true_position_per_query:
            if position <= k:
                mrr_query += 1 / position
        if num_true_queries != 0:
            mrr_total += mrr_query / num_true_queries
        else:
            mrr_total += 0.0
        
    return mrr_total/num_queries

def main():
    print("----- Load bi-encoder model ------")
    bi_encoder = SentenceTransformer("bkai-foundation-models/vietnamese-bi-encoder")
    print("----- Load cross-encoder model ------")
    cross_encoder = CrossEncoder("itdainb/PhoRanker", max_length=256)

    print("----- Load data ------")
    corpus_path = "data/corpus.csv"
    train_path = "data/train.csv"
    with open('data/sentence_tokenized_final.pkl', 'rb') as file:
        documents_tokenized_sentence = pickle.load(file)

    documents_dir, queries_dir, queries_doc, queries_map, documents_id, documents_list, queries_id, queries_list = read_data(corpus_path, train_path)
    queries_dir, queries_doc, documents_id, documents_list, queries_id, queries_list = remove_null_elements(queries_dir, queries_doc, documents_id, documents_list, queries_id, queries_list, documents_tokenized_sentence)

    print("----- Retrieval ------")
    true_position_all_queries = cross_encoder_on_documents(bi_encoder, cross_encoder, queries_list, documents_tokenized_sentence, documents_id, queries_doc, queries_map)

    mrr_score = calculate_mrr_at_10(true_position_all_queries)
    return mrr_score

if __name__=="__main__":
    main()