#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv
import json
import logging
import sys

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.phrases import Phraser
from gensim.similarities import SparseTermSimilarityMatrix
from tqdm import tqdm
from scipy.sparse import dok_matrix

from arqmath_eval import get_judged_documents

sys.path.append('scm_at_arqmath')

from scm_at_arqmath.scripts.common import read_corpora
from scm_at_arqmath.scripts.configuration import CSV_PARAMETERS, ARQMATH_TASK1_TEST_POSTS_NUM_DOCUMENTS as NUM_QUERIES, ARQMATH_COLLECTION_POSTS_NUM_DOCUMENTS as NUM_DOCUMENTS


LOGGER = logging.getLogger(__name__)


def read_input_data():
    input_data = {
        'dictionary': Dictionary.load('dictionary'),
        'tfidf_queries': TfidfModel.load('tfidf-queries'),
        'tfidf_documents': TfidfModel.load('tfidf-documents'),
        'word_similarities': dok_matrix(SparseTermSimilarityMatrix.load('word-similarities').matrix),
        'phraser': Phraser.load('phraser'),
        'results': read_results('results'),
    }
    return input_data


def read_results(filename, topn=5):
    results = dict()
    with open(filename, 'rt', newline='') as csv_file:
        csv_reader = csv.reader(csv_file, **CSV_PARAMETERS)
        for topic_id, post_id, rank, score, description in csv_reader:
            if topic_id not in results:
                results[topic_id] = []
            if len(results[topic_id]) < topn:
                results[topic_id].append(post_id)
    LOGGER.info('loaded search results from {}'.format(filename))
    return results


def get_reader_configuration(input_data):
    tfidf_queries = input_data['tfidf_queries']
    tfidf_documents = input_data['tfidf_documents']
    phraser = input_data['phraser']
    results = input_data['results']

    topic_ids = set(results.keys())
    document_ids = get_judged_documents(task='task1-votes.V1.2', subset='all')

    configuration = {
        'topic_corpus_filename': 'queries.json.gz',
        'topic_corpus_num_documents': NUM_QUERIES,
        'topic_ids': topic_ids,
        'topic_transformer': lambda topic: topic,
        'document_corpus_filename': 'documents.json.gz',
        'document_corpus_num_documents': NUM_DOCUMENTS,
        'document_ids': document_ids,
        'document_transformer': lambda document: document,
        'parallelize_transformers': False,
    }
    reader_kwargs = {
        'phraser': phraser,        
    }
    return (configuration, reader_kwargs)


def produce_json_document(input_data, corpora, f=sys.stdout):
    results = input_data['results']

    dictionary = input_data['dictionary']
    seen_text_ids = set().union(*map(set, results.values())).union(set(results.keys()))
    queries, documents = corpora
    texts = {
        text_id: [
            str(dictionary.token2id[token])
            for token
            in text
            if token in dictionary.token2id
        ]
        for (text_id, text)
        in {**documents, **queries}.items()
        if text_id in seen_text_ids
    }
    LOGGER.info('produced json_document["texts"]')

    tfidf_queries = input_data['tfidf_queries']
    tfidf_documents = input_data['tfidf_documents']
    texts_bow = {
        **{
            query_id: dict(tfidf_queries[dictionary.doc2bow(query)])
            for (query_id, query)
            in queries.items()
            if query_id in seen_text_ids
        }, **{
            document_id: dict(tfidf_documents[dictionary.doc2bow(document)])
            for (document_id, document)
            in documents.items()
            if document_id in seen_text_ids
        }
    }
    LOGGER.info('produced json_document["texts_bow"]')

    seen_terms = set().union(*map(set, queries.values())).union(*map(set, documents.values()))
    dictionary = {
        dictionary.token2id[term]: term
        for term
        in seen_terms
        if term in dictionary.token2id
    }
    LOGGER.info('produced json_document["dictionary"]')

    word_similarities = dict()
    term1_ids, term2_ids = input_data['word_similarities'].nonzero()
    term1_ids, term2_ids = map(int, term1_ids), map(int, term2_ids)
    for term1_id, term2_id in zip(term1_ids, term2_ids):
        if term1_id not in dictionary:
            continue
        if term2_id not in dictionary:
            continue
        if term1_id >= term2_id:
            continue
        if term1_id not in word_similarities:
            word_similarities[term1_id] = dict()
        word_similarity = input_data['word_similarities'][term1_id, term2_id]
        word_similarity = float(word_similarity)
        word_similarities[term1_id][term2_id] = word_similarity
    LOGGER.info('produced json_document["word_similarities"]')

    json_document = {
        'version': '1',
        'results': results,
        'texts': texts,
        'texts_bow': texts_bow,
        'dictionary': dictionary,
        'word_similarities': word_similarities,
    }

    json.dump(json_document, f, sort_keys=True, indent=4)
    


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    input_data = read_input_data()
    reader_configuration = get_reader_configuration(input_data)
    corpora = read_corpora(*reader_configuration)
    produce_json_document(input_data, corpora)
