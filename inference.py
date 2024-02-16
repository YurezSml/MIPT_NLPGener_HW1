import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModel
from gensim.models import Word2Vec

import pickle
import dill

import model
from model import CrossEncoderBert

import string

import os
import time
import random

import webbrowser
from flask import Flask, render_template, request, jsonify

MAX_LENGTH = 512
N_NEIGHBORS = 500
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"

# Загружаем модель и токенайзер с дополнительным токеном [U_token]

MODEL_FOLDER = 'model'

with open(os.path.join(MODEL_FOLDER, 'friends_model.pkl'), 'rb') as file:
    serialized = file.read()
model = dill.loads(serialized)
model.tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_FOLDER, 'tokenizer'))
model.to(device)

with open(os.path.join(MODEL_FOLDER, 'charact_corpus.pkl'), 'rb') as file:
    corpus = pickle.load(file)

# Удаляем знаки препинания из корпуса реплик для улучшения качества поиска ближайших соседей
fixed_corpus = [str(x).translate(str.maketrans('', '', string.punctuation)) for x in corpus]
splited_corpus = [str(x).split(' ') for x in fixed_corpus]
word2vec_model = Word2Vec(splited_corpus, min_count=1)

def get_vec_for_word(word: str):
    try:
        word_vec = model.wv[word]
        return word_vec
    except:
        return np.zeros((100,))

def get_query_as_vec(query: str):
    fixed_query = query.translate(str.maketrans('', '', string.punctuation))
    splited_question = fixed_query.split(' ')
    vec_quer = np.mean([get_vec_for_word(word) for word in splited_question], axis=0)
    vec_quer = np.expand_dims(vec_quer, axis=0)

vec_corp = np.array([np.mean([get_vec_for_word(word) for word in sent], axis=0) for sent in splited_corpus])

nbrs = NearestNeighbors(n_neighbors=N_NEIGHBORS, algorithm='ball_tree')
nbrs.fit(vec_corp)

def get_ranked_docs(model,
                    query: str,
                    corpus: list[str],
                    num_answers: int
) -> None:
    queries = [query] * len(corpus)
    tokenized_texts = model.tokenizer(
        queries, corpus, max_length=MAX_LENGTH, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        ce_scores = model(tokenized_texts['input_ids'], tokenized_texts['attention_mask']).squeeze(-1)
        ce_scores = torch.sigmoid(ce_scores)

    scores = ce_scores.cpu().numpy()
    scores_ix = np.argsort(scores)[::-1]
    best_answers = []
    for ix in scores_ix[:num_answers]:
        best_answers.append((scores[ix], corpus[ix]))
    
    return best_answers


def get_best_answer(query: str, type: str):
    answer = get_n_relevant_answers(query, type, 1)
    return answer[0]


def get_n_relevant_answers(query: str, type: str, num_answers: int):
    fixed_corpus = []
    if type == 'fast':
        vec_quer = get_query_as_vec(query)
        for neighb in nbrs.kneighbors(vec_quer)[1][0]:
            fixed_corpus.append(corpus[neighb])
    else:
        fixed_corpus = corpus
    idx = 0
    num_iter = len(fixed_corpus) // N_NEIGHBORS + (len(fixed_corpus) % N_NEIGHBORS != 0)
    print(f'num_iter: {num_iter}')
    n_relevant_answers_with_score = []
    while idx < num_iter:
        curr_corpus = []
        if (idx + 1) * N_NEIGHBORS > len(fixed_corpus):
            curr_corpus = fixed_corpus[idx*N_NEIGHBORS:]
        else:
            curr_corpus = fixed_corpus[idx*N_NEIGHBORS:(idx+1)*N_NEIGHBORS]
        relevant_answers = get_ranked_docs(model, query, curr_corpus, num_answers)
        n_relevant_answers_with_score = n_relevant_answers_with_score + relevant_answers
        n_relevant_answers_with_score = sorted(n_relevant_answers_with_score, key=lambda tup: tup[0])[:num_answers]
        idx += 1 
    n_relevant_answers = [str(x[1]) for x in n_relevant_answers_with_score]
    return n_relevant_answers

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/request', methods=['GET', 'POST'])
def request_answer():
    query = request.args.get('query')
    type = request.args.get('type')
    start_time = time.time()
    answer = get_best_answer(query, type)
    end_time = time.time()
    return jsonify({
        "response_code": "200",
        "request": query,
        "response": answer,
        "time": str(end_time - start_time),
    })


@app.route('/retrieve', methods=['GET', 'POST'])
def request_relevant_answer():
    query = request.args.get('query')
    type = request.args.get('type')
    num_answers = request.args.get('num_answers')    
    start_time = time.time()
    answers = get_n_relevant_answers(query, type, int(num_answers))
    end_time = time.time()
    return jsonify({
        "response_code": "200",
        "request": query,        
        "response": answers,
        "time": str(end_time - start_time),
    })


@app.route('/dialog', methods=['GET', 'POST'])
def request_dialog_answers():
    queries = []
    queries.append(request.args.get('query_1'))
    queries.append(request.args.get('query_2'))
    queries.append(request.args.get('query_3'))

    type = request.args.get('type')
    
    start_time = time.time()
    answers = []
    context_query = ''
    for query in queries:
        context_query = context_query + query
        answer = get_best_answer(context_query, type)
        answers.append(answer)
        context_query = context_query + ' [U_token] ' + answer + ' [U_token] '
    end_time = time.time()
    return jsonify({
        "response_code": "200",
        "requests": queries,
        "response": answers,
        "time": str(end_time - start_time),
    })


@app.route('/ping')
def request_ping():
    return jsonify({
        "status": 'working!'
    })


if __name__ == '__main__':
    webbrowser.open_new('http://127.0.0.1:5000')
    app.run('localhost', 5000)