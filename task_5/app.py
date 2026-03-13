# app.py
import os
import json
import re
import math
from collections import Counter, defaultdict
from flask import Flask, render_template, request, jsonify

import pymorphy3

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "task_3", "inverted_index.json")
TF_IDF_FOLDER = os.path.join(BASE_DIR, "..", "task_4", "tf_idf")
PAGES_COUNT = 100

morph = pymorphy3.MorphAnalyzer()

INDEX = None
LEMMA_VECTORS = None
LEMMA_IDF = None
DOC_TERM_COUNTS = None 


def get_inverted_index():
    """Загрузка инвертированного индекса"""
    global INDEX
    if INDEX is not None:
        return INDEX
    
    INDEX = defaultdict(set)
    
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
            
            for lemma, docs in data.items():
                for doc in docs:
                    if isinstance(doc, str):
                        try:
                            doc_num = int(doc)
                            INDEX[lemma].add(doc_num)
                        except ValueError:
                            INDEX[lemma].add(doc)
                    else:
                        INDEX[lemma].add(doc)
        
        print(f"Загружен инвертированный индекс: {len(INDEX)} лемм")
        return INDEX
        
    except Exception as e:
        print(f"Ошибка загрузки индекса: {e}")
        return defaultdict(set)


def load_tf_idf():
    """Загрузка TF-IDF векторов для документов"""
    global LEMMA_VECTORS, LEMMA_IDF, DOC_TERM_COUNTS
    
    if LEMMA_VECTORS is not None and LEMMA_IDF is not None:
        return LEMMA_VECTORS, LEMMA_IDF
    
    LEMMA_VECTORS = defaultdict(dict)
    LEMMA_IDF = {}
    DOC_TERM_COUNTS = {}
    
    loaded_files = 0
    
    for i in range(1, PAGES_COUNT + 1):
        doc_num = f"{i:03d}"
        
        lemma_path = os.path.join(TF_IDF_FOLDER, "lemmas", f"tf_{doc_num}.txt")
        
        if os.path.exists(lemma_path):
            try:
                term_count = 0
                with open(lemma_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        parts = line.split()
                        if len(parts) == 3:
                            term, idf_str, tfidf_str = parts
                            try:
                                tfidf = float(tfidf_str)
                                idf = float(idf_str)
                                
                                LEMMA_VECTORS[i][term] = tfidf
                                term_count += 1
                                
                                if term not in LEMMA_IDF:
                                    LEMMA_IDF[term] = idf
                                    
                            except ValueError:
                                continue
                
                DOC_TERM_COUNTS[i] = term_count
                loaded_files += 1
                    
            except Exception as e:
                print(f"Ошибка загрузки {lemma_path}: {e}")
    
    print(f"Загружено документов: {loaded_files}")
    print(f"Уникальных лемм: {len(LEMMA_IDF)}")
    
    return LEMMA_VECTORS, LEMMA_IDF


def cosine_similarity(vec1, vec2):
    """Вычисление косинусного сходства между векторами"""
    common_terms = set(vec1.keys()) & set(vec2.keys())
    
    if not common_terms:
        return 0.0
    
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
    
    norm1 = math.sqrt(sum(w * w for w in vec1.values()))
    norm2 = math.sqrt(sum(w * w for w in vec2.values()))
    
    if norm1 > 0 and norm2 > 0:
        return dot_product / (norm1 * norm2)
    return 0.0


def lemmatize_query(query):
    """Лемматизация запроса"""
    tokens = re.findall(r'\b[а-яa-z0-9]+\b', query.lower())
    
    lemmas = []
    for token in tokens:
        try:
            lemma = morph.parse(token)[0].normal_form
            lemmas.append(lemma)
        except Exception:
            lemmas.append(token)
    
    stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'а', 'но', 'или', 'не', 
                  'у', 'к', 'о', 'об', 'от', 'из', 'за', 'над', 'под', 'как',
                  'весь', 'этот', 'быть', 'что', 'это', 'то', 'который', 'так',
                  'же', 'такой', 'свой', 'его', 'ее', 'их', 'ты', 'вы', 'мы'}
    
    lemmas = [l for l in lemmas if l not in stop_words and len(l) > 1]
    
    return lemmas


def query_to_vector(lemmas, idf_dict):
    """Преобразование запроса в TF-IDF вектор"""
    if not lemmas:
        return {}
    
    tf = Counter(lemmas)
    total_terms = len(lemmas)
    
    query_vec = {}
    for lemma, count in tf.items():
        if lemma in idf_dict:
            tf_val = count / total_terms
            idf_val = idf_dict[lemma]
            query_vec[lemma] = tf_val * idf_val
        else:
            pass
    
    return query_vec


def vector_search(query, lemma_vectors, lemma_idf, index, top_k=10):
    """Векторный поиск по запросу"""
    print(f"Поиск по запросу: '{query}'")
    
    query_lemmas = lemmatize_query(query)
    print(f"Леммы запроса: {query_lemmas}")
    
    if not query_lemmas:
        return [], [], []
    
    candidates = set()
    query_terms_info = []
    
    for lemma in query_lemmas:
        if lemma in index:
            doc_count = len(index[lemma])
            candidates.update(index[lemma])
            query_terms_info.append({
                'lemma': lemma,
                'found': True,
                'doc_count': doc_count,
                'idf': lemma_idf.get(lemma, 0)
            })
        else:
            query_terms_info.append({
                'lemma': lemma,
                'found': False,
                'doc_count': 0,
                'idf': 0
            })
    
    if not candidates:
        return [], query_lemmas, query_terms_info
    
    query_vec = query_to_vector(query_lemmas, lemma_idf)
    
    if not query_vec:
        return [], query_lemmas, query_terms_info
    
    scores = []
    for doc_id in candidates:
        if doc_id in lemma_vectors:
            doc_vec = lemma_vectors[doc_id]
            similarity = cosine_similarity(query_vec, doc_vec)
            
            if similarity > 0:
                # Проверяем, сколько терминов запроса есть в документе
                matched_terms = sum(1 for term in query_vec if term in doc_vec)
                term_coverage = matched_terms / len(query_vec) if query_vec else 0
                
                similarity = similarity * (0.8 + 0.4 * term_coverage)
                
                scores.append((doc_id, similarity, matched_terms))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    for i, (doc_id, score, matched_terms) in enumerate(scores[:top_k]):
        doc_num = f"{doc_id:03d}"
        
        doc_top_terms = []
        if doc_id in lemma_vectors:
            doc_vec = lemma_vectors[doc_id]
            doc_top_terms = sorted(doc_vec.items(), key=lambda x: x[1], reverse=True)[:5]
        
        results.append({
            'rank': i + 1,
            'doc_id': doc_id,
            'doc_num': doc_num,
            'score': round(score, 6),
            'matched_terms': matched_terms,
            'total_terms': len(query_vec),
            'coverage': f"{matched_terms}/{len(query_vec)}",
            'top_terms': [{'term': t, 'weight': round(w, 4)} for t, w in doc_top_terms]
        })
    
    return results, query_lemmas, query_terms_info


print("Загрузка поисковой системы...")

inverted_index = get_inverted_index()
lemma_vectors, lemma_idf = load_tf_idf()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    """Обработка поискового запроса"""
    query = request.form.get('query', '').strip()
    
    if not query:
        return jsonify({'error': 'Пустой запрос'})
    
    try:
        results, query_lemmas, query_terms_info = vector_search(
            query, lemma_vectors, lemma_idf, inverted_index
        )
        
        return jsonify({
            'success': True,
            'query': query,
            'query_lemmas': query_lemmas,
            'query_terms': query_terms_info,
            'results': results,
            'total_results': len(results)
        })
        
    except Exception as e:
        print(f"Ошибка поиска: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route('/document/<int:doc_id>')
def get_document(doc_id):
    """Получение информации о документе"""
    doc_num = f"{doc_id:03d}"
    
    if doc_id not in lemma_vectors:
        return jsonify({'error': 'Документ не найден'})
    
    doc_vec = lemma_vectors[doc_id]
    top_terms = sorted(doc_vec.items(), key=lambda x: x[1], reverse=True)[:20]
    
    doc_text = None
    doc_path = os.path.join(BASE_DIR, "..", "task_1", "clean", f"{doc_num}.txt")
    if os.path.exists(doc_path):
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc_text = content[:500] + "..." if len(content) > 500 else content
        except Exception as e:
            print(f"Ошибка загрузки текста документа: {e}")
    
    return jsonify({
        'doc_id': doc_id,
        'doc_num': doc_num,
        'term_count': DOC_TERM_COUNTS.get(doc_id, 0) if DOC_TERM_COUNTS else 0,
        'top_terms': [{'term': t, 'weight': round(w, 4)} for t, w in top_terms],
        'preview': doc_text
    })


@app.route('/stats')
def get_stats():
    """Статистика системы"""
    if lemma_vectors and DOC_TERM_COUNTS:
        avg_length = round(sum(DOC_TERM_COUNTS.values()) / len(lemma_vectors), 1)
    else:
        avg_length = 0
    
    return jsonify({
        'total_docs': len(lemma_vectors),
        'unique_lemmas': len(lemma_idf),
        'avg_doc_length': avg_length,
        'index_size': len(inverted_index)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
