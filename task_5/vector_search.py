import os
import json
import re
import math
from collections import Counter, defaultdict

import pymorphy3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(BASE_DIR, "..", "task_3", "inverted_index.json")
TF_IDF_FOLDER = os.path.join(BASE_DIR, "..", "task_4", "tf_idf")
PAGES_FOLDER = os.path.join(BASE_DIR, "..", "task_1", "clean")
PAGES_COUNT = 100

morph = pymorphy3.MorphAnalyzer()


def get_inverted_index():
    index = defaultdict(set)
    
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
            
            for lemma, docs in data.items():
                for doc in docs:
                    if isinstance(doc, str):
                        try:
                            doc_num = int(doc)
                            index[lemma].add(doc_num)
                        except ValueError:
                            index[lemma].add(doc)
                    else:
                        index[lemma].add(doc)
        
        print(f"Загружен инвертированный индекс из JSON: {len(index)} лемм")
        
        print("Примеры лемм из индекса:")
        for i, (lemma, docs) in enumerate(list(index.items())[:5]):
            print(f"  {lemma}: {sorted(docs)[:5]}")
        
        return index
        
    except FileNotFoundError:
        print(f"Файл индекса {INDEX_PATH} не найден")
        return defaultdict(set)
    except json.JSONDecodeError as e:
        print(f"Ошибка при парсинге JSON: {e}")
        return defaultdict(set)
    except Exception as e:
        print(f"Ошибка при загрузке индекса: {e}")
        return defaultdict(set)


def load_tf_idf():
    vectors = defaultdict(dict)
    idf_dict = {}

    loaded_files = 0
    
    for i in range(1, PAGES_COUNT + 1):
        doc_num = f"{i:03d}"
        
        possible_paths = [
            os.path.join(TF_IDF_FOLDER, "lemmas", f"tf_{doc_num}.txt"),
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if not file_path:
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
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
                            
                            vectors[i][term] = tfidf
                            
                            if term not in idf_dict:
                                idf_dict[term] = idf
                                
                        except ValueError:
                            continue
                        
            loaded_files += 1
        except Exception as e:
            print(f"Ошибка при загрузке {file_path}: {e}")
    
    print(f"Загружено TF-IDF векторов: {loaded_files} документов")
    print(f"Загружено уникальных лемм: {len(idf_dict)}")
    
    return vectors, idf_dict


def cosine_similarity(vec1, vec2):
    common_terms = set(vec1.keys()) & set(vec2.keys())
    
    if not common_terms:
        return 0.0
    
    # Скалярное произведение
    dot_product = sum(vec1[term] * vec2[term] for term in common_terms)
    
    # Нормы векторов
    norm1 = math.sqrt(sum(w * w for w in vec1.values()))
    norm2 = math.sqrt(sum(w * w for w in vec2.values()))
    
    if norm1 > 0 and norm2 > 0:
        return dot_product / (norm1 * norm2)
    return 0.0


def lemmatize_query(query):
    tokens = re.findall(r'\b[а-яa-z]+\b', query.lower())
    
    lemmas = []
    for token in tokens:
        try:
            lemma = morph.parse(token)[0].normal_form
            lemmas.append(lemma)
        except Exception:
            lemmas.append(token)
    
    stop_words = {'и', 'в', 'на', 'с', 'по', 'для', 'а', 'но', 'или', 'не', 
                  'у', 'к', 'о', 'об', 'от', 'из', 'за', 'над', 'под', 'как',
                  'весь', 'этот', 'быть'}
    lemmas = [l for l in lemmas if l not in stop_words and len(l) > 1]
    
    return lemmas


def query_to_vector(lemmas, idf_dict):
    if not lemmas:
        return {}
    
    # Подсчет частоты терминов в запросе
    tf = Counter(lemmas)
    total_terms = len(lemmas)
    
    query_vec = {}
    for lemma, count in tf.items():
        if lemma in idf_dict:
            tf_val = count / total_terms
            idf_val = idf_dict[lemma]
            # TF-IDF
            query_vec[lemma] = tf_val * idf_val
    
    return query_vec


def vector_search(query, lemma_vectors, lemma_idf, index, top_k=10):
    print(f"\nЗапрос: '{query}'")
    
    query_lemmas = lemmatize_query(query)
    print(f"Леммы запроса: {query_lemmas}")
    
    if not query_lemmas:
        print("Запрос не содержит значимых лемм")
        return []
    
    candidates = set()
    for lemma in query_lemmas:
        if lemma in index:
            candidates.update(index[lemma])
            print(f"  Лемма '{lemma}' найдена в {len(index[lemma])} документах")
        else:
            print(f"  Лемма '{lemma}' не найдена в индексе")
    
    if not candidates:
        print("Нет документов, содержащих леммы запроса")
        return []
    
    print(f"Найдено кандидатов: {len(candidates)}")
    
    query_vec = query_to_vector(query_lemmas, lemma_idf)
    
    if not query_vec:
        print("Не удалось создать вектор запроса")
        return []
    
    scores = {}
    for doc_id in candidates:
        if doc_id in lemma_vectors:
            doc_vec = lemma_vectors[doc_id]
            similarity = cosine_similarity(query_vec, doc_vec)
            if similarity > 0.0001:
                scores[doc_id] = similarity
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked[:top_k]


def print_results(results, query):
    if not results:
        print("\nДокументов не найдено")
        return
    
    print("\n" + "-"*10)
    print("РЕЗУЛЬТАТЫ ПОИСКА")
    print("-"*10)
    
    for i, (doc_id, score) in enumerate(results, 1):
        doc_num = f"{doc_id:03d}"
        print(f"{i:2d}. Документ {doc_num} | Релевантность: {score:.6f}")
    
    print("-"*10)
    
    if results:
        top_score = results[0][1]
        print(f"\nМаксимальная релевантность: {top_score:.6f}")
        
        # if top_score > 0.5:
        #     print("  ✓ Отличное совпадение")
        # elif top_score > 0.3:
        #     print("  ✓ Хорошее совпадение")
        # elif top_score > 0.15:
        #     print("  ✓ Среднее совпадение")
        # elif top_score > 0.05:
        #     print("  ⚠ Слабое совпадение")
        # else:
        #     print("  ⚠ Очень слабое совпадение")


def diagnose_term(term, index, idf_dict, lemma_vectors, top_n=5):
    print(f"\n=== ДИАГНОСТИКА ТЕРМИНА '{term}' ===")
    
    if term in index:
        docs = index[term]
        print(f"В индексе: ДА")
        print(f"Встречается в {len(docs)} документах")
        
        if term in idf_dict:
            idf = idf_dict[term]
            print(f"IDF = {idf:.4f}")
        
        print(f"\nТоп-{top_n} документов с наибольшим весом:")
        
        term_weights = []
        for doc_id in docs:
            if doc_id in lemma_vectors and term in lemma_vectors[doc_id]:
                weight = lemma_vectors[doc_id][term]
                term_weights.append((doc_id, weight))
        
        term_weights.sort(key=lambda x: x[1], reverse=True)
        
        for doc_id, weight in term_weights[:top_n]:
            doc_num = f"{doc_id:03d}"
            print(f"  Документ {doc_num}: {weight:.6f}")
    else:
        print(f"Термин '{term}' НЕ НАЙДЕН в индексе")



index = get_inverted_index()
lemma_vectors, lemma_idf = load_tf_idf()

print(f"Документов: {len(lemma_vectors)}")
print(f"Уникальных лемм: {len(lemma_idf)}")
print(f"Лемм в индексе: {len(index)}")

print("\n" + "-"*10)
print("ВЕКТОРНАЯ ПОИСКОВАЯ СИСТЕМА")
print("-"*10)
print("\nКоманды:")
print("  /stats - статистика")
print("  /diag <термин> - диагностика термина")
print("  /exit - выход")
print("-"*10)

while True:
    try:
        query = input("\nВведите запрос: ").strip().lower()
        
        if query in ['/exit', 'exit', '/quit', 'quit']:
            print("До свидания!")
            break
        
        if query == '/stats':
            print(f"\nСтатистика:")
            print(f"  Документов: {len(lemma_vectors)}")
            print(f"  Уникальных лемм: {len(lemma_idf)}")
            if lemma_vectors:
                avg_len = sum(len(v) for v in lemma_vectors.values()) / len(lemma_vectors)
                print(f"  Средняя длина документа: {avg_len:.1f} лемм")
            continue
        
        if query.startswith('/diag '):
            term = query[6:].strip()
            diagnose_term(term, index, lemma_idf, lemma_vectors)
            continue
        
        if not query:
            continue
        
        results = vector_search(query, lemma_vectors, lemma_idf, index)
        print_results(results, query)
        
        if results:
            print(f"\nДля просмотра деталей документа введите его номер (1-{len(results)})")
            choice = input("> ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(results):
                doc_id, score = results[int(choice)-1]
                doc_num = f"{doc_id:03d}"
                print(f"\nДетали документа {doc_num}:")
                print(f"Релевантность: {score:.6f}")
                
                if doc_id in lemma_vectors:
                    doc_vec = lemma_vectors[doc_id]
                    top_terms = sorted(doc_vec.items(), key=lambda x: x[1], reverse=True)[:5]
                    print("Топ-5 лемм в документе:")
                    for term, weight in top_terms:
                        print(f"  {term}: {weight:.6f}")
    
    except KeyboardInterrupt:
        print("\n\nПоиск прерван")
        break
    except Exception as e:
        print(f"\nОшибка: {e}")
        import traceback
        traceback.print_exc()
