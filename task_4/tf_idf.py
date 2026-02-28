import math
import os
import json
import traceback
from collections import defaultdict

N = 100
digits = 7
base = 10
INDEX = {}


def format_doc_number(num):
    return f"{num:03d}"

def solve_idf(doc_count):
    if doc_count == 0:
        return 0
    return round(math.log(N / doc_count, base), 4)

def parse_index(path="../task_3/inverted_index.json"):
    global INDEX
    try:
        with open(path, 'r', encoding='utf-8') as f:
            json_index = json.load(f)
            
            for term, docs in json_index.items():
                INDEX[term] = set()
                for doc in docs:
                    if isinstance(doc, str) and doc.startswith('page_'):
                        try:
                            doc_num = int(doc.replace('page_', ''))
                            INDEX[term].add(doc_num)
                        except ValueError:
                            INDEX[term].add(doc)
                    else:
                        try:
                            INDEX[term].add(int(doc))
                        except (ValueError, TypeError):
                            INDEX[term].add(doc)
            
        print(f"Загружено {len(INDEX)} терминов из JSON индекса")
        
    except FileNotFoundError:
        print(f"Файл индекса {path} не найден")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Ошибка при парсинге JSON: {e}")
        exit(1)

def parse_tokens(doc_num):
    formatted_num = format_doc_number(doc_num)
    
    possible_paths = [
        f"../task_2/tokens/{formatted_num}.txt",
    ]
    
    tokens = []
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f.readlines():
                        token = line.strip()
                        if token:
                            tokens.append(token)
                print(f"  Загружены токены из: {path} (найдено {len(tokens)} токенов)")
                return tokens
            except Exception as e:
                print(f"  Ошибка при чтении {path}: {e}")
    
    print(f"  Файл с токенами для документа {doc_num} (формат: {formatted_num}) не найден")
    return []

def get_idf_for_term(term):
    if term in INDEX:
        return solve_idf(len(INDEX[term]))
    return 0

def get_idf_for_lemma(lemma, tokens_of_lemma):
    if not tokens_of_lemma:
        return 0
    
    docs_with_lemma = set()
    for token in tokens_of_lemma:
        if token in INDEX:
            docs_with_lemma.update(INDEX[token])
    
    return solve_idf(len(docs_with_lemma))

def parse_lemmas(doc_num):
    formatted_num = format_doc_number(doc_num)
    
    possible_paths = [
        f"../task_2/lemmas/{formatted_num}.txt",
    ]
    
    lemmas_to_tokens = {}
    token_to_lemma = {}

    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            lemma = parts[0]
                            tokens = parts[1:]
                            
                            lemmas_to_tokens[lemma] = tokens
                            for token in tokens:
                                token_to_lemma[token] = lemma
                print(f"  Загружены леммы из: {path}")
                return lemmas_to_tokens, token_to_lemma
            except Exception as e:
                print(f"  Ошибка при чтении {path}: {e}")
    
    print(f"  Файл с леммами для документа {doc_num} не найден")
    return {}, {}

def create_output_directories():
    os.makedirs("tf_idf", exist_ok=True)
    
    os.makedirs("tf_idf/tokens", exist_ok=True)
    os.makedirs("tf_idf/lemmas", exist_ok=True)


create_output_directories()

print("\nЗагрузка инвертированного индекса...")
parse_index()

processed_docs = 0
skipped_docs = 0
docs_with_tokens = 0

print(f"\nНачинаем обработку {N} документов...")
print("-" * 10)

for i in range(1, N + 1):
    try:
        formatted_num = format_doc_number(i)
        print(f"\nОбработка документа {i} (формат: {formatted_num})...")
        
        tf_lemmas = defaultdict(int)
        tf_tokens = defaultdict(int)
        
        tokens = parse_tokens(i)
        if not tokens:
            print(f"  Документ {i} не содержит токенов, пропускаем")
            skipped_docs += 1
            continue
            
        doc_size = len(tokens)
        docs_with_tokens += 1
        print(f"  Найдено токенов: {doc_size}")
        
        lemmas_to_tokens, token_to_lemma = parse_lemmas(i)
        
        for token in tokens:
            tf_tokens[token] += 1
            if token in token_to_lemma:
                tf_lemmas[token_to_lemma[token]] += 1
        
        print(f"  Уникальных терминов: {len(tf_tokens)}")
        print(f"  Уникальных лемм: {len(tf_lemmas)}")
        
        term_file = f"tf_idf/tokens/tf_{formatted_num}.txt"
        with open(term_file, "w", encoding="utf-8") as f:
            term_count = 0
            for token, count in sorted(tf_tokens.items(), 
                                        key=lambda x: x[1], reverse=True):
                idf = get_idf_for_term(token)
                tf_idf = round((count / doc_size) * idf, digits)
                if tf_idf > 0.0000001:
                    f.write(f"{token} {idf:.4f} {tf_idf:.7f}\n")
                    term_count += 1
            print(f"  Сохранено терминов в {term_file}: {term_count}")
        
        lemma_file = f"tf_idf/lemmas/tf_{formatted_num}.txt"
        with open(lemma_file, "w", encoding="utf-8") as f:
            lemma_count = 0
            for lemma, count in sorted(tf_lemmas.items(), 
                                        key=lambda x: x[1], reverse=True):
                if lemma in lemmas_to_tokens:
                    idf = get_idf_for_lemma(lemma, lemmas_to_tokens[lemma])
                    tf_idf = round((count / doc_size) * idf, digits)
                    if tf_idf > 0.0000001:
                        f.write(f"{lemma} {idf:.4f} {tf_idf:.7f}\n")
                        lemma_count += 1
            print(f"  Сохранено лемм в {lemma_file}: {lemma_count}")
        
        processed_docs += 1
        
    except Exception as e:
        print(f"  ОШИБКА при обработке документа {i}: {e}")
        traceback.print_exc()
        skipped_docs += 1

print("\n" + "-" * 10)
print(f"Всего документов в корпусе: {N}")
print(f"Документов с токенами: {docs_with_tokens}")
print(f"Успешно обработано: {processed_docs}")
print(f"Пропущено/с ошибками: {skipped_docs}")
print("\nСтруктура выходных файлов:")
