import json
import re

class BooleanSearchEngine:
    def __init__(self, index_file='inverted_index.json'):
        self.index_file = index_file
        self.index = self._load_index()
        self.all_documents = self._get_all_documents()
        
    def _load_index(self):
        try:
            with open(self.index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
            
            return {term: set(docs) for term, docs in index.items()}
        except FileNotFoundError:
            print(f"Файл индекса {self.index_file} не найден")
            print("Сначала выполните index.py для построения индекса")
            return {}
    
    def _get_all_documents(self):
        all_docs = set()

        for docs in self.index.values():
            all_docs.update(docs)
        return all_docs
    
    def search(self, query):
        query = query.strip()
        
        if not query:
            return set()
        
        result = self._evaluate_expression(query)
        
        return result
    
    def _evaluate_expression(self, expr):
        expr = expr.strip()
        
        if not expr:
            return set()
        
        if expr.startswith('(') and expr.endswith(')'):
            if self._is_balanced(expr):
                return self._evaluate_expression(expr[1:-1].strip())
        
        depth = 0
        for i, char in enumerate(expr):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif depth == 0:
                if i + 2 < len(expr) and expr[i:i+3] == ' OR ':
                    left = expr[:i]
                    right = expr[i+3:]
                    left_result = self._evaluate_expression(left)
                    right_result = self._evaluate_expression(right)
                    return left_result | right_result
        
        depth = 0
        for i, char in enumerate(expr):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif depth == 0:
                if i + 4 < len(expr) and expr[i:i+5] == ' AND ':
                    left = expr[:i]
                    right = expr[i+5:]
                    left_result = self._evaluate_expression(left)
                    right_result = self._evaluate_expression(right)
                    return left_result & right_result
        
        if expr.startswith('NOT '):
            term = expr[4:].strip()
            term_result = self._evaluate_term(term)
            return self.all_documents - term_result
        
        return self._evaluate_term(expr)
    
    def _is_balanced(self, expr):
        count = 0
        for char in expr:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0
    
    def _evaluate_term(self, term):
        term = term.strip()
        if not term:
            return set()
        
        term_lower = term.lower()
        
        if term in self.index:
            return self.index[term]
        
        for key, docs in self.index.items():
            if key.lower() == term_lower:
                return docs
        
        return set()
    
    def interactive_search(self):
        print("Поддерживаемые операторы:")
        print("  AND - логическое И")
        print("  OR  - логическое ИЛИ")
        print("  NOT - логическое НЕ")
        print("\nДля выхода введите 'exit'")
        print("-" * 10)
        
        while True:
            try:
                query = input("\nВведите запрос: ").strip()
                
                if query.lower() in ['exit']:
                    break
                
                if not query:
                    continue
                
                result = self.search(query)
                
                if result:
                    print(f"\nНайдено документов: {len(result)}")
                    print("Документы:")
                    for doc in sorted(result):
                        print(f"  - {doc}")
                else:
                    print("\nДокументов не найдено")
                    
            except KeyboardInterrupt:
                print("\n\nПоиск прерван")
                break
            except Exception as e:
                print(f"\nОшибка при обработке запроса: {e}")
                print("Проверьте синтаксис запроса")


engine = BooleanSearchEngine()

if not engine.index:
    print("Не удалось загрузить индекс")

engine.interactive_search()
