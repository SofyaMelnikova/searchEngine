import json
import os
import re
from collections import defaultdict

class InvertedIndexBuilder:
    def __init__(self, lemmas_dir='../task_2/lemmas/', output_file='inverted_index.json'):
        self.lemmas_dir = lemmas_dir
        self.output_file = output_file
        self.inverted_index = defaultdict(set)
        
    def build_index(self):
        if not os.path.exists(self.lemmas_dir):
            raise FileNotFoundError(f"Директория {self.lemmas_dir} не найдена")
        
        lemma_files = [f for f in os.listdir(self.lemmas_dir) 
                      if f.endswith('.txt')]
        
        for filename in lemma_files:
            file_path = os.path.join(self.lemmas_dir, filename)
            
            doc_name = self._extract_doc_name(filename)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lemmas = self._extract_lemmas(content)
            
            for lemma in lemmas:
                self.inverted_index[lemma].add(doc_name)
        
        serializable_index = {
            term: list(docs) for term, docs in self.inverted_index.items()
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_index, f, ensure_ascii=False, indent=2)
        
        print(f"Индекс построен и сохранен в {self.output_file}")
        print(f"Всего терминов: {len(serializable_index)}")
        
        return serializable_index
    
    def _extract_doc_name(self, filename):
        name = filename.replace('.txt', '')
        return name
    
    def _extract_lemmas(self, content):
        lemmas = set()
        
        for line in content.split('\n'):
            line = line.strip()

            if not line:
                continue
            
            lemma = line.split(' ')[0].strip()
            if lemma:
                lemmas.add(lemma)
        
        return lemmas

builder = InvertedIndexBuilder()
try:
    index = builder.build_index()
    
    print("\nПример первых 10 терминов:")
    for i, (term, docs) in enumerate(list(index.items())[:10]):
        print(f"{term}: {docs}")
        
except FileNotFoundError as e:
    print(f"Ошибка: {e}")
    print("Убедитесь, что папка с леммами существует и содержит файлы")
