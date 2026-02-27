import os
import re
from bs4 import BeautifulSoup
import pymorphy3
from nltk.corpus import stopwords
import nltk

class TextProcessor:
    def __init__(self, pages_dir='../downloads', output_dir='.'):
        self.pages_dir = pages_dir
        self.output_dir = output_dir

        self.tokens_dir = os.path.join(output_dir, 'tokens')
        self.lemmas_dir = os.path.join(output_dir, 'lemmas')
        self.clean_text_dir = os.path.join(output_dir, 'clean')

        os.makedirs(self.tokens_dir, exist_ok=True)
        os.makedirs(self.lemmas_dir, exist_ok=True)
        os.makedirs(self.clean_text_dir, exist_ok=True)

        nltk.download('stopwords', quiet=True)
        self.stop_words = set(stopwords.words('russian'))

        self.morph = pymorphy3.MorphAnalyzer()

    def extract_text_from_html(self, html_content):
        soup = BeautifulSoup(html_content, 'html.parser')
        
        for element in soup(['script', 'style', 'meta', 'link', 'noscript', 
                             'header', 'footer', 'nav', 'aside', 'form', 
                             'button', 'iframe']):
            element.decompose()
        
        for element in soup.find_all(class_=re.compile(r'(sidebar|menu|nav|footer|header|banner|adv|advert|popup|modal)')):
            element.decompose()
        
        main_content = None
        
        possible_content = soup.find('div', class_=re.compile(r'content|article|mw-parser-output|page|main'))
        if possible_content:
            main_content = possible_content
        else:
            main_content = soup.find('body')
        
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        text = re.sub(r'\s+', ' ', text)
        
        lines = text.split('. ')
        meaningful_lines = []
        for line in lines:
            if re.search(r'https?://|\.(jpg|png|svg|css|js)|\d+px|data:image|window\.|function\(', line, re.IGNORECASE):
                continue
            if re.search(r'[{}\[\]<>]|var |let |const |return |if\(', line):
                continue
            if line.strip():
                meaningful_lines.append(line.strip())
        
        clean_text = '. '.join(meaningful_lines)
        
        clean_text = re.sub(r'\s+', ' ', clean_text)
        
        return clean_text

    def save_clean_text(self, text, output_file):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Очищенный текст сохранен в {output_file}")
        except Exception as e:
            print(f"Ошибка при сохранении очищенного текста: {e}")

    def tokenize(self, text):
        words = re.findall(r'\b[а-яё]{2,30}\b', text.lower())
        
        clean_words = []
        for word in words:
            if word in self.stop_words:
                continue
                
            if len(word) > 25:
                continue
                
            clean_words.append(word)
        
        return clean_words

    def lemmatize_words(self, words):
        lemma_dict = {}
        
        for word in words:
            try:
                parsed = self.morph.parse(word)[0]
                lemma = parsed.normal_form
                
                if len(lemma) > 30:
                    continue
                    
                if lemma not in lemma_dict:
                    lemma_dict[lemma] = set()
                lemma_dict[lemma].add(word)
            except Exception as e:
                print(f"Ошибка лемматизации слова '{word}': {e}")
                continue
        
        result = {}
        for lemma, tokens in lemma_dict.items():
            sorted_tokens = sorted(list(tokens))
            result[lemma] = sorted_tokens
        
        return result

    def get_html_files(self):
        html_files = []
        
        if not os.path.exists(self.pages_dir):
            print(f"Директория {self.pages_dir} не существует!")
            return []
            
        for file in os.listdir(self.pages_dir):
            if file.endswith(('.html', '.htm')):
                html_files.append(os.path.join(self.pages_dir, file))
        
        html_files.sort()
        return html_files

    def get_page_number(self, file_path):
        basename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(basename)[0]
        numbers = re.findall(r'\d+', name_without_ext)

        if numbers:
            return numbers[0]
        return name_without_ext

    def process_file(self, html_file_path):
        try:
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            clean_text = self.extract_text_from_html(html_content)
            
            tokens = self.tokenize(clean_text)
            
            unique_tokens = []
            seen = set()
            for token in tokens:
                if token not in seen:
                    seen.add(token)
                    unique_tokens.append(token)
            
            unique_tokens.sort()
            
            lemmas = self.lemmatize_words(unique_tokens)
            
            return unique_tokens, lemmas, clean_text
            
        except Exception as e:
            print(f"Ошибка при обработке файла {html_file_path}: {e}")
            return [], {}, ""

    def save_tokens(self, tokens, output_file):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for token in tokens:
                    f.write(f"{token}\n")
            print(f"Токены сохранены в {output_file}")
        except Exception as e:
            print(f"Ошибка при сохранении токенов: {e}")

    def save_lemmas(self, lemmas, output_file):
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for lemma, words in sorted(lemmas.items()):
                    f.write(f"{lemma} {' '.join(words)}\n")
            print(f"Леммы сохранены в {output_file}")
        except Exception as e:
            print(f"Ошибка при сохранении лемм: {e}")

    def process_all_pages(self):
        html_files = self.get_html_files()
        
        if not html_files:
            print(f"HTML-файлы не найдены в директории: {self.pages_dir}")
            return
        
        print(f"Найдено HTML-файлов: {len(html_files)}")
        print("-" * 10)
        
        total_tokens = 0
        total_lemmas = 0
        total_files_processed = 0
        
        for html_file in html_files:
            page_num = self.get_page_number(html_file)
            
            print(f"Обработка файла: {os.path.basename(html_file)}")
            
            tokens, lemmas, clean_text = self.process_file(html_file)
            
            if tokens and clean_text:
                clean_file = os.path.join(self.clean_text_dir, f"{page_num}.txt")
                self.save_clean_text(clean_text, clean_file)
                
                tokens_file = os.path.join(self.tokens_dir, f"{page_num}.txt")
                self.save_tokens(tokens, tokens_file)
                
                lemmas_file = os.path.join(self.lemmas_dir, f"{page_num}.txt")
                self.save_lemmas(lemmas, lemmas_file)
                
                total_tokens += len(tokens)
                total_lemmas += len(lemmas)
                total_files_processed += 1
                
                print(f"Найдено: {len(tokens)} уникальных токенов, {len(lemmas)} лемм")
                print(f"Длина очищенного текста: {len(clean_text)} символов")
            else:
                print(f"Не удалось извлечь данные из файла")
            
            print("-" * 10)

processor = TextProcessor(pages_dir='../downloads', output_dir='.')
processor.process_all_pages()
