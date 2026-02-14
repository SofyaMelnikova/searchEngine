import os
import time
import requests
from urllib.parse import quote

BASE_URL = "https://ru.ruwiki.ru/wiki/"
OUTPUT_DIR = "downloads"
INDEX_FILE = "index.txt"
REQUEST_DELAY = 10

CITIES = [
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
    "Нижний Новгород", "Челябинск", "Самара", "Омск", "Ростов-на-Дону",
    "Уфа", "Красноярск", "Воронеж", "Пермь", "Волгоград",
    "Краснодар", "Саратов", "Тюмень", "Тольятти", "Ижевск",
    "Барнаул", "Ульяновск", "Иркутск", "Хабаровск", "Ярославль",
    "Владивосток", "Махачкала", "Томск", "Оренбург", "Кемерово",
    "Новокузнецк", "Рязань", "Астрахань", "Набережные Челны", "Пенза",
    "Липецк", "Киров", "Чебоксары", "Тула", "Калининград",
    "Балашиха", "Курск", "Ставрополь", "Улан-Удэ", "Тверь",
    "Магнитогорск", "Сочи", "Иваново", "Брянск", "Белгород",
    "Сургут", "Владимир", "Нижний Тагил", "Архангельск", "Чита",
    "Симферополь", "Калуга", "Смоленск", "Волжский", "Курган",
    "Орёл", "Череповец", "Вологда", "Владикавказ", "Саранск",
    "Мурманск", "Якутск", "Тамбов", "Грозный", "Стерлитамак",
    "Кострома", "Петрозаводск", "Нижневартовск", "Йошкар-Ола", "Новороссийск",
    "Комсомольск-на-Амуре", "Таганрог", "Сыктывкар", "Химки", "Нальчик",
    "Шахты", "Дзержинск", "Орск", "Братск", "Ангарск",
    "Энгельс", "Благовещенск", "Псков", "Бийск", "Прокопьевск",
    "Рыбинск", "Балаково", "Армавир", "Северодвинск", "Королёв",
    "Подольск", "Петропавловск-Камчатский", "Норильск", "Сызрань", "Мытищи"
]

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if os.path.exists(INDEX_FILE):
    os.remove(INDEX_FILE)

session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
})

for i, city in enumerate(CITIES[:100], 1):
    url = f"{BASE_URL}/{city}"
    
    filename = f"page_{i:03d}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    print(f"[{i}/100] Загружаю: {city}")
    
    try:
        response = session.get(url, timeout=15)
        response.encoding = 'utf-8'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        with open(INDEX_FILE, 'a', encoding='utf-8') as f:
            f.write(f"{filename}\t{url}\n")
        
        print(f"Сохранено: {filename}")
        
        if i < 100:
            time.sleep(REQUEST_DELAY)
            
    except Exception as e:
        print(f"Ошибка: {e}")
