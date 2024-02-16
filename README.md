# Разработка Retrieval-Based чат-бота на примере персонажей сериала "Друзья"

**Источник данных:** https://www.kaggle.com/datasets/gopinath15/friends-netflix-script-data 

**Количество записей**: 69974

**Датасет содержит 5 полей:**

* Text – реплика персонажа и технический текст
* Speaker – персонаж, говорящий реплику
* Episode – номер и название эпизода
* Season – номер сезона
* Show – название шоу

## Примеры работы чат-бота

<img src="./valid/Test_4.jpg" width=auto height=auto/>

## Структура репозитория
    ├── data                               # Данные
    ├────── Friends_processed.csv          # Обработанный датасет
    ├────── data.pkl                       # Данные для обучения модели
    ├── model                              # Модель
    ├────── tokenizer                      # Дообученный токенайзер модели
    ├────── charact_corpus.pkl             # Корпус реплик персонажа
    ├────── friends_model.pkl              # Дообученная модель (доступна по ссылке в отчете)
    ├── templates                          # Шаблоны html страниц
    ├────── index.html                     # шаблон главной страницы с ссылками на тесты
    ├── valid                              # Результаты валидации на тестах
    ├── Analysis.ipynb                     # Анализ данных
    ├── Processing.ipynb                   # Обработка данных
    ├── README.md                          # Краткое описание проекта
    ├── Report_ShmelkovYB.pdf              # Отчет
    ├── inference.py                       # Инференс
    ├── model.py                           # Модель
    ├── requirements.txt                   # Используемые библиотеки с версиями
    └── train.py                           # Обучение модели
