# Разработка Retrieval-Based чат-бота на примере персонажей сериала "Друзья"

## Описание датасета, анализ и предобработка

**Источник данных:** https://www.kaggle.com/datasets/gopinath15/friends-netflix-script-data 

**Количество записей**: 69974

**Датасет содержит 5 полей:**

* Text – реплика персонажа и технический текст
* Speaker – персонаж, говорящий реплику
* Episode – номер и название эпизода
* Season – номер сезона
* Show – название шоу


## Анализ и предобработка


## Примеры работы чат-бота

<img src="./resources/Example.jpg" width="400" height="700"/>

## Структура репозитория
    ├── templates                          # Шаблоны html страниц
    ├── data                               # Данные
    ├────── data.pkl                       # Данные для обучения модели
    ├────── Friends_processed.csv          # Обработанный датасет
    ├── model                              # Модель
    ├────── charact_corpus.pkl             # Корпус реплик персонажа
    ├── Analysis.ipynb                     # Анализ данных
    ├── Processing.ipynb                   # Обработка данных
    ├── train.py                           # Обучение модели
    ├── inference.py                       # Инференс
    ├── Report_ShmelkovYB.pdf              # Отчет
    └── README.md                          # Краткое описание проекта
