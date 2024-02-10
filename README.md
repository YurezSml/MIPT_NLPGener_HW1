# Разработка Retrieval-Based чат-бота на примере персонажей сериала "Друзья"

## Описание датасета, анализ и предобработка
**Источник данных:** https://www.kaggle.com/datasets/gopinath15/friends-netflix-script-data 
**Датасет состоит из 5 столбцов:**
Text – реплика персонажа и технический текст
Speaker – персонаж, говорящий реплику
Episode – номер и название эпизода
Season – номер сезона
Show – название шоу
Всего датасет содержит **69974 записи**.

## Анализ и предобработка


## Примеры работы чат-бота

<img src="./resources/Example.jpg" width="400" height="700"/>

## Структура репозитория
    ├── templates                          # Шаблоны html страниц
    ├── data                               # Данные
    ├────── charact_corpus.pkl             # Корпус реплик персонажа
    ├────── data.pkl                       # Данные для обучения модели
    ├────── Friends_processed.csv          # Обработанный датасет
    ├── Analysis.ipynb                     # Анализ данных
    ├── Processing.ipynb                   # Обработка данных
    ├── Learning.ipynb                     # Обучение модели
    ├── Report_ShmelkovYB.pdf              # Отчет
    └── README.md                          # Краткое описание проекта
