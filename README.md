# evals-researcher

#### Запустить презентацию 
`quarto preview presentation/evals-presentation.ipynb`

#### Запустить phoenix
`phoenix serve`

#### Запустить annotation tool для разметки итоговых отчётов
`marimo edit annotation.py`

Его исходный код лежит в [репо](https://github.com/Asya02/molabel) - форке [репо](https://github.com/koaning/molabel)

#### Запустить прогон на датасете
`python run_queries.py`

#### Запустить annotation tool для разметки промежуточных шагов пайплайна
`streamlit run annotation_tool/app.py`

## Из чего ещё состоит репозиторий
- В папке `notebooks` представлены по порядку все шаги
- В папке `fact_eval` реализация метрик для оценки соответствия написаному тексту информации по ссылке. Метрика взята из [статьи](https://deepresearch-bench.github.io/) и реализована под работу с GigaChat.
