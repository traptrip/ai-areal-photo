# AI Aeral photo hackathon 2022

## Final solution

./experiments/sota_train/
  - train.py - файл с обучением модели
  - config.py - файл с конфигурацией для обучения
  - model.py - файл с моделями 
  - losses.py - файл с лосс функцией/ отрицательной метрикой
  - transforms.py - файл с отдельными аугментациями 
  - presets.py - файл с аугментациями для обучения/тестирования
  - stacking_train_eval.py - файл с обучением ансамбля моделей и его валидацией
  - stacking_inference.py - файл с инференсом ансамбля моделей
  
  - evaluate.py - файл с валидацией одной конкретной модели
  - inference.py - файл с инференсом одной модели
  - bagging_evaluate.py - файл с валидацией бэггинга моделей (простое усреднение результатов)
  - bagging_inference.py - файл с инференсом бэггинга моделей
  
