# Modeling Turbulent Radiative Layer Dynamics Using Neural Operators

## О проекте

Исследование посвящено прогнозированию динамики `turbulent_radiative_layer_2D` с помощью нейросетевых моделей для физических полей. Основная задача проекта - по нескольким предыдущим временным шагам предсказывать будущую эволюцию системы и уменьшать деградацию качества при авторегрессионном rollout.

В текущей версии проекта рассматриваются 4 поля:

- `density`
- `pressure`
- `velocity_x`
- `velocity_y`

В проекте исследованы и сравнены две архитектурные ветки: семейство `FNO`-моделей и `ConvNextU-Net`.

## Статус

- Текущий этап: чекпоинт 6
- Уже сделано: репозиторий и план работ, обзор литературы, подготовка датасета и EDA, baseline и метрики, серия улучшенных `FNO`-экспериментов, обучение и сравнение `ConvNextU-Net`
- Текущие лучшие решения: `ConvNextU-Net_4_2_1_48_7` по `1-step` и по `4-step rollout`
- Итог: `ConvNextU-Net` превосходит все `FNO`-конфигурации по всем ключевым метрикам

## Участники

- Студент: Эмиль Фахретдинов
- Научный руководитель: Каюмов Руслан Асхатович

## Данные и постановка задачи

- Датасет: `turbulent_radiative_layer_2D`
- Источник: [The Well](https://polymathic-ai.org/the_well/datasets/turbulent_radiative_layer_2D/)
- Вход модели: 4 предыдущих шага
- Текущая оценка качества: `1-step` прогноз и `4-step autoregressive rollout`
- Нормализация: `Z-score`

Связанные материалы:

- `dataset.ipynb` - загрузка данных, структура полей, подготовка выборок
- `data_explore.ipynb` - EDA и визуальный анализ траекторий
- `models_eval.ipynb` - сравнение baseline и улучшенных моделей

## Прогресс по чекпоинтам

### Чекпоинт 1

Сформирован репозиторий проекта и верхнеуровневый план работы. В `README.md` зафиксированы тема, участники, структура этапов и логика движения от данных и baseline к улучшенным архитектурам.

### Чекпоинт 2

Проведен обзор направлений, релевантных задаче прогнозирования физических полей:

- neural operators и `FNO`
- сверточные архитектуры уровня `U-Net`
- авторегрессионное предсказание для PDE и физически мотивированных симуляций
- метрики сравнения `baseline` и устойчивости rollout

Итог этого этапа использован для выбора семейства моделей: сначала серия `FNO`, далее - сравнение с `ConvNextU-Net`.

### Чекпоинт 3

Выполнены подготовка данных и EDA:

- изучена структура датасета и состав полей
- собраны выборки для обучения и валидации
- проведен анализ пространственно-временной динамики
- проверено поведение полей при rollout
- `dataset.ipynb`
- `data_explore.ipynb`

### Чекпоинт 4

Собран baseline и зафиксированы метрики для дальнейшего сравнения:

- baseline для сравнения: `Baseline (FNO)`
- метрики: `VRMSE`, `Rel L2`, `R2`, `RMSE`, `MAE`
- подготовлен сценарий сравнения baseline с улучшенными моделями
- `models_eval.ipynb`
- `results/`

### Чекпоинт 5

На этом этапе выполнено улучшение решения через серию экспериментов с `FNO`:

- протестированы `delta`- и `full-frame`-варианты предсказания
- сравнены несколько конфигураций по числу мод и слоев
- зафиксированы результаты как для `1-step`, так и для `4-step rollout`
- проведено сравнение не только с baseline, но и с внешним ориентиром `Baseline (FNO)`

Лучшие конфигурации по итогам чекпоинта 5:

- `Delta_64_4` - лучший `1-step`: `VRMSE = 0.2799`
- `Delta_32_4` - лучший `4-step rollout`: `VRMSE = 0.3411`

### Чекпоинт 6

На текущем этапе выполнено обучение и сравнение `ConvNextU-Net` с лучшими `FNO`-конфигурациями:

- обучены `full-frame`- и `delta`-варианты `ConvNextU-Net`
- проведено сравнение по `1-step` и `4-step rollout` метрикам
- `ConvNextU-Net` превзошел все `FNO`-модели по всем ключевым метрикам

Лучшая модель: `ConvNextU-Net_4_2_1_48_7` (`full-frame`):

- `1-step VRMSE = 0.2173` (vs `0.2799` у лучшей `FNO`)
- `4-step rollout VRMSE = 0.2905` (vs `0.3411` у лучшей `FNO`)

## Итоги чекпоинта 6

После добавления `ConvNextU-Net` картина результатов:

- лучший `1-step` результат показывает `ConvNextU-Net_4_2_1_48_7`: `VRMSE = 0.2173` против `0.5121` у `Baseline (FNO)` и `0.2799` у лучшей `FNO`-конфигурации
- лучший `4-step rollout` также у `ConvNextU-Net_4_2_1_48_7`: `VRMSE = 0.2905` против `1.6885` у `Baseline (FNO)` и `0.3411` у лучшей `FNO`-конфигурации

Это дает примерно:

- `~22%` улучшения по `1-step VRMSE` относительно лучшей `FNO`
- `~15%` улучшения по `4-step rollout VRMSE` относительно лучшей `FNO`
- `~58%` улучшения по `1-step VRMSE` и `~83%` по rollout относительно `Baseline (FNO)`

Интересно, что `full-frame`-вариант `ConvNextU-Net` превосходит `delta`-вариант той же архитектуры, тогда как для `FNO` ситуация была обратной. Это указывает на различие в том, как архитектуры используют контекст предыдущих шагов.

## Сравнение моделей

### 1-step prediction

Источники: `results/FNO/Model Comparison (1-Step Prediction).csv`, `results/U-net/Model Comparison (1-Step Prediction).csv`


| Model                          | VRMSE ↓    | Rel L2 ↓   | R2 ↑       | RMSE ↓     |
| ------------------------------ | ---------- | ---------- | ---------- | ---------- |
| Baseline (FNO)                 | 0.5121     | 0.5114     | 0.7207     | 0.5227     |
| Delta_16_4                     | 0.2896     | 0.2892     | 0.9096     | 0.2957     |
| Delta_32_4                     | 0.2816     | 0.2812     | 0.9137     | 0.2876     |
| Full_16_4                      | 0.3090     | 0.3086     | 0.8969     | 0.3155     |
| Full_32_4                      | 0.2997     | 0.2993     | 0.9019     | 0.3061     |
| Full_32_5                      | 0.3079     | 0.3075     | 0.8964     | 0.3145     |
| Full_64_4                      | 0.2973     | 0.2969     | 0.9031     | 0.3037     |
| Delta_64_4                     | 0.2799     | 0.2795     | 0.9144     | 0.2859     |
| ConvNextU-Net_delta_4_2_1_42_7 | 0.2491     | 0.2487     | 0.9311     | 0.2545     |
| ConvNextU-Net_4_2_1_42_7       | 0.2221     | 0.2219     | 0.9463     | 0.2268     |
| ConvNextU-Net_4_2_1_48_7       | **0.2173** | **0.2170** | **0.9482** | **0.2220** |


### 4-step autoregressive rollout

Источники: `results/FNO/Model Comparison (Rollout 4-step).csv`, `results/U-net/Model Comparison (Rollout 4-step).csv`


| Model                          | VRMSE ↓    | Rel L2 ↓   | R2 ↑       | RMSE ↓     | MAE ↓      |
| ------------------------------ | ---------- | ---------- | ---------- | ---------- | ---------- |
| Baseline (FNO)                 | 1.6885     | 1.6864     | -2.1406    | 1.7192     | 0.5966     |
| Delta_16_4                     | 0.3483     | 0.3478     | 0.8696     | 0.3552     | 0.1376     |
| Delta_32_4                     | 0.3411     | 0.3406     | 0.8741     | 0.3479     | 0.1307     |
| Full_16_4                      | 0.3624     | 0.3618     | 0.8588     | 0.3695     | 0.1520     |
| Full_32_4                      | 0.3550     | 0.3544     | 0.8638     | 0.3620     | 0.1390     |
| Full_32_5                      | 0.3594     | 0.3588     | 0.8604     | 0.3665     | 0.1436     |
| Full_64_4                      | 0.3522     | 0.3517     | 0.8658     | 0.3592     | 0.1341     |
| Delta_64_4                     | 0.3416     | 0.3411     | 0.8734     | 0.3484     | 0.1269     |
| ConvNextU-Net_delta_4_2_1_42_7 | 0.3198     | 0.3193     | 0.8889     | 0.3263     | 0.1153     |
| ConvNextU-Net_4_2_1_42_7       | 0.2912     | 0.2908     | 0.9081     | 0.2971     | 0.1084     |
| ConvNextU-Net_4_2_1_48_7       | **0.2905** | **0.2901** | **0.9083** | **0.2963** | **0.1071** |


### Покомпонентное сравнение: 1-step VRMSE

Источники: `results/FNO/Per-field Comparison (1-Step).csv`, `results/U-net/Per-field Comparison (1-Step).csv`


| Model                          | density    | pressure   | velocity_x | velocity_y |
| ------------------------------ | ---------- | ---------- | ---------- | ---------- |
| Baseline (FNO)                 | 0.3346     | 0.6925     | 0.4643     | 0.5568     |
| Delta_16_4                     | 0.1551     | 0.3681     | 0.3028     | 0.3323     |
| Delta_32_4                     | 0.1423     | 0.3624     | 0.2954     | 0.3262     |
| Full_16_4                      | 0.1661     | 0.3979     | 0.3207     | 0.3513     |
| Full_32_4                      | 0.1502     | 0.3939     | 0.3110     | 0.3436     |
| Full_32_5                      | 0.1547     | 0.4044     | 0.3181     | 0.3545     |
| Full_64_4                      | 0.1459     | 0.3908     | 0.3085     | 0.3438     |
| Delta_64_4                     | 0.1381     | 0.3595     | 0.2954     | 0.3267     |
| ConvNextU-Net_delta_4_2_1_42_7 | 0.1083     | 0.3190     | 0.2703     | 0.2987     |
| ConvNextU-Net_4_2_1_42_7       | 0.1092     | 0.2727     | 0.2423     | 0.2644     |
| ConvNextU-Net_4_2_1_48_7       | **0.1021** | **0.2714** | **0.2371** | **0.2587** |


### Покомпонентное сравнение: 4-step rollout VRMSE

Источники: `results/FNO/Per-field Comparison (Rollout).csv`, `results/U-net/Per-field Comparison (Rollout).csv`


| Model                          | density    | pressure   | velocity_x | velocity_y |
| ------------------------------ | ---------- | ---------- | ---------- | ---------- |
| Baseline (FNO)                 | 1.9597     | 1.4643     | 0.9464     | 2.3837     |
| Delta_16_4                     | 0.1929     | 0.4444     | 0.3532     | 0.4027     |
| Delta_32_4                     | 0.1818     | 0.4391     | 0.3478     | 0.3958     |
| Full_16_4                      | 0.2014     | 0.4659     | 0.3657     | 0.4165     |
| Full_32_4                      | 0.1929     | 0.4615     | 0.3562     | 0.4093     |
| Full_32_5                      | 0.1942     | 0.4653     | 0.3616     | 0.4164     |
| Full_64_4                      | 0.1903     | 0.4588     | 0.3522     | 0.4076     |
| Delta_64_4                     | 0.1800     | 0.4440     | 0.3465     | 0.3960     |
| ConvNextU-Net_delta_4_2_1_42_7 | 0.1656     | 0.4084     | 0.3259     | 0.3794     |
| ConvNextU-Net_4_2_1_42_7       | 0.1503     | 0.3658     | 0.3037     | 0.3452     |
| ConvNextU-Net_4_2_1_48_7       | **0.1478** | **0.3676** | **0.3047** | **0.3419** |


### Рост ошибки по горизонту rollout

Источники: `results/FNO/Error Growth (Rollout Stability).csv`, `results/U-net/Error Growth (Rollout Stability).csv`


| Model                          | Step 1     | Step 2     | Step 3     | Step 4     |
| ------------------------------ | ---------- | ---------- | ---------- | ---------- |
| Baseline                       | 0.5098     | 0.7238     | 1.1606     | 2.9894     |
| Delta_16_4                     | 0.2870     | 0.3337     | 0.3670     | 0.3947     |
| Delta_32_4                     | 0.2787     | 0.3260     | 0.3598     | 0.3885     |
| Full_16_4                      | 0.3065     | 0.3475     | 0.3793     | 0.4070     |
| Full_32_4                      | 0.2968     | 0.3398     | 0.3727     | 0.4005     |
| Full_32_5                      | 0.3048     | 0.3454     | 0.3762     | 0.4022     |
| Full_64_4                      | 0.2942     | 0.3378     | 0.3692     | 0.3976     |
| Delta_64_4                     | 0.2770     | 0.3258     | 0.3607     | 0.3906     |
| ConvNextU-Net_delta_4_2_1_42_7 | 0.2473     | 0.3014     | 0.3409     | 0.3735     |
| ConvNextU-Net_4_2_1_42_7       | 0.2209     | 0.2701     | 0.3103     | 0.3466     |
| ConvNextU-Net_4_2_1_48_7       | **0.2160** | **0.2694** | **0.3109** | **0.3471** |


## Визуальное сравнение rollout

Ниже приведены GIF-анимации rollout по полю `density` для одной и той же траектории.

### FNO модели

| Baseline (FNO) | Delta_32_4 | Delta_64_4 | Full_64_4 |
| --- | --- | --- | --- |
| ![](plot_gifs_out/FNO/delta_gifs/baseline_model/traj000_density_pretrained_delta/sequence.gif) | ![](plot_gifs_out/FNO/delta_gifs/delta_32_4/traj000_density_delta_last_delta_model/sequence.gif) | ![](plot_gifs_out/FNO/delta_gifs/delta_64_4/traj000_density_delta_final_model_delta/sequence.gif) | ![](plot_gifs_out/FNO/full_gifs/full_64_4/traj000_density_full_final_model_full_frame/sequence.gif) |

Дополнительные GIF для FNO: `plot_gifs_out/FNO/delta_gifs/` и `plot_gifs_out/FNO/full_gifs/`

### ConvNextU-Net модели

| ConvNextU-Net_4_2_1_42_7 | ConvNextU-Net_4_2_1_48_7 | ConvNextU-Net_delta_4_2_1_42_7 |
| --- | --- | --- |
| ![](plot_gifs_out/U-net/full_gifs/unet_full_4_2_1_42_7/traj000_density_cnextunet_full_best_cnextunet_by_valid_rollout_vrmse/sequence.gif) | ![](plot_gifs_out/U-net/full_gifs/unet_full_4_2_1_48_7/traj000_density_cnextunet_full_best_cnextunet_by_valid_rollout_vrmse/sequence.gif) | ![](plot_gifs_out/U-net/delta_gifs/unet_delta_4_2_1_42_7/traj000_density_cnextunet_delta_best_cnextunet_delta_by_valid_rollout_vrmse/sequence.gif) |

Дополнительные GIF для U-Net: `plot_gifs_out/U-net/delta_gifs/` и `plot_gifs_out/U-net/full_gifs/`

## Основной вывод

По итогам всех экспериментов наилучшее качество демонстрирует `ConvNextU-Net_4_2_1_48_7` (`full-frame`):

- лучший `1-step VRMSE = 0.2173` — на `~22%` лучше, чем лучшая `FNO`-конфигурация (`Delta_64_4`, `0.2799`)
- лучший `4-step rollout VRMSE = 0.2905` — на `~15%` лучше, чем лучшая `FNO`-конфигурация (`Delta_32_4`, `0.3411`)
- по всем покомпонентным метрикам (`density`, `pressure`, `velocity_x`, `velocity_y`) `ConvNextU-Net` опережает все `FNO`-конфигурации

Примечательно, что для `ConvNextU-Net` лучшим оказался `full-frame`-вариант, тогда как для `FNO` лидировали `delta`-конфигурации.

## Структура репозитория

- `dataset.ipynb` - работа с датасетом и подготовка выборок
- `data_explore.ipynb` - EDA и первичная визуализация
- `models_eval.ipynb` - сравнение baseline и улучшенных моделей
- `training_scripts/train_fno_delta.py` - обучение `delta`-вариантов `FNO`
- `training_scripts/train_fno_full_frame.py` - обучение `full-frame`-вариантов `FNO`
- `training_scripts/train_cnextunet_delta.py` - обучение `delta`-вариантов `ConvNextU-Net`
- `training_scripts/train_cnextunet_full_frame.py` - обучение `full-frame`-вариантов `ConvNextU-Net`
- `autoregressive_pretrained_fno.py` - rollout и визуализация предобученной baseline-модели и FNO моделей
- `autoregressive_cnextunet.py` - rollout и визуализация `ConvNextU-Net`
- `results/FNO/` - итоговые таблицы по метрикам для `FNO`-моделей
- `results/U-net/` - итоговые таблицы по метрикам для `ConvNextU-Net`-моделей
- `plot_gifs_out/FNO/` - GIF-анимации rollout для `FNO`-моделей
- `plot_gifs_out/U-net/` - GIF-анимации rollout для `ConvNextU-Net`-моделей
- `models_trained/` - сохраненные чекпоинты моделей

