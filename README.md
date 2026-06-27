# Autenticación Automática de Carteras de Lujo mediante Deep Learning y Visión por Computadora

## Descripción


Este proyecto desarrolla un sistema de autenticación automática de carteras de lujo mediante Inteligencia Artificial y análisis de imágenes capturadas con teléfonos móviles.
El enfoque utiliza modelos de Deep Learning capaces de analizar tanto imágenes generales de la cartera como componentes específicos del producto, incluyendo logotipos, texturas, costuras, herrajes y turnlocks.
Para la implementación se empleó transferencia de aprendizaje con la arquitectura EfficientNet-B0, utilizando datasets organizados en las categorías genuine y fake.
Además, se implementó un pipeline completo de entrenamiento, validación y evaluación del modelo, incorporando métricas como:

Accuracy
Precision
Recall
F1-Score
Matriz de Confusión

El sistema fue probado en componentes de carteras Chanel 2.55 y Boy, obteniendo resultados iniciales que permiten evaluar la viabilidad del enfoque de autenticación automática basado en imágenes.

## Objetivo General

Desarrollar un sistema de autenticación automática de artículos de moda, orientado a identificar falsificaciones de alta calidad mediante modelos de Deep Learning aplicados a imágenes generales del producto y a componentes específicos como logo y turnlock.

## Tecnologías Utilizadas

- Python — lenguaje principal de desarrollo
- TensorFlow / Keras — entrenamiento y construcción del modelo de Deep Learning
- EfficientNet-B0 — arquitectura CNN utilizada mediante Transfer Learning
- NumPy — procesamiento numérico y manejo de arreglos
- Scikit-learn — métricas, evaluación y balanceo de clases
- Matplotlib — visualización de métricas y gráficas de entrenamiento
- Google Colab — entorno de entrenamiento en la nube con GPU
- Google Drive — almacenamiento y gestión del dataset y modelos entrenados


## Modelo Utilizado

Se utiliza EfficientNet-B0 como backbone preentrenado mediante transferencia de aprendizaje (Transfer Learning), aprovechando pesos entrenados previamente en ImageNet.

El modelo fue implementado para la autenticación de componentes específicos de carteras Chanel, utilizando imágenes de logo y turnlock pertenecientes a los modelos Chanel 2.55 y Chanel Boy.

El enfoque corresponde a una clasificación binaria supervisada:

- Fake
- Genuine

Además, se aplicaron técnicas de aumento de datos y balanceo de clases mediante class weights para reducir el sesgo generado por el desbalance del dataset.

## Documentación del Pipeline

[Proyecto Pipeline](docs/Proyecto_Pipeline.pdf)

## Flujo en base al notebook
1. Dataset organizado en clases genuine y fake, incluyendo componentes logo y turnlock de carteras Chanel 2.55 y Boy
2. División automática del dataset en conjuntos de entrenamiento y validación
3. Preprocesamiento y aumento de datos de imágenes (flip horizontal y normalización interna de EfficientNetB0)
4. Implementación del modelo de transferencia de aprendizaje EfficientNet-B0
5. Entrenamiento supervisado utilizando pesos de clase para balancear el dataset
6. Validación del modelo durante el entrenamiento
7. Evaluación del rendimiento mediante accuracy y loss
8. Generación de métricas de clasificación y matriz de confusión para las clases genuine y fake


## MODELO DE DETECCIÓN 
LINK CUADERNO: [Cuaderno](notebooks/yolo_chanel.ipynb)

## Entorno de ejecucion

- Plataforma: Google Colab
- Acelerador: GPU
- GPU utilizada: NVIDIA A100-SXM4-40GB
- Libreria principal: Ultralytics YOLO
- Version registrada: `ultralytics 8.4.51`
- Modelo base: `yolov8n.pt`

## Dataset

El dataset se carga desde Google Drive como archivo comprimido `DATASET.zip` y se descomprime en `/content/work/DATASET`.

Estructura detectada:

| Division | Imagenes | Anotaciones TXT |
|---|---:|---:|
| Entrenamiento | 713 | 713 |
| Validacion | 108 | 108 |
| **Total** | **821** | **821** |

Resumen de validacion del dataset:

- Imagenes corruptas: 0
- Imagenes sin anotaciones: 0
- Total de instancias en validacion: 110
- Clases del modelo:
  - `logo`
  - `turnlock`

Distribucion de instancias en validacion:

| Clase | Instancias |
|---|---:|
| logo | 14 |
| turnlock | 96 |

## Configuracion del entrenamiento

| Parametro | Valor |
|---|---:|
| Modelo | `yolov8n.pt` |
| Tamano de imagen | 640 |
| Epocas configuradas | 80 |
| Batch size | 64 |
| Workers | 8 |
| Optimizador | AdamW automatico |
| AMP | Activado |
| Cache | Activado |
| Paciencia early stopping | 10 |

El entrenamiento estaba configurado para 80 epocas, pero se detuvo automaticamente en la epoca 12 por falta de mejora durante 10 epocas consecutivas.

## Resultados

Metricas finales del mejor modelo en validacion:

| Metrica | Valor |
|---|---:|
| Precision | 0.00298 |
| Recall | 0.88690 |
| mAP50 | 0.75852 |
| mAP50-95 | 0.46104 |

Resultados por clase en mAP50-95:

| Clase | mAP50-95 |
|---|---:|
| logo | 0.44577 |
| turnlock | 0.47631 |



## Dataset

https://drive.google.com/drive/folders/1rO40Dx3AqRcatTunt6by7Mrim1Oj-I5T?usp=sharing

## Comparación v03 vs v04

Notebooks comparados:

- [Autenticacion_Carteras__Metricas_v03](notebooks/Autenticacion_Carteras__Metricas_v03.ipynb)
- [Autenticacion_Carteras__Metricas_v04](notebooks/Autenticacion_Carteras__Metricas_v04.ipynb)

### Modelo general de autenticación

| Métrica | v03 | v04 |
|---|---:|---:|
| Test loss | 0.35175031423568726 | 0.35175031423568726 |
| Test accuracy | 0.8469342589378357 | 0.8469342589378357 |
| Accuracy reportada | 0.85 | 0.85 |
| Macro F1-score | 0.84 | 0.84 |
| Weighted F1-score | 0.85 | 0.85 |

### Modelo de partes

| Métrica | v03 | v04 |
|---|---:|---:|
| Total de imágenes | 316 | 2129 |
| Train | 253 | 1704 |
| Validation | 63 | 425 |
| Validation loss | 0.5976 | 0.3709 |
| Validation accuracy | 0.6984 | 0.9035 |
| Macro F1-score | 0.66 | 0.68 |
| Weighted F1-score | 0.70 | 0.93 |

### Distribución del dataset de partes en v04

| Carpeta | Cantidad de imágenes |
|---|---:|
| genuine/bag---boy | 1632 |
| genuine/bag---2.55 | 414 |
| fake/bag---boy | 51 |
| fake/bag---2.55 | 32 |
| **Total** | **2129** |

### Reporte por clase del modelo de partes en v04

| Clase | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Fake | 0.27 | 0.94 | 0.42 | 16 |
| Genuine | 1.00 | 0.90 | 0.95 | 409 |

Promedios del reporte:

| Promedio | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Macro avg | 0.64 | 0.92 | 0.68 | 425 |
| Weighted avg | 0.97 | 0.90 | 0.93 | 425 |


## Descripcion de la Actualizacion v6

El objetivo principal de esta version fue determinar que arquitectura ofrece mejor rendimiento para la autenticacion de partes visuales de carteras, considerando componentes asociados a modelos Chanel 2.55 y Chanel Boy.

Las arquitecturas evaluadas fueron:

* EfficientNetB0
* MobileNetV2
* ResNet50

Todas las arquitecturas fueron utilizadas mediante transferencia de aprendizaje, aprovechando pesos preentrenados en ImageNet y adaptando la capa final para una clasificacion binaria.

## Dataset Utilizado

Para esta version se utilizo el dataset `Chanel_Parts.zip`, cargado desde Google Drive y descomprimido en el entorno de Google Colab.

El dataset contiene imagenes organizadas en dos clases principales:

* fake
* genuine

Cada clase contiene imagenes correspondientes a los modelos Chanel Boy y Chanel 2.55.

### Distribucion del Dataset de Partes en v06

| Carpeta | Cantidad de Imagenes |
|---|---:|
| genuine/bag---boy | 1632 |
| genuine/bag---2.55 | 414 |
| fake/bag---boy | 200 |
| fake/bag---2.55 | 200 |
| Total | 2446 |

La distribucion muestra que el dataset continua presentando desbalance entre clases, especialmente por la mayor cantidad de imagenes genuinas frente a imagenes falsas.

Para reducir el efecto de este desbalance durante el entrenamiento, se aplicaron pesos de clase mediante `class_weight`.

## Division del Dataset

El dataset fue dividido automaticamente en entrenamiento y validacion usando `validation_split=0.2`.

| Conjunto | Cantidad de Imagenes |
|---|---:|
| Entrenamiento | 1957 |
| Validacion | 489 |
| Total | 2446 |

La division se realizo con una semilla fija (`seed=123`) para mantener reproducibilidad en la separacion de datos.

## Configuracion General del Entrenamiento

Las tres arquitecturas fueron entrenadas bajo una configuracion similar para permitir una comparacion directa.

| Parametro | Valor |
|---|---|
| Entorno | Google Colab |
| Tipo de problema | Clasificacion binaria |
| Clases | fake, genuine |
| Tamano de imagen | 224 x 224 |
| Batch size | 16 |
| Division de validacion | 20% |
| Optimizador | Adam |
| Learning rate | 0.0001 |
| Funcion de perdida | binary_crossentropy |
| Metrica principal | accuracy |
| Aumento de datos | RandomFlip horizontal |
| Regularizacion | Dropout 0.3 |
| Transfer learning | Pesos ImageNet |
| Backbone entrenable | No |
| Pesos de clase | Activados |

## Modelo EfficientNetB0

La arquitectura fue cargada sin la capa superior (`include_top=False`) y con pesos preentrenados en ImageNet. Posteriormente se agregaron capas personalizadas para la clasificacion binaria:

* RandomFlip horizontal
* EfficientNetB0 como extractor de caracteristicas
* GlobalAveragePooling2D
* Dropout de 0.3
* Capa Dense con activacion sigmoid

### Resultados de EfficientNetB0

| Metrica | Valor |
|---|---:|
| Accuracy de validacion | 0.885481 |
| Accuracy de validacion (%) | 88.548058 |
| Loss de validacion | 0.338831 |
| Error (%) | 11.451942 |

EfficientNetB0 obtuvo un rendimiento solido, aunque fue superado por ResNet50 en la evaluacion comparativa final.

## Modelo MobileNetV2

El modelo fue entrenado mediante transferencia de aprendizaje, manteniendo congelado el backbone preentrenado y agregando capas finales para clasificacion binaria.

Tambien se incorporo EarlyStopping con monitoreo de `val_loss`, paciencia de 5 epocas y restauracion de los mejores pesos.

### Resultados de MobileNetV2

| Metrica | Valor |
|---|---:|
| Mejor accuracy de validacion | 0.834356 |
| Accuracy final de validacion | 0.828221 |
| Accuracy final de validacion (%) | 82.822084 |
| Loss de validacion | 0.434094 |
| Error (%) | 17.177916 |

MobileNetV2 fue el modelo con menor rendimiento dentro de la comparacion. Sin embargo, sigue siendo una alternativa relevante si se prioriza ligereza y menor costo de inferencia frente a mayor precision.

## Modelo ResNet50

Al igual que los otros modelos, se utilizo transferencia de aprendizaje con pesos ImageNet, congelando el backbone y entrenando solamente las capas superiores personalizadas para clasificacion binaria.

Tambien se incorporo EarlyStopping monitoreando `val_loss`, con paciencia de 5 epocas y restauracion de los mejores pesos.

### Resultados de ResNet50

| Metrica | Valor |
|---|---:|
| Mejor accuracy de validacion | 0.926380 |
| Accuracy final de validacion | 0.926380 |
| Accuracy final de validacion (%) | 92.638040 |
| Loss de validacion | 0.227940 |
| Error (%) | 7.361960 |

ResNet50 obtuvo el mejor rendimiento de la version v06, superando tanto a EfficientNetB0 como a MobileNetV2.

## Comparacion General de Modelos en v06

| Ranking | Modelo | Accuracy | Loss | Accuracy (%) | Error (%) |
|---:|---|---:|---:|---:|---:|
| 1 | ResNet50 | 0.926380 | 0.227940 | 92.638040 | 7.361960 |
| 2 | EfficientNetB0 | 0.885481 | 0.338831 | 88.548058 | 11.451942 |
| 3 | MobileNetV2 | 0.828221 | 0.434094 | 82.822084 | 17.177916 |


## Actualizacion v07

La version v07 mantiene el dataset `Chanel_Parts.zip` con 2446 imagenes y refuerza la evaluacion con graficas comparativas, matriz de confusion y validacion K-Fold para buscar umbrales globales.

| Modelo | Accuracy val | Loss val | Mejor epoca accuracy | Mejor epoca loss |
|---|---:|---:|---:|---:|
| ResNet50 | 0.926380 | 0.235909 | 18 | 18 |
| EfficientNetB0 | 0.873211 | 0.357648 | 20 | 20 |
| MobileNetV2 | 0.844581 | 0.430654 | 19 | 19 |

Resumen K-Fold v07 para umbral global:

| Modelo | Umbral promedio | Accuracy promedio | Precision promedio | Recall promedio | F1 promedio |
|---|---:|---:|---:|---:|---:|
| ResNet50 | 0.682 | 0.662571 | 0.866330 | 0.704456 | 0.768646 |
| EfficientNetB0 | 0.918 | 0.245740 | 1.000000 | 0.091268 | 0.152726 |
| MobileNetV2 | 0.908 | 0.210562 | 1.000000 | 0.049353 | 0.090633 |

## Actualizacion v08

La version v08 reemplaza las arquitecturas ligeras/anteriormente usadas por modelos mas actuales y registra tiempos de entrenamiento. Se evaluan:

- EfficientNetB3 con imagenes de 300 x 300
- MobileNetV3Large con imagenes de 224 x 224
- ResNet50 con imagenes de 224 x 224

Dataset y division usados en v08:

| Conjunto / clase | Cantidad |
|---|---:|
| Total de imagenes | 2446 |
| Entrenamiento | 1957 |
| Validacion | 489 |
| Fake en validacion | 83 |
| Genuine en validacion | 406 |

Distribucion del dataset:

| Carpeta | Imagenes |
|---|---:|
| genuine/bag---boy | 1632 |
| genuine/bag---2.55 | 414 |
| fake/bag---boy | 200 |
| fake/bag---2.55 | 200 |
| Total | 2446 |

### Ranking v08 por evaluacion final

| Ranking | Modelo | Accuracy | Loss | Accuracy (%) | Error (%) |
|---:|---|---:|---:|---:|---:|
| 1 | ResNet50 | 0.932515 | 0.219140 | 93.251532 | 6.748468 |
| 2 | MobileNetV3 | 0.912065 | 0.270404 | 91.206545 | 8.793455 |
| 3 | EfficientNetB3 | 0.883436 | 0.343516 | 88.343561 | 11.656439 |

### Mejores metricas de validacion v08

| Modelo | Mejor Val Accuracy | Mejor Val Loss | Epocas | Tiempo total |
|---|---:|---:|---:|---:|
| ResNet50 | 0.932515 | 0.219140 | 20 | 65.42 s |
| MobileNetV3 | 0.918200 | 0.270405 | 20 | 81.78 s |
| EfficientNetB3 | 0.883436 | 0.343516 | 20 | 188.02 s |

### Reporte de clasificacion v08

| Modelo | Clase | Precision | Recall | F1-score | Support |
|---|---|---:|---:|---:|---:|
| EfficientNetB3 | Fake | 0.62 | 0.80 | 0.70 | 83 |
| EfficientNetB3 | Genuine | 0.96 | 0.90 | 0.93 | 406 |
| MobileNetV3 | Fake | 0.68 | 0.90 | 0.78 | 83 |
| MobileNetV3 | Genuine | 0.98 | 0.91 | 0.95 | 406 |
| ResNet50 | Fake | 0.78 | 0.84 | 0.81 | 83 |
| ResNet50 | Genuine | 0.97 | 0.95 | 0.96 | 406 |

| Modelo | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| ResNet50 | 0.93 | 0.88 | 0.93 |
| MobileNetV3 | 0.91 | 0.86 | 0.92 |
| EfficientNetB3 | 0.88 | 0.81 | 0.89 |

### K-Fold de umbral global v08

| Modelo | Umbral promedio | Umbral mediana | Accuracy promedio | Precision promedio | Recall promedio | F1 promedio |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV3 | 0.536 | 0.50 | 0.897517 | 0.985737 | 0.889100 | 0.933766 |
| ResNet50 | 0.636 | 0.63 | 0.897559 | 0.989702 | 0.886540 | 0.933656 |
| EfficientNetB3 | 0.652 | 0.65 | 0.785357 | 0.994118 | 0.746281 | 0.850238 |

## Comparacion v07 vs v08

| Aspecto | v07 | v08 |
|---|---:|---:|
| Mejor modelo por accuracy | ResNet50 | ResNet50 |
| Mejor accuracy final | 0.926380 | 0.932515 |
| Mejor loss final | 0.235909 | 0.219140 |
| Mejor F1 promedio K-Fold | 0.768646 | 0.933766 |
| Mejor umbral promedio K-Fold | 0.682 | 0.536 |
| F1 Fake del mejor modelo | 0.16 | 0.81 |

En v08 el mejor resultado global sigue siendo ResNet50, pero MobileNetV3 queda muy cerca en K-Fold. La mejora principal frente a v07 esta en la clase `Fake`: el F1 del mejor modelo sube de 0.16 a 0.81 y el F1 promedio K-Fold sube de 0.768646 a 0.933766.



## MATRIZ DE CONSISTENCIA
LINK MATRIZ: [Matriz](docs/MATRIZ_CONSISTENCIA.pdf)

