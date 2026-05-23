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
