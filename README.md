# Autenticación Automática de Carteras de Lujo mediante Deep Learning y Visión por Computadora

## Descripción

Este proyecto desarrolla un sistema de autenticación automática de carteras de lujo mediante inteligencia artificial y análisis de imágenes capturadas con teléfonos móviles.  
El enfoque utiliza modelos de aprendizaje profundo capaces de analizar múltiples componentes del producto, como logotipos, costuras, texturas y herrajes, permitiendo una evaluación más precisa frente a falsificaciones de alta calidad.  
Se implementa un pipeline completo que incluye procesamiento de datos, entrenamiento, validación y evaluación con métricas como AUC-ROC y F1-score.  
El sistema busca mejorar la precisión, objetividad y escalabilidad del proceso de autenticación en escenarios reales del contexto peruano.

## Objetivo General

Desarrollar un sistema de autenticación automática de artículos de moda frente a falsificaciones de alta calidad, basado en el análisis de imágenes capturadas mediante teléfonos celulares, que permita mejorar la precisión, objetividad y escalabilidad del proceso de verificación en el contexto peruano.

## Tecnologías Utilizadas

- **Python**: Lenguaje principal utilizado para el desarrollo del pipeline de procesamiento, entrenamiento y evaluación del modelo.
- **PyTorch**: Framework de deep learning empleado para la construcción y entrenamiento del modelo.
- **Torchvision**: Librería utilizada para modelos preentrenados (EfficientNet-B0) y transformaciones de imágenes.
- **NumPy**: Manejo de operaciones numéricas y cálculo de distancias entre embeddings.
- **Pillow (PIL)**: Carga y procesamiento básico de imágenes.
- **Scikit-learn**: Evaluación del modelo mediante métricas como ROC-AUC y curva ROC.
- **Matplotlib**: Visualización de resultados y métricas.
- **TQDM**: Monitoreo del progreso durante el entrenamiento.

## Modelo Utilizado

El sistema utiliza **EfficientNet-B0** como extractor de características visuales, generando embeddings representativos de cada imagen.

Se implementa un enfoque de **One-Class Learning**, donde el modelo se entrena únicamente con imágenes de carteras auténticas. A partir de ello, se calcula un centro de embeddings y se evalúan nuevas imágenes midiendo su distancia respecto a este centro.

La decisión final se basa en un umbral definido por percentiles:

- **Auténtico**: si la distancia es menor al umbral
- **Rechazado**: si la distancia supera el umbral

Este enfoque permite detectar falsificaciones de alta calidad de manera eficiente, objetiva y escalable.

## Documentación del Sprint 1

- Reporte completo del proyecto:  
  [Proyecto Pipeline](docs/Proyecto_Pipeline.pdf)

Este documento incluye:
- Resumen ejecutivo
- Pipeline de datos
- EDA
- Modelo baseline (EfficientNet-B0)
- Métricas (ROC-AUC, pérdidas, distancias)
- Resultados experimentales

## Flujo en base al notebook

1. Dataset genuino organizado por modelo y cartera.
2. Split por cartera en `train`, `val` y `test`.
3. Aumentación en entrenamiento con:
   - RandomResizedCrop
   - ColorJitter
   - GaussianBlur
   - RandomErasing
4. Modelo EfficientNet-B0 con embedding de 256 dimensiones.
5. Cálculo del centro one-class.
6. Entrenamiento con OneClassLoss.
7. Cálculo de distancias.
8. Cálculo de threshold por percentil.
9. Generación de fake hard con oclusión, affine warp, color jitter y blur.
10. Evaluación ROC-AUC.
11. Predicción por cartera.

## Dataset

El dataset no se incluye en este repositorio debido a su tamaño (>1GB).

Se encuentra disponible en Google Drive:

[Descargar dataset](https://drive.google.com/file/d/1LdI5wrySA2Rrj32BoNPPDrwc6fM8g4QV/view?usp=sharing)

## Estructura del dataset

Colocar el dataset genuino aquí:

```text
data/genuine/chanel/
└── modelo/
    └── bag_id/
        ├── img1.jpg
        ├── img2.jpg
        └── ...
```

Ejemplo:

```text
data/genuine/chanel/classic/
└── 112/
    ├── front.jpg
    ├── logo.jpg
    └── zipper.jpg
```

## Ejecutar pipeline completo

```bash
pip install -r requirements.txt
python src/main.py
```

## Predicción de una cartera

Coloca imágenes de una cartera en:

```text
handbag/112/
```

Ejecuta:

```bash
python src/predict_bag.py --bag_dir handbag/112
```
