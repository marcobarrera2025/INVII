# Autenticación Automática de Carteras de Lujo mediante Deep Learning y Visión por Computadora

## Descripción

Este proyecto desarrolla un sistema de autenticación automática de carteras de lujo mediante inteligencia artificial y análisis de imágenes capturadas con teléfonos móviles.  
El enfoque utiliza modelos de aprendizaje profundo capaces de analizar múltiples componentes del producto, como logotipos, costuras, texturas y herrajes.  
Se implementa un pipeline completo con métricas como accuracy, precision, recall y F1-score.

## Objetivo General

Desarrollar un sistema de autenticación automática de artículos de moda frente a falsificaciones de alta calidad.

## Tecnologías Utilizadas

- Python  
- TensorFlow / Keras  
- NumPy  
- Pillow  
- Scikit-learn  
- Matplotlib  

## Modelo Utilizado

Se utiliza EfficientNet-B0 como backbone preentrenado.

El modelo sigue un enfoque de clasificación binaria supervisada:

- Fake  
- Genuine  

## Flujo en base al notebook

1. Dataset organizado en genuine y fake  
2. Split por carpeta (train, val, test)  
3. Preprocesamiento de imágenes  
4. Modelo EfficientNet-B0  
5. Entrenamiento supervisado  
6. Validación  
7. Evaluación en test  
8. Métricas y matriz de confusión  

## Dataset

https://drive.google.com/file/d/1LdI5wrySA2Rrj32BoNPPDrwc6fM8g4QV/view

