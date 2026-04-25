# Autenticaci車n Autom芍tica de Carteras de Lujo mediante Deep Learning y Visi車n por Computadora

## Descripci車n

Este proyecto desarrolla un sistema de autenticaci車n autom芍tica de carteras de lujo mediante inteligencia artificial y an芍lisis de im芍genes capturadas con tel谷fonos m車viles.  
El enfoque utiliza modelos de aprendizaje profundo capaces de analizar m迆ltiples componentes del producto, como logotipos, costuras, texturas y herrajes, permitiendo una evaluaci車n m芍s precisa frente a falsificaciones de alta calidad.  
Se implementa un pipeline completo que incluye procesamiento de datos, entrenamiento, validaci車n y evaluaci車n con m谷tricas como AUC-ROC y F1-score.  
El sistema busca mejorar la precisi車n, objetividad y escalabilidad del proceso de autenticaci車n en escenarios reales del contexto peruano.

## Objetivo General

Desarrollar un sistema de autenticaci車n autom芍tica de art赤culos de moda frente a falsificaciones de alta calidad, basado en el an芍lisis de im芍genes capturadas mediante tel谷fonos celulares, que permita mejorar la precisi車n, objetividad y escalabilidad del proceso de verificaci車n en el contexto peruano.

## Tecnolog赤as Utilizadas

- **Python**: Lenguaje principal utilizado para el desarrollo del pipeline de procesamiento, entrenamiento y evaluaci車n del modelo.
- **PyTorch**: Framework de deep learning empleado para la construcci車n y entrenamiento del modelo.
- **Torchvision**: Librer赤a utilizada para modelos preentrenados (EfficientNet-B0) y transformaciones de im芍genes.
- **NumPy**: Manejo de operaciones num谷ricas y c芍lculo de distancias entre embeddings.
- **Pillow (PIL)**: Carga y procesamiento b芍sico de im芍genes.
- **Scikit-learn**: Evaluaci車n del modelo mediante m谷tricas como ROC-AUC y curva ROC.
- **Matplotlib**: Visualizaci車n de resultados y m谷tricas.
- **TQDM**: Monitoreo del progreso durante el entrenamiento.

## Modelo Utilizado

El sistema utiliza **EfficientNet-B0** como extractor de caracter赤sticas visuales, generando embeddings representativos de cada imagen.

Se implementa un enfoque de **One-Class Learning**, donde el modelo se entrena 迆nicamente con im芍genes de carteras aut谷nticas. A partir de ello, se calcula un centro de embeddings y se eval迆an nuevas im芍genes midiendo su distancia respecto a este centro.

La decisi車n final se basa en un umbral definido por percentiles:

- **Aut谷ntico**: si la distancia es menor al umbral  
- **Rechazado**: si la distancia supera el umbral  

Este enfoque permite detectar falsificaciones de alta calidad de manera eficiente, objetiva y escalable.

## Documentaci車n del Sprint 1

- Reporte completo del proyecto:  
  [Proyecto Pipeline](docs/Proyecto_Pipeline.pdf)

Este documento incluye:
- Resumen ejecutivo
- Pipeline de datos
- EDA
- Modelo baseline (EfficientNet-B0)
- M谷tricas (ROC-AUC, p谷rdidas, distancias)
- Resultados experimentales

## Flujo en base al notebook

1. Dataset genuino organizado por modelo y cartera.
2. Split por cartera en `train`, `val` y `test`.
3. Aumentaci車n en entrenamiento con:
   - RandomResizedCrop
   - ColorJitter
   - GaussianBlur
   - RandomErasing
4. Modelo EfficientNet-B0 con embedding de 256 dimensiones.
5. C芍lculo del centro one-class.
6. Entrenamiento con OneClassLoss.
7. C芍lculo de distancias.
8. C芍lculo de threshold por percentil.
9. Generaci車n de fake hard con oclusi車n, affine warp, color jitter y blur.
10. Evaluaci車n ROC-AUC.
11. Predicci車n por cartera.

## Dataset

El dataset no se incluye en este repositorio debido a su tama?o (>1GB).

Se encuentra disponible en Google Drive:

 [Descargar dataset](https://drive.google.com/file/d/1LdI5wrySA2Rrj32BoNPPDrwc6fM8g4QV/view?usp=sharing)

## Estructura del dataset

Colocar el dataset genuino aqu赤:

```text
data/genuine/chanel/
 modelo/
     bag_id/
         img1.jpg
         img2.jpg
         ...