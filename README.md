# INVII - Autenticación de Carteras

Este repositorio es una versión modular del notebook `Autenticacion_Carteras.ipynb`.

## Documentación

- 📘 Reporte completo del proyecto:  
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

El dataset no se incluye en este repositorio debido a su tama?o (>1GB).

Se encuentra disponible en Google Drive:

?? [Link al dataset]https://drive.google.com/file/d/1LdI5wrySA2Rrj32BoNPPDrwc6fM8g4QV/view?usp=sharing

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

## Nota

El dataset no se sube a GitHub porque pesa más de 1GB.
