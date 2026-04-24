# Dataset

## Dataset usado en el notebook

El notebook usa una estructura por:

- Marca/categoría: `chanel`
- Modelo
- ID de cartera
- Imágenes por cartera

## Estructura

```text
data/genuine/chanel/
└── model_name/
    └── bag_id/
        ├── image_1.jpg
        ├── image_2.jpg
        └── image_n.jpg
```

## Split

El notebook separa por **cartera**, no por imagen:

- train: 70%
- val: 15%
- test: 15%

Esto evita que imágenes de la misma cartera aparezcan en train y test al mismo tiempo.
