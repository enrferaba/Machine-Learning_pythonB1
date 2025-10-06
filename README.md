# Machine Learning B1 – Recursos de práctica

Este repositorio contiene los cuadernos y utilidades que usamos en clase.
Para poder ejecutar los ejercicios que dependen de las imágenes de práctica
necesitas reconstruir la carpeta `Files-20250930 (2)/files/` con el paquete
oficial del profesor.

## Preparar las imágenes sin subir binarios

El campus virtual distribuye un archivo `Files-20251006.zip` con la siguiente
estructura interna:

```
files/
  images.tar
  ...
```

GitHub suele rechazar archivos binarios grandes, por lo que el ZIP y el TAR no
se incluyen en el repositorio. En su lugar:

1. Descarga `Files-20251006.zip` desde la plataforma del curso.
2. Copia el archivo dentro de `Files-20250930 (2)/` (no se subirá gracias a
   `.gitignore`).
3. Abre y ejecuta `Files-20250930 (2)/prepare_official_files.ipynb` para
   descomprimir el paquete y dejar listas las imágenes junto a la carpeta de
   salida `files/reduced_images/`.
4. (Opcional) Ejecuta la última celda del cuaderno para comprobar la integridad
   del ZIP (`zipfile.ZipFile(...).testzip()` devuelve `None` cuando todo está
   correcto).

Una vez ejecutado el cuaderno, los notebooks de trabajo (`trabajo.ipynb` y
`trabajo_step_by_step.ipynb`) encontrarán automáticamente las imágenes porque
`class_helpers.ensure_image_resources()` prepara las rutas necesarias.

## Notebooks principales

- `trabajo.ipynb`: Resolución principal del proyecto.
- `trabajo_step_by_step.ipynb`: Versión guiada con explicaciones paso a paso.

Consulta la carpeta `Files-20250930 (2)/` para ver el resto de material auxiliar
que utilizamos en las sesiones.
