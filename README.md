# Libro Aprendizaje Profundo

Estos apuntes están hechos utilizando Jupyter Book v2 y cubren los aspectos más fundamentales de un curso de Aprendizaje Profundo a nivel de pregrado, utilizando las bibliotecas PyTorch y Keras. El libro está publicado en https://www.jorgeanais.cl/libro_aprendizaje_profundo/ 


## Comandos útiles Jupyter book


Crear un repo en Github y copiarlo localmente
```bash 
git clone git@github.com:jorgeanais/libro_aprendizaje_profundo.git  # Reemplazar según corresponda
cd libro_aprendizaje_profundo
```

Instalar las dependencias y entorno virtual
```bash 
python -m venv venv     # opcional
source venv/bin/active  # opcional
pip install jupyter-book ghp-import
```

Inicializar proyecto
```bash
jupyter book init
```

Agrega archivos .md o .ipynb normalmente dentro del proyecto
```bash
libro_aprendizaje_profundo/
├── myst.yml
├── index.md              # portada
└── capitulos/
    ├── introduccion.md
    └── redes_neuronales.ipynb
```

Agrega tus capítulos al myst.yml
```yaml
project:
  title: Aprendizaje Profundo
  toc:
    - file: index.md
    - title: Fundamentos
      children:
        - file: cap1/000_motivacion.md
        - file: cap1/010_perceptron.ipynb
        - file: cap1/020_perceptron_multicapa.md
        - file: cap1/030_GPUvsCPU.ipynb
        ...
```

Previsualiza en local
```bash
jupyter book start
```

## Compila y publica en GitHub Pages (primera vez)

1. Conectar carpeta local con el repo (solo por una vez)
```bash
git add .
git commit -m 'primer commit'
git branch -M main
git push -u origin main.
```

2. Genera la GitHub Action de despliegue
```bash
jupyter book init --gh-pages
```
El prompt preguntará la rama (main) y el nombre del archivo de Action (deja el default deploy.yml). Esto genera `.github/workflows/deploy.yml` automáticamente.

3. Sube el archivo de la GitHub Action
```bash
git add .github/workflows/deploy.yml
git commit -m 'agrega deploy action'
git push
```

Tomar un café y esperar el despliegue automático. En la pestaña 'Actions' del repo se puede ver el build corriendo. 
Al terminar, tu libro estará en https://TU-USUARIO.github.io/libro-aprendizaje-profundo/. 

## Workflow
1. Agregar archivos `.md` o `.ipynb` en el directorio (yo lo organice en carpetas cap1, cap2, etc...)
2. Actualizar el documento myst.yml con la estructura deseada
3. Una vez listos los cambios
```bash
git add .
git commit -m "mensaje descriptivo"
git git push
```
4. Cada vez que hagas git push a main, la Action recompila y republica sola la nueva versión del libro
5. Disfrutar y tomarse un cafecito

## Documentación

Revisa la documentación oficial de [Jupyter Book](https://jupyterbook.org/stable/)

