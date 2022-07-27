# Udemy_courses
A brief analysis about the courses offered by Udemy

# Vertex AI: Serverless framework for MLOPs (ESP / ENG)

## Español

### Qué es esto?
Este repo contiene un pipeline end to end diseñado usando el SDK de Kubeflow Pipelines (KFP). En el contexto del uso de Vertex AI como solución, la idea es construir una arquitectura de machine learning lo más automatizada posible, integrando algunos de los principales servicios de Google Cloud Platform (GCP) tales como BigQuery (data warehousing), Google Cloud Storage (almacenamiento de objetos) y Container Registry (repositorio de inágenes de Docker).

### Cómo lo corro?
- Primero, ejecutar la notebook **pipeline_setup.ipynb**. Contiene la configuración de la infraestructura que será utilizada: se crean datasets en BigQuery y buckets en GCS y se instalan librerías necesarias. Además se crean imágenes de Docker y se pushea a Container Registry para los jobs de tuneos de hiperparámetros. 
- Segundo, dentro de la carpeta *components* se encuentra la notebook **components_definition.ipynb** que deberá ejecutarse para generar los .yamls que serán invocados en la notebook principal de ejecución. 
- Por último, seguir los pasos indicados en **pipeline_run.ipynb**. Algunos parámetros como la cantidad de trials de hiperparámetros o los tipos de máquina deseadas para algunos pasos pueden ser fácilmente modificables.

### TO-DO
agregar costo estimado
permisos
