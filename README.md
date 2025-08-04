# Insight-Buddy MVP

Un agente de análisis de datos que proporciona respuestas en lenguaje natural y visualizaciones simples sobre datasets tabulares locales.

## Características

- Análisis de datos CSV/Parquet con consultas en lenguaje natural
- Generación automática de visualizaciones
- Interfaz web con Streamlit
- Búsqueda semántica con FAISS
- Respuestas contextuales usando LangChain + OpenAI

## Stack Tecnológico

- Python 3.11
- LangChain ≥ 0.1.x
- OpenAI API
- FAISS (búsqueda vectorial)
- Pandas/Polars (manipulación de datos)
- DuckDB (opcional)
- Streamlit (interfaz web)
- Matplotlib (visualizaciones)

## Instalación

1. Crear y activar entorno virtual:
```bash
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1
# Linux/Mac
source venv/bin/activate
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno:
```bash
# Crear archivo .env con tu API key de OpenAI
OPENAI_API_KEY=tu_api_key_aqui
```

## Uso

### CLI
```bash
# Ingestar datos
python ingest.py path/to/file.csv

# Hacer preguntas
python ask.py "¿Cuál fue el total de ventas en 2024?"
```

### Interfaz Web
```bash
streamlit run streamlit_app.py
```

## Estructura del Proyecto

```
├── app/                 # Código principal de la aplicación
├── data/               # Datasets y índices FAISS
├── notebooks/          # Jupyter notebooks de ejemplo
├── outputs/            # Gráficos generados
├── requirements.txt    # Dependencias
├── README.md          # Este archivo
└── LICENSE            # Licencia del proyecto
```

## Supuestos

- Datasets ≤ 100 MB
- Máquina local ≥ 8 GB RAM
- Conexión a internet para API de OpenAI

## Licencia

MIT License