# 🤖 Agente Data Analyst

*Un agente inteligente para análisis de datos automatizado con visualizaciones adaptativas*

[🚀 Demo](#demo) • [📋 Características](#características) • [⚡ Instalación](#instalación) • [🎯 Uso](#uso)


## 🌟 Descripción

**Agente Data Analyst** es una aplicación inteligente que combina el poder de la inteligencia artificial con análisis de datos automatizado. Permite a los usuarios hacer preguntas en lenguaje natural sobre sus datos y obtener visualizaciones automáticas y análisis detallados.

### ✨ Características Principales

- 🧠 **IA Conversacional**: Interactúa con tus datos usando lenguaje natural
- 📊 **Visualizaciones Automáticas**: Genera gráficos adaptativos basados en el contexto
- 🔍 **Detección Inteligente**: Reconoce automáticamente tipos de columnas y patrones
- 🌐 **Interfaz Web**: Aplicación moderna construida con Streamlit
- 📈 **Múltiples Tipos de Gráficos**: Barras, líneas, dispersión, histogramas y más
- 🎯 **Análisis Contextual**: Comprende preguntas específicas sobre género, tiempo, categorías

## 🚀 Demo

### Interfaz Principal
![Interfaz de la Aplicación](./assets/Insight%201.png)

### Carga de Datos y Análisis
![Carga de Datos](./assets/Insight%202.png)

### Generación de Gráficos
![Generación de Gráficos](./assets/Insight%203.png)

### Resultados y Visualizaciones
![Resultados](./assets/Insight%204.png)

*Ejemplo: "Genera un gráfico para conocer cuántos hombres y mujeres han sido arrestados"*

## 🛠️ Tecnologías

- **Backend**: Python 3.8+
- **Frontend**: Streamlit
- **IA**: OpenAI GPT-4
- **Visualización**: Matplotlib, Seaborn
- **Procesamiento**: Pandas, NumPy

## ⚡ Instalación

### Prerrequisitos

- Python 3.8 o superior
- Clave API de OpenAI

### Pasos de Instalación

1. **Clona el repositorio**
   ```bash
   git clone https://github.com/tu-usuario/agente-data-analyst.git
   cd agente-data-analyst
   ```

2. **Instala las dependencias**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configura las variables de entorno**
   ```bash
   cp .env.example .env
   ```
   
   Edita el archivo `.env` y añade tu clave API de OpenAI:
   ```
   OPENAI_API_KEY=tu_clave_api_aqui
   ```

4. **Ejecuta la aplicación**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Abre tu navegador** en `http://localhost:8501`

## 🎯 Uso

### Carga de Datos

1. Sube tu archivo CSV usando el widget de carga
2. El agente analizará automáticamente la estructura de tus datos

### Análisis Conversacional

```
👤 Usuario: "Muestra la distribución de ventas por región"
🤖 Agente: [Genera automáticamente un gráfico de barras con las ventas por región]

👤 Usuario: "¿Cuál es la tendencia de ventas en los últimos meses?"
🤖 Agente: [Crea un gráfico de líneas mostrando la evolución temporal]
```

### Ejemplos de Preguntas

- 📊 "Genera un gráfico de barras para las categorías de productos"
- 📈 "Muestra la tendencia de ventas por mes"
- 🔍 "Analiza la distribución por género"
- 📉 "Crea un histograma de los precios"
- 🎯 "Compara las ventas entre regiones"

## 📁 Estructura del Proyecto

```
agente-data-analyst/
├── app/
│   ├── __init__.py
│   ├── agent.py          # Lógica del agente IA
│   ├── data_loader.py    # Carga y procesamiento de datos
│   └── plotter.py        # Generación de visualizaciones
├── data/
│   └── raw/              # Datos de ejemplo
├── .env.example          # Plantilla de variables de entorno
├── .gitignore
├── requirements.txt      # Dependencias del proyecto
├── streamlit_app.py      # Aplicación principal
└── README.md
```

## 🔧 Configuración Avanzada

### Variables de Entorno

| Variable | Descripción | Requerido |
|----------|-------------|----------|
| `OPENAI_API_KEY` | Clave API de OpenAI | ✅ |
| `OPENAI_MODEL` | Modelo a usar (default: gpt-4) | ❌ |

### Personalización

Puedes personalizar el comportamiento del agente modificando:

- **`app/agent.py`**: Lógica de procesamiento de preguntas
- **`app/plotter.py`**: Tipos de gráficos y estilos
- **`streamlit_app.py`**: Interfaz de usuario

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🙏 Agradecimientos

- [OpenAI](https://openai.com/) por la API de GPT-4
- [Streamlit](https://streamlit.io/) por el framework de aplicaciones web
- [Matplotlib](https://matplotlib.org/) y [Seaborn](https://seaborn.pydata.org/) por las capacidades de visualización

---

<div align="center">

**¿Te gusta el proyecto? ¡Dale una ⭐!**

[Reportar Bug](https://github.com/tu-usuario/agente-data-analyst/issues) • [Solicitar Feature](https://github.com/tu-usuario/agente-data-analyst/issues) • [Documentación](https://github.com/tu-usuario/agente-data-analyst/wiki)

</div>