import streamlit as st
import os
import tempfile
from pathlib import Path
from app.agent import TabularQAAgent
from app.data_loader import DataLoader
import pandas as pd
from PIL import Image

# Configuración de la página
st.set_page_config(
    page_title="Insight-Buddy MVP",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📊 Insight-Buddy MVP")
st.markdown("**Sube tu CSV → Haz preguntas → Obtén insights automáticos**")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Verificar variables de entorno
    if not os.getenv('OPENAI_API_KEY'):
        st.error("⚠️ OPENAI_API_KEY no configurada")
        st.info("Configura tu API key en las variables de entorno")
        st.stop()
    
    st.success("✅ API Key configurada")
    
    # Información del proyecto
    st.markdown("---")
    st.markdown("### 📋 Funcionalidades")
    st.markdown("""
    - 📁 Carga de archivos CSV
    - 🤖 Análisis con IA
    - 📊 Gráficos automáticos
    - 💬 Respuestas en lenguaje natural
    """)

# Inicializar session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'store_name' not in st.session_state:
    st.session_state.store_name = None
if 'current_dataframe' not in st.session_state:
    st.session_state.current_dataframe = None

# Sección de carga de archivos
st.header("📁 Carga tu Dataset")

uploaded_file = st.file_uploader(
    "Selecciona un archivo CSV",
    type=['csv'],
    help="Sube un archivo CSV para comenzar el análisis"
)

if uploaded_file is not None:
    try:
        # Guardar archivo temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        # Mostrar información del archivo
        st.success(f"✅ Archivo cargado: {uploaded_file.name}")
        
        # Cargar datos y crear agente
        with st.spinner("🔄 Procesando datos y creando índice..."):
            # Cargar datos
            data_loader = DataLoader()
            df = data_loader.load_csv(temp_path)
            
            # Crear vector store con nombre único basado en el archivo
            import hashlib
            file_hash = hashlib.md5(uploaded_file.name.encode()).hexdigest()[:8]
            store_name = f"streamlit_{file_hash}"
            
            # Procesar archivo y crear vector store
            vector_store = data_loader.process_file(temp_path, store_name)
            
            # Crear agente
            agent = TabularQAAgent()
            
            # Guardar el store_name para usar en las consultas
            st.session_state.store_name = store_name
            
            # Guardar en session state
            st.session_state.agent = agent
            st.session_state.data_loaded = True
            st.session_state.current_dataframe = df  # Guardar DataFrame para gráficos
            st.session_state.dataset_info = {
                'filename': uploaded_file.name,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict()
            }
        
        # Limpiar archivo temporal
        os.unlink(temp_path)
        
        st.success("🎉 ¡Dataset procesado exitosamente!")
        
        # Mostrar información del dataset
        with st.expander("📊 Información del Dataset", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Filas", st.session_state.dataset_info['shape'][0])
                st.metric("Columnas", st.session_state.dataset_info['shape'][1])
            
            with col2:
                st.write("**Columnas disponibles:**")
                for col in st.session_state.dataset_info['columns']:
                    st.write(f"• {col}")
            
            # Mostrar preview de los datos
            st.write("**Vista previa:**")
            st.dataframe(df.head(10), use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {str(e)}")
        st.session_state.data_loaded = False

# Sección de preguntas
st.header("💬 Haz tu Pregunta")

if st.session_state.data_loaded:
    # Ejemplos de preguntas
    st.markdown("**💡 Ejemplos de preguntas:**")
    example_questions = [
        "¿Cuáles son los productos más vendidos?",
        "Muestra la evolución de las ventas por mes",
        "¿Cuál es el promedio de ventas por categoría?",
        "Haz un gráfico de barras de los top 5 productos"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"📝 {question}", key=f"example_{i}"):
                st.session_state.current_question = question
    
    # Caja de texto para preguntas personalizadas
    question = st.text_area(
        "Escribe tu pregunta:",
        value=st.session_state.get('current_question', ''),
        height=100,
        placeholder="Ejemplo: ¿Cuáles son los productos más vendidos? Haz un gráfico"
    )
    
    # Botón para procesar pregunta
    if st.button("🚀 Analizar", type="primary", disabled=not question.strip()):
        if question.strip():
            with st.spinner("🤖 Analizando datos y generando respuesta..."):
                try:
                    # Obtener respuesta del agente
                    response = st.session_state.agent.ask(
                        question, 
                        st.session_state.store_name, 
                        df=st.session_state.current_dataframe
                    )
                    
                    # Mostrar respuesta
                    st.header("📋 Respuesta")
                    st.markdown(response['answer'])
                    
                    # Mostrar gráfico si existe
                    if 'chart' in response and response['chart']:
                        chart_path = response['chart']
                        if os.path.exists(chart_path):
                            st.header("📊 Visualización")
                            image = Image.open(chart_path)
                            st.image(image, caption="Gráfico generado automáticamente", use_column_width=True)
                    
                    # Mostrar información adicional
                    if 'sources' in response and response['sources']:
                        with st.expander("📚 Fuentes consultadas"):
                            for i, source in enumerate(response['sources'], 1):
                                st.write(f"**Fuente {i}:**")
                                st.write(source)
                                st.write("---")
                    
                    # Mostrar metadata
                    with st.expander("ℹ️ Información técnica"):
                        st.json({
                            'dataset': st.session_state.dataset_info['filename'],
                            'modelo': response.get('model', 'N/A'),
                            'timestamp': response.get('timestamp', 'N/A')
                        })
                
                except Exception as e:
                    st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                    st.info("💡 Intenta reformular tu pregunta o verifica que el dataset contenga la información solicitada.")
else:
    st.info("👆 Primero carga un archivo CSV para comenzar")
    
    # Mostrar información de ayuda
    with st.expander("❓ ¿Cómo usar Insight-Buddy?"):
        st.markdown("""
        ### 🚀 Pasos para usar la aplicación:
        
        1. **📁 Carga tu CSV**: Usa el botón de arriba para subir tu archivo
        2. **👀 Revisa los datos**: Verifica que la información se cargó correctamente
        3. **💬 Haz preguntas**: Escribe preguntas en lenguaje natural sobre tus datos
        4. **📊 Obtén insights**: Recibe respuestas y gráficos automáticos
        
        ### 💡 Tipos de preguntas que puedes hacer:
        
        - **Análisis descriptivo**: "¿Cuál es el promedio de ventas?"
        - **Comparaciones**: "¿Qué producto se vende más?"
        - **Tendencias**: "Muestra la evolución por mes"
        - **Visualizaciones**: "Haz un gráfico de barras"
        
        ### 📋 Requisitos del archivo:
        
        - Formato CSV
        - Tamaño máximo recomendado: 100 MB
        - Encabezados en la primera fila
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"  
    "💡 Insight-Buddy MVP - Análisis de datos con IA | "  
    "Desarrollado con Streamlit + LangChain + OpenAI"  
    "</div>", 
    unsafe_allow_html=True
)