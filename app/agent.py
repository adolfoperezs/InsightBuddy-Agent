"""Agente QA Tabular para Insight-Buddy MVP.

Este módulo implementa un agente de preguntas y respuestas que:
1. Recupera contextos relevantes del índice FAISS
2. Construye prompts estructurados para el LLM
3. Devuelve respuestas con citas de filas/columnas
4. Maneja errores y casos de fallback
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from dotenv import load_dotenv

from .data_loader import DataLoader
from .plotter import DataPlotter

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class TabularQAAgent:
    """Agente de preguntas y respuestas para datos tabulares."""
    
    def __init__(self, 
                 model_name: str = None,
                 temperature: float = 0.1,
                 max_tokens: int = 1000):
        """
        Inicializar el agente QA.
        
        Args:
            model_name: Nombre del modelo OpenAI (por defecto desde .env)
            temperature: Temperatura para la generación (0.0-1.0)
            max_tokens: Máximo número de tokens en la respuesta
        """
        self.model_name = model_name or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Inicializar el modelo de chat
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Inicializar el data loader
        self.data_loader = DataLoader()
        
        # Inicializar el plotter
        self.plotter = DataPlotter()
        
        # Plantilla de prompt personalizada
        self.prompt_template = self._create_prompt_template()
        
        logger.info(f"Agente QA inicializado con modelo: {self.model_name}")
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Crear plantilla de prompt estructurada para QA tabular."""
        
        template = """Eres un analista de datos experto que ayuda a responder preguntas sobre datasets tabulares.

CONTEXTO DE DATOS:
{context}

INSTRUCCIONES:
1. Analiza cuidadosamente los datos proporcionados en el contexto
2. Responde la pregunta basándote ÚNICAMENTE en la información disponible
3. Si los datos contienen información relevante, proporciona una respuesta clara y específica
4. Incluye números, fechas y detalles específicos cuando sea posible
5. Si la información no está disponible en los datos, indica claramente que no puedes responder
6. Cita las columnas o filas relevantes cuando sea apropiado

PREGUNTA: {question}

RESPUESTA:"""
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def load_vector_store(self, store_name: str):
        """Cargar el vector store especificado.
        
        Args:
            store_name: Nombre del store a cargar
            
        Returns:
            Vector store cargado o None si hay error
        """
        try:
            vector_store = self.data_loader.load_vector_store(store_name)
            logger.info(f"Vector store '{store_name}' cargado exitosamente")
            return vector_store
        except Exception as e:
            logger.error(f"Error cargando vector store '{store_name}': {e}")
            return None
    
    def create_qa_chain(self, vector_store, k: int = 3):
        """Crear cadena de QA con el vector store.
        
        Args:
            vector_store: Vector store para recuperación
            k: Número de documentos a recuperar
            
        Returns:
            Cadena RetrievalQA configurada
        """
        if vector_store is None:
            raise ValueError("Vector store no puede ser None")
        
        # Configurar el retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Crear la cadena QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={
                "prompt": self.prompt_template
            },
            return_source_documents=True
        )
        
        return qa_chain
    
    def ask(self, 
            question: str, 
            store_name: str, 
            k: int = 3) -> Dict[str, Any]:
        """Hacer una pregunta al agente.
        
        Args:
            question: Pregunta a realizar
            store_name: Nombre del vector store a usar
            k: Número de documentos a recuperar
            
        Returns:
            Diccionario con respuesta, fuentes y metadatos
        """
        try:
            # Cargar vector store
            vector_store = self.load_vector_store(store_name)
            if vector_store is None:
                return {
                    "answer": "Error: No se pudo cargar el dataset especificado.",
                    "sources": [],
                    "error": f"Vector store '{store_name}' no encontrado",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Crear cadena QA
            qa_chain = self.create_qa_chain(vector_store, k=k)
            
            # Ejecutar consulta
            logger.info(f"Procesando pregunta: {question[:100]}...")
            result = qa_chain({"query": question})
            
            # Procesar fuentes
            sources = []
            for doc in result.get("source_documents", []):
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            response = {
                "answer": result["result"],
                "sources": sources,
                "question": question,
                "store_name": store_name,
                "timestamp": datetime.now().isoformat(),
                "model_used": self.model_name
            }
            
            # Detectar si la pregunta requiere un gráfico y generarlo automáticamente
            chart_path = self.plotter.generate_chart_if_needed(question, store_name)
            if chart_path:
                response["chart"] = chart_path
                response["answer"] += f"\n\n📊 Gráfico generado: {chart_path}"
            
            logger.info("Pregunta procesada exitosamente")
            return response
            
        except Exception as e:
            logger.error(f"Error procesando pregunta: {e}")
            return {
                "answer": "Lo siento, ocurrió un error al procesar tu pregunta. Por favor, verifica que el dataset esté disponible e intenta nuevamente.",
                "sources": [],
                "error": str(e),
                "question": question,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_available_stores(self) -> list:
        """Obtener lista de stores disponibles.
        
        Returns:
            Lista de nombres de stores disponibles
        """
        try:
            index_path = os.getenv('FAISS_INDEX_PATH', 'data/index/')
            if not os.path.exists(index_path):
                return []
            
            stores = []
            for item in os.listdir(index_path):
                item_path = os.path.join(index_path, item)
                if os.path.isdir(item_path):
                    # Verificar que contenga archivos FAISS
                    if (os.path.exists(os.path.join(item_path, 'index.faiss')) and 
                        os.path.exists(os.path.join(item_path, 'index.pkl'))):
                        stores.append(item)
            
            return stores
            
        except Exception as e:
            logger.error(f"Error obteniendo stores disponibles: {e}")
            return []
    
    def search_similar(self, 
                      query: str, 
                      store_name: str, 
                      k: int = 5) -> list:
        """Buscar documentos similares sin generar respuesta.
        
        Args:
            query: Consulta de búsqueda
            store_name: Nombre del vector store
            k: Número de resultados
            
        Returns:
            Lista de documentos similares
        """
        try:
            vector_store = self.load_vector_store(store_name)
            if vector_store is None:
                return []
            
            results = vector_store.similarity_search(query, k=k)
            
            similar_docs = []
            for doc in results:
                similar_docs.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error en búsqueda de similitud: {e}")
            return []


def main():
    """Función principal para pruebas."""
    agent = TabularQAAgent()
    
    # Mostrar stores disponibles
    stores = agent.get_available_stores()
    print(f"Stores disponibles: {stores}")
    
    if stores:
        # Ejemplo de pregunta
        question = "¿Cuáles son los productos más vendidos?"
        result = agent.ask(question, stores[0])
        
        print(f"\nPregunta: {question}")
        print(f"Respuesta: {result['answer']}")
        print(f"Fuentes encontradas: {len(result['sources'])}")


if __name__ == "__main__":
    main()