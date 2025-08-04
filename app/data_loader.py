import pandas as pd
import numpy as np
from pathlib import Path
import os
from typing import List, Dict, Any
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

class DataLoader:
    """Clase para cargar, procesar y vectorizar datos tabulares."""
    
    def __init__(self, chunk_size: int = 50):
        """
        Inicializar el DataLoader.
        
        Args:
            chunk_size: Número de filas por chunk para dividir los datos
        """
        self.chunk_size = chunk_size
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        self.index_path = Path(os.getenv('FAISS_INDEX_PATH', 'data/index/'))
        self.index_path.mkdir(parents=True, exist_ok=True)
        
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Cargar archivo CSV y detectar tipos de datos automáticamente.
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        try:
            # Cargar CSV con detección automática de tipos
            df = pd.read_csv(file_path)
            
            # Limpiar espacios en blanco en columnas de texto
            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].astype(str).str.strip()
            
            # Intentar convertir columnas de fecha
            date_columns = ['Date', 'date', 'fecha', 'Date_Time']
            for col in df.columns:
                if col in date_columns or 'date' in col.lower():
                    try:
                        df[col] = pd.to_datetime(df[col], dayfirst=True)
                    except:
                        logger.warning(f"No se pudo convertir {col} a fecha")
            
            # Detectar y convertir columnas numéricas
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Intentar convertir a numérico
                    numeric_series = pd.to_numeric(df[col], errors='coerce')
                    if not numeric_series.isna().all():
                        df[col] = numeric_series
            
            logger.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
            logger.info(f"Columnas: {list(df.columns)}")
            logger.info(f"Tipos de datos:\n{df.dtypes}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error al cargar CSV: {e}")
            raise
    
    def create_chunks(self, df: pd.DataFrame) -> List[Document]:
        """
        Dividir DataFrame en chunks y crear documentos con metadatos.
        
        Args:
            df: DataFrame a dividir
            
        Returns:
            Lista de documentos LangChain
        """
        documents = []
        
        # Información general del dataset
        dataset_summary = self._create_dataset_summary(df)
        documents.append(Document(
            page_content=dataset_summary,
            metadata={
                'type': 'dataset_summary',
                'total_rows': len(df),
                'columns': list(df.columns)
            }
        ))
        
        # Dividir en chunks de filas
        for i in range(0, len(df), self.chunk_size):
            chunk_df = df.iloc[i:i + self.chunk_size]
            
            # Crear descripción del chunk
            chunk_description = self._create_chunk_description(chunk_df, i)
            
            # Crear documento
            doc = Document(
                page_content=chunk_description,
                metadata={
                    'type': 'data_chunk',
                    'chunk_id': i // self.chunk_size,
                    'start_row': i,
                    'end_row': min(i + self.chunk_size - 1, len(df) - 1),
                    'num_rows': len(chunk_df)
                }
            )
            documents.append(doc)
        
        logger.info(f"Creados {len(documents)} documentos ({len(documents)-1} chunks + 1 resumen)")
        return documents
    
    def _create_dataset_summary(self, df: pd.DataFrame) -> str:
        """
        Crear resumen general del dataset.
        
        Args:
            df: DataFrame a resumir
            
        Returns:
            Descripción textual del dataset
        """
        summary_parts = [
            f"RESUMEN DEL DATASET DE VENTAS:",
            f"Total de registros: {len(df)}",
            f"Período: {df['Date'].min()} a {df['Date'].max()}" if 'Date' in df.columns else "",
            f"Columnas disponibles: {', '.join(df.columns)}"
        ]
        
        # Estadísticas por columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary_parts.append("\nESTADÍSTICAS NUMÉRICAS:")
            for col in numeric_cols:
                if col in ['Price', 'Quantity']:
                    summary_parts.append(
                        f"{col}: Min={df[col].min():.2f}, Max={df[col].max():.2f}, "
                        f"Promedio={df[col].mean():.2f}, Total={df[col].sum():.2f}"
                    )
        
        # Valores únicos en columnas categóricas
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary_parts.append("\nCATEGORÍAS PRINCIPALES:")
            for col in categorical_cols:
                unique_vals = df[col].value_counts().head(5)
                summary_parts.append(f"{col}: {', '.join([f'{k} ({v})' for k, v in unique_vals.items()])}")
        
        return "\n".join(filter(None, summary_parts))
    
    def _create_chunk_description(self, chunk_df: pd.DataFrame, start_idx: int) -> str:
        """
        Crear descripción textual de un chunk de datos.
        
        Args:
            chunk_df: DataFrame del chunk
            start_idx: Índice de inicio del chunk
            
        Returns:
            Descripción textual del chunk
        """
        description_parts = [
            f"CHUNK DE DATOS (filas {start_idx} a {start_idx + len(chunk_df) - 1}):"
        ]
        
        # Convertir filas a texto descriptivo
        for idx, row in chunk_df.iterrows():
            row_desc = f"Orden {row.get('Order ID', idx)}: "
            
            # Agregar información clave de cada fila
            if 'Date' in row:
                row_desc += f"Fecha: {row['Date']}, "
            if 'Product' in row:
                row_desc += f"Producto: {row['Product']}, "
            if 'Price' in row and 'Quantity' in row:
                total = float(row['Price']) * float(row['Quantity'])
                row_desc += f"Precio: ${row['Price']}, Cantidad: {row['Quantity']}, Total: ${total:.2f}, "
            if 'Purchase Type' in row:
                row_desc += f"Tipo: {row['Purchase Type']}, "
            if 'Payment Method' in row:
                row_desc += f"Pago: {row['Payment Method']}, "
            if 'Manager' in row:
                row_desc += f"Manager: {row['Manager']}, "
            if 'City' in row:
                row_desc += f"Ciudad: {row['City']}"
            
            description_parts.append(row_desc.rstrip(', '))
        
        return "\n".join(description_parts)
    
    def create_vector_store(self, documents: List[Document], store_name: str = "sales_data") -> FAISS:
        """
        Crear y guardar índice FAISS con los documentos.
        
        Args:
            documents: Lista de documentos a vectorizar
            store_name: Nombre del almacén vectorial
            
        Returns:
            Índice FAISS creado
        """
        try:
            logger.info(f"Creando embeddings para {len(documents)} documentos...")
            
            # Crear índice FAISS
            vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Guardar índice
            store_path = self.index_path / store_name
            vector_store.save_local(str(store_path))
            
            logger.info(f"Índice FAISS guardado en: {store_path}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error al crear vector store: {e}")
            raise
    
    def load_vector_store(self, store_name: str = "sales_data") -> FAISS:
        """
        Cargar índice FAISS existente.
        
        Args:
            store_name: Nombre del almacén vectorial
            
        Returns:
            Índice FAISS cargado
        """
        try:
            store_path = self.index_path / store_name
            if not store_path.exists():
                raise FileNotFoundError(f"No se encontró índice en: {store_path}")
            
            vector_store = FAISS.load_local(
                str(store_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info(f"Índice FAISS cargado desde: {store_path}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error al cargar vector store: {e}")
            raise
    
    def process_file(self, file_path: str, store_name: str = "sales_data") -> FAISS:
        """
        Procesar archivo completo: cargar, chunking y vectorización.
        
        Args:
            file_path: Ruta al archivo CSV
            store_name: Nombre del almacén vectorial
            
        Returns:
            Índice FAISS creado
        """
        logger.info(f"Procesando archivo: {file_path}")
        
        # Cargar datos
        df = self.load_csv(file_path)
        
        # Crear chunks
        documents = self.create_chunks(df)
        
        # Crear y guardar vector store
        vector_store = self.create_vector_store(documents, store_name)
        
        logger.info("Procesamiento completado exitosamente")
        return vector_store


if __name__ == "__main__":
    # Ejemplo de uso
    loader = DataLoader(chunk_size=50)
    
    # Procesar archivo de ventas
    csv_path = "data/raw/Sales-Data-Analysis.csv"
    if Path(csv_path).exists():
        vector_store = loader.process_file(csv_path)
        print(f"Procesamiento completado. Índice guardado en: {loader.index_path}")
    else:
        print(f"Archivo no encontrado: {csv_path}")