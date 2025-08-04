"""Módulo de visualización para Insight-Buddy MVP.

Este módulo detecta intenciones de gráfico en preguntas y genera
visualizaciones automáticas usando matplotlib y pandas.
"""

import os
import re
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Usar backend sin GUI
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar matplotlib para mejor apariencia
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataPlotter:
    """Generador de visualizaciones automáticas para datos tabulares."""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Inicializar el plotter.
        
        Args:
            output_dir: Directorio donde guardar las visualizaciones
        """
        # Usar ruta absoluta para el directorio de salida
        if not os.path.isabs(output_dir):
            self.output_dir = os.path.abspath(output_dir)
        else:
            self.output_dir = output_dir
        
        # Crear directorio si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Patrones para detectar intenciones de gráfico
        self.chart_patterns = {
            'bar': [
                r'gráfico de barras?',
                r'barras?',
                r'comparar.*por',
                r'distribución.*por',
                r'ventas.*por.*ciudad',
                r'ventas.*por.*producto',
                r'por.*manager',
                r'ranking',
                r'top.*\d+'
            ],
            'line': [
                r'gráfico de líneas?',
                r'líneas?',
                r'tendencia',
                r'evolución',
                r'a lo largo del tiempo',
                r'por.*fecha',
                r'por.*mes',
                r'histórico'
            ],
            'pie': [
                r'gráfico de pastel',
                r'gráfico circular',
                r'pie',
                r'proporción',
                r'porcentaje.*de',
                r'distribución.*total'
            ],
            'scatter': [
                r'dispersión',
                r'scatter',
                r'correlación',
                r'relación.*entre',
                r'precio.*vs.*cantidad',
                r'vs\.',
                r'contra'
            ]
        }
        
        logger.info(f"DataPlotter inicializado. Directorio de salida: {self.output_dir}")
    
    def detect_chart_intent(self, question: str) -> Optional[str]:
        """Detectar intención de gráfico en la pregunta.
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Tipo de gráfico detectado o None
        """
        question_lower = question.lower()
        
        # Buscar palabras clave explícitas
        explicit_keywords = {
            'gráfico': True,
            'grafico': True,
            'chart': True,
            'plot': True,
            'visualiza': True,
            'muestra': True,
            'dibuja': True
        }
        
        has_explicit = any(keyword in question_lower for keyword in explicit_keywords)
        
        # Si no hay palabras clave explícitas, ser más restrictivo
        if not has_explicit:
            # Solo detectar para patrones muy específicos
            specific_patterns = [
                r'ventas.*por.*ciudad',
                r'productos.*más.*vendidos',
                r'top.*\d+',
                r'ranking',
                r'distribución.*por',
                r'comparar.*ventas'
            ]
            
            if not any(re.search(pattern, question_lower) for pattern in specific_patterns):
                return None
        
        # Detectar tipo de gráfico
        for chart_type, patterns in self.chart_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question_lower):
                    logger.info(f"Intención de gráfico detectada: {chart_type} para '{question}'")
                    return chart_type
        
        # Si hay palabras clave explícitas pero no se detectó tipo específico, usar barras por defecto
        if has_explicit:
            logger.info(f"Intención de gráfico genérica detectada, usando barras para '{question}'")
            return 'bar'
        
        return None
    
    def load_data_from_csv(self, csv_path: str) -> pd.DataFrame:
        """Cargar datos desde CSV.
        
        Args:
            csv_path: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos
        """
        try:
            df = pd.read_csv(csv_path)
            
            # Convertir fechas si es necesario
            if 'Date' in df.columns:
                 try:
                     # Intentar varios formatos de fecha comunes
                     df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
                 except ValueError:
                     try:
                         df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%Y')
                     except ValueError:
                         try:
                             df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
                         except ValueError:
                             df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True)
            
            logger.info(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
            return df
            
        except Exception as e:
            logger.error(f"Error cargando datos desde {csv_path}: {e}")
            raise
    
    def create_bar_chart(self, df: pd.DataFrame, question: str) -> str:
        """Crear gráfico de barras adaptativo.
        
        Args:
            df: DataFrame con los datos
            question: Pregunta original
            
        Returns:
            Ruta del archivo PNG generado
        """
        plt.figure(figsize=(12, 8))
        
        question_lower = question.lower()
        
        # Detectar columnas categóricas y numéricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Buscar palabras clave en la pregunta para determinar qué columna usar
        target_col = None
        value_col = None
        
        # Buscar columnas mencionadas en la pregunta
        for col in categorical_cols:
            if col.lower() in question_lower or any(word in col.lower() for word in ['gender', 'sex', 'género', 'sexo', 'hombre', 'mujer', 'male', 'female']):
                target_col = col
                break
        
        # Si no se encuentra una columna específica, usar la primera categórica
        if target_col is None and categorical_cols:
            target_col = categorical_cols[0]
        
        # Buscar columna de valores
        if numeric_cols:
            # Priorizar columnas que parezcan conteos o cantidades
            for col in numeric_cols:
                if any(word in col.lower() for word in ['count', 'quantity', 'amount', 'total', 'cantidad']):
                    value_col = col
                    break
            if value_col is None:
                value_col = numeric_cols[0]
        
        if target_col is None:
            raise ValueError(f"No se encontraron columnas categóricas apropiadas en el dataset. Columnas disponibles: {df.columns.tolist()}")
        
        # Crear el gráfico
        # Para columnas categóricas como género, siempre usar conteo de frecuencias
        if target_col and any(word in target_col.lower() for word in ['gender', 'sex', 'género', 'sexo', 'type', 'category', 'categoría']):
            # Contar frecuencias para columnas categóricas
            grouped_data = df[target_col].value_counts().sort_values(ascending=False)
            ylabel = 'Cantidad'
        elif value_col and target_col:
            # Solo usar suma de valores si tiene sentido (valores no son todos ceros)
            sum_data = df.groupby(target_col)[value_col].sum().sort_values(ascending=False)
            if sum_data.sum() > 0:  # Si hay valores significativos
                grouped_data = sum_data
                ylabel = f'Total {value_col}'
            else:
                # Si los valores son ceros, usar conteo
                grouped_data = df[target_col].value_counts().sort_values(ascending=False)
                ylabel = 'Cantidad'
        else:
            # Contar frecuencias por defecto
            grouped_data = df[target_col].value_counts().sort_values(ascending=False)
            ylabel = 'Cantidad'
        
        # Limitar a top 10 para mejor visualización
        if len(grouped_data) > 10:
            grouped_data = grouped_data.head(10)
        
        grouped_data.plot(kind='bar', color='skyblue')
        plt.title(f'Distribución por {target_col}', fontsize=16, fontweight='bold')
        plt.xlabel(target_col, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bar_chart_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de barras guardado: {filepath}")
        return filepath
    
    def create_line_chart(self, df: pd.DataFrame, question: str) -> str:
        """Crear gráfico de líneas adaptativo para series temporales.
        
        Args:
            df: DataFrame con los datos
            question: Pregunta original
            
        Returns:
            Ruta del archivo PNG generado
        """
        plt.figure(figsize=(12, 8))
        
        # Buscar columnas de fecha
        date_cols = []
        for col in df.columns:
            if any(word in col.lower() for word in ['date', 'fecha', 'time', 'tiempo']):
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    continue
        
        # Buscar columnas numéricas
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if date_cols and numeric_cols:
            # Usar la primera columna de fecha encontrada
            date_col = date_cols[0]
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
            # Buscar la columna numérica más apropiada
            value_col = None
            for col in numeric_cols:
                if any(word in col.lower() for word in ['price', 'sales', 'amount', 'total', 'precio', 'ventas', 'cantidad']):
                    value_col = col
                    break
            if value_col is None:
                value_col = numeric_cols[0]
            
            # Agrupar por fecha y sumar valores
            daily_data = df.groupby(date_col)[value_col].sum()
            
            daily_data.plot(kind='line', marker='o', linewidth=2, markersize=4)
            plt.title(f'Evolución de {value_col} en el Tiempo', fontsize=16, fontweight='bold')
            plt.xlabel(date_col, fontsize=12)
            plt.ylabel(value_col, fontsize=12)
            
        elif numeric_cols:
            # Si no hay fechas, usar la primera columna numérica
            value_col = numeric_cols[0]
            df[value_col].plot(kind='line', marker='o', linewidth=2, markersize=4)
            plt.title(f'Evolución de {value_col}', fontsize=16, fontweight='bold')
            plt.xlabel('Índice', fontsize=12)
            plt.ylabel(value_col, fontsize=12)
        else:
            raise ValueError(f"No se encontraron columnas numéricas apropiadas para el gráfico de líneas. Columnas disponibles: {df.columns.tolist()}")
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"line_chart_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de líneas guardado: {filepath}")
        return filepath
    
    def create_pie_chart(self, df: pd.DataFrame, question: str) -> str:
        """Crear gráfico de pastel adaptativo.
        
        Args:
            df: DataFrame con los datos
            question: Pregunta original
            
        Returns:
            Ruta del archivo PNG generado
        """
        plt.figure(figsize=(10, 8))
        
        question_lower = question.lower()
        
        # Detectar columnas categóricas
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Buscar la columna más apropiada basada en la pregunta
        target_col = None
        
        # Buscar columnas mencionadas en la pregunta
        for col in categorical_cols:
            if col.lower() in question_lower or any(word in col.lower() for word in ['gender', 'sex', 'género', 'sexo', 'type', 'tipo', 'category', 'categoría']):
                target_col = col
                break
        
        # Si no se encuentra una columna específica, usar la primera categórica
        if target_col is None and categorical_cols:
            target_col = categorical_cols[0]
        
        if target_col is None:
            raise ValueError(f"No se encontraron columnas categóricas apropiadas para el gráfico de pastel. Columnas disponibles: {df.columns.tolist()}")
        
        # Crear el gráfico de pastel
        data = df[target_col].value_counts()
        
        # Limitar a top 8 categorías para mejor visualización
        if len(data) > 8:
            data = data.head(8)
            other_count = df[target_col].value_counts().iloc[8:].sum()
            if other_count > 0:
                data['Otros'] = other_count
        
        title = f'Distribución por {target_col}'
        
        # Crear gráfico de pastel
        colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
        wedges, texts, autotexts = plt.pie(data.values, labels=data.index, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        
        plt.title(title, fontsize=16, fontweight='bold')
        
        # Mejorar legibilidad
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.axis('equal')
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pie_chart_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de pastel guardado: {filepath}")
        return filepath
    
    def create_scatter_plot(self, df: pd.DataFrame, question: str) -> str:
        """Crear gráfico de dispersión adaptativo.
        
        Args:
            df: DataFrame con los datos
            question: Pregunta original
            
        Returns:
            Ruta del archivo PNG generado
        """
        plt.figure(figsize=(10, 8))
        
        # Buscar columnas numéricas
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Usar las dos primeras columnas numéricas
            x_col = numeric_cols[0]
            y_col = numeric_cols[1]
            
            # Buscar columnas específicas mencionadas en la pregunta
            question_lower = question.lower()
            for col in numeric_cols:
                if any(word in col.lower() for word in ['price', 'precio', 'cost', 'costo']):
                    x_col = col
                    break
            
            for col in numeric_cols:
                if any(word in col.lower() for word in ['quantity', 'cantidad', 'amount', 'total']):
                    y_col = col
                    break
            
            plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
            plt.xlabel(x_col, fontsize=12)
            plt.ylabel(y_col, fontsize=12)
            plt.title(f'Relación entre {x_col} y {y_col}', fontsize=16, fontweight='bold')
            
            # Agregar línea de tendencia si hay suficientes datos
            if len(df) > 1:
                try:
                    z = np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1)
                    p = np.poly1d(z)
                    plt.plot(df[x_col], p(df[x_col]), "r--", alpha=0.8)
                except:
                    pass  # Si no se puede calcular la tendencia, continuar sin ella
            
        else:
             raise ValueError(f"Se necesitan al menos 2 columnas numéricas para el gráfico de dispersión. Columnas numéricas disponibles: {numeric_cols}")
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Guardar gráfico
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scatter_plot_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de dispersión guardado: {filepath}")
        return filepath
    
    def generate_chart_if_needed(self, question: str, store_name: str, df: pd.DataFrame = None) -> Optional[str]:
        """
        Generar un gráfico si la pregunta lo requiere.
        
        Args:
            question: Pregunta del usuario
            store_name: Nombre del store de datos
            df: DataFrame con los datos (opcional)
            
        Returns:
            Ruta del archivo PNG generado o None si no se requiere gráfico
        """
        chart_type = self.detect_chart_intent(question)
        
        if not chart_type:
            return None
        
        try:
            # Si no se proporciona DataFrame, intentar cargar desde CSV
            if df is None:
                # Intentar cargar datos desde el CSV original
                csv_path = f"data/raw/{store_name.replace('_data', '')}-Data-Analysis.csv"
                if not os.path.exists(csv_path):
                    csv_path = f"data/raw/Sales-Data-Analysis.csv"  # Fallback
                
                if not os.path.exists(csv_path):
                    logger.warning(f"No se encontró archivo CSV para {store_name} y no se proporcionó DataFrame")
                    return None
                
                df = self.load_data_from_csv(csv_path)
            
            # Generar el gráfico según el tipo detectado
            if chart_type == 'bar':
                return self.create_bar_chart(df, question)
            elif chart_type == 'line':
                return self.create_line_chart(df, question)
            elif chart_type == 'pie':
                return self.create_pie_chart(df, question)
            elif chart_type == 'scatter':
                return self.create_scatter_plot(df, question)
            
        except Exception as e:
            logger.error(f"Error generando gráfico para '{question}': {e}")
            return None
        
        return None
    
    def generate_chart(self, question: str, csv_path: str = None, df: pd.DataFrame = None) -> Optional[Dict[str, Any]]:
        """Generar gráfico basado en la pregunta.
        
        Args:
            question: Pregunta del usuario
            csv_path: Ruta al archivo CSV (opcional)
            df: DataFrame con datos (opcional)
            
        Returns:
            Diccionario con información del gráfico generado o None
        """
        try:
            # Detectar intención de gráfico
            chart_type = self.detect_chart_intent(question)
            if not chart_type:
                return None
            
            # Cargar datos
            if df is None:
                if csv_path is None:
                    csv_path = "data/raw/Sales-Data-Analysis.csv"
                df = self.load_data_from_csv(csv_path)
            
            # Generar gráfico según el tipo
            chart_methods = {
                'bar': self.create_bar_chart,
                'line': self.create_line_chart,
                'pie': self.create_pie_chart,
                'scatter': self.create_scatter_plot
            }
            
            if chart_type in chart_methods:
                filepath = chart_methods[chart_type](df, question)
                
                return {
                    'chart_generated': True,
                    'chart_type': chart_type,
                    'chart_path': filepath,
                    'chart_filename': os.path.basename(filepath),
                    'question': question,
                    'timestamp': datetime.now().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Error generando gráfico: {e}")
            return {
                'chart_generated': False,
                'error': str(e),
                'question': question
            }
        
        return None


def main():
    """Función principal para pruebas."""
    plotter = DataPlotter()
    
    # Preguntas de prueba
    test_questions = [
        "Muestra un gráfico de ventas por ciudad",
        "¿Cuáles son los productos más vendidos? Haz un gráfico",
        "Gráfico de líneas de ventas en el tiempo",
        "Distribución porcentual por producto",
        "Relación entre precio y cantidad"
    ]
    
    for question in test_questions:
        print(f"\nPregunta: {question}")
        result = plotter.generate_chart(question)
        if result:
            if result.get('chart_generated'):
                print(f"✅ Gráfico generado: {result['chart_filename']}")
            else:
                print(f"❌ Error: {result.get('error')}")
        else:
            print("ℹ️ No se detectó intención de gráfico")


if __name__ == "__main__":
    main()