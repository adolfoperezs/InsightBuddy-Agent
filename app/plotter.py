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
        """Crear gráfico de barras.
        
        Args:
            df: DataFrame con los datos
            question: Pregunta original
            
        Returns:
            Ruta del archivo PNG generado
        """
        plt.figure(figsize=(12, 8))
        
        question_lower = question.lower()
        
        # Determinar qué graficar basado en la pregunta
        if 'ciudad' in question_lower:
            # Ventas por ciudad
            if 'Price' in df.columns and 'Quantity' in df.columns:
                df['Total_Sales'] = df['Price'] * df['Quantity']
                city_sales = df.groupby('City')['Total_Sales'].sum().sort_values(ascending=False)
            else:
                city_sales = df['City'].value_counts()
            
            city_sales.plot(kind='bar', color='skyblue')
            plt.title('Ventas por Ciudad', fontsize=16, fontweight='bold')
            plt.xlabel('Ciudad', fontsize=12)
            plt.ylabel('Ventas Totales' if 'Total_Sales' in locals() else 'Número de Órdenes', fontsize=12)
            
        elif 'producto' in question_lower:
            # Ventas por producto
            if 'Price' in df.columns and 'Quantity' in df.columns:
                df['Total_Sales'] = df['Price'] * df['Quantity']
                product_sales = df.groupby('Product')['Total_Sales'].sum().sort_values(ascending=False)
            else:
                product_sales = df['Product'].value_counts()
            
            product_sales.plot(kind='bar', color='lightcoral')
            plt.title('Ventas por Producto', fontsize=16, fontweight='bold')
            plt.xlabel('Producto', fontsize=12)
            plt.ylabel('Ventas Totales' if 'Total_Sales' in locals() else 'Número de Órdenes', fontsize=12)
            
        elif 'manager' in question_lower:
            # Ventas por manager
            if 'Price' in df.columns and 'Quantity' in df.columns:
                df['Total_Sales'] = df['Price'] * df['Quantity']
                manager_sales = df.groupby('Manager')['Total_Sales'].sum().sort_values(ascending=False)
            else:
                manager_sales = df['Manager'].value_counts()
            
            manager_sales.plot(kind='bar', color='lightgreen')
            plt.title('Ventas por Manager', fontsize=16, fontweight='bold')
            plt.xlabel('Manager', fontsize=12)
            plt.ylabel('Ventas Totales' if 'Total_Sales' in locals() else 'Número de Órdenes', fontsize=12)
            
        else:
            # Gráfico genérico - productos más vendidos
            if 'Quantity' in df.columns:
                product_qty = df.groupby('Product')['Quantity'].sum().sort_values(ascending=False)
            else:
                product_qty = df['Product'].value_counts()
            
            product_qty.plot(kind='bar', color='gold')
            plt.title('Productos Más Vendidos', fontsize=16, fontweight='bold')
            plt.xlabel('Producto', fontsize=12)
            plt.ylabel('Cantidad Total', fontsize=12)
        
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
        """Crear gráfico de líneas.
        
        Args:
            df: DataFrame con los datos
            question: Pregunta original
            
        Returns:
            Ruta del archivo PNG generado
        """
        plt.figure(figsize=(12, 8))
        
        if 'Date' in df.columns:
            # Ventas a lo largo del tiempo
            if 'Price' in df.columns and 'Quantity' in df.columns:
                df['Total_Sales'] = df['Price'] * df['Quantity']
                daily_sales = df.groupby('Date')['Total_Sales'].sum()
            else:
                daily_sales = df.groupby('Date').size()
            
            daily_sales.plot(kind='line', marker='o', linewidth=2, markersize=4)
            plt.title('Evolución de Ventas en el Tiempo', fontsize=16, fontweight='bold')
            plt.xlabel('Fecha', fontsize=12)
            plt.ylabel('Ventas Totales' if 'Total_Sales' in locals() else 'Número de Órdenes', fontsize=12)
            
        else:
            # Si no hay fecha, crear línea con índice
            if 'Price' in df.columns:
                df['Price'].plot(kind='line', marker='o')
                plt.title('Evolución de Precios', fontsize=16, fontweight='bold')
                plt.ylabel('Precio', fontsize=12)
            else:
                plt.text(0.5, 0.5, 'No hay datos temporales disponibles', 
                        ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
                plt.title('Datos Temporales No Disponibles', fontsize=16)
        
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
        """Crear gráfico de pastel.
        
        Args:
            df: DataFrame con los datos
            question: Pregunta original
            
        Returns:
            Ruta del archivo PNG generado
        """
        plt.figure(figsize=(10, 8))
        
        question_lower = question.lower()
        
        if 'producto' in question_lower:
            # Distribución por producto
            if 'Price' in df.columns and 'Quantity' in df.columns:
                df['Total_Sales'] = df['Price'] * df['Quantity']
                data = df.groupby('Product')['Total_Sales'].sum()
            else:
                data = df['Product'].value_counts()
            title = 'Distribución de Ventas por Producto'
            
        elif 'ciudad' in question_lower:
            # Distribución por ciudad
            data = df['City'].value_counts()
            title = 'Distribución por Ciudad'
            
        else:
            # Distribución genérica por producto
            data = df['Product'].value_counts()
            title = 'Distribución por Producto'
        
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
        """Crear gráfico de dispersión.
        
        Args:
            df: DataFrame con los datos
            question: Pregunta original
            
        Returns:
            Ruta del archivo PNG generado
        """
        plt.figure(figsize=(10, 8))
        
        if 'Price' in df.columns and 'Quantity' in df.columns:
            plt.scatter(df['Price'], df['Quantity'], alpha=0.6, s=50)
            plt.xlabel('Precio', fontsize=12)
            plt.ylabel('Cantidad', fontsize=12)
            plt.title('Relación entre Precio y Cantidad', fontsize=16, fontweight='bold')
            
            # Agregar línea de tendencia
            z = np.polyfit(df['Price'], df['Quantity'], 1)
            p = np.poly1d(z)
            plt.plot(df['Price'], p(df['Price']), "r--", alpha=0.8)
            
        else:
            plt.text(0.5, 0.5, 'No hay datos numéricos suficientes para dispersión', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('Datos Insuficientes para Dispersión', fontsize=16)
        
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
    
    def generate_chart_if_needed(self, question: str, store_name: str) -> Optional[str]:
        """
        Generar un gráfico si la pregunta lo requiere.
        
        Args:
            question: Pregunta del usuario
            store_name: Nombre del store de datos
            
        Returns:
            Ruta del archivo PNG generado o None si no se requiere gráfico
        """
        chart_type = self.detect_chart_intent(question)
        
        if not chart_type:
            return None
        
        try:
            # Cargar datos desde el CSV original
            csv_path = f"data/raw/{store_name.replace('_data', '')}-Data-Analysis.csv"
            if not os.path.exists(csv_path):
                csv_path = f"data/raw/Sales-Data-Analysis.csv"  # Fallback
            
            if not os.path.exists(csv_path):
                logger.warning(f"No se encontró archivo CSV para {store_name}")
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