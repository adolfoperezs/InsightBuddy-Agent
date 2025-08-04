#!/usr/bin/env python3
"""
Script CLI para ingestar y vectorizar datos CSV/Parquet.

Uso:
    python ingest.py path/to/file.csv
    python ingest.py path/to/file.csv --store-name custom_name --chunk-size 100
"""

import argparse
import sys
from pathlib import Path
from app.data_loader import DataLoader
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal del script CLI."""
    parser = argparse.ArgumentParser(
        description='Ingestar y vectorizar datos CSV/Parquet para Insight-Buddy'
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        help='Ruta al archivo CSV o Parquet a procesar'
    )
    
    parser.add_argument(
        '--store-name',
        type=str,
        default='sales_data',
        help='Nombre del almacén vectorial (default: sales_data)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=50,
        help='Número de filas por chunk (default: 50)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Sobrescribir índice existente si ya existe'
    )
    
    args = parser.parse_args()
    
    # Validar archivo de entrada
    file_path = Path(args.file_path)
    if not file_path.exists():
        logger.error(f"Archivo no encontrado: {file_path}")
        sys.exit(1)
    
    if file_path.suffix.lower() not in ['.csv', '.parquet']:
        logger.error(f"Formato de archivo no soportado: {file_path.suffix}")
        logger.info("Formatos soportados: .csv, .parquet")
        sys.exit(1)
    
    try:
        # Crear loader
        loader = DataLoader(chunk_size=args.chunk_size)
        
        # Verificar si ya existe el índice
        store_path = loader.index_path / args.store_name
        if store_path.exists() and not args.force:
            response = input(f"El índice '{args.store_name}' ya existe. ¿Sobrescribir? (y/N): ")
            if response.lower() != 'y':
                logger.info("Operación cancelada")
                sys.exit(0)
        
        # Procesar archivo
        logger.info(f"Iniciando procesamiento de: {file_path}")
        logger.info(f"Configuración: chunk_size={args.chunk_size}, store_name={args.store_name}")
        
        if file_path.suffix.lower() == '.csv':
            vector_store = loader.process_file(str(file_path), args.store_name)
        else:
            # Para archivos Parquet (implementación futura)
            logger.error("Soporte para Parquet aún no implementado")
            sys.exit(1)
        
        # Mostrar estadísticas
        logger.info("\n" + "="*50)
        logger.info("PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("="*50)
        logger.info(f"Archivo procesado: {file_path}")
        logger.info(f"Índice guardado en: {store_path}")
        logger.info(f"Nombre del almacén: {args.store_name}")
        
        # Probar búsqueda rápida
        try:
            test_results = vector_store.similarity_search("ventas totales", k=1)
            logger.info(f"Prueba de búsqueda exitosa: {len(test_results)} resultados encontrados")
        except Exception as e:
            logger.warning(f"Error en prueba de búsqueda: {e}")
        
        logger.info("\nPuedes usar el índice con:")
        logger.info(f"  python ask.py '¿Cuáles fueron las ventas totales?'")
        logger.info(f"  streamlit run streamlit_app.py")
        
    except KeyboardInterrupt:
        logger.info("\nOperación interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error durante el procesamiento: {e}")
        logger.exception("Detalles del error:")
        sys.exit(1)

if __name__ == "__main__":
    main()