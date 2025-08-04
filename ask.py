#!/usr/bin/env python3
"""Script CLI para hacer preguntas a los datos usando el agente QA tabular.

Ejemplos de uso:
    python ask.py "¿Cuál fue el total de ventas en 2024?" --store sales_data
    python ask.py "¿Qué productos se vendieron más?" --store sales_data --verbose
    python ask.py "Muestra estadísticas de ventas por ciudad" --store sales_data -k 5
"""

import argparse
import json
import sys
from typing import Dict, Any

from app.agent import TabularQAAgent


def format_response(response: Dict[str, Any], verbose: bool = False) -> str:
    """Formatear la respuesta para mostrar en consola.
    
    Args:
        response: Respuesta del agente
        verbose: Si mostrar información detallada
        
    Returns:
        String formateado para mostrar
    """
    output = []
    
    # Respuesta principal
    output.append("=" * 60)
    output.append("RESPUESTA:")
    output.append("=" * 60)
    output.append(response.get('answer', 'Sin respuesta'))
    output.append("")
    
    # Información adicional si está en modo verbose
    if verbose:
        output.append("-" * 40)
        output.append("INFORMACIÓN ADICIONAL:")
        output.append("-" * 40)
        
        # Metadatos
        if 'store_name' in response:
            output.append(f"Dataset usado: {response['store_name']}")
        if 'model_used' in response:
            output.append(f"Modelo: {response['model_used']}")
        if 'timestamp' in response:
            output.append(f"Timestamp: {response['timestamp']}")
        
        # Fuentes
        sources = response.get('sources', [])
        if sources:
            output.append(f"\nFuentes consultadas ({len(sources)}):")
            for i, source in enumerate(sources, 1):
                output.append(f"\n{i}. Tipo: {source.get('metadata', {}).get('type', 'unknown')}")
                if 'metadata' in source and 'chunk_id' in source['metadata']:
                    output.append(f"   Chunk: {source['metadata']['chunk_id']}")
                content = source.get('content', '')[:150]
                output.append(f"   Contenido: {content}...")
        
        # Errores
        if 'error' in response:
            output.append(f"\n⚠️  Error: {response['error']}")
    
    return "\n".join(output)


def main():
    """Función principal del CLI."""
    parser = argparse.ArgumentParser(
        description="Hacer preguntas a datasets usando IA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ejemplos:
  python ask.py "¿Cuáles son las ventas totales?" --store sales_data
  python ask.py "¿Qué ciudades tienen más ventas?" -s sales_data -v
  python ask.py "Estadísticas por producto" --store sales_data -k 5
        """
    )
    
    parser.add_argument(
        "question",
        nargs="?",
        help="Pregunta a realizar sobre los datos"
    )
    
    parser.add_argument(
        "-s", "--store",
        default="sales_data",
        help="Nombre del dataset/store a consultar (default: sales_data)"
    )
    
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=3,
        help="Número de documentos relevantes a recuperar (default: 3)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Mostrar información detallada incluyendo fuentes"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Salida en formato JSON"
    )
    
    parser.add_argument(
        "--list-stores",
        action="store_true",
        help="Listar datasets disponibles"
    )
    
    parser.add_argument(
        "--model",
        help="Modelo OpenAI a usar (sobrescribe .env)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperatura del modelo (0.0-1.0, default: 0.1)"
    )
    
    args = parser.parse_args()
    
    try:
        # Inicializar agente
        agent = TabularQAAgent(
            model_name=args.model,
            temperature=args.temperature
        )
        
        # Listar stores si se solicita
        if args.list_stores:
            stores = agent.get_available_stores()
            if stores:
                print("Datasets disponibles:")
                for store in stores:
                    print(f"  - {store}")
            else:
                print("No se encontraron datasets. Ejecuta 'python ingest.py' primero.")
            return
        
        # Validar que se proporcionó una pregunta
        if not args.question:
            print("❌ Error: Debes proporcionar una pregunta.")
            print("   Usa --help para ver ejemplos de uso.")
            sys.exit(1)
        
        # Validar que existe el store
        available_stores = agent.get_available_stores()
        if not available_stores:
            print("❌ Error: No hay datasets disponibles.")
            print("   Ejecuta 'python ingest.py path/to/data.csv' primero.")
            sys.exit(1)
        
        if args.store not in available_stores:
            print(f"❌ Error: Dataset '{args.store}' no encontrado.")
            print(f"   Datasets disponibles: {', '.join(available_stores)}")
            sys.exit(1)
        
        # Hacer la pregunta
        print(f"🤔 Pregunta: {args.question}")
        print(f"📊 Dataset: {args.store}")
        print("🔍 Buscando respuesta...\n")
        
        response = agent.ask(
            question=args.question,
            store_name=args.store,
            k=args.top_k
        )
        
        # Mostrar respuesta
        if args.json:
            print(json.dumps(response, indent=2, ensure_ascii=False))
        else:
            print(format_response(response, verbose=args.verbose))
        
        # Código de salida basado en si hubo error
        if 'error' in response:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Operación cancelada por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()