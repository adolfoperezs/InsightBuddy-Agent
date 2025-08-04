#!/usr/bin/env python3
"""Ejemplos de uso del agente QA tabular.

Este script demuestra diferentes tipos de preguntas que se pueden hacer
al agente QA y muestra las respuestas obtenidas.
"""

import time
from app.agent import TabularQAAgent


def run_example_questions():
    """Ejecutar una serie de preguntas de ejemplo."""
    
    # Inicializar agente
    print("🤖 Inicializando agente QA...")
    agent = TabularQAAgent()
    
    # Verificar stores disponibles
    stores = agent.get_available_stores()
    if not stores:
        print("❌ No hay datasets disponibles. Ejecuta 'python ingest.py' primero.")
        return
    
    store_name = stores[0]
    print(f"📊 Usando dataset: {store_name}\n")
    
    # Lista de preguntas de ejemplo
    example_questions = [
        "¿Cuántos registros hay en total?",
        "¿Cuáles son los productos disponibles?",
        "¿Qué ciudades están incluidas en los datos?",
        "¿Cuál es el rango de fechas de las ventas?",
        "¿Cuáles son los productos más vendidos?",
        "¿Qué manager tiene más ventas?",
        "¿Cuál es el precio promedio de los productos?",
        "¿Qué métodos de pago se utilizan?",
        "¿Cuáles son las ventas totales?",
        "¿Hay diferencias en las ventas por ciudad?"
    ]
    
    print("=" * 80)
    print("🔍 EJECUTANDO PREGUNTAS DE EJEMPLO")
    print("=" * 80)
    
    for i, question in enumerate(example_questions, 1):
        print(f"\n[{i}/{len(example_questions)}] 🤔 {question}")
        print("-" * 60)
        
        try:
            # Hacer la pregunta
            response = agent.ask(question, store_name, k=2)
            
            # Mostrar respuesta
            print(f"💡 {response['answer']}")
            
            # Mostrar fuentes si están disponibles
            sources = response.get('sources', [])
            if sources:
                print(f"\n📚 Fuentes consultadas: {len(sources)}")
                for j, source in enumerate(sources, 1):
                    source_type = source.get('metadata', {}).get('type', 'unknown')
                    print(f"   {j}. Tipo: {source_type}")
            
            # Pausa entre preguntas
            if i < len(example_questions):
                time.sleep(1)
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 80)
    print("✅ EJEMPLOS COMPLETADOS")
    print("=" * 80)


def demonstrate_search_functionality():
    """Demostrar la funcionalidad de búsqueda semántica."""
    
    print("\n🔍 DEMOSTRANDO BÚSQUEDA SEMÁNTICA")
    print("=" * 50)
    
    agent = TabularQAAgent()
    stores = agent.get_available_stores()
    
    if not stores:
        print("❌ No hay datasets disponibles.")
        return
    
    store_name = stores[0]
    
    # Términos de búsqueda de ejemplo
    search_terms = [
        "hamburguesas",
        "ventas Madrid",
        "Pablo Perez",
        "bebidas",
        "diciembre 2022"
    ]
    
    for term in search_terms:
        print(f"\n🔎 Buscando: '{term}'")
        results = agent.search_similar(term, store_name, k=2)
        
        if results:
            print(f"   Encontrados: {len(results)} documentos relevantes")
            for i, result in enumerate(results, 1):
                doc_type = result.get('metadata', {}).get('type', 'unknown')
                content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                print(f"   {i}. [{doc_type}] {content_preview}")
        else:
            print("   No se encontraron resultados")


def main():
    """Función principal."""
    print("🚀 INSIGHT-BUDDY MVP - EJEMPLOS DE USO")
    print("=" * 50)
    
    try:
        # Ejecutar preguntas de ejemplo
        run_example_questions()
        
        # Demostrar búsqueda semántica
        demonstrate_search_functionality()
        
        print("\n🎉 ¡Demostración completada exitosamente!")
        print("\n💡 Puedes hacer tus propias preguntas usando:")
        print("   python ask.py \"Tu pregunta aquí\" --store sales_data")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demostración cancelada por el usuario.")
    except Exception as e:
        print(f"\n❌ Error durante la demostración: {e}")


if __name__ == "__main__":
    main()