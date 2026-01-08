# verification.py
import os
import sys

def verify_structure():
    """Verifica se a estrutura está completa."""
    required_files = [
        'src/cleaning/__init__.py',
        'src/cleaning/base_cleaner.py',
        'src/cleaning/cleaner_factory.py',
        'src/cleaning/cleaning_orchestrator.py',
        'src/cleaning/date_cleaner.py',
        'src/cleaning/categorical_cleaner.py',
        'src/cleaning/text_cleaner.py',
        'src/cleaning/id_cleaner.py',
        'src/cleaning/hybrid_corrector.py',
        'main.py',
        'requirements.txt'
    ]
    
    print("🔍 Verificando estrutura do projeto...")
    
    all_ok = True
    for filepath in required_files:
        if os.path.exists(filepath):
            print(f"  ✅ {filepath}")
        else:
            print(f"  ❌ {filepath} (FALTANDO)")
            all_ok = False
    
    if all_ok:
        print("\n🎉 Estrutura completa! Você pode rodar:")
        print("   1. streamlit run main.py")
        print("   2. pytest tests/ (para testes)")
    else:
        print("\n⚠️  Alguns arquivos estão faltando.")
        print("   Execute o script de implantação primeiro.")
    
    return all_ok

if __name__ == "__main__":
    verify_structure()