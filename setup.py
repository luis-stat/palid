#!/usr/bin/env python
"""
Script de inicialização para configurar o projeto.
"""
import os
import sys
import subprocess

def setup_project():
    """Configura a estrutura inicial do projeto."""
    print("Configurando")
    
    # 1. Criar diretórios necessários
    directories = [
        'data/raw',
        'data/external', 
        'data/interim',
        'data/processed',
        'models',
        'reports/figures',
        'tests'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Criado: {directory}")
    
    # 2. Criar arquivos essenciais
    essential_files = {
        'data/external/gabarito_master.csv': 'nome_arquivo,nome_coluna,tipo_real\n',
        'data/raw/exemplo.csv': 'coluna_data,coluna_cidade,coluna_texto\n2023-01-01,São Paulo,Exemplo texto\n',
        '.env': '# Variáveis de ambiente\n'
    }
    
    for filepath, content in essential_files.items():
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Criado: {filepath}")
    
    # 3. Criar modelo dummy (se não existir)
    model_path = 'models/semantic_type_classifier.pkl'
    if not os.path.exists(model_path):
        print("Modelo não encontrado. Execute:")
        print("     python -c \"exec(open('create_dummy_model.py').read())\"")
    
    print("\nConfiguração completa!")
    print("Próximos passos:")
    print("   1. Instale dependências: pip install -r requirements.txt")
    print("   2. Execute: streamlit run main.py")
    print("   3. Para testes: pytest tests/")

if __name__ == "__main__":
    setup_project()