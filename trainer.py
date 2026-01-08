# Arquivo: pipeline_treino.py
import os
from src.features import build_features_dataset
from src.modeling.train import ModelTrainer

# --- Configurações de Caminhos ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Entradas
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
GABARITO_PATH = os.path.join(BASE_DIR, 'data', 'external', 'gabarito_master.csv')

# Saídas
FEATURES_OUTPUT = os.path.join(BASE_DIR, 'data', 'processed', 'training_features.csv')
MODEL_OUTPUT = os.path.join(BASE_DIR, 'models', 'semantic_type_classifier.pkl')

def executar_pipeline():
    print("Iniciando Pipeline de Treinamento")

    # 1. Gera as Features (Lê os CSVs brutos e cria o training_features.csv)
    print("\n[1/2] Gerando Dataset de Features...")
    if not os.path.exists(os.path.dirname(FEATURES_OUTPUT)):
        os.makedirs(os.path.dirname(FEATURES_OUTPUT))
        
    build_features_dataset(
        raw_data_dir=RAW_DATA_DIR,
        gabarito_path=GABARITO_PATH,
        output_path=FEATURES_OUTPUT
    )
    
    # 2. Treina o Modelo (Lê o training_features.csv e cria o .pkl)
    print("\n[2/2] Treinando Modelo LightGBM...")
    trainer = ModelTrainer(
        features_path=FEATURES_OUTPUT,
        output_model_path=MODEL_OUTPUT
    )
    trainer.run()
    
    print("\nPipeline finalizado.")
    print(f"Modelo salvo em: {MODEL_OUTPUT}")

if __name__ == "__main__":
    executar_pipeline()