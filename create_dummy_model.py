import pickle
import os
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def create_dummy_model():
    """Cria um modelo dummy para testes."""
    print("Criando modelo dummy")
    
    # Classes que o modelo deve reconhecer
    classes = [
        'DATA_HORA',
        'CATEGORICO_NOMINAL', 
        'CATEGORICO_ORDINAL',
        'CATEGORICO_ESTADO',
        'TEXTO_LIVRE',
        'ID',
        'NUMERICO'
    ]
    
    # Criar encoder
    encoder = LabelEncoder()
    encoder.fit(classes)
    
    # Criar modelo dummy (sempre prediz a primeira classe)
    model = DummyClassifier(strategy="constant", constant=0)
    
    # "Treinar" com dados dummy
    X_dummy = np.random.randn(100, 10)
    y_dummy = np.zeros(100)  # Todas a classe 0 (DATA_HORA)
    model.fit(X_dummy, y_dummy)
    
    # Criar colunas de features dummy
    feature_columns = [f'feature_{i}' for i in range(10)]
    
    # Salvar
    os.makedirs('models', exist_ok=True)
    
    with open('models/semantic_type_classifier.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'label_encoder': encoder,
            'feature_columns': feature_columns
        }, f)
    
    print(f"Modelo dummy salvo em: models/semantic_type_classifier.pkl")
    print(f"Classes: {classes}")
    print("Este é um modelo dummy. Para uso real, treine com:")
    print("    python trainer.py")

if __name__ == "__main__":
    create_dummy_model()