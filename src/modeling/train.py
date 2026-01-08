import pandas as pd
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

class ModelTrainer:
    def __init__(self, features_path: str, output_model_path: str):
        self.features_path = features_path
        self.output_model_path = output_model_path
        self.label_encoder = LabelEncoder()
        
    def run(self):
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features não encontradas em {self.features_path}")
            
        df = pd.read_csv(self.features_path)
        
        # Remove colunas de metadados
        X = df.drop(['target', 'nome_arquivo', 'nome_coluna'], axis=1, errors='ignore').fillna(0)
        y = df['target']
        
        # Converte colunas object remanescentes para numérico
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
        y_enc = self.label_encoder.fit_transform(y)
        feature_cols = list(X.columns)
        
        # --- CORREÇÃO DO ERRO DE ESTRATIFICAÇÃO ---
        # Verifica se alguma classe tem menos de 2 exemplos
        class_counts = np.bincount(y_enc)
        if np.min(class_counts) < 2:
            print("⚠️ Aviso: Algumas classes têm apenas 1 exemplo. Estratificação desativada.")
            stratify_param = None
        else:
            stratify_param = y_enc

        # Treino
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc, 
            test_size=0.2, 
            random_state=42, 
            stratify=stratify_param  # Usa None se houver poucos dados
        )
        
        clf = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, class_weight='balanced', verbose=-1)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50)])
        
        # Avaliação
        y_pred = clf.predict(X_test)
        print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
        
        # Salvar
        os.makedirs(os.path.dirname(self.output_model_path), exist_ok=True)
        with open(self.output_model_path, 'wb') as f:
            pickle.dump({'model': clf, 'label_encoder': self.label_encoder, 'feature_columns': feature_cols}, f)
        print(f"Modelo salvo em: {self.output_model_path}")