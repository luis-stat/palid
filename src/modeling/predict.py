import pandas as pd
import pickle
import os
from src.features import MetaFeatureExtractor

class ModelPredictor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.feature_columns = []
        self.extractor = MetaFeatureExtractor()
        self._load_model()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}")
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.feature_columns = data.get('feature_columns', [])
        
    def predict_column_types(self, df: pd.DataFrame) -> dict:
        """Recebe um DF e retorna {coluna: tipo}."""
        # Ajusta extrator (opcionalmente poderia usar fit aqui, mas usamos genérico)
        # self.extractor.fit(df.columns.tolist()) 
        
        feats_list = []
        cols = []
        for col in df.columns:
            f = self.extractor.extract(df[col], col)
            feats_list.append(f)
            cols.append(col)
            
        X = pd.DataFrame(feats_list).fillna(0)
        # Alinha colunas com o treino
        for c in self.feature_columns:
            if c not in X.columns: X[c] = 0
        X = X[self.feature_columns]
        
        preds_enc = self.model.predict(X)
        types = self.label_encoder.inverse_transform(preds_enc)
        return dict(zip(cols, types))