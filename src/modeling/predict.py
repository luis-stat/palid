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
        self.extractor = None
        self._load_model()
        self._load_extractor()
        
    def _load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo não encontrado em {self.model_path}")
        with open(self.model_path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.feature_columns = data.get('feature_columns', [])

    def _load_extractor(self):
        base_dir = os.path.dirname(self.model_path)
        ext_path = os.path.join(base_dir, 'feature_extractor.pkl')
        
        if os.path.exists(ext_path):
            with open(ext_path, 'rb') as f:
                self.extractor = pickle.load(f)
        else:
            self.extractor = MetaFeatureExtractor()
        
    def predict_column_types(self, df: pd.DataFrame) -> dict:
        feats_list = []
        cols = []
        for col in df.columns:
            f = self.extractor.extract(df[col], col)
            feats_list.append(f)
            cols.append(col)
            
        X = pd.DataFrame(feats_list).fillna(0)
        for c in self.feature_columns:
            if c not in X.columns: X[c] = 0
        X = X[self.feature_columns]
        
        preds_enc = self.model.predict(X)
        types = self.label_encoder.inverse_transform(preds_enc)
        return dict(zip(cols, types))