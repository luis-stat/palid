import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any, List
import os
import warnings
from src.dataset import DataLoader

class MetaFeatureExtractor:
    def __init__(self):
        self.tfidf = TfidfVectorizer(
            max_features=50, lowercase=True, analyzer='char_wb', ngram_range=(2, 4)
        )
        self.is_fitted = False
        
    def _can_convert_numeric(self, val):
        try:
            if pd.isna(val): return False
            float(str(val).replace(',', '.'))
            return True
        except: return False
            
    def _can_convert_date(self, val):
        try:
            if pd.isna(val): return False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pd.to_datetime(str(val), dayfirst=True)
            return True
        except: return False
        
    def fit(self, column_names: List[str]):
        if column_names:
            try:
                self.tfidf.fit(column_names)
                self.is_fitted = True
            except: self.is_fitted = False
                
    def extract(self, series: pd.Series, col_name: str = "") -> Dict[str, Any]:
        features = {}
        series_clean = series.dropna()
        total = len(series)
        non_null = len(series_clean)
        
        features['non_null_ratio'] = non_null / total if total > 0 else 0
        features['cardinality_ratio'] = series_clean.nunique() / non_null if non_null > 0 else 0
        
        sample = series_clean.sample(min(len(series_clean), 1000), random_state=42) if non_null > 0 else series_clean
        
        features['numeric_ratio'] = sample.apply(self._can_convert_numeric).sum() / len(sample) if len(sample) > 0 else 0
        features['date_ratio'] = sample.apply(self._can_convert_date).sum() / len(sample) if len(sample) > 0 else 0
        
        str_lens = sample.astype(str).str.len()
        features['avg_len'] = str_lens.mean() if not str_lens.empty else 0
        
        map_dtype = {'object': [1,0,0], 'int64': [0,1,0], 'float64': [0,0,1]}
        feats_d = map_dtype.get(str(series.dtype), [0,0,0])
        features['is_obj'], features['is_int'], features['is_float'] = feats_d
        
        if self.is_fitted and col_name:
            try:
                emb = self.tfidf.transform([col_name]).toarray()[0]
                for i, v in enumerate(emb): features[f'nm_emb_{i}'] = v
            except:
                for i in range(50): features[f'nm_emb_{i}'] = 0
        else:
            for i in range(50): features[f'nm_emb_{i}'] = 0
                
        return features

def build_features_dataset(raw_data_dir: str, gabarito_path: str, output_path: str):
    """Pipeline para ler CSVs brutos e criar o dataset de treino."""
    extractor = MetaFeatureExtractor()
    loader = DataLoader()
    
    if not os.path.exists(gabarito_path):
        raise FileNotFoundError("Gabarito não encontrado.")
    gabarito_df = pd.read_csv(gabarito_path)
    
    files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if f.endswith('.csv')]
    
    # 1. Fit no TF-IDF
    all_cols = []
    for fpath in files:
        try:
            enc = loader.detect_encoding(fpath)
            sep = loader.detect_separator(fpath, enc)
            df = pd.read_csv(fpath, sep=sep, encoding=enc, nrows=1)
            all_cols.extend(df.columns.tolist())
        except: pass
    extractor.fit(all_cols)
    
    # 2. Extração
    dataset = []
    print(f"Processando {len(files)} arquivos...")
    
    for fpath in files:
        fname = os.path.basename(fpath)
        try:
            enc = loader.detect_encoding(fpath)
            sep = loader.detect_separator(fpath, enc)
            df = pd.read_csv(fpath, sep=sep, encoding=enc, low_memory=False)
            
            for col in df.columns:
                target_row = gabarito_df[(gabarito_df['nome_arquivo'] == fname) & (gabarito_df['nome_coluna'] == col)]
                if not target_row.empty:
                    target = target_row['tipo_real'].iloc[0]
                    feats = extractor.extract(df[col], col)
                    feats['target'] = target
                    feats['nome_arquivo'] = fname
                    feats['nome_coluna'] = col
                    dataset.append(feats)
        except Exception as e:
            warnings.warn(f"Erro em {fname}: {e}")
            
    if not dataset:
        raise ValueError("Nenhum dado foi extraído. Verifique se os nomes dos arquivos em 'data/raw' correspondem ao 'gabarito_master.csv'.")
            
    df_final = pd.DataFrame(dataset)
    df_final.to_csv(output_path, index=False)
    print(f"Features geradas com sucesso: {output_path}")