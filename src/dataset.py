import pandas as pd
import numpy as np
import chardet
import random
from typing import Tuple

class DataLoader:
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'

    @staticmethod
    def detect_separator(file_path: str, encoding: str) -> str:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                line = f.readline()
            if line.count(';') > line.count(','):
                return ';'
            return ','
        except Exception:
            return ','

    def load_data(self, uploaded_file) -> Tuple[pd.DataFrame, str, str]:
        """Carrega dados de um arquivo carregado pelo Streamlit ou caminho local."""
        try:
            # Se for upload do streamlit (bytes)
            if hasattr(uploaded_file, 'read'):
                # Salva temporariamente para detecção (ou lê direto se ajustar a lógica)
                # Aqui assumimos leitura direta pandas para simplificar no streamlit
                df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                return df, ',', 'utf-8' # Simplificação para stream
            
            # Se for caminho local
            enc = self.detect_encoding(uploaded_file)
            sep = self.detect_separator(uploaded_file, enc)
            df = pd.read_csv(uploaded_file, sep=sep, encoding=enc, on_bad_lines='skip')
            return df, sep, enc
        except Exception as e:
            # Fallback
            return pd.read_csv(uploaded_file, on_bad_lines='skip'), ',', 'utf-8'

def generate_synthetic_data(output_dir: str, n=300):
    """Gera os dados sintéticos para treinamento."""
    random.seed(2024)
    np.random.seed(2024)
    
    def random_dates(start, end, n):
        start_u = start.value//10**9
        end_u = end.value//10**9
        return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

    # Exemplo de geração
    df_av = pd.DataFrame({
        'flight_no': [f'VOO-{random.randint(100,999)}' for _ in range(n)],
        'companhia': np.random.choice(['Gol', 'Latam', 'Azul', 'TAP'], n),
        'origem': np.random.choice(['GRU', 'GIG', 'MIA', 'LIS'], n),
        'destino': np.random.choice(['JFK', 'LHR', 'CDG', 'DXB'], n),
        'atraso_minutos': np.random.randint(0, 180, n),
        'data_partida': random_dates(pd.to_datetime('2024-01-01'), pd.to_datetime('2024-12-31'), n),
        'cancelado': np.random.choice([True, False], n, p=[0.05, 0.95])
    })
    df_av.to_csv(f'{output_dir}/viagens_aereas.csv', index=False)
    print(f"Gerado: viagens_aereas.csv em {output_dir}")