import streamlit as st
import pandas as pd
import os
from src.dataset import DataLoader
from src.cleaning import DataCleaner
from src.modeling.predict import ModelPredictor

# Configurações de Caminho
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'semantic_type_classifier.pkl')
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
GABARITO_PATH = os.path.join(BASE_DIR, 'data', 'external', 'gabarito_master.csv')

def load_reference_dicts(target_cols):
    """Carrega gabaritos da pasta data/raw baseado no arquivo mestre."""
    dicts = {}
    if not os.path.exists(GABARITO_PATH): return dicts
    
    try:
        master = pd.read_csv(GABARITO_PATH)
        for col in target_cols:
            matches = master[master['nome_coluna'] == col]
            if not matches.empty:
                fname = matches.iloc[0]['nome_arquivo']
                fpath = os.path.join(DATA_RAW_DIR, fname)
                if os.path.exists(fpath):
                    ref_df = pd.read_csv(fpath, usecols=[col])
                    vals = ref_df[col].dropna().astype(str).unique().tolist()
                    dicts[col] = [v for v in vals if v.strip()]
    except Exception as e:
        st.error(f"Erro carregando gabaritos: {e}")
    return dicts

def main():
    st.set_page_config(page_title="AutoClean", layout="wide")
    st.title("Sistema Inteligente de Limpeza de Dados")
    
    with st.sidebar:
        st.header("Configurações")
        threshold = st.slider("Sensibilidade (Híbrido)", 60, 100, 85)
        st.info("Arquitetura Modular v2.0")

    uploaded_file = st.file_uploader("Suba seu CSV sujo", type=['csv'])
    
    if uploaded_file:
        loader = DataLoader()
        df, _, _ = loader.load_data(uploaded_file)
        
        # Cache de Predição
        if 'preds' not in st.session_state:
            try:
                predictor = ModelPredictor(MODEL_PATH)
                st.session_state['preds'] = predictor.predict_column_types(df)
            except Exception as e:
                st.error(f"Erro ao carregar modelo: {e}")
                st.session_state['preds'] = {}

        # Cache de Dicionários
        if 'dicts' not in st.session_state:
            st.session_state['dicts'] = load_reference_dicts(df.columns)
            
        # Preview
        c1, c2 = st.columns([3, 1])
        c1.dataframe(df.head())
        with c2:
            st.info(f"{len(df)} linhas")
            st.info(f"{len(st.session_state['dicts'])} gabaritos encontrados")

        if st.button("Executar Limpeza"):
            cleaner = DataCleaner()
            
            with st.spinner("Limpando dados com arquitetura modular"):
                df_clean, report = cleaner.clean_dataset(
                    df, 
                    st.session_state['preds'], 
                    st.session_state['dicts'], 
                    threshold
                )
            
            st.success("Concluído!")
            st.subheader("Relatório")
            
            resumo = []
            for col, stats in report['columns_cleaned'].items():
                if stats['rows_corrected'] > 0 or stats['rows_removed'] > 0:
                    resumo.append({
                        'Coluna': col, 'Método': stats['method'], 
                        'Corrigidos': stats['rows_corrected'], 'Removidos': stats['rows_removed']
                    })
            
            if resumo: 
                st.dataframe(pd.DataFrame(resumo))
            else: 
                st.info("Nenhuma correção necessária.")
            
            st.download_button("Baixar CSV", df_clean.to_csv(index=False), "limpo.csv", "text/csv")

if __name__ == "__main__":
    main()