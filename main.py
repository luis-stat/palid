import streamlit as st
import pandas as pd
import os
from src.dataset import DataLoader
from src.cleaning import DataCleaner
from src.modeling.predict import ModelPredictor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'semantic_type_classifier.pkl')
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
GABARITO_PATH = os.path.join(BASE_DIR, 'data', 'external', 'gabarito_master.csv')

def load_gabarito_rules(filepath):
    if not os.path.exists(filepath): return {}
    try:
        df_gabarito = pd.read_csv(filepath)
        def clean_text(text): return str(text).replace('"', '').replace("'", "").strip().lower()
        df_gabarito['nome_coluna'] = df_gabarito['nome_coluna'].apply(clean_text)
        if 'tipo' in df_gabarito.columns:
            df_gabarito['tipo'] = df_gabarito['tipo'].apply(clean_text)
            return dict(zip(df_gabarito['nome_coluna'], df_gabarito['tipo']))
        return {}
    except Exception as e:
        st.error(f"Erro lendo gabarito: {e}"); return {}

def load_reference_dicts(target_cols):
    dicts = {}
    if not os.path.exists(GABARITO_PATH): return dicts
    try:
        master = pd.read_csv(GABARITO_PATH)
        master['nome_coluna_norm'] = master['nome_coluna'].astype(str).str.replace('"', '').str.replace("'", "").str.strip().str.lower()
        for col in target_cols:
            col_norm = str(col).replace('"', '').replace("'", "").strip().lower()
            matches = master[master['nome_coluna_norm'] == col_norm]
            if not matches.empty:
                fname = matches.iloc[0]['nome_arquivo']
                if pd.isna(fname) or str(fname).strip() == '': continue
                fpath = os.path.join(DATA_RAW_DIR, str(fname))
                if os.path.exists(fpath):
                    try:
                        ref_df = pd.read_csv(fpath)
                        vals = ref_df.iloc[:, 0].dropna().astype(str).unique().tolist()
                        dicts[col] = [v for v in vals if v.strip()]
                    except: pass
    except Exception as e: st.error(f"Erro carregando dicionários: {e}")
    return dicts

def main():
    st.set_page_config(page_title="PALID", layout="wide")
    st.title("PALID - PET Estatística")

    with st.sidebar:
        st.header("Configurações")
        
        st.subheader("Leitura do Arquivo")
        sep_manual = st.selectbox(
            "Separador do CSV", 
            options=[None, ',', ';', '\t'], 
            format_func=lambda x: 'Automático' if x is None else x,
        )
        
        st.divider()
        st.subheader("Limpeza")
        threshold = st.slider("Sensibilidade:", 0, 100, 85, help="Quanto menor, mais agressiva é a correção.")
        st.info("Para uso no Fuzzy Match")

    uploaded_file = st.file_uploader("Suba seu CSV para análise", type=['csv'])

    if uploaded_file:
        try:
            if sep_manual:
                df = pd.read_csv(uploaded_file, sep=sep_manual, on_bad_lines='skip')
            else:
                loader = DataLoader()
                df, _, _ = loader.load_data(uploaded_file)
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            return

        if 'init_done' not in st.session_state:
            manual_rules = load_gabarito_rules(GABARITO_PATH)
            
            try:
                predictor = ModelPredictor(MODEL_PATH)
                ai_preds = predictor.predict_column_types(df)
            except: ai_preds = {}

            final_preds = ai_preds.copy()
            for col in df.columns:
                col_clean = str(col).replace('"', '').replace("'", "").strip().lower()
                if col_clean in manual_rules:
                    final_preds[col] = manual_rules[col_clean].upper()
            
            st.session_state['preds'] = final_preds
            st.session_state['dicts'] = {} 
            st.session_state['init_done'] = True
        
        st.subheader("Pré-visualização dos dados")
        st.dataframe(df, use_container_width=True)
        st.divider()

        st.subheader("Regras de padronização")
        st.info("Compare os valores encontrados com o que você deseja e preencha a caixa de texto para corrigir.")

        with st.expander("Painel de edição de colunas", expanded=True):
            user_preds = st.session_state['preds'].copy()
            user_dicts = st.session_state['dicts'].copy()
            
            available_types = ['TEXTO_LIVRE', 'NUMERICO', 'DATA', 'CATEGORICO', 'ID', 'EMAIL', 'BOOLEANO']

            for col in df.columns:
                c_info, c_input = st.columns([1, 1], gap="large")
                
                with c_info:
                    st.markdown(f"### `{col}`")
                    
                    unique_vals = df[col].dropna().unique().tolist()
                    preview_vals = unique_vals[:20]
                    preview_str = ", ".join(map(str, preview_vals))
                    if len(unique_vals) > 20:
                        preview_str += f" ... (+{len(unique_vals)-20})"
                    
                    st.caption("Valores encontrados no arquivo:")
                    st.code(preview_str, language=None)

                with c_input:
                    current_type = user_preds.get(col, 'TEXTO_LIVRE')
                    if current_type not in available_types: available_types.append(current_type)
                    
                    new_type = st.selectbox(
                        "Tipo de Dado", 
                        options=available_types, 
                        index=available_types.index(current_type),
                        key=f"type_{col}"
                    )
                    user_preds[col] = new_type

                    current_user_input = user_dicts.get(col, [])
                    vals_str = ", ".join(current_user_input)
                    
                    new_vals_str = st.text_area(
                        f"Valores esperados (Separar por vírgula)", 
                        value=vals_str,
                        height=100,
                        key=f"dict_{col}",
                        placeholder="Digite os valores corretos aqui. Ex: Fortaleza, Caucaia, Maracanaú",
                        help="O sistema usará Fuzzy Match para transformar os valores da esquerda nestes valores."
                    )
                    
                    if new_vals_str.strip():
                        user_dicts[col] = [x.strip() for x in new_vals_str.split(',') if x.strip()]
                        if user_preds[col] != 'CATEGORICO':
                            user_preds[col] = 'CATEGORICO'
                    else:
                        if col in user_dicts: del user_dicts[col]
                
                st.markdown("---")

            st.session_state['preds'] = user_preds
            st.session_state['dicts'] = user_dicts

        st.write("")
        executar_btn = st.button("Executar Limpeza", type="primary", use_container_width=True)

        if executar_btn:
            cleaner = DataCleaner()
            with st.spinner("Padronizando dados"):
                df_clean, report = cleaner.clean_dataset(
                    df, 
                    st.session_state['preds'], 
                    st.session_state['dicts'], 
                    threshold
                )
            
            st.success("Limpeza Concluída!")
            st.divider()

            tab1, tab2, tab3 = st.tabs(["Dados Limpos", "Relatório", "Comparativo"])

            with tab1:
                st.dataframe(df_clean, use_container_width=True)
                st.download_button("Baixar CSV", df_clean.to_csv(index=False), "limpo.csv", "text/csv")

            with tab2:
                resumo = []
                for col, stats in report['columns_cleaned'].items():
                    if stats.get('rows_corrected', 0) > 0 or stats.get('rows_removed', 0) > 0:
                        resumo.append({
                            'Coluna': col, 
                            'Ação': stats.get('method', 'N/A'), 
                            'Corrigidos': stats.get('rows_corrected', 0), 
                            'Removidos': stats.get('rows_removed', 0)
                        })
                if resumo:
                    st.dataframe(pd.DataFrame(resumo), use_container_width=True)
                else:
                    st.info("Nenhuma alteração necessária.")

            with tab3:
                cols_comp = st.multiselect("Escolha as colunas para comparar:", df.columns)
                
                if cols_comp:
                    data_compare = {}
                    for col in cols_comp:
                        data_compare[f"{col} (Original)"] = df[col].astype(str).head(20).reset_index(drop=True)
                        data_compare[f"{col} (Limpo)"] = df_clean[col].astype(str).head(20).reset_index(drop=True)
                    
                    df_view = pd.DataFrame(data_compare)
                    
                    st.dataframe(df_view, use_container_width=True)
                else:
                    st.info("Selecione pelo menos uma coluna acima para visualizar a comparação.")

if __name__ == "__main__":
    main()