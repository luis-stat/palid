import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from typing import Dict, List, Optional, Set, Tuple, Any
import unicodedata
import re

class HybridCorrector:
    """
    Módulo independente para correção híbrida (exato + overlap + fuzzy).
    Pode ser reutilizado por múltiplos cleaners.
    """
    
    def __init__(self):
        self.cache = {}
    
    def normalize_text(self, text: Any) -> str:
        """Normaliza texto removendo acentos, caracteres especiais, etc."""
        if pd.isna(text):
            return ""
        text = str(text)
        text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
        text = text.lower().strip()
        text = re.sub(r'[^a-z0-9 ]', ' ', text)
        return re.sub(r'\s+', ' ', text)
    
    def get_tokens(self, text: str) -> Set[str]:
        """Extrai tokens (palavras) do texto."""
        if not text or pd.isna(text):
            return set()
        return set(text.split())
    
    def hierarchical_matcher(self, query: str, candidates_data: List[Dict], threshold: float) -> Optional[str]:
        """
        Corretor hierárquico em 3 níveis:
        1. Match exato (normalizado)
        2. Overlap de tokens
        3. Fuzzy matching
        """
        if not query or pd.isna(query):
            return None
        
        q_norm = self.normalize_text(query)
        if not q_norm:
            return None
        
        q_tokens = self.get_tokens(q_norm)
        
        # 1. Match exato (após normalização)
        for cand in candidates_data:
            if cand['norm'] == q_norm:
                return cand['orig']
        
        # 2. Overlap de tokens (Jaccard similarity simplificada)
        best_overlap = 0
        overlap_candidates = []
        
        if q_tokens:
            for cand in candidates_data:
                if not cand['tokens']:
                    continue
                
                # Calcula interseção de tokens
                intersection = len(q_tokens.intersection(cand['tokens']))
                union = len(q_tokens.union(cand['tokens']))
                
                if intersection > 0:  # Tem pelo menos algum overlap
                    overlap_score = intersection / union if union > 0 else 0
                    
                    if overlap_score > best_overlap:
                        best_overlap = overlap_score
                        overlap_candidates = [cand]
                    elif overlap_score == best_overlap and overlap_score > 0:
                        overlap_candidates.append(cand)
        
        # 3. Fuzzy matching no pool reduzido
        if overlap_candidates:
            final_pool = overlap_candidates
        else:
            final_pool = candidates_data
        
        if not final_pool:
            return None
        
        # Prepara strings para fuzzy matching
        choices = [c['norm'] for c in final_pool]
        
        try:
            # Usa token sort ratio que é mais robusto para ordem diferente
            match = process.extractOne(q_norm, choices, scorer=fuzz.token_sort_ratio)
            
            if match:
                matched_str, score, idx = match
                if score >= threshold:
                    return final_pool[idx]['orig']
        except Exception as e:
            # Fallback para ratio normal se token_sort_ratio falhar
            try:
                match = process.extractOne(q_norm, choices, scorer=fuzz.ratio)
                if match:
                    matched_str, score, idx = match
                    if score >= threshold:
                        return final_pool[idx]['orig']
            except:
                pass
        
        return None
    
    def correct(
        self,
        series: pd.Series,
        reference_values: List[str],
        threshold: float = 85
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Aplica correção híbrida a uma série.
        
        Args:
            series: Série de dados a corrigir
            reference_values: Lista de valores de referência válidos
            threshold: Limiar de similaridade (0-100)
            
        Returns:
            Tuple: (série corrigida, estatísticas de correção)
        """
        series_clean = series.copy()
        
        # Prepara candidatos
        candidates = []
        for ref_val in reference_values:
            if pd.isna(ref_val):
                continue
                
            norm_text = self.normalize_text(ref_val)
            candidates.append({
                'orig': ref_val,
                'norm': norm_text,
                'tokens': self.get_tokens(norm_text)
            })
        
        # Processa valores únicos
        unique_vals = []
        for val in series.dropna().unique():
            if pd.notna(val):
                unique_vals.append(str(val))
        
        corrections = {}
        
        for val in unique_vals:
            # Se já está na lista de referência, não precisa corrigir
            if val in reference_values:
                continue
            
            # Usa cache para performance
            if val not in self.cache:
                self.cache[val] = self.hierarchical_matcher(val, candidates, threshold)
            
            match = self.cache[val]
            if match and match != val:
                corrections[val] = match
        
        # Aplica correções
        if corrections:
            # Para manter tipos de dados, usamos replace mas tratamos NaN
            mask = series.isin(corrections.keys())
            series_clean = series.copy()
            
            for wrong_val, correct_val in corrections.items():
                # Substitui apenas valores não nulos que são iguais ao wrong_val
                series_clean = series_clean.replace(wrong_val, correct_val)
        
        # Estatísticas
        rows_affected = 0
        if corrections:
            # Conta quantas linhas foram afetadas
            for wrong_val in corrections.keys():
                rows_affected += (series == wrong_val).sum()
        
        stats = {
            'rows_corrected': int(rows_affected),
            'corrections_made': corrections.copy(),
            'rows_removed': 0
        }
        
        return series_clean, stats
    
    def batch_correct(
        self,
        values: List[str],
        reference_values: List[str],
        threshold: float = 85
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Versão para lista de strings (útil para testes).
        
        Returns:
            Tuple: (valores corrigidos, estatísticas)
        """
        series = pd.Series(values)
        corrected_series, stats = self.correct(series, reference_values, threshold)
        return corrected_series.tolist(), stats
    
    def clear_cache(self) -> None:
        """Limpa o cache do corretor."""
        self.cache.clear()