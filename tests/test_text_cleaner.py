import pytest
import pandas as pd
import numpy as np
from src.cleaning import TextCleaner
from src.cleaning import HybridCorrector

class TestTextCleaner:
    """Testes para o TextCleaner."""
    
    def test_basic_normalization(self, sample_text_series):
        """Testa normalização básica de texto."""
        # Arrange
        cleaner = TextCleaner(
            normalize_accents=True,
            remove_special_chars=True,
            enable_spell_check=False
        )
        
        # Act
        cleaned_series, stats = cleaner.clean(sample_text_series)
        
        # Assert
        assert stats['method'] == 'text_cleaning'
        assert stats['normalization_applied'] == True
        
        # Verifica normalização
        assert cleaned_series.iloc[0] == 'ola mundo'  # Remove acentos e pontuação
        assert cleaned_series.iloc[1] == 'cafe com acucar'  # Remove acentos
        assert cleaned_series.iloc[8] == 'acao reacao'  # Remove acentos e caracteres especiais
        
        # Verifica tratamento de espaços
        assert cleaned_series.iloc[7] == 'texto com espacos'  # Remove espaços extras
    
    def test_without_normalization(self):
        """Testa limpeza sem normalização."""
        # Arrange
        cleaner = TextCleaner(
            normalize_accents=False,
            remove_special_chars=False,
            enable_spell_check=False
        )
        
        series = pd.Series(['café com açúcar', 'São Paulo - SP'])
        
        # Act
        cleaned_series, stats = cleaner.clean(series)
        
        # Assert
        assert stats['normalization_applied'] == False
        assert cleaned_series.iloc[0] == 'café com açúcar'  # Mantém acentos
        assert cleaned_series.iloc[1] == 'São Paulo - SP'  # Mantém caracteres especiais
    
    def test_hybrid_correction(self):
        """Testa correção híbrida em texto."""
        # Arrange
        corrector = HybridCorrector()
        cleaner = TextCleaner(corrector=corrector)
        
        series = pd.Series(['sao paulo', 'rio janero', 'curitiba'])
        reference = ['São Paulo', 'Rio de Janeiro', 'Curitiba']
        
        # Act
        cleaned_series, stats = cleaner.clean(
            series,
            reference_values=reference,
            similarity_threshold=70
        )
        
        # Assert
        assert stats['method'] == 'text_with_hybrid_correction'
        assert stats['hybrid_corrections'] > 0
    
    def test_remove_empty_rows(self):
        """Testa remoção de linhas vazias."""
        # Arrange
        cleaner = TextCleaner()
        
        series = pd.Series(['texto', '', '   ', np.nan, 'outro texto'])
        
        # Act
        cleaned_series, stats = cleaner.clean(
            series,
            remove_empty_rows=True
        )
        
        # Assert
        assert len(cleaned_series) == 2  # Apenas 'texto' e 'outro texto'
        assert stats['rows_removed'] == 3
    
    def test_can_handle_text_type(self):
        """Testa detecção de tipo TEXTO_LIVRE."""
        # Arrange
        cleaner = TextCleaner()
        
        # Caso 1: Tipo explícito
        sample1 = pd.Series(['algum texto'])
        assert cleaner.can_handle('TEXTO_LIVRE', sample1) == True
        
        # Caso 2: Cardinalidade alta + strings
        sample2 = pd.Series([f'texto_{i}' for i in range(100)])
        assert cleaner.can_handle('OUTRO_TIPO', sample2) == True
        
        # Caso 3: Poucas strings
        sample3 = pd.Series([1, 2, 3, 4, 5])
        assert cleaner.can_handle('OUTRO_TIPO', sample3) == False
    
    @pytest.mark.skipif(True, reason="Requer pyspellchecker instalado")
    def test_spell_checking(self):
        """Testa correção ortográfica (requer pyspellchecker)."""
        # Arrange
        cleaner = TextCleaner(enable_spell_check=True)
        
        series = pd.Series(['casa', 'kaza', 'mesa', 'mza'])
        
        # Act
        cleaned_series, stats = cleaner.clean(series)
        
        # Assert
        assert stats['spell_corrections'] > 0
        assert stats['method'] == 'text_with_spell_check'
        # 'kaza' deve virar 'casa', 'mza' pode não ter correção
    
    def test_dynamic_options(self):
        """Testa configuração dinâmica de opções."""
        # Arrange
        cleaner = TextCleaner(
            normalize_accents=True,
            remove_special_chars=True
        )
        
        series = pd.Series(['café com açúcar!'])
        
        # Act - Limpa com configuração inicial
        cleaned1, _ = cleaner.clean(series)
        
        # Muda configuração dinamicamente
        cleaner.set_options(normalize_accents=False)
        cleaned2, _ = cleaner.clean(series)
        
        # Assert
        assert cleaned1.iloc[0] == 'cafe com acucar'  # Com normalização
        assert cleaned2.iloc[0] == 'café com açúcar!'  # Sem normalização (mantém acentos)