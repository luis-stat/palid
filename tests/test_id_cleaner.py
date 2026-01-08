import pytest
import pandas as pd
import numpy as np
from src.cleaning import IDCleaner

class TestIDCleaner:
    """Testes para o IDCleaner."""
    
    def test_basic_id_cleaning(self, sample_id_series):
        """Testa limpeza básica de IDs."""
        # Arrange
        cleaner = IDCleaner(
            strip_whitespace=True,
            case_sensitive=False
        )
        
        # Act
        cleaned_series, stats = cleaner.clean(sample_id_series)
        
        # Assert
        assert stats['method'] == 'id_cleaning'
        
        # Verifica transformações básicas
        assert cleaned_series.iloc[0] == 'ABC123'  # Mantém
        assert cleaned_series.iloc[1] == 'ABC123'  # lowercase -> uppercase
        assert cleaned_series.iloc[2] == 'ABC123'  # remove espaço
        assert cleaned_series.iloc[7] == 'ID002'  # remove espaços
        
        # Verifica formato de validação
        assert 'format_validation' in stats
    
    def test_duplicate_handling(self):
        """Testa tratamento de duplicatas."""
        # Arrange
        cleaner = IDCleaner()
        
        series = pd.Series(['ID001', 'ID001', 'ID002', 'ID003', 'ID001'])
        
        # Act - Marcar duplicatas
        cleaned_marked, stats_marked = cleaner.clean(
            series,
            handle_duplicates='mark'
        )
        
        # Act - Remover duplicatas
        cleaned_removed, stats_removed = cleaner.clean(
            series,
            handle_duplicates='remove'
        )
        
        # Assert
        assert stats_marked['duplicates_found'] == 3
        assert 'ID001_DUPLICATE_1' in cleaned_marked.values
        assert 'ID001_DUPLICATE_2' in cleaned_marked.values
        
        assert stats_removed['rows_removed'] == 3
        assert len(cleaned_removed) == 2  # Apenas IDs únicos
        assert 'ID001' in cleaned_removed.values
        assert 'ID002' in cleaned_removed.values
    
    def test_invalid_id_removal(self):
        """Testa remoção de IDs inválidos."""
        # Arrange
        cleaner = IDCleaner()
        
        series = pd.Series(['VALID123', '', '   ', 'TOOLONG' * 20, 'ID-001'])
        
        # Act - Remover inválidos
        cleaned_removed, stats_removed = cleaner.clean(
            series,
            remove_invalid_ids=True
        )
        
        # Act - Marcar inválidos como NA
        cleaned_marked, stats_marked = cleaner.clean(
            series,
            remove_invalid_ids=False
        )
        
        # Assert
        assert stats_removed['rows_removed'] > 0
        assert len(cleaned_removed) < len(series)
        
        assert stats_marked['rows_corrected'] > 0
        assert pd.isna(cleaned_marked.iloc[1])  # String vazia -> NA
        assert pd.isna(cleaned_marked.iloc[2])  # Só espaços -> NA
    
    def test_format_validation(self):
        """Testa validação de formatos específicos."""
        # Arrange
        cleaner = IDCleaner()
        
        series = pd.Series([
            '123.456.789-00',  # CPF
            '11.222.333/4444-55',  # CNPJ
            '550e8400-e29b-41d4-a716-446655440000',  # UUID
            'email@exemplo.com',  # Email
            'ABC123',  # Código genérico
        ])
        
        # Act
        cleaned_series, stats = cleaner.clean(series)
        
        # Assert
        validation = stats['format_validation']
        
        # Deve detectar alguns formatos (depende da regex)
        assert validation.get('cpf', 0) >= 1
        assert validation.get('cnpj', 0) >= 1
        assert validation.get('uuid', 0) >= 1
    
    def test_can_handle_id_type(self):
        """Testa detecção de tipo ID."""
        # Arrange
        cleaner = IDCleaner()
        
        # Caso 1: Tipo explícito
        sample1 = pd.Series(['ID001', 'ID002'])
        assert cleaner.can_handle('ID', sample1) == True
        
        # Caso 2: Alta cardinalidade (>95%)
        sample2 = pd.Series([f'ID_{i:03d}' for i in range(100)])
        assert cleaner.can_handle('OUTRO_TIPO', sample2) == True
        
        # Caso 3: Parece código (mix letras/números)
        sample3 = pd.Series(['AB123', 'CD456', 'EF789'])
        assert cleaner.can_handle('OUTRO_TIPO', sample3) == True
        
        # Caso 4: Não parece código
        sample4 = pd.Series(['Nome 1', 'Nome 2', 'Nome 3'])
        assert cleaner.can_handle('OUTRO_TIPO', sample4) == False
    
    def test_case_sensitive_option(self):
        """Testa opção case_sensitive."""
        # Arrange
        sensitive_cleaner = IDCleaner(case_sensitive=True)
        insensitive_cleaner = IDCleaner(case_sensitive=False)
        
        series = pd.Series(['AbC123', 'aBc123', 'ABC123'])
        
        # Act
        cleaned_sensitive, _ = sensitive_cleaner.clean(series)
        cleaned_insensitive, _ = insensitive_cleaner.clean(series)
        
        # Assert
        # Case sensitive mantém o caso original
        assert cleaned_sensitive.iloc[0] == 'AbC123'
        assert cleaned_sensitive.iloc[1] == 'aBc123'
        
        # Case insensitive converte para maiúsculas
        assert cleaned_insensitive.iloc[0] == 'ABC123'
        assert cleaned_insensitive.iloc[1] == 'ABC123'