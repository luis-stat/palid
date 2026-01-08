import pytest
import pandas as pd
from src.cleaning import DataCleaner, CleanerSelector

class TestDataCleaner:
    """Testes para o orquestrador principal."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """DataFrame de exemplo com múltiplos tipos."""
        return pd.DataFrame({
            'data': ['15/07/2023', '2023-07-16', 'invalid'],
            'categoria': ['são paulo', 'RIO DE JANEIRO', 'curitiba'],
            'texto': ['Olá, mundo!', 'café com açúcar', 'teste'],
            'id': ['ABC123', 'abc123', 'ID-001'],
            'numero': [1, 2, 3],
        })
    
    @pytest.fixture
    def type_predictions(self):
        """Predições de tipo para as colunas."""
        return {
            'data': 'DATA_HORA',
            'categoria': 'CATEGORICO_NOMINAL',
            'texto': 'TEXTO_LIVRE',
            'id': 'ID',
            'numero': 'NUMERICO',
        }
    
    def test_full_pipeline(self, sample_dataframe, type_predictions):
        """Testa o pipeline completo de limpeza."""
        # Arrange
        commandCleaner = DataCleaner()
        
        # Act
        cleaned_df, report = commandCleaner.clean_dataset(
            sample_dataframe,
            type_predictions,
            similarity_threshold=85
        )
        
        # Assert
        assert 'columns_cleaned' in report
        assert 'total_rows_corrected' in report
        assert 'total_rows_removed' in report
        
        # Verifica que todas as colunas preditas foram processadas
        for col in type_predictions:
            assert col in report['columns_cleaned']
        
        # Verifica transformações específicas
        assert cleaned_df['data'].iloc[0] == '15/07/2023'
        assert cleaned_df['categoria'].iloc[0] == 'São Paulo'
        assert cleaned_df['id'].iloc[1] == 'ABC123'  # lowercase -> uppercase
        
        # Coluna 'numero' sem tipo predito não deve ser limpa
        assert 'numero' not in report['columns_cleaned']
    
    def test_with_reference_dictionaries(self, sample_dataframe, type_predictions):
        """Testa com dicionários de referência."""
        # Arrange
        commandCleaner = DataCleaner()
        
        reference_dicts = {
            'categoria': ['São Paulo', 'Rio de Janeiro', 'Curitiba']
        }
        
        # Act
        cleaned_df, report = commandCleaner.clean_dataset(
            sample_dataframe,
            type_predictions,
            reference_dictionaries=reference_dicts,
            similarity_threshold=80
        )
        
        # Assert
        cat_stats = report['columns_cleaned']['categoria']
        assert cat_stats['method'] == 'hybrid_dictionary'
        assert cat_stats.get('rows_corrected', 0) >= 0
    
    def test_custom_factory(self, sample_dataframe, type_predictions):
        """Testa com fábrica de cleaners customizada."""
        # Arrange
        from src.cleaning import DateCleaner, CategoricalCleaner
        
        # Cria cleaners customizados
        date_cleaner = DateCleaner(date_format='%Y-%m-%d')
        cat_cleaner = CategoricalCleaner()
        
        factory = CleanerSelector(cleaners=[date_cleaner, cat_cleaner])
        commandCleaner = DataCleaner(factory=factory)
        
        # Act
        cleaned_df, report = commandCleaner.clean_dataset(
            sample_dataframe,
            type_predictions,
            similarity_threshold=85
        )
        
        # Assert
        # Com fábrica customizada, apenas datas e categorias são limpas
        assert 'data' in report['columns_cleaned']
        assert 'categoria' in report['columns_cleaned']
        
        # Formato de data customizado
        assert cleaned_df['data'].iloc[1] == '2023-07-16'  # Formato YYYY-MM-DD
    
    def test_empty_dataframe(self):
        """Testa com DataFrame vazio."""
        # Arrange
        commandCleaner = DataCleaner()
        empty_df = pd.DataFrame()
        empty_predictions = {}
        
        # Act
        cleaned_df, report = commandCleaner.clean_dataset(
            empty_df,
            empty_predictions
        )
        
        # Assert
        assert len(cleaned_df) == 0
        assert report['total_rows_corrected'] == 0
        assert report['total_rows_removed'] == 0
        assert len(report['columns_cleaned']) == 0
    
    def test_missing_type_predictions(self, sample_dataframe):
        """Testa quando faltam predições de tipo para algumas colunas."""
        # Arrange
        commandCleaner = DataCleaner()
        
        # Predições apenas para algumas colunas
        partial_predictions = {
            'data': 'DATA_HORA',
            'categoria': 'CATEGORICO_NOMINAL',
        }
        
        # Act
        cleaned_df, report = commandCleaner.clean_dataset(
            sample_dataframe,
            partial_predictions
        )
        
        # Assert
        # Apenas colunas com predição devem ser processadas
        assert 'data' in report['columns_cleaned']
        assert 'categoria' in report['columns_cleaned']
        assert 'texto' not in report['columns_cleaned']
        assert 'id' not in report['columns_cleaned']
    
    def test_error_handling(self):
        """Testa tratamento de erros durante a limpeza."""
        # Arrange
        commandCleaner = DataCleaner()
        
        # DataFrame que pode causar erro
        problematic_df = pd.DataFrame({
            'coluna_estranha': [object(), object(), object()]  # Objetos não serializáveis
        })
        
        problematic_predictions = {
            'coluna_estranha': 'TEXTO_LIVRE'
        }
        
        # Act & Assert (não deve lançar exceção, deve continuar)
        try:
            cleaned_df, report = commandCleaner.clean_dataset(
                problematic_df,
                problematic_predictions
            )
            # Se chegou aqui, tratou o erro
            assert True
        except Exception as e:
            pytest.fail(f"Cleaners não tratou erro satisfatoriamente: {e}")