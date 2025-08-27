"""
CFEM Analytics - Análise da Compensação Financeira pela Exploração de Recursos Minerais

Este pacote fornece ferramentas para análise avançada de dados CFEM, incluindo:
- Processamento e limpeza de dados
- Análises estatísticas e de machine learning
- Visualizações interativas
- Análises geoespaciais

Módulos:
    data_processor: Processamento e limpeza de dados CFEM
    analytics: Análises estatísticas e machine learning
    visualizations: Criação de gráficos e dashboards
    geo_analysis: Análises geoespaciais avançadas

Versão: 1.0.0
Autor: CFEM Analytics Team
Email: contato@cfem-analytics.com
"""

__version__ = "1.0.0"
__author__ = "CFEM Analytics Team"
__email__ = "contato@cfem-analytics.com"

# Importações principais
from .data_processor import CFEMDataProcessor
from .analytics import CFEMAnalytics
from .visualizations import CFEMVisualizations
from .geo_analysis import CFEMGeoAnalysis

# Lista de exportações públicas
__all__ = [
    'CFEMDataProcessor',
    'CFEMAnalytics', 
    'CFEMVisualizations',
    'CFEMGeoAnalysis',
    '__version__',
    '__author__',
    '__email__'
]

# Configurações e constantes globais
DEFAULT_CONFIG = {
    'processing': {
        'remove_outliers': False,
        'min_cfem_value': 0.01,
        'coordinate_validation': True
    },
    'analytics': {
        'clustering_eps_km': 50.0,
        'min_samples': 3,
        'anomaly_contamination': 0.1
    },
    'visualizations': {
        'theme': 'plotly_white',
        'color_palette': 'viridis',
        'figure_height': 500
    },
    'geo_analysis': {
        'hotspot_radius_km': 100.0,
        'spatial_threshold': 0.05
    }
}

# Metadados dos dados CFEM
CFEM_COLUMNS = {
    'required': [
        'TITULAR',
        'MUNICIPIO(S)', 
        'ESTADO',
        'PRIMEIRODESUBS',
        'CFEM',
        'LONGITUDE',
        'LATITUDE'
    ],
    'optional': [
        'DATA',
        'SEGUNDODESUBS',
        'TERCEIRODESUBS'
    ],
    'calculated': [
        'CFEM_FAIXA',
        'REGIAO',
        'PORTE_EMPRESA',
        'DENSIDADE_ESTADO',
        'DIVERSIFICACAO_EMPRESA'
    ]
}

# Mapeamento de regiões brasileiras
REGIOES_BRASIL = {
    # Norte
    'AC': 'Norte', 'AP': 'Norte', 'AM': 'Norte', 'PA': 'Norte', 
    'RO': 'Norte', 'RR': 'Norte', 'TO': 'Norte',
    # Nordeste
    'AL': 'Nordeste', 'BA': 'Nordeste', 'CE': 'Nordeste', 'MA': 'Nordeste',
    'PB': 'Nordeste', 'PE': 'Nordeste', 'PI': 'Nordeste', 'RN': 'Nordeste', 'SE': 'Nordeste',
    # Centro-Oeste
    'GO': 'Centro-Oeste', 'MT': 'Centro-Oeste', 'MS': 'Centro-Oeste', 'DF': 'Centro-Oeste',
    # Sudeste
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste', 'SP': 'Sudeste',
    # Sul
    'PR': 'Sul', 'RS': 'Sul', 'SC': 'Sul'
}

# Validação de importações
try:
    import pandas as pd
    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px
    import folium
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import IsolationForest, RandomForestRegressor
    from scipy import stats
except ImportError as e:
    print(f"Aviso: Algumas dependências podem não estar instaladas: {e}")
    print("Execute: pip install -r requirements.txt")

def get_sample_data():
    """
    Retorna dados de exemplo para testes
    
    Returns:
        pd.DataFrame: DataFrame com dados de exemplo
    """
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    estados = ['MG', 'PA', 'GO', 'MT', 'BA', 'RS', 'PR', 'SP']
    substancias = ['FERRO', 'OURO', 'CALCÁRIO', 'AREIA', 'BAUXITA']
    empresas = ['EMPRESA A LTDA', 'MINERADORA B S.A.', 'COMPANHIA C', 'EXTRATORA D']
    
    data = []
    for i in range(100):
        data.append({
            'TITULAR': np.random.choice(empresas),
            'MUNICIPIO(S)': f'MUNICÍPIO {i+1}',
            'ESTADO': np.random.choice(estados),
            'PRIMEIRODESUBS': np.random.choice(substancias),
            'CFEM': np.random.lognormal(mean=10, sigma=2),
            'LONGITUDE': np.random.uniform(-74, -32),
            'LATITUDE': np.random.uniform(-34, 6)
        })
    
    return pd.DataFrame(data)

def validate_cfem_data(df):
    """
    Valida se o DataFrame possui as colunas necessárias para análise CFEM
    
    Args:
        df (pd.DataFrame): DataFrame a ser validado
        
    Returns:
        tuple: (bool, list) - (é válido, lista de colunas faltantes)
    """
    required_cols = CFEM_COLUMNS['required']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    is_valid = len(missing_cols) == 0
    
    return is_valid, missing_cols

def create_cfem_pipeline():
    """
    Cria pipeline completo de análise CFEM
    
    Returns:
        dict: Dicionário com instâncias das classes principais
    """
    return {
        'processor': CFEMDataProcessor(),
        'analytics': CFEMAnalytics(),
        'visualizations': CFEMVisualizations(),
        'geo_analysis': CFEMGeoAnalysis()
    }

# Informações sobre o projeto
PROJECT_INFO = {
    'name': 'CFEM Analytics',
    'description': 'Plataforma de análise avançada para dados CFEM',
    'version': __version__,
    'author': __author__,
    'email': __email__,
    'license': 'MIT',
    'python_requires': '>=3.8',
    'keywords': ['CFEM', 'mineração', 'análise', 'visualização', 'geoespacial'],
    'url': 'https://github.com/cfem-analytics/cfem-analytics'
}

def print_project_info():
    """Exibe informações do projeto"""
    print(f"📊 {PROJECT_INFO['name']} v{PROJECT_INFO['version']}")
    print(f"📝 {PROJECT_INFO['description']}")
    print(f"👨‍💻 Autor: {PROJECT_INFO['author']}")
    print(f"📧 Email: {PROJECT_INFO['email']}")
    print(f"🐍 Python: {PROJECT_INFO['python_requires']}")
    print(f"📄 Licença: {PROJECT_INFO['license']}")

if __name__ == "__main__":
    print_project_info()