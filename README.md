# 🏗️ CFEM Analytics - Sistema de Análise Minerária

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Sobre o Projeto

O **Geominer Analytics** é uma aplicação web interativa desenvolvida em **Streamlit** para análise abrangente dos dados de **Exploração de Recursos Minerais (CFEM)** no Brasil. 

O sistema oferece visualizações dinâmicas, análises estatísticas avançadas, insights geoespaciais e modelos de machine learning para compreender o panorama da mineração brasileira.

### 🎯 Objetivos

- **Transparência**: Facilitar o acesso e compreensão dos dados de CFEM
- **Insights**: Gerar análises aprofundadas sobre o setor mineral
- **Interatividade**: Proporcionar exploração dinâmica dos dados
- **Inteligência**: Aplicar técnicas de ML para descobrir padrões

## ✨ Funcionalidades Principais

### 📊 Dashboard Executivo
- **KPIs em tempo real**: Total CFEM, número de empresas, estados ativos
- **Mapas interativos**: Distribuição geográfica das operações
- **Rankings dinâmicos**: Maiores arrecadadores e estados mais ativos
- **Filtros avançados**: Por empresa, estado, substância e valor

### 🗺️ Análises Geoespaciais
- **Mapas de calor**: Concentração de atividades minerárias
- **Clustering espacial**: Agrupamento de operações similares
- **Análise regional**: Comparativos entre regiões/estados
- **Densidade operacional**: Operações por área geográfica

### 🧠 Machine Learning & Analytics
- **Clustering inteligente**: Segmentação automática de operações
- **Detecção de anomalias**: Identificação de outliers
- **Modelos preditivos**: Previsão de valores CFEM
- **Análise de concentração**: Índices de mercado (HHI, Gini)

### 📈 Visualizações Avançadas
- **Gráficos interativos**: Plotly para máxima interatividade
- **Análises estatísticas**: Distribuições, correlações, testes
- **Comparativos temporais**: Evolução histórica (se disponível)
- **Export de relatórios**: PDF e CSV para análises offline

## 🚀 Começando

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git (opcional, para clonagem do repositório)

### Instalação

1. **Clone o repositório:**
```bash
git clone https://github.com/niloRoch/cfem-analytics.git
cd cfem-analytics
```

2. **Crie um ambiente virtual:**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Instale as dependências:**
```bash
pip install -r requirements.txt
```

4. **Execute a aplicação:**
```bash
streamlit run app.py
```

5. **Acesse no navegador:**
```
http://localhost:8501
```

### 📁 Estrutura de Dados

Coloque seu arquivo Excel `Emp-CFEM.xlsx` na pasta `data/raw/` com as seguintes colunas:

| Coluna | Descrição | Tipo |
|--------|-----------|------|
| `Titular` | Nome da empresa mineradora | Texto |
| `Municipio(s)` | Município onde ocorre a exploração | Texto |
| `Estado` | Unidade Federativa (sigla) | Texto |
| `PrimeiroDeSUBS` | Substância mineral extraída | Texto |
| `LONGITUDE` | Coordenada de longitude | Numérico |
| `LATITUDE` | Coordenada de latitude | Numérico |
| `CFEM` | Valor da compensação financeira | Numérico |

## 🏗️ Arquitetura do Sistema

```
cfem-analytics/
├── app.py                      # 🎯 Aplicação principal
├── requirements.txt            # 📦 Dependências
├── README.md                  # 📖 Documentação
├── data/
│   ├── raw/                   # 📊 Dados originais
│   └── processed/             # 🔄 Dados processados
├── src/
│   ├── data_processor.py      # 🛠️ Processamento de dados
│   ├── visualizations.py     # 📊 Visualizações
│   ├── analytics.py          # 🧮 Analytics & ML
│   └── geo_analysis.py       # 🗺️ Análises geoespaciais
├── assets/
│   ├── images/               # 🖼️ Imagens
│   └── styles/               # 🎨 Estilos CSS
└── tests/
    └── test_*.py             # 🧪 Testes unitários
```

## 📊 Casos de Uso

### Análise Básica

```python
from src.data_processor import CFEMDataProcessor
from src.visualizations import CFEMVisualizations

# Carregar e processar dados
processor = CFEMDataProcessor()
df = processor.load_excel_data('data/raw/Emp-CFEM.xlsx')
df_clean = processor.clean_data(df)
df_enriched = processor.enrich_data(df_clean)

# Criar visualizações
viz = CFEMVisualizations()
dashboard_figs = viz.create_executive_dashboard(df_enriched, stats)
```

### Machine Learning

```python
from src.analytics import CFEMAnalytics

# Análises avançadas
analytics = CFEMAnalytics()

# Clustering
clustering_results = analytics.perform_clustering_analysis(df_enriched)

# Detecção de anomalias
anomaly_results = analytics.detect_anomalies(df_enriched)

# Modelo preditivo
model_results = analytics.build_predictive_model(df_enriched)
```

## 🔧 Configuração Avançada

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto:

```env
# Configurações da aplicação
STREAMLIT_THEME=light
DEBUG_MODE=False
CACHE_TTL=3600

# Configurações de dados
DATA_PATH=data/raw/Emp-CFEM.xlsx
PROCESSED_PATH=data/processed/

# Configurações de ML
RANDOM_SEED=42
MODEL_CACHE=True
```

### Personalização de Temas

Modifique `assets/styles/custom.css` para personalizar a aparência:

```css
/* Tema personalizado */
:root {
    --primary-color: #1f4e79;
    --secondary-color: #667eea;
    --accent-color: #764ba2;
}

.main-header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 2rem;
    border-radius: 10px;
}
```

## 📈 Métricas e KPIs

O sistema calcula automaticamente diversos indicadores:

### Indicadores Financeiros
- **Valor Total CFEM**: Soma de todas as compensações
- **CFEM Médio**: Valor médio por operação
- **Concentração de Receita**: % das top 10 empresas

### Indicadores Operacionais
- **Número de Operações**: Total de registros únicos
- **Diversidade de Substâncias**: Tipos de minerais diferentes
- **Cobertura Geográfica**: Estados e municípios ativos

### Índices de Concentração
- **HHI (Herfindahl-Hirschman)**: Concentração de mercado
- **CR4/CR8**: Razão de concentração top 4/8 empresas
- **Coeficiente de Gini**: Desigualdade na distribuição

## 🧪 Testes

Execute os testes unitários:

```bash
# Todos os testes
python -m pytest tests/

# Teste específico
python -m pytest tests/test_data_processor.py -v

# Com cobertura
python -m pytest tests/ --cov=src/
```

## 🚀 Deploy

### Streamlit Cloud

1. Faça push do código para GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte seu repositório
4. Configure as variáveis de ambiente
5. Deploy automático!

### Docker

```bash
# Build da imagem
docker build -t cfem-analytics .

# Execute o container
docker run -p 8501:8501 cfem-analytics
```

### Heroku

```bash
# Login no Heroku
heroku login

# Crie o app
heroku create cfem-analytics-app

# Deploy
git push heroku main
```

## 🤝 Contribuindo

Contribuições são muito bem-vindas! Veja como participar:

1. **Fork** o projeto
2. **Crie** uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. **Push** para a branch (`git push origin feature/AmazingFeature`)
5. **Abra** um Pull Request

### Diretrizes

- Siga o padrão PEP 8 para Python
- Escreva testes para novas funcionalidades
- Documente mudanças no README
- Use conventional commits

## 🐛 Reportando Bugs

Encontrou um bug? Ajude-nos a melhorar:

1. **Verifique** se o bug já foi reportado nas [Issues](https://github.com/niloRoch/cfem-analytics/issues)
2. **Crie** uma nova issue com:
   - Descrição clara do problema
   - Passos para reproduzir
   - Screenshots (se aplicável)
   - Informações do ambiente (OS, Python, etc.)

## 📚 Documentação Detalhada

### APIs Principais

#### CFEMDataProcessor

```python
class CFEMDataProcessor:
    """Processamento e limpeza de dados CFEM"""
    
    def load_excel_data(file_path: str) -> pd.DataFrame:
        """Carrega dados do Excel"""
        
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e valida dados"""
        
    def enrich_data(df: pd.DataFrame) -> pd.DataFrame:
        """Enriquece dados com informações calculadas"""
```

#### CFEMVisualizations

```python
class CFEMVisualizations:
    """Criação de visualizações avançadas"""
    
    def create_executive_dashboard(df, stats) -> Dict[str, go.Figure]:
        """Dashboard executivo completo"""
        
    def create_geospatial_analysis(df) -> Dict[str, object]:
        """Análises geoespaciais"""
        
    def create_statistical_analysis(df) -> Dict[str, go.Figure]:
        """Análises estatísticas avançadas"""
```

#### CFEMAnalytics

```python
class CFEMAnalytics:
    """Machine Learning e analytics avançados"""
    
    def perform_clustering_analysis(df) -> Dict[str, Any]:
        """Análise de clustering"""
        
    def detect_anomalies(df) -> Dict[str, Any]:
        """Detecção de anomalias"""
        
    def build_predictive_model(df) -> Dict[str, Any]:
        """Modelo preditivo"""
```

### Configurações de Performance

Para otimizar a performance em datasets grandes:

```python
# No início do app.py
import streamlit as st

# Configurações de cache
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data():
    return processor.load_excel_data('data/raw/Emp-CFEM.xlsx')

# Configurações de memória
st.set_page_config(
    page_title="CFEM Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

### Personalização de Filtros

Adicione novos filtros no sidebar:

```python
# Filtro customizado
st.sidebar.subheader("Filtros Avançados")

# Filtro por porte da empresa
if 'PORTE_EMPRESA' in df.columns:
    portes_selecionados = st.sidebar.multiselect(
        'Porte da Empresa:',
        options=df['PORTE_EMPRESA'].unique(),
        default=df['PORTE_EMPRESA'].unique()
    )

# Filtro por região
if 'REGIAO' in df.columns:
    regioes_selecionadas = st.sidebar.multiselect(
        'Regiões:',
        options=df['REGIAO'].unique(),
        default=df['REGIAO'].unique()
    )
```

## 🔍 Casos de Uso

### 1. Análise Governamental
- **Transparência**: Monitoramento da arrecadação CFEM
- **Política Pública**: Identificação de oportunidades de desenvolvimento
- **Compliance**: Verificação de cumprimento de obrigações

### 2. Análise Empresarial
- **Benchmarking**: Comparação com concorrentes
- **Estratégia**: Identificação de oportunidades de mercado
- **Due Diligence**: Avaliação de investimentos

### 3. Pesquisa Acadêmica
- **Estudos Econômicos**: Base para análises setoriais
- **Geografia Econômica**: Distribuição espacial da atividade
- **Sustentabilidade**: Impactos ambientais e sociais

## 📊 Exemplos de Análises

### Concentração de Mercado

```python
# Calcular índices de concentração
concentration = analytics.calculate_market_concentration(df)

print(f"HHI: {concentration['hhi']:.4f}")
print(f"CR4: {concentration['cr4']:.4f}")
print(f"Interpretação: {concentration['market_interpretation']}")
```

### Análise Geoespacial

```python
# Criar mapa de calor
heatmap = viz._create_heatmap(df)
st_folium(heatmap, width=700, height=500)

# Análise de clusters espaciais
cluster_map = viz._create_cluster_map(df)
```

### Machine Learning Pipeline

```python
# Pipeline completo de ML
pipeline_results = {
    'clustering': analytics.perform_clustering_analysis(df),
    'anomalies': analytics.detect_anomalies(df),
    'predictions': analytics.build_predictive_model(df),
    'statistics': analytics.perform_statistical_tests(df)
}

# Gerar relatório de insights
insights = analytics.generate_insights_report(df, **pipeline_results)
```

## 🎨 Customização Visual

### Temas Disponíveis

```python
# Tema Corporativo
CORPORATE_THEME = {
    'primary': '#1f4e79',
    'secondary': '#4a90e2',
    'accent': '#f39c12',
    'background': '#ffffff',
    'text': '#2c3e50'
}

# Tema Dark
DARK_THEME = {
    'primary': '#bb86fc',
    'secondary': '#03dac6',
    'accent': '#cf6679',
    'background': '#121212',
    'text': '#ffffff'
}
```

### Gráficos Customizados

```python
def create_custom_chart(df, theme=CORPORATE_THEME):
    fig = px.bar(df, x='Estado', y='CFEM')
    fig.update_layout(
        plot_bgcolor=theme['background'],
        paper_bgcolor=theme['background'],
        font_color=theme['text'],
        colorway=[theme['primary'], theme['secondary'], theme['accent']]
    )
    return fig
```

## 🔐 Segurança e Privacidade

### Proteção de Dados

- **Anonimização**: Opção para anonimizar dados sensíveis
- **Criptografia**: Dados em trânsito protegidos por HTTPS
- **Acesso**: Controle de acesso via autenticação (opcional)

### Configuração de Segurança

```python
# config/security.py
SECURITY_CONFIG = {
    'ENABLE_AUTH': False,  # Ativar autenticação
    'ANONYMIZE_DATA': False,  # Anonimizar dados
    'RATE_LIMITING': True,  # Limitação de taxa
    'SSL_REQUIRED': True  # Requer HTTPS
}
```

## 📈 Roadmap

### Versão 2.0 (Próximas Funcionalidades)

- [ ] **API RESTful**: Endpoints para integração externa
- [ ] **Dashboard Mobile**: Versão otimizada para mobile
- [ ] **Alertas Inteligentes**: Notificações baseadas em ML
- [ ] **Integração BI**: Conectores para PowerBI/Tableau
- [ ] **Análise de Sentimentos**: Análise de notícias do setor
- [ ] **Previsões Avançadas**: Modelos de deep learning

### Versão 2.1 (Melhorias)

- [ ] **Performance**: Otimizações para datasets > 1M registros
- [ ] **Colaboração**: Compartilhamento de dashboards
- [ ] **Automação**: Pipeline de dados automatizado
- [ ] **Multi-idioma**: Suporte a inglês e espanhol


## 📞 Suporte

- **LinkedIn**: [Nilo Rocha](https://linkedin.com/in/nilo-rocha-)
- **Issues**: [GitHub Issues](https://github.com/niloRoch/cfem-analytics/issues)


## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

```
MIT License

Copyright (c) 2024 Seu Nome

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

[![Website](https://img.shields.io/badge/Website-4c1d95?style=for-the-badge&logo=firefox&logoColor=a855f7)](https://www.nilorocha.tech)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/nilo-rocha-)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/niloRoch)
[![CV](https://img.shields.io/badge/Bold-312e81?style=for-the-badge&logo=readthedocs&logoColor=8b5cf6)](https://bold.pro/my/nilo-rocha)
[![Email](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:nilo.roch4@gmail.com)

---

<div align="center">

<img src="https://github.com/niloRoch/datasets/blob/main/assets/LOGO.png" alt="Tech Vision" width="99" style="border-radius: 10px; box-shadow: 0 4px 8px rgba(139, 92, 246, 0.3);"/>

</div>





