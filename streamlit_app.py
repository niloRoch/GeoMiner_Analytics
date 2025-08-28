import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_processor import CFEMDataProcessor
from geo_analysis import CFEMGeoAnalysis
from visualizations import CFEMVisualizations
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="CFEM Analytics Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': 'mailto:admin@cfem.com',
        'About': "Sistema de análise CFEM desenvolvido para análise de dados minerários"
    }
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #667eea 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sidebar-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        background: #fff3cd;
        color: #856404;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #ffeaa7;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 4px;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown("""
<div class="main-header">
    <h1>🏭 CFEM Analytics Dashboard</h1>
    <p>Sistema Avançado de Análise da Compensação Financeira pela Exploração de Recursos Minerais</p>
    <p><em>Análise de dados, visualizações interativas e insights para o setor mineral brasileiro</em></p>
</div>
""", unsafe_allow_html=True)

# Inicializar classes de processamento
@st.cache_resource
def initialize_processors():
    """Inicializa os processadores de dados"""
    processor = CFEMDataProcessor()
    geo_analyzer = CFEMGeoAnalysis()
    visualizer = CFEMVisualizations()
    return processor, geo_analyzer, visualizer

# Função para carregar dados
@st.cache_data
def load_and_process_data(uploaded_file):
    """Carrega e processa dados do arquivo Excel"""
    try:
        processor, _, _ = initialize_processors()
        
        # Carregar dados
        df_raw = pd.read_excel(uploaded_file)
        
        # Processar dados
        df_clean = processor.clean_data(df_raw)
        df_enriched = processor.enrich_data(df_clean)
        
        # Calcular estatísticas
        stats = processor.calculate_statistics(df_enriched)
        
        # Relatório de qualidade
        quality_report = processor.validate_data_quality(df_enriched)
        
        return df_enriched, stats, quality_report, df_raw
        
    except Exception as e:
        st.error(f"Erro ao processar dados: {str(e)}")
        return None, None, None, None

def main():
    """Função principal do aplicativo"""
    
    # Sidebar
    st.sidebar.markdown("## 📂 Upload de Dados")
    
    uploaded_file = st.sidebar.file_uploader(
        "Escolha o arquivo Excel com dados CFEM:",
        type=['xlsx', 'xls'],
        help="Carregue um arquivo Excel contendo os dados de CFEM para análise"
    )
    
    if uploaded_file is None:
        # Página inicial sem dados
        st.markdown("## 👋 Bem-vindo ao CFEM Analytics!")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### 🎯 O que é o CFEM Analytics?
            
            O CFEM Analytics é uma ferramenta avançada de análise de dados da **Compensação Financeira 
            pela Exploração de Recursos Minerais (CFEM)**. Nossa plataforma oferece:
            
            #### 📊 Funcionalidades Principais:
            - **Dashboard Executivo**: KPIs principais e visão geral dos dados
            - **Análises Estatísticas**: Distribuições, correlações e análises avançadas
            - **Análises Geoespaciais**: Mapas interativos, clustering e hotspots
            - **Machine Learning**: Previsões e análises preditivas
            - **Configurações**: Personalização e exportação de dados
            
            #### 🚀 Como começar:
            1. Carregue seu arquivo Excel na barra lateral
            2. Aguarde o processamento dos dados
            3. Navegue pelas diferentes páginas de análise
            4. Exporte relatórios e visualizações
            """)
            
        with col2:
            st.markdown("""
            <div class="sidebar-info">
                <h4>📋 Requisitos do Arquivo</h4>
                <ul>
                    <li><strong>Formato:</strong> Excel (.xlsx/.xls)</li>
                    <li><strong>Colunas esperadas:</strong>
                        <ul>
                            <li>TITULAR</li>
                            <li>MUNICIPIO(S)</li>
                            <li>ESTADO</li>
                            <li>LONGITUDE</li>
                            <li>LATITUDE</li>
                            <li>CFEM</li>
                            <li>PRIMEIRODESUBS</li>
                        </ul>
                    </li>
                    <li><strong>Tamanho:</strong> Até 200MB</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Dados de exemplo
        st.markdown("### 📊 Exemplo de Estrutura de Dados")
        
        sample_data = pd.DataFrame({
            'TITULAR': ['EMPRESA A LTDA', 'EMPRESA B S.A.', 'EMPRESA C EIRELI'],
            'MUNICIPIO(S)': ['Belo Horizonte', 'Rio de Janeiro', 'São Paulo'],
            'ESTADO': ['MG', 'RJ', 'SP'],
            'LONGITUDE': [-43.9378, -43.1729, -46.6333],
            'LATITUDE': [-19.9208, -22.9068, -23.5505],
            'CFEM': [150000.50, 280000.75, 95000.25],
            'PRIMEIRODESUBS': ['Ferro', 'Petróleo', 'Areia']
        })
        
        st.dataframe(sample_data, use_container_width=True)
        
        return
    
    # Carregar dados
    with st.spinner("🔄 Processando dados... Isso pode levar alguns minutos."):
        data, stats, quality_report, raw_data = load_and_process_data(uploaded_file)
    
    if data is None:
        st.error("❌ Falha no processamento dos dados. Verifique o formato do arquivo.")
        return
    
    # Armazenar dados na sessão
    st.session_state.data = data
    st.session_state.stats = stats
    st.session_state.quality_report = quality_report
    st.session_state.raw_data = raw_data
    
    # Sidebar com informações dos dados
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Informações dos Dados")
    
    st.sidebar.metric("Total de Registros", f"{len(data):,}")
    st.sidebar.metric("Total de Empresas", f"{data['TITULAR'].nunique():,}")
    st.sidebar.metric("Total de Estados", f"{data['ESTADO'].nunique():,}")
    st.sidebar.metric("Valor Total CFEM", f"R$ {data['CFEM'].sum():,.2f}")
    
    # Filtros interativos
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔍 Filtros")
    
    # Filtro por estado
    selected_states = st.sidebar.multiselect(
        "Estados:",
        options=sorted(data['ESTADO'].unique()),
        default=sorted(data['ESTADO'].unique())
    )
    
    # Filtro por substância
    selected_substances = st.sidebar.multiselect(
        "Substâncias:",
        options=sorted(data['PRIMEIRODESUBS'].unique()),
        default=sorted(data['PRIMEIRODESUBS'].unique())[:10]  # Limitar inicialmente
    )
    
    # Filtro por faixa de valor CFEM
    min_cfem, max_cfem = st.sidebar.slider(
        "Faixa de Valor CFEM (R$):",
        min_value=float(data['CFEM'].min()),
        max_value=float(data['CFEM'].max()),
        value=(float(data['CFEM'].min()), float(data['CFEM'].max())),
        format="%.2f"
    )
    
    # Aplicar filtros
    filtered_data = data[
        (data['ESTADO'].isin(selected_states)) &
        (data['PRIMEIRODESUBS'].isin(selected_substances)) &
        (data['CFEM'] >= min_cfem) &
        (data['CFEM'] <= max_cfem)
    ]
    
    st.session_state.filtered_data = filtered_data
    
    # Área principal
    if len(filtered_data) == 0:
        st.warning("⚠️ Nenhum registro encontrado com os filtros aplicados. Ajuste os filtros na barra lateral.")
        return
    
    # Resumo dos dados filtrados
    st.markdown("## 📊 Resumo dos Dados Carregados")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Total de Registros</h3>
            <h2>{:,}</h2>
            <p>registros após filtros</p>
        </div>
        """.format(len(filtered_data)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Valor Total CFEM</h3>
            <h2>R$ {:,.0f}</h2>
            <p>valor total arrecadado</p>
        </div>
        """.format(filtered_data['CFEM'].sum()), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Empresas Ativas</h3>
            <h2>{:,}</h2>
            <p>empresas únicas</p>
        </div>
        """.format(filtered_data['TITULAR'].nunique()), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Estados Envolvidos</h3>
            <h2>{:,}</h2>
            <p>estados com operações</p>
        </div>
        """.format(filtered_data['ESTADO'].nunique()), unsafe_allow_html=True)
    
    # Qualidade dos dados
    st.markdown("## 🔍 Qualidade dos Dados")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Completude dos dados
        completude_data = []
        for col, completude in quality_report['completude'].items():
            completude_data.append({'Coluna': col, 'Completude (%)': completude})
        
        completude_df = pd.DataFrame(completude_data)
        
        fig_completude = px.bar(
            completude_df,
            x='Completude (%)',
            y='Coluna',
            orientation='h',
            title='Completude dos Dados por Coluna',
            color='Completude (%)',
            color_continuous_scale='RdYlGn'
        )
        fig_completude.update_layout(height=400)
        
        st.plotly_chart(fig_completude, use_container_width=True)
    
    with col2:
        st.markdown("### 📋 Relatório de Qualidade")
        
        # Duplicatas
        total_duplicatas = quality_report['duplicatas']['total_duplicatas']
        if total_duplicatas > 0:
            st.markdown(f"""
            <div class="warning-message">
                <strong>⚠️ Duplicatas Encontradas:</strong><br>
                {total_duplicatas} registros duplicados
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-message">
                <strong>✅ Sem Duplicatas:</strong><br>
                Nenhuma duplicata encontrada
            </div>
            """, unsafe_allow_html=True)
        
        # Outliers
        if 'outliers' in quality_report:
            outliers_pct = quality_report['outliers']['percentual_outliers']
            if outliers_pct > 5:
                st.markdown(f"""
                <div class="warning-message">
                    <strong>⚠️ Outliers Detectados:</strong><br>
                    {outliers_pct:.1f}% dos registros são outliers
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-message">
                    <strong>✅ Poucos Outliers:</strong><br>
                    {outliers_pct:.1f}% dos registros são outliers
                </div>
                """, unsafe_allow_html=True)
    
    # Prévia dos dados
    st.markdown("## 📋 Prévia dos Dados")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_raw = st.checkbox("Mostrar dados originais", value=False)
        num_rows = st.selectbox("Linhas para exibir:", [10, 25, 50, 100], index=0)
    
    with col1:
        display_data = raw_data.head(num_rows) if show_raw else filtered_data.head(num_rows)
        st.dataframe(display_data, use_container_width=True, height=400)
    
    # Navegação para páginas
    st.markdown("## 🧭 Navegação")
    st.markdown("""
    Os dados foram carregados com sucesso! Use o menu lateral para navegar pelas diferentes análises:
    
    - **📊 Dashboard Executivo**: Visão geral com KPIs principais
    - **📈 Análises Estatísticas**: Distribuições, correlações e estatísticas descritivas  
    - **🌍 Análises Geoespaciais**: Mapas interativos, clustering espacial e hotspots
    - **🤖 Machine Learning**: Modelos preditivos e análises avançadas
    - **⚙️ Configurações**: Exportar dados e configurar relatórios
    """)
    
    # Instruções
    st.markdown("---")
    st.markdown("""
    ### 💡 Dicas de Uso:
    - Use os **filtros na barra lateral** para focar em dados específicos
    - **Navegue pelas páginas** usando o menu lateral
    - **Exporte relatórios** na página de Configurações
    - **Interaja com os gráficos** - muitos são interativos com zoom, pan e tooltips
    """)

if __name__ == "__main__":
    main()





