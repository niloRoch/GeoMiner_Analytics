import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import logging
import os
from pathlib import Path
from cfem_analytics.data_processor import CFEMDataProcessor, check_data_health

# Configurar página
st.set_page_config(
    page_title="CFEM Analytics",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Importar módulos locais
try:
    from src.data_processor import CFEMDataProcessor
    from src.analytics import CFEMAnalytics
    from src.visualizations import CFEMVisualizations
    from src.geo_analysis import CFEMGeoAnalysis
except ImportError:
    st.error("Erro ao importar módulos. Verifique se todos os arquivos estão no diretório 'src/'")
    st.stop()


# Inicializa o processador
processor = CFEMDataProcessor()

# Upload do arquivo
uploaded_file = st.file_uploader("📂 Envie um arquivo Excel com dados CFEM", type=["xlsx"])

if uploaded_file:
    # 1. Carrega dados brutos
    df_raw = processor.load_excel_data(uploaded_file)
    check_data_health(df_raw, "Dados brutos")   # <-- debug inicial

    # 2. Limpa os dados
    df_clean = processor.clean_data(df_raw)
    check_data_health(df_clean, "Dados limpos") # <-- debug após limpeza

    # 3. Continua fluxo normal (enriquecer, análises, gráficos etc.)
    df_enriched = processor.enrich_data(df_clean)
    check_data_health(df_enriched, "Dados enriquecidos")

# Configurar logging
logging.basicConfig(level=logging.INFO)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .insight-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Funções auxiliares
@st.cache_data
def load_data():
    """
    Carrega dados CFEM.
    Prioridade:
    1. data/processed/cfem_cleaned.csv
    2. data/raw/Emp-CFEM.xlsx
    3. data/raw/Emp-CFEM.csv
    4. Dados de exemplo (get_sample_data)
    """
    try:
        processor = CFEMDataProcessor()
        processed_path = "data/processed/cfem_cleaned.csv"
        raw_excel = "data/raw/Emp-CFEM.xlsx"
        raw_csv = "data/raw/Emp-CFEM.csv"

        # 1. CSV já processado
        if os.path.exists(processed_path):
            df = pd.read_csv(processed_path)

        # 2. Excel bruto
        elif os.path.exists(raw_excel):
            df = processor.load_excel_data(raw_excel)
            df = processor.clean_data(df)
            df = processor.enrich_data(df)
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv(processed_path, index=False)

        # 3. CSV bruto
        elif os.path.exists(raw_csv):
            # tenta com vírgula como separador decimal
            try:
                df = pd.read_csv(raw_csv, sep=";", decimal=",")
            except Exception:
                # fallback comum
                df = pd.read_csv(raw_csv, sep=",", decimal=".")

            df = processor.clean_data(df)
            df = processor.enrich_data(df)
            os.makedirs("data/processed", exist_ok=True)
            df.to_csv(processed_path, index=False)

        # 4. Dados de exemplo
        else:
            from src import get_sample_data
            st.warning("Nenhum arquivo de dados encontrado. Usando dados de exemplo.")
            df = get_sample_data()

        # --- Normaliza colunas essenciais ---
        df.columns = df.columns.str.strip().str.upper()
        alias = {
            'UF': 'ESTADO',
            'ESTADO(S)': 'ESTADO',
            'MUNICIPIO': 'MUNICIPIO(S)',
            'MUNICÍPIO': 'MUNICIPIO(S)',
            'PRIMEIRO DE SUBS': 'PRIMEIRODESUBS',
            'SUBSTÂNCIA': 'PRIMEIRODESUBS',
            'SUBSTANCIA': 'PRIMEIRODESUBS'
        }
        df = df.rename(columns=alias)

        # --- Valida colunas obrigatórias ---
        from src import CFEM_COLUMNS
        required = set(CFEM_COLUMNS['required'])
        missing = required - set(df.columns)
        if missing:
            st.error(f"Colunas obrigatórias ausentes: {sorted(missing)}")
            st.stop()

        return df

    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None


@st.cache_data
def calculate_basic_stats(df):
    """Calcula estatísticas básicas"""
    processor = CFEMDataProcessor()
    return processor.calculate_statistics(df)

@st.cache_data
def perform_analytics(df):
    """Executa análises avançadas"""
    analytics = CFEMAnalytics()
    
    results = {}
    
    # Clustering
    try:
        results['clustering'] = analytics.perform_clustering_analysis(df)
    except Exception as e:
        st.warning(f"Erro no clustering: {str(e)}")
        results['clustering'] = None
    
    # Detecção de anomalias
    try:
        results['anomalies'] = analytics.detect_anomalies(df)
    except Exception as e:
        st.warning(f"Erro na detecção de anomalias: {str(e)}")
        results['anomalies'] = None
    
    # Modelo preditivo
    try:
        results['prediction'] = analytics.build_predictive_model(df)
    except Exception as e:
        st.warning(f"Erro no modelo preditivo: {str(e)}")
        results['prediction'] = None
    
    # Concentração de mercado
    try:
        results['concentration'] = analytics.calculate_market_concentration(df)
    except Exception as e:
        st.warning(f"Erro na análise de concentração: {str(e)}")
        results['concentration'] = None
    
    return results

def main():
    """Função principal da aplicação"""
    
    # Header
    st.markdown('<h1 class="main-header">⛏️ CFEM Analytics</h1>', unsafe_allow_html=True)
    st.markdown("### Análise da Compensação Financeira pela Exploração de Recursos Minerais")
    
    # Sidebar
    st.sidebar.title("🛠️ Configurações")
    
    # Carregar dados
    with st.spinner("Carregando dados..."):
        df = load_data()
    
    if df is None:
        st.write("Colunas disponíveis:", df.columns.tolist())
        st.write("Amostra de dados:", df.head())
    
    # Sidebar - Filtros
    st.sidebar.subheader("🔍 Filtros")
    
    # Filtro por estado
    estados_disponveis = ['Todos'] + sorted(df['ESTADO'].unique().tolist())
    estado_selecionado = st.sidebar.selectbox("Estado:", estados_disponveis)
    
    # Filtro por substância
    substancias_disponveis = ['Todas'] + sorted(df['PRIMEIRODESUBS'].unique().tolist())
    substancia_selecionada = st.sidebar.selectbox("Substância:", substancias_disponveis)
    
    # Filtro por faixa de CFEM
    cfem_min, cfem_max = float(df['CFEM'].min()), float(df['CFEM'].max())
    faixa_cfem = st.sidebar.slider(
        "Faixa de CFEM (R$):",
        min_value=cfem_min,
        max_value=cfem_max,
        value=(cfem_min, cfem_max),
        format="%.2f"
    )
    
    # Aplicar filtros
    df_filtered = df.copy()
    
    if estado_selecionado != 'Todos':
        df_filtered = df_filtered[df_filtered['ESTADO'] == estado_selecionado]
    
    if substancia_selecionada != 'Todas':
        df_filtered = df_filtered[df_filtered['PRIMEIRODESUBS'] == substancia_selecionada]
    
    df_filtered = df_filtered[
        (df_filtered['CFEM'] >= faixa_cfem[0]) & 
        (df_filtered['CFEM'] <= faixa_cfem[1])
    ]
    
    # Verificar se ainda há dados após filtros
    if len(df_filtered) == 0:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
        return
    
    # Calcular estatísticas
    stats = calculate_basic_stats(df_filtered)
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", 
        "🗺️ Análise Geográfica", 
        "🤖 Machine Learning", 
        "📈 Estatísticas", 
        "📋 Relatórios"
    ])
    
    # Tab 1: Dashboard
    with tab1:
        render_dashboard(df_filtered, stats)
    
    # Tab 2: Análise Geográfica
    with tab2:
        render_geographic_analysis(df_filtered)
    
    # Tab 3: Machine Learning
    with tab3:
        render_ml_analysis(df_filtered)
    
    # Tab 4: Estatísticas
    with tab4:
        render_statistical_analysis(df_filtered)
    
    # Tab 5: Relatórios
    with tab5:
        render_reports(df_filtered, stats)

def render_dashboard(df, stats):
    """Renderiza dashboard principal"""
    st.subheader("📊 Dashboard Executivo")
    
    # KPIs principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total CFEM</h3>
            <h2>R$ {stats['cfem_total']:,.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Empresas</h3>
            <h2>{stats['total_empresas']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Estados</h3>
            <h2>{stats['total_estados']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Substâncias</h3>
            <h2>{stats['total_substancias']}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Empresas")
        top_companies = df.groupby('TITULAR')['CFEM'].sum().nlargest(10).reset_index()
        fig_companies = px.bar(
            top_companies, 
            x='CFEM', 
            y='TITULAR',
            orientation='h',
            color='CFEM',
            color_continuous_scale='viridis'
        )
        fig_companies.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_companies, use_container_width=True)
    
    with col2:
        st.subheader("🌍 Distribuição por Estado")
        state_data = df.groupby('ESTADO')['CFEM'].sum().reset_index()
        fig_states = px.pie(
            state_data,
            values='CFEM',
            names='ESTADO',
            title="Distribuição do CFEM por Estado"
        )
        fig_states.update_layout(height=400)
        st.plotly_chart(fig_states, use_container_width=True)
    
    # Análise por substância
    st.subheader("⚡ Análise por Substância")
    substance_data = df.groupby('PRIMEIRODESUBS').agg({
        'CFEM': ['sum', 'count', 'mean'],
        'TITULAR': 'nunique'
    }).reset_index()
    
    substance_data.columns = ['SUBSTANCIA', 'CFEM_TOTAL', 'NUM_OPERACOES', 'CFEM_MEDIO', 'NUM_EMPRESAS']
    
    fig_substance = px.treemap(
        substance_data,
        path=['SUBSTANCIA'],
        values='CFEM_TOTAL',
        color='NUM_EMPRESAS',
        title="Treemap - Valor CFEM por Substância (cor = número de empresas)"
    )
    st.plotly_chart(fig_substance, use_container_width=True)
    
    # Insights automáticos
    st.subheader("💡 Principais Insights")
    
    # Calcular insights
    top_company = max(stats['top_empresas'], key=stats['top_empresas'].get)
    top_state = max(stats['top_estados'], key=stats['top_estados'].get)
    top_substance = max(stats['top_substancias'], key=stats['top_substancias'].get)
    
    concentration_level = "Alto" if stats['hhi_empresas'] > 0.25 else "Médio" if stats['hhi_empresas'] > 0.15 else "Baixo"
    
    st.markdown(f"""
    <div class="insight-box">
        <h4>🎯 Insights Principais:</h4>
        <ul>
            <li><strong>Empresa líder:</strong> {top_company} domina o mercado</li>
            <li><strong>Estado dominante:</strong> {top_state} concentra a maior arrecadação</li>
            <li><strong>Substância principal:</strong> {top_substance} é a mais valiosa</li>
            <li><strong>Concentração de mercado:</strong> {concentration_level} (HHI: {stats['hhi_empresas']:.4f})</li>
            <li><strong>Valor médio:</strong> R$ {stats['cfem_medio']:,.2f} por operação</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_geographic_analysis(df):
    """Renderiza análise geográfica"""
    st.subheader("🗺️ Análise Geoespacial")
    
    # Verificar se há dados de coordenadas
    df_coords = df.dropna(subset=['LONGITUDE', 'LATITUDE'])
    
    if len(df_coords) == 0:
        st.warning("Dados de coordenadas não disponíveis.")
        return
    
    # Análise geográfica
    geo_analyzer = CFEMGeoAnalysis()
    
    # Mapa interativo
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Mapa de Operações")
        
        # Criar mapa base
        center_lat = df_coords['LATITUDE'].mean()
        center_lon = df_coords['LONGITUDE'].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
        
        # Adicionar marcadores
        for idx, row in df_coords.iterrows():
            # Definir cor baseada no valor CFEM
            if row['CFEM'] > df_coords['CFEM'].quantile(0.9):
                color = 'red'
                size = 10
            elif row['CFEM'] > df_coords['CFEM'].quantile(0.7):
                color = 'orange'
                size = 8
            else:
                color = 'blue'
                size = 6
            
            popup_text = f"""
            <b>{row['TITULAR']}</b><br>
            Local: {row['MUNICIPIO(S)']} - {row['ESTADO']}<br>
            Substância: {row['PRIMEIRODESUBS']}<br>
            CFEM: R$ {row['CFEM']:,.2f}
            """
            
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=size,
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=row['TITULAR']
            ).add_to(m)
        
        # Exibir mapa
        map_data = st_folium(m, width=700, height=500)
    
    with col2:
        st.subheader("📊 Estatísticas Espaciais")
        
        # Calcular estatísticas espaciais
        spatial_stats = geo_analyzer.calculate_spatial_statistics(df_coords)
        
        if 'error' not in spatial_stats:
            center = spatial_stats['geographic_center']
            st.metric("Centro Geográfico", f"{center['latitude']:.2f}, {center['longitude']:.2f}")
            
            dispersion = spatial_stats['spatial_dispersion']
            st.metric("Dispersão Média (km)", f"{dispersion['mean_distance_to_center']:.1f}")
            st.metric("Densidade (ops/km²)", f"{spatial_stats['density_metrics']['points_per_km2']:.3f}")
    
    # Análise de hotspots
    st.subheader("🔥 Análise de Hotspots")
    
    try:
        hotspot_results = geo_analyzer.hotspot_analysis(df_coords)
        
        if 'error' not in hotspot_results:
            hotspot_data = hotspot_results['data_with_hotspots']
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de categorias de hotspot
                hotspot_counts = hotspot_data['Hotspot_Category'].value_counts()
                fig_hotspot = px.bar(
                    x=hotspot_counts.index,
                    y=hotspot_counts.values,
                    title="Distribuição de Hotspots",
                    color=hotspot_counts.values,
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_hotspot, use_container_width=True)
            
            with col2:
                # Top hotspots
                st.subheader("🏆 Top Hotspots")
                top_hotspots = hotspot_results['top_hotspots'].head(5)
                for idx, row in top_hotspots.iterrows():
                    st.write(f"**{row['TITULAR']}**")
                    st.write(f"Local: {row['MUNICIPIO(S)']} - {row['ESTADO']}")
                    st.write(f"Score: {row['Hotspot_Score']:.1f}")
                    st.write("---")
        
    except Exception as e:
        st.warning(f"Erro na análise de hotspots: {str(e)}")

def render_ml_analysis(df):
    """Renderiza análises de Machine Learning"""
    st.subheader("🤖 Análises de Machine Learning")
    
    # Executar análises
    with st.spinner("Executando análises avançadas..."):
        ml_results = perform_analytics(df)
    
    # Clustering Analysis
    if ml_results['clustering']:
        st.subheader("🎯 Análise de Clustering")
        
        clustering_results = ml_results['clustering']
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'kmeans' in clustering_results:
                kmeans_data = clustering_results['kmeans']
                st.metric("Número Ótimo de Clusters", kmeans_data['optimal_k'])
                st.metric("Silhouette Score", f"{kmeans_data['silhouette_score']:.3f}")
                
                # Visualizar clusters se há dados de coordenadas
                if 'data_with_clusters' in kmeans_data:
                    cluster_data = kmeans_data['data_with_clusters'].dropna(subset=['LONGITUDE', 'LATITUDE'])
                    if len(cluster_data) > 0:
                        fig_clusters = px.scatter_mapbox(
                            cluster_data,
                            lat='LATITUDE',
                            lon='LONGITUDE',
                            color='Cluster_KMeans',
                            size='CFEM',
                            hover_data=['TITULAR', 'ESTADO'],
                            mapbox_style='open-street-map',
                            title="Clusters Espaciais",
                            zoom=3
                        )
                        st.plotly_chart(fig_clusters, use_container_width=True)
        
        with col2:
            if 'cluster_analysis' in clustering_results['kmeans']:
                st.subheader("📊 Análise dos Clusters")
                cluster_analysis = clustering_results['kmeans']['cluster_analysis']
                st.dataframe(cluster_analysis)
    
    # Detecção de Anomalias
    if ml_results['anomalies']:
        st.subheader("🚨 Detecção de Anomalias")
        
        anomaly_results = ml_results['anomalies']
        anomaly_analysis = anomaly_results['anomaly_analysis']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total de Anomalias", anomaly_analysis['total_anomalies'])
        
        with col2:
            st.metric("Percentual de Anomalias", f"{anomaly_analysis['anomaly_percentage']:.1f}%")
        
        with col3:
            normal_mean = anomaly_analysis['anomaly_stats']['cfem_mean_normal']
            st.metric("CFEM Médio Normal", f"R$ {normal_mean:,.2f}")
        
        # Gráfico de anomalias
        if 'data_with_anomalies' in anomaly_results:
            anomaly_data = anomaly_results['data_with_anomalies']
            fig_anomaly = px.scatter(
                anomaly_data,
                x=range(len(anomaly_data)),
                y='CFEM',
                color='Is_Anomaly',
                title="Detecção de Anomalias nos Valores CFEM",
                labels={'x': 'Índice', 'CFEM': 'Valor CFEM (R$)'}
            )
            st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # Modelo Preditivo
    if ml_results['prediction']:
        st.subheader("📈 Modelo Preditivo")
        
        prediction_results = ml_results['prediction']
        metrics = prediction_results['metrics']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Métricas do Modelo")
            st.metric("R² Score (Teste)", f"{metrics['test_r2']:.3f}")
            st.metric("MAE (Teste)", f"R$ {metrics['test_mae']:,.2f}")
            
            # Interpretação da performance
            r2_score = metrics['test_r2']
            if r2_score > 0.8:
                performance = "Excelente"
                color = "green"
            elif r2_score > 0.6:
                performance = "Boa"
                color = "orange"
            else:
                performance = "Precisa melhorar"
                color = "red"
            
            st.markdown(f"**Performance:** <span style='color:{color}'>{performance}</span>", unsafe_allow_html=True)
        
        with col2:
            st.subheader("🎯 Feature Importance")
            feature_importance = prediction_results['feature_importance'].head(5)
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Importância das Features"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    # Concentração de Mercado
    if ml_results['concentration']:
        st.subheader("📊 Análise de Concentração")
        
        concentration_results = ml_results['concentration']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Índice HHI", f"{concentration_results['hhi']:.4f}")
        
        with col2:
            st.metric("CR4 (Top 4)", f"{concentration_results['cr4']:.1%}")
        
        with col3:
            st.metric("Coef. Gini", f"{concentration_results['gini_coefficient']:.3f}")
        
        with col4:
            st.markdown(f"**Interpretação:** {concentration_results['market_interpretation']}")

def render_statistical_analysis(df):
    """Renderiza análises estatísticas"""
    st.subheader("📈 Análises Estatísticas")
    
    # Análise descritiva
    st.subheader("📊 Estatísticas Descritivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("CFEM - Estatísticas")
        desc_stats = df['CFEM'].describe()
        for stat, value in desc_stats.items():
            st.metric(stat.title(), f"R$ {value:,.2f}")
    
    with col2:
        st.subheader("📈 Distribuição CFEM")
        fig_hist = px.histogram(
            df,
            x='CFEM',
            nbins=30,
            title="Distribuição dos Valores CFEM",
            labels={'CFEM': 'Valor CFEM (R$)', 'count': 'Frequência'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Box plots por categoria
    st.subheader("📦 Análise por Categorias")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plot por estado (top 10)
        top_states = df['ESTADO'].value_counts().head(10).index
        df_top_states = df[df['ESTADO'].isin(top_states)]
        
        fig_box_state = px.box(
            df_top_states,
            x='ESTADO',
            y='CFEM',
            title="Distribuição CFEM por Estado (Top 10)"
        )
        fig_box_state.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box_state, use_container_width=True)
    
    with col2:
        # Box plot por substância (top 8)
        top_substances = df['PRIMEIRODESUBS'].value_counts().head(8).index
        df_top_substances = df[df['PRIMEIRODESUBS'].isin(top_substances)]
        
        fig_box_substance = px.box(
            df_top_substances,
            x='PRIMEIRODESUBS',
            y='CFEM',
            title="Distribuição CFEM por Substância (Top 8)"
        )
        fig_box_substance.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box_substance, use_container_width=True)
    
    # Matriz de correlação
    st.subheader("🔗 Análise de Correlação")
    
    # Selecionar apenas colunas numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            title="Matriz de Correlação",
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Análise temporal (se disponível)
    if 'DATA' in df.columns:
        st.subheader("📅 Análise Temporal")
        
        df_temporal = df.copy()
        df_temporal['DATA'] = pd.to_datetime(df_temporal['DATA'], errors='coerce')
        df_temporal = df_temporal.dropna(subset=['DATA'])
        
        if len(df_temporal) > 0:
            df_temporal['ANO'] = df_temporal['DATA'].dt.year
            df_temporal['MES'] = df_temporal['DATA'].dt.month
            
            # Evolução anual
            annual_data = df_temporal.groupby('ANO')['CFEM'].agg(['sum', 'count', 'mean']).reset_index()
            
            fig_timeline = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_timeline.add_trace(
                go.Bar(x=annual_data['ANO'], y=annual_data['sum'], name='Total CFEM'),
                secondary_y=False
            )
            
            fig_timeline.add_trace(
                go.Scatter(x=annual_data['ANO'], y=annual_data['count'], 
                          mode='lines+markers', name='Número de Operações'),
                secondary_y=True
            )
            
            fig_timeline.update_xaxes(title_text="Ano")
            fig_timeline.update_yaxes(title_text="CFEM Total (R$)", secondary_y=False)
            fig_timeline.update_yaxes(title_text="Número de Operações", secondary_y=True)
            
            st.plotly_chart(fig_timeline, use_container_width=True)

def render_reports(df, stats):
    """Renderiza seção de relatórios"""
    st.subheader("📋 Relatórios e Exportação")
    
    # Relatório executivo
    st.subheader("📄 Relatório Executivo")
    
    # Calcular insights para o relatório
    top_company = max(stats['top_empresas'], key=stats['top_empresas'].get)
    top_state = max(stats['top_estados'], key=stats['top_estados'].get)
    top_substance = max(stats['top_substancias'], key=stats['top_substancias'].get)
    
    # Criar relatório
    report_content = f"""
    ## Relatório CFEM Analytics
    
    ### Resumo Executivo
    
    **Período de Análise:** {len(df)} operações analisadas
    
    **Principais Métricas:**
    - **Valor Total CFEM:** R$ {stats['cfem_total']:,.2f}
    - **Número de Empresas:** {stats['total_empresas']}
    - **Estados Envolvidos:** {stats['total_estados']}
    - **Substâncias Minerais:** {stats['total_substancias']}
    - **Valor Médio por Operação:** R$ {stats['cfem_medio']:,.2f}
    
    ### Principais Players
    
    **Empresa Líder:** {top_company}
    - Valor CFEM: R$ {stats['top_empresas'][top_company]:,.2f}
    - Participação: {(stats['top_empresas'][top_company]/stats['cfem_total']*100):.1f}%
    
    **Estado Dominante:** {top_state}
    - Valor CFEM: R$ {stats['top_estados'][top_state]:,.2f}
    - Participação: {(stats['top_estados'][top_state]/stats['cfem_total']*100):.1f}%
    
    **Substância Principal:** {top_substance}
    - Valor CFEM: R$ {stats['top_substancias'][top_substance]:,.2f}
    - Participação: {(stats['top_substancias'][top_substance]/stats['cfem_total']*100):.1f}%
    
    ### Análise de Concentração
    
    **Índice Herfindahl-Hirschman (HHI):** {stats['hhi_empresas']:.4f}
    
    Interpretação: {'Alto nível de concentração' if stats['hhi_empresas'] > 0.25 else 'Médio nível de concentração' if stats['hhi_empresas'] > 0.15 else 'Baixo nível de concentração'}
    
    ### Distribuição Geográfica
    
    **Top 5 Estados por Arrecadação:**
    """
    
    # Adicionar top 5 estados
    top_5_states = sorted(stats['top_estados'].items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (state, value) in enumerate(top_5_states, 1):
        report_content += f"\n{i}. **{state}:** R$ {value:,.2f}"
    
    report_content += "\n\n### Recomendações\n\n"
    
    # Gerar recomendações baseadas nos dados
    if stats['hhi_empresas'] > 0.25:
        report_content += "- **Alta Concentração:** Considerar políticas para promover maior competição no setor.\n"
    
    if stats['total_estados'] < 10:
        report_content += "- **Concentração Regional:** Explorar oportunidades de desenvolvimento em outras regiões.\n"
    
    report_content += "- **Monitoramento:** Implementar acompanhamento regular das principais operações.\n"
    report_content += "- **Análise Temporal:** Desenvolver análise de tendências com dados históricos.\n"
    
    # Exibir relatório
    st.markdown(report_content)
    
    # Opções de exportação
    st.subheader("💾 Exportar Dados")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Exportar CSV"):
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="cfem_data.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📄 Exportar Relatório"):
            st.download_button(
                label="Download Relatório",
                data=report_content,
                file_name="relatorio_cfem.md",
                mime="text/markdown"
            )
    
    with col3:
        if st.button("📈 Exportar Estatísticas"):
            stats_json = pd.Series(stats).to_json(indent=2)
            st.download_button(
                label="Download Estatísticas",
                data=stats_json,
                file_name="estatisticas_cfem.json",
                mime="application/json"
            )

if __name__ == "__main__":

    main()




