import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="CFEM Analytics",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .kpi-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .sidebar-header {
        color: #1f4e79;
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega e processa os dados de CFEM"""
    # Dados de exemplo baseados na amostra fornecida
    data = {
        'Titular': [
            'Agrocity Mineração Ltda',
            'Alcoa World Alumina Brasil Ltda.',
            "Almeida's Mineração e Terraplanagem Ltda",
            'Alto Vale de Mineraçao Ltda Me',
            'Amg Mineração S.a.',
            'Anglo American Minério de Ferro Brasil S.a',
            'Anglo American Níquel Brasil Ltda.',
            'Anglogold Ashanti Córrego do Sítio Mineração S.a.',
            'Aperam Inox América do Sul S.a.'
        ],
        'Municipio': [
            'GOUVEIA', 'JURUTI', 'NOVA SANTA HELENA', 'TROMBUDO CENTRAL',
            'BOM SUCESSO', 'ALVORADA DE MINAS', 'BARRO ALTO', 'BARÃO DE COCAIS',
            'SENADOR MODESTINO GONÇALVES'
        ],
        'Estado': ['MG', 'PA', 'MT', 'SC', 'MG', 'MG', 'GO', 'MG', 'MG'],
        'PrimeiroDeSUBS': [
            'QUARTZO', 'BAUXITA', 'OURO', 'FOLHELHO', 'CASSITERITA',
            'FERRO', 'NÍQUEL', 'FERRO', 'MANGANÊS'
        ],
        'LONGITUDE': [
            -43.7339870, -56.0949980, -55.1836860, -49.7924780, -51.7644650,
            -43.3660840, -48.9157070, -43.4848820, -43.2229750
        ],
        'LATITUDE': [
            -18.4429730, -2.1552530, -10.8510000, -27.3030200, -23.7074370,
            -18.7335930, -14.9737090, -19.9358470, -17.9460250
        ],
        'CFEM': [
            19829.00, 6620519.00, 6078.00, 8164.00, 343082.00,
            68370284.00, 4000860.00, 9066913.00, 28709.00
        ]
    }
    
    return pd.DataFrame(data)

def calculate_kpis(df: pd.DataFrame) -> Dict:
    """Calcula os principais KPIs"""
    return {
        'total_cfem': df['CFEM'].sum(),
        'total_empresas': df['Titular'].nunique(),
        'total_estados': df['Estado'].nunique(),
        'total_municipios': df['Municipio'].nunique(),
        'total_substancias': df['PrimeiroDeSUBS'].nunique(),
        'cfem_medio': df['CFEM'].mean()
    }

def create_kpi_cards(kpis: Dict):
    """Cria cards com os principais KPIs"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Total CFEM",
            f"R$ {kpis['total_cfem']:,.0f}".replace(',', '.'),
            delta=None
        )
    
    with col2:
        st.metric(
            "🏢 Empresas",
            f"{kpis['total_empresas']:,}",
            delta=None
        )
    
    with col3:
        st.metric(
            "🗺️ Estados",
            f"{kpis['total_estados']}",
            delta=None
        )
    
    with col4:
        st.metric(
            "⛏️ Substâncias",
            f"{kpis['total_substancias']}",
            delta=None
        )

def create_map(df: pd.DataFrame):
    """Cria mapa interativo com as operações"""
    # Centro do Brasil aproximadamente
    center_lat, center_lon = -15.0, -50.0
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    
    # Normalize CFEM values for marker size
    cfem_normalized = (df['CFEM'] - df['CFEM'].min()) / (df['CFEM'].max() - df['CFEM'].min())
    
    for idx, row in df.iterrows():
        # Tamanho do marcador baseado no valor CFEM
        marker_size = max(5, cfem_normalized.iloc[idx] * 50)
        
        # Cor baseada na substância
        color_map = {
            'FERRO': 'red',
            'BAUXITA': 'orange', 
            'OURO': 'yellow',
            'NÍQUEL': 'green',
            'CASSITERITA': 'blue',
            'QUARTZO': 'purple',
            'FOLHELHO': 'gray',
            'MANGANÊS': 'black'
        }
        
        color = color_map.get(row['PrimeiroDeSUBS'], 'blue')
        
        popup_text = f"""
        <b>{row['Titular']}</b><br>
        Município: {row['Municipio']} - {row['Estado']}<br>
        Substância: {row['PrimeiroDeSUBS']}<br>
        CFEM: R$ {row['CFEM']:,.2f}
        """
        
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=marker_size,
            popup=folium.Popup(popup_text, max_width=300),
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=2
        ).add_to(m)
    
    return m

def create_charts(df: pd.DataFrame):
    """Cria gráficos analíticos"""
    
    # Gráfico 1: Top 10 Empresas por CFEM
    top_empresas = df.nlargest(10, 'CFEM')
    fig1 = px.bar(
        top_empresas,
        x='CFEM',
        y='Titular',
        orientation='h',
        title='Top 10 Empresas por Valor CFEM',
        labels={'CFEM': 'Valor CFEM (R$)', 'Titular': 'Empresa'}
    )
    fig1.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
    
    # Gráfico 2: Distribuição por Estado
    estado_cfem = df.groupby('Estado')['CFEM'].agg(['sum', 'count']).reset_index()
    fig2 = px.bar(
        estado_cfem,
        x='Estado',
        y='sum',
        title='Arrecadação CFEM por Estado',
        labels={'sum': 'Total CFEM (R$)', 'Estado': 'Estado'}
    )
    
    # Gráfico 3: Distribuição por Substância
    subs_cfem = df.groupby('PrimeiroDeSUBS')['CFEM'].sum().reset_index()
    fig3 = px.pie(
        subs_cfem,
        values='CFEM',
        names='PrimeiroDeSUBS',
        title='Distribuição do CFEM por Substância'
    )
    
    # Gráfico 4: Scatter Plot - Localização vs CFEM
    fig4 = px.scatter(
        df,
        x='LONGITUDE',
        y='LATITUDE',
        size='CFEM',
        color='PrimeiroDeSUBS',
        hover_data=['Titular', 'Municipio', 'Estado'],
        title='Distribuição Geográfica das Operações (tamanho = valor CFEM)'
    )
    
    return fig1, fig2, fig3, fig4

def main():
    # Header principal
    st.markdown('<h1 class="main-header">⛏️ CFEM Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Sistema de Análise da Compensação Financeira pela Exploração de Recursos Minerais
    </div>
    """, unsafe_allow_html=True)
    
    # Carregamento dos dados
    df = load_data()
    kpis = calculate_kpis(df)
    
    # Sidebar com filtros
    st.sidebar.markdown('<div class="sidebar-header">🔍 Filtros</div>', 
                       unsafe_allow_html=True)
    
    # Filtro por Estado
    estados_selecionados = st.sidebar.multiselect(
        'Selecione os Estados:',
        options=sorted(df['Estado'].unique()),
        default=sorted(df['Estado'].unique())
    )
    
    # Filtro por Substância
    substancias_selecionadas = st.sidebar.multiselect(
        'Selecione as Substâncias:',
        options=sorted(df['PrimeiroDeSUBS'].unique()),
        default=sorted(df['PrimeiroDeSUBS'].unique())
    )
    
    # Filtro por valor CFEM
    cfem_range = st.sidebar.slider(
        'Faixa de Valor CFEM (R$):',
        min_value=int(df['CFEM'].min()),
        max_value=int(df['CFEM'].max()),
        value=(int(df['CFEM'].min()), int(df['CFEM'].max())),
        format='R$ %d'
    )
    
    # Aplicar filtros
    df_filtered = df[
        (df['Estado'].isin(estados_selecionados)) &
        (df['PrimeiroDeSUBS'].isin(substancias_selecionadas)) &
        (df['CFEM'] >= cfem_range[0]) &
        (df['CFEM'] <= cfem_range[1])
    ]
    
    if df_filtered.empty:
        st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados.")
        return
    
    # Recalcular KPIs com dados filtrados
    kpis_filtered = calculate_kpis(df_filtered)
    
    # Exibir KPIs
    create_kpi_cards(kpis_filtered)
    
    st.markdown("---")
    
    # Layout principal com duas colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Distribuição Geográfica das Operações")
        mapa = create_map(df_filtered)
        st_folium(mapa, width=700, height=500)
    
    with col2:
        st.subheader("📊 Resumo dos Dados Filtrados")
        st.write(f"**Total de registros:** {len(df_filtered)}")
        st.write(f"**Valor médio CFEM:** R$ {df_filtered['CFEM'].mean():,.2f}")
        st.write(f"**Valor máximo:** R$ {df_filtered['CFEM'].max():,.2f}")
        st.write(f"**Valor mínimo:** R$ {df_filtered['CFEM'].min():,.2f}")
        
        # Top 5 empresas
        st.write("**Top 5 Empresas:**")
        top5 = df_filtered.nlargest(5, 'CFEM')[['Titular', 'CFEM']]
        for idx, row in top5.iterrows():
            st.write(f"• {row['Titular']}: R$ {row['CFEM']:,.0f}")
    
    st.markdown("---")
    
    # Gráficos analíticos
    fig1, fig2, fig3, fig4 = create_charts(df_filtered)
    
    # Layout em grid 2x2
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
    
    st.markdown("---")
    
    # Tabela de dados
    st.subheader("📋 Dados Detalhados")
    
    # Opções de exibição
    col1, col2, col3 = st.columns(3)
    with col1:
        mostrar_todas = st.checkbox("Mostrar todas as colunas", value=True)
    with col2:
        ordenar_por = st.selectbox("Ordenar por:", ['CFEM', 'Titular', 'Estado', 'PrimeiroDeSUBS'])
    with col3:
        ordem_desc = st.checkbox("Ordem decrescente", value=True)
    
    # Preparar dados para exibição
    df_display = df_filtered.copy()
    df_display['CFEM_Formatado'] = df_display['CFEM'].apply(lambda x: f"R$ {x:,.2f}")
    
    if not mostrar_todas:
        df_display = df_display[['Titular', 'Municipio', 'Estado', 'PrimeiroDeSUBS', 'CFEM_Formatado']]
    
    df_display = df_display.sort_values(by=ordenar_por, ascending=not ordem_desc)
    
    # Exibir tabela
    st.dataframe(df_display, use_container_width=True, height=400)
    
    # Botão de download
    csv = df_filtered.to_csv(index=False)
    st.download_button(
        label="📥 Baixar dados filtrados (CSV)",
        data=csv,
        file_name=f"cfem_dados_filtrados.csv",
        mime="text/csv"
    )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>CFEM Analytics - Desenvolvido para análise de dados de mineração brasileira</p>
        <p>Dados baseados na Compensação Financeira pela Exploração de Recursos Minerais</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()