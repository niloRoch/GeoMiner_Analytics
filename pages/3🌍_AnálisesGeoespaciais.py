import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import streamlit_folium as st_folium
from geo_analysis import CFEMGeoAnalysis
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Configuração da página
st.set_page_config(
    page_title="Análises Geoespaciais - CFEM Analytics",
    page_icon="🌍",
    layout="wide"
)

def create_interactive_map(data, map_type="operacoes"):
    """Cria mapas interativos usando Folium"""
    # Centro do Brasil
    center_lat, center_lon = -15.0, -50.0
    
    # Filtrar dados com coordenadas válidas
    data_with_coords = data.dropna(subset=['LONGITUDE', 'LATITUDE']).copy()
    
    if len(data_with_coords) == 0:
        return None
    
    # Criar mapa base
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=4,
        tiles='OpenStreetMap'
    )
    
    if map_type == "operacoes":
        # Mapa com clusters de operações
        marker_cluster = plugins.MarkerCluster(
            name="Operações CFEM",
            overlay=True,
            control=True
        ).add_to(m)
        
        # Definir cores baseadas no valor CFEM
        def get_color(cfem_value, percentiles):
            if cfem_value >= percentiles[0.9]:
                return 'red'
            elif cfem_value >= percentiles[0.7]:
                return 'orange'
            elif cfem_value >= percentiles[0.5]:
                return 'yellow'
            else:
                return 'blue'
        
        percentiles = data_with_coords['CFEM'].quantile([0.5, 0.7, 0.9])
        
        for idx, row in data_with_coords.iterrows():
            color = get_color(row['CFEM'], percentiles)
            
            popup_html = f"""
            <div style="width: 200px;">
                <b>{row['TITULAR']}</b><br>
                <b>Local:</b> {row['MUNICIPIO(S)']} - {row['ESTADO']}<br>
                <b>Substância:</b> {row['PRIMEIRODESUBS']}<br>
                <b>CFEM:</b> R$ {row['CFEM']:,.2f}<br>
                <b>Coordenadas:</b> {row['LATITUDE']:.4f}, {row['LONGITUDE']:.4f}
            </div>
            """
            
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=8,
                popup=folium.Popup(popup_html, max_width=250),
                color=color,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(marker_cluster)
    
    elif map_type == "heatmap":
        # Mapa de calor
        heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['CFEM']] 
                    for idx, row in data_with_coords.iterrows()]
        
        plugins.HeatMap(
            heat_data, 
            name="Densidade CFEM",
            radius=15, 
            blur=10,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
            overlay=True,
            control=True
        ).add_to(m)
    
    elif map_type == "densidade":
        # Mapa de densidade populacional
        coordinates = [[row['LATITUDE'], row['LONGITUDE']] 
                      for idx, row in data_with_coords.iterrows()]
        
        plugins.HeatMap(
            coordinates,
            name="Densidade de Operações",
            radius=20,
            blur=15,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'},
            overlay=True,
            control=True
        ).add_to(m)
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    # Adicionar legenda
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Legenda CFEM</b></p>
    <p><i class="fa fa-circle" style="color:red"></i> Alto (Top 10%)</p>
    <p><i class="fa fa-circle" style="color:orange"></i> Alto (Top 30%)</p>
    <p><i class="fa fa-circle" style="color:yellow"></i> Médio (Top 50%)</p>
    <p><i class="fa fa-circle" style="color:blue"></i> Baixo</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def perform_spatial_clustering(data, eps_km=50, min_samples=3):
    """Executa clustering espacial usando DBSCAN"""
    # Filtrar dados com coordenadas válidas
    data_coords = data.dropna(subset=['LONGITUDE', 'LATITUDE']).copy()
    
    if len(data_coords) < min_samples:
        return None, None
    
    # Preparar dados para clustering
    coords = data_coords[['LATITUDE', 'LONGITUDE']].values
    
    # Converter eps de km para graus (aproximação)
    eps_degrees = eps_km / 111.0  # 1 grau ≈ 111 km
    
    # Aplicar DBSCAN
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)
    
    clustering = DBSCAN(eps=eps_degrees, min_samples=min_samples)
    cluster_labels = clustering.fit_predict(coords_scaled)
    
    # Adicionar labels ao dataframe
    data_coords['cluster'] = cluster_labels
    data_coords['is_noise'] = cluster_labels == -1
    
    # Calcular estatísticas dos clusters
    cluster_stats = {}
    for cluster_id in set(cluster_labels):
        if cluster_id != -1:  # Ignorar ruído
            cluster_data = data_coords[data_coords['cluster'] == cluster_id]
            cluster_stats[cluster_id] = {
                'size': len(cluster_data),
                'total_cfem': cluster_data['CFEM'].sum(),
                'avg_cfem': cluster_data['CFEM'].mean(),
                'center_lat': cluster_data['LATITUDE'].mean(),
                'center_lon': cluster_data['LONGITUDE'].mean(),
                'companies': cluster_data['TITULAR'].nunique(),
                'dominant_substance': cluster_data['PRIMEIRODESUBS'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A'
            }
    
    return data_coords, cluster_stats

def create_cluster_map(data_with_clusters):
    """Cria mapa com clusters espaciais"""
    if data_with_clusters is None or len(data_with_clusters) == 0:
        return None
    
    center_lat = data_with_clusters['LATITUDE'].mean()
    center_lon = data_with_clusters['LONGITUDE'].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=5)
    
    # Cores para os clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred',
              'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white',
              'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    # Adicionar pontos coloridos por cluster
    for idx, row in data_with_clusters.iterrows():
        if row['cluster'] == -1:  # Ruído
            color = 'black'
            fillColor = 'black'
        else:
            color_idx = row['cluster'] % len(colors)
            color = colors[color_idx]
            fillColor = colors[color_idx]
        
        popup_text = f"""
        <b>Cluster:</b> {row['cluster'] if row['cluster'] != -1 else 'Ruído'}<br>
        <b>Empresa:</b> {row['TITULAR']}<br>
        <b>Local:</b> {row['MUNICIPIO(S)']} - {row['ESTADO']}<br>
        <b>CFEM:</b> R$ {row['CFEM']:,.2f}
        """
        
        folium.CircleMarker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=6,
            popup=folium.Popup(popup_text, max_width=200),
            color=color,
            fillColor=fillColor,
            fillOpacity=0.7,
            weight=2
        ).add_to(m)
    
    return m

def create_hotspot_analysis(data):
    """Cria análise de hotspots"""
    data_coords = data.dropna(subset=['LONGITUDE', 'LATITUDE']).copy()
    
    if len(data_coords) < 5:
        return None
    
    # Calcular densidade local para cada ponto
    hotspot_scores = []
    radius_km = 100  # Raio de 100km
    
    for idx, row in data_coords.iterrows():
        lat, lon = row['LATITUDE'], row['LONGITUDE']
        
        # Calcular distâncias usando aproximação simples
        distances = []
        for _, other_row in data_coords.iterrows():
            other_lat, other_lon = other_row['LATITUDE'], other_row['LONGITUDE']
            
            # Fórmula de Haversine simplificada
            dlat = np.radians(other_lat - lat)
            dlon = np.radians(other_lon - lon)
            
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(lat)) * np.cos(np.radians(other_lat)) * 
                 np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371 * c  # Raio da Terra em km
            
            distances.append(distance)
        
        # Encontrar vizinhos dentro do raio
        distances = np.array(distances)
        neighbors = data_coords[distances <= radius_km]
        
        # Calcular score do hotspot
        if len(neighbors) > 1:
            local_sum = neighbors['CFEM'].sum()
            local_density = len(neighbors)
            hotspot_score = local_sum * np.log(local_density + 1)
        else:
            hotspot_score = 0
        
        hotspot_scores.append(hotspot_score)
    
    data_coords['hotspot_score'] = hotspot_scores
    
    # Classificar hotspots
    percentiles = np.percentile(hotspot_scores, [70, 85, 95])
    
    def classify_hotspot(score):
        if score >= percentiles[2]:
            return 'Very Hot'
        elif score >= percentiles[1]:
            return 'Hot'
        elif score >= percentiles[0]:
            return 'Warm'
        else:
            return 'Cold'
    
    data_coords['hotspot_category'] = data_coords['hotspot_score'].apply(classify_hotspot)
    
    return data_coords

def create_accessibility_analysis(data):
    """Cria análise de acessibilidade (distância para capitais)"""
    # Centroides das capitais estaduais (aproximados)
    state_capitals = {
        'AC': (-9.0238, -70.8120),   'AL': (-9.5713, -36.7820),
        'AP': (1.4050, -51.7700),    'AM': (-3.4168, -65.8561),
        'BA': (-12.5797, -41.7007),  'CE': (-5.4984, -39.3206),
        'DF': (-15.7998, -47.8645),  'ES': (-19.1834, -40.3089),
        'GO': (-15.8270, -49.8362),  'MA': (-4.9609, -45.2744),
        'MT': (-12.6819, -56.9211),  'MS': (-20.7722, -54.7852),
        'MG': (-18.5122, -44.5550),  'PA': (-3.9999, -51.9000),
        'PB': (-7.2399, -36.7820),   'PR': (-24.8932, -51.4386),
        'PE': (-8.8137, -36.9541),   'PI': (-8.5000, -42.9000),
        'RJ': (-22.3094, -42.4508),  'RN': (-5.4026, -36.9541),
        'RS': (-30.0346, -51.2177),  'RO': (-10.9472, -61.9857),
        'RR': (1.9981, -61.3300),    'SC': (-27.2423, -50.2189),
        'SP': (-23.6821, -46.8755),  'SE': (-10.5741, -37.3857),
        'TO': (-10.1753, -48.2982)
    }
    
    data_coords = data.dropna(subset=['LONGITUDE', 'LATITUDE']).copy()
    
    if len(data_coords) == 0:
        return None
    
    # Calcular distância para capital do estado
    distances_to_capital = []
    
    for idx, row in data_coords.iterrows():
        state = row['ESTADO']
        lat, lon = row['LATITUDE'], row['LONGITUDE']
        
        if state in state_capitals:
            cap_lat, cap_lon = state_capitals[state]
            
            # Fórmula de Haversine
            dlat = np.radians(cap_lat - lat)
            dlon = np.radians(cap_lon - lon)
            
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(lat)) * np.cos(np.radians(cap_lat)) * 
                 np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            distance = 6371 * c  # km
            
            distances_to_capital.append(distance)
        else:
            distances_to_capital.append(np.nan)
    
    data_coords['distance_to_capital'] = distances_to_capital
    
    # Classificar acessibilidade
    def classify_accessibility(distance):
        if pd.isna(distance):
            return 'Desconhecido'
        elif distance < 100:
            return 'Muito Acessível'
        elif distance < 300:
            return 'Acessível'
        elif distance < 600:
            return 'Moderadamente Acessível'
        else:
            return 'Remoto'
    
    data_coords['accessibility_category'] = data_coords['distance_to_capital'].apply(classify_accessibility)
    
    return data_coords

def create_spatial_statistics_chart(data):
    """Cria gráficos de estatísticas espaciais"""
    data_coords = data.dropna(subset=['LONGITUDE', 'LATITUDE']).copy()
    
    if len(data_coords) == 0:
        return None
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Distribuição Latitudinal', 'Distribuição Longitudinal',
                       'Dispersão Espacial', 'Densidade por Quadrante'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Distribuição Latitudinal
    fig.add_trace(
        go.Histogram(
            x=data_coords['LATITUDE'],
            nbinsx=30,
            name='Latitude',
            marker_color='rgba(102, 126, 234, 0.7)'
        ),
        row=1, col=1
    )
    
    # Distribuição Longitudinal
    fig.add_trace(
        go.Histogram(
            x=data_coords['LONGITUDE'],
            nbinsx=30,
            name='Longitude',
            marker_color='rgba(118, 75, 162, 0.7)'
        ),
        row=1, col=2
    )
    
    # Dispersão Espacial (scatter plot)
    fig.add_trace(
        go.Scatter(
            x=data_coords['LONGITUDE'],
            y=data_coords['LATITUDE'],
            mode='markers',
            marker=dict(
                size=data_coords['CFEM']/data_coords['CFEM'].max() * 20 + 5,
                color=data_coords['CFEM'],
                colorscale='Viridis',
                opacity=0.6,
                showscale=True
            ),
            name='Operações',
            text=[f"CFEM: R$ {val:,.0f}" for val in data_coords['CFEM']],
            hovertemplate='<b>Longitude:</b> %{x}<br><b>Latitude:</b> %{y}<br>%{text}<extra></extra>'
        ),
        row=2, col=1
    )
    
    # Densidade por Quadrante
    center_lat = data_coords['LATITUDE'].mean()
    center_lon = data_coords['LONGITUDE'].mean()
    
    quadrants = {
        'NE': len(data_coords[(data_coords['LATITUDE'] >= center_lat) & 
                             (data_coords['LONGITUDE'] >= center_lon)]),
        'NW': len(data_coords[(data_coords['LATITUDE'] >= center_lat) & 
                             (data_coords['LONGITUDE'] < center_lon)]),
        'SE': len(data_coords[(data_coords['LATITUDE'] < center_lat) & 
                             (data_coords['LONGITUDE'] >= center_lon)]),
        'SW': len(data_coords[(data_coords['LATITUDE'] < center_lat) & 
                             (data_coords['LONGITUDE'] < center_lon)])
    }
    
    fig.add_trace(
        go.Bar(
            x=list(quadrants.keys()),
            y=list(quadrants.values()),
            name='Quadrantes',
            marker_color='rgba(76, 205, 196, 0.7)'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        title_text="Estatísticas de Distribuição Espacial",
        showlegend=False
    )
    
    return fig

def main():
    """Função principal da página"""
    
    st.title("Análises Geoespaciais")
    st.markdown("Análises espaciais avançadas dos dados CFEM")
    
    # Verificar se os dados estão disponíveis
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        st.warning("Nenhum dado carregado. Por favor, carregue os dados na página principal.")
        return
    
    data = st.session_state.filtered_data
    
    # Verificar se há dados com coordenadas
    data_with_coords = data.dropna(subset=['LONGITUDE', 'LATITUDE'])
    
    if len(data_with_coords) == 0:
        st.error("Nenhum dado com coordenadas válidas encontrado.")
        return
    
    st.success(f"Dados carregados: {len(data_with_coords)} registros com coordenadas válidas")
    
    # Sidebar com opções
    st.sidebar.markdown("## Opções de Análise")
    
    analysis_type = st.sidebar.selectbox(
        "Escolha o tipo de análise:",
        ["Mapas Interativos", "Clustering Espacial", "Análise de Hotspots", 
         "Análise de Acessibilidade", "Estatísticas Espaciais"]
    )
    
    if analysis_type == "Mapas Interativos":
        st.markdown("## Mapas Interativos")
        
        # Opções do mapa
        col1, col2 = st.columns([2, 1])
        
        with col2:
            map_type = st.selectbox(
                "Tipo de Mapa:",
                ["operacoes", "heatmap", "densidade"]
            )
            
            map_descriptions = {
                "operacoes": "Mapa com marcadores das operações agrupados por clusters",
                "heatmap": "Mapa de calor baseado nos valores de CFEM",
                "densidade": "Mapa de densidade populacional das operações"
            }
            
            st.info(map_descriptions[map_type])
        
        with col1:
            # Criar e exibir mapa
            with st.spinner("Carregando mapa..."):
                interactive_map = create_interactive_map(data_with_coords, map_type)
                
                if interactive_map:
                    st_data = st_folium.st_folium(
                        interactive_map, 
                        width=700, 
                        height=500,
                        returned_objects=["last_object_clicked"]
                    )
        
        # Estatísticas do mapa
        st.markdown("### Estatísticas do Mapa")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total de Operações", len(data_with_coords))
        
        with col2:
            extent_lat = data_with_coords['LATITUDE'].max() - data_with_coords['LATITUDE'].min()
            st.metric("Extensão Latitudinal", f"{extent_lat:.2f}°")
        
        with col3:
            extent_lon = data_with_coords['LONGITUDE'].max() - data_with_coords['LONGITUDE'].min()
            st.metric("Extensão Longitudinal", f"{extent_lon:.2f}°")
        
        with col4:
            center_lat = data_with_coords['LATITUDE'].mean()
            center_lon = data_with_coords['LONGITUDE'].mean()
            st.metric("Centro Geográfico", f"{center_lat:.2f}, {center_lon:.2f}")
    
    elif analysis_type == "Clustering Espacial":
        st.markdown("## Clustering Espacial")
        
        # Parâmetros do clustering
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Parâmetros")
            eps_km = st.slider("Raio do cluster (km):", min_value=10, max_value=200, value=50, step=10)
            min_samples = st.slider("Mínimo de amostras:", min_value=2, max_value=10, value=3)
            
            if st.button("Executar Clustering"):
                with st.spinner("Executando clustering espacial..."):
                    data_clustered, cluster_stats = perform_spatial_clustering(
                        data_with_coords, eps_km, min_samples
                    )
                    
                    if data_clustered is not None:
                        st.session_state.data_clustered = data_clustered
                        st.session_state.cluster_stats = cluster_stats
                        st.success(f"Clustering concluído! {len(cluster_stats)} clusters encontrados.")
                    else:
                        st.error("Não foi possível executar o clustering com os parâmetros fornecidos.")
        
        with col2:
            if 'data_clustered' in st.session_state:
                # Criar mapa dos clusters
                cluster_map = create_cluster_map(st.session_state.data_clustered)
                
                if cluster_map:
                    st_folium.st_folium(cluster_map, width=600, height=400)
        
        # Estatísticas dos clusters
        if 'cluster_stats' in st.session_state and st.session_state.cluster_stats:
            st.markdown("### Estatísticas dos Clusters")
            
            cluster_data = []
            for cluster_id, stats in st.session_state.cluster_stats.items():
                cluster_data.append({
                    'Cluster': f"Cluster {cluster_id}",
                    'Tamanho': stats['size'],
                    'CFEM Total': f"R$ {stats['total_cfem']:,.0f}",
                    'CFEM Médio': f"R$ {stats['avg_cfem']:,.0f}",
                    'Empresas': stats['companies'],
                    'Substância Principal': stats['dominant_substance'],
                    'Centro (Lat, Lon)': f"{stats['center_lat']:.4f}, {stats['center_lon']:.4f}"
                })
            
            if cluster_data:
                cluster_df = pd.DataFrame(cluster_data)
                st.dataframe(cluster_df, use_container_width=True, hide_index=True)
            
            # Gráfico dos clusters
            fig_clusters = go.Figure()
            
            cluster_sizes = [stats['size'] for stats in st.session_state.cluster_stats.values()]
            cluster_cfem = [stats['total_cfem'] for stats in st.session_state.cluster_stats.values()]
            cluster_ids = [f"Cluster {cid}" for cid in st.session_state.cluster_stats.keys()]
            
            fig_clusters.add_trace(go.Scatter(
                x=cluster_sizes,
                y=cluster_cfem,
                mode='markers+text',
                marker=dict(size=15, opacity=0.7, color='rgba(102, 126, 234, 0.7)'),
                text=cluster_ids,
                textposition="top center",
                name='Clusters'
            ))
            
            fig_clusters.update_layout(
                title='Relação: Tamanho vs Valor CFEM dos Clusters',
                xaxis_title='Número de Operações',
                yaxis_title='CFEM Total (R$)',
                height=400
            )
            
            st.plotly_chart(fig_clusters, use_container_width=True)
    
    elif analysis_type == "Análise de Hotspots":
        st.markdown("## Análise de Hotspots")
        
        with st.spinner("Executando análise de hotspots..."):
            hotspot_data = create_hotspot_analysis(data_with_coords)
        
        if hotspot_data is not None:
            # Estatísticas dos hotspots
            hotspot_counts = hotspot_data['hotspot_category'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Distribuição dos Hotspots")
                
                fig_hotspots = px.pie(
                    values=hotspot_counts.values,
                    names=hotspot_counts.index,
                    title="Classificação das Áreas",
                    color_discrete_map={
                        'Very Hot': '#FF0000',
                        'Hot': '#FF8000',
                        'Warm': '#FFFF00',
                        'Cold': '#0080FF'
                    }
                )
                
                st.plotly_chart(fig_hotspots, use_container_width=True)
            
            with col2:
                st.markdown("### Top 10 Hotspots")
                
                top_hotspots = hotspot_data.nlargest(10, 'hotspot_score')[
                    ['TITULAR', 'MUNICIPIO(S)', 'ESTADO', 'CFEM', 'hotspot_category', 'hotspot_score']
                ]
                
                # Formatação
                top_hotspots_display = top_hotspots.copy()
                top_hotspots_display['CFEM'] = top_hotspots_display['CFEM'].apply(lambda x: f"R$ {x:,.0f}")
                top_hotspots_display['hotspot_score'] = top_hotspots_display['hotspot_score'].apply(lambda x: f"{x:,.0f}")
                
                st.dataframe(top_hotspots_display, use_container_width=True, hide_index=True)
            
            # Mapa dos hotspots
            st.markdown("### Mapa dos Hotspots")
            
            center_lat = hotspot_data['LATITUDE'].mean()
            center_lon = hotspot_data['LONGITUDE'].mean()
            
            hotspot_map = folium.Map(location=[center_lat, center_lon], zoom_start=5)
            
            # Cores para categorias de hotspot
            hotspot_colors = {
                'Very Hot': 'red',
                'Hot': 'orange', 
                'Warm': 'yellow',
                'Cold': 'blue'
            }
            
            for idx, row in hotspot_data.iterrows():
                color = hotspot_colors.get(row['hotspot_category'], 'gray')
                
                popup_text = f"""
                <b>Categoria:</b> {row['hotspot_category']}<br>
                <b>Score:</b> {row['hotspot_score']:,.0f}<br>
                <b>Empresa:</b> {row['TITULAR']}<br>
                <b>CFEM:</b> R$ {row['CFEM']:,.2f}
                """
                
                folium.CircleMarker(
                    location=[row['LATITUDE'], row['LONGITUDE']],
                    radius=8,
                    popup=folium.Popup(popup_text, max_width=200),
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(hotspot_map)
            
            st_folium.st_folium(hotspot_map, width=700, height=400)
        else:
            st.error("Não foi possível executar a análise de hotspots.")
    
    elif analysis_type == "Análise de Acessibilidade":
        st.markdown("## Análise de Acessibilidade")
        
        with st.spinner("Calculando acessibilidade..."):
            accessibility_data = create_accessibility_analysis(data_with_coords)
        
        if accessibility_data is not None:
            # Estatísticas de acessibilidade
            access_stats = accessibility_data.groupby('accessibility_category').agg({
                'CFEM': ['count', 'sum', 'mean'],
                'TITULAR': 'nunique'
            }).round(2)
            
            access_stats.columns = ['Num_Operações', 'CFEM_Total', 'CFEM_Médio', 'Num_Empresas']
            access_stats = access_stats.reset_index()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Distribuição por Acessibilidade")
                
                fig_access = px.bar(
                    access_stats,
                    x='accessibility_category',
                    y='Num_Operações',
                    title="Número de Operações por Categoria de Acessibilidade",
                    color='accessibility_category'
                )
                
                st.plotly_chart(fig_access, use_container_width=True)
            
            with col2:
                st.markdown("### CFEM por Acessibilidade")
                
                fig_cfem_access = px.bar(
                    access_stats,
                    x='accessibility_category',
                    y='CFEM_Total',
                    title="CFEM Total por Categoria de Acessibilidade",
                    color='accessibility_category'
                )
                
                st.plotly_chart(fig_cfem_access, use_container_width=True)
            
            # Tabela resumo
            st.markdown("### Resumo da Análise de Acessibilidade")
            
            # Formatação da tabela
            access_display = access_stats.copy()
            access_display['CFEM_Total'] = access_display['CFEM_Total'].apply(lambda x: f"R$ {x:,.0f}")
            access_display['CFEM_Médio'] = access_display['CFEM_Médio'].apply(lambda x: f"R$ {x:,.0f}")
            
            st.dataframe(access_display, use_container_width=True, hide_index=True)
            
            # Estatísticas gerais
            avg_distance = accessibility_data['distance_to_capital'].mean()
            median_distance = accessibility_data['distance_to_capital'].median()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Distância Média à Capital", f"{avg_distance:.1f} km")
            
            with col2:
                st.metric("Distância Mediana à Capital", f"{median_distance:.1f} km")
            
            with col3:
                remote_pct = len(accessibility_data[accessibility_data['accessibility_category'] == 'Remoto']) / len(accessibility_data) * 100
                st.metric("% Operações Remotas", f"{remote_pct:.1f}%")
        else:
            st.error("Não foi possível executar a análise de acessibilidade.")
    
    elif analysis_type == "Estatísticas Espaciais":
        st.markdown("## Estatísticas Espaciais")
        
        # Gráficos de distribuição espacial
        fig_spatial_stats = create_spatial_statistics_chart(data_with_coords)
        
        if fig_spatial_stats:
            st.plotly_chart(fig_spatial_stats, use_container_width=True)
        
        # Métricas espaciais
        st.markdown("### Métricas de Distribuição Espacial")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Centro geográfico
        center_lat = data_with_coords['LATITUDE'].mean()
        center_lon = data_with_coords['LONGITUDE'].mean()
        
        with col1:
            st.metric("Centro Geográfico", f"({center_lat:.2f}, {center_lon:.2f})")
        
        with col2:
            # Extensão geográfica
            lat_range = data_with_coords['LATITUDE'].max() - data_with_coords['LATITUDE'].min()
            lon_range = data_with_coords['LONGITUDE'].max() - data_with_coords['LONGITUDE'].min()
            st.metric("Extensão (Lat x Lon)", f"{lat_range:.2f}° x {lon_range:.2f}°")
        
        with col3:
            # Dispersão
            lat_std = data_with_coords['LATITUDE'].std()
            lon_std = data_with_coords['LONGITUDE'].std()
            dispersion = np.sqrt(lat_std**2 + lon_std**2)
            st.metric("Índice de Dispersão", f"{dispersion:.4f}")
        
        with col4:
            # Densidade
            area_approx = lat_range * lon_range * 12321  # Conversão aproximada para km²
            density = len(data_with_coords) / area_approx if area_approx > 0 else 0
            st.metric("Densidade (ops/1000km²)", f"{density*1000:.2f}")
        
        # Análise por região/estado
        st.markdown("### Distribuição por Estado/Região")
        
        if 'REGIAO' in data_with_coords.columns:
            regional_spatial = data_with_coords.groupby('REGIAO').agg({
                'LATITUDE': ['mean', 'std'],
                'LONGITUDE': ['mean', 'std'],
                'CFEM': ['count', 'sum']
            }).round(4)
            
            regional_spatial.columns = ['Lat_Média', 'Lat_Desvio', 'Lon_Média', 'Lon_Desvio', 'Num_Ops', 'CFEM_Total']
            regional_spatial = regional_spatial.reset_index()
            
            st.dataframe(regional_spatial, use_container_width=True)
        else:
            state_spatial = data_with_coords.groupby('ESTADO').agg({
                'LATITUDE': ['mean', 'std'],
                'LONGITUDE': ['mean', 'std'],  
                'CFEM': ['count', 'sum']
            }).round(4)
            
            state_spatial.columns = ['Lat_Média', 'Lat_Desvio', 'Lon_Média', 'Lon_Desvio', 'Num_Ops', 'CFEM_Total']
            state_spatial = state_spatial.reset_index()
            
            # Mostrar apenas top 15 estados
            state_spatial = state_spatial.nlargest(15, 'CFEM_Total')
            st.dataframe(state_spatial, use_container_width=True)

if __name__ == "__main__":
    main()