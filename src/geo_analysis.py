import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Dict, List, Optional, Tuple, Any
from scipy.spatial.distance import pdist, squareform
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import folium
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from shapely.geometry import Point, Polygon
from shapely.ops import voronoi_diagram
import warnings
warnings.filterwarnings('ignore')

class CFEMGeoAnalysis:
    """
    Classe para análises geoespaciais avançadas dos dados CFEM
    """
    
    def __init__(self):
        self.brazil_bounds = {
            'lat_min': -34, 'lat_max': 6,
            'lon_min': -74, 'lon_max': -32
        }
        self.state_centroids = self._get_state_centroids()
        self.region_mapping = self._get_region_mapping()
    
    def spatial_clustering(self, df: pd.DataFrame, 
                         eps_km: float = 50.0, 
                         min_samples: int = 3) -> Dict[str, Any]:
        """
        Realiza clustering espacial usando DBSCAN
        
        Args:
            df: DataFrame com coordenadas
            eps_km: Raio em quilômetros para agrupamento
            min_samples: Número mínimo de amostras por cluster
            
        Returns:
            Resultados do clustering espacial
        """
        # Filtrar dados com coordenadas válidas
        df_coords = df.dropna(subset=['LONGITUDE', 'LATITUDE']).copy()
        
        if len(df_coords) < min_samples:
            return {"error": "Dados insuficientes para clustering"}
        
        # Converter coordenadas para array
        coords = df_coords[['LATITUDE', 'LONGITUDE']].values
        
        # Converter eps de km para graus (aproximação)
        eps_degrees = eps_km / 111.0  # 1 grau ≈ 111 km
        
        # Aplicar DBSCAN
        scaler = StandardScaler()
        coords_scaled = scaler.fit_transform(coords)
        
        clustering = DBSCAN(eps=eps_degrees, min_samples=min_samples)
        cluster_labels = clustering.fit_predict(coords_scaled)
        
        # Adicionar labels ao dataframe
        df_result = df_coords.copy()
        df_result['Spatial_Cluster'] = cluster_labels
        df_result['Is_Noise'] = cluster_labels == -1
        
        # Analisar clusters
        cluster_stats = self._analyze_spatial_clusters(df_result)
        
        return {
            'data_with_clusters': df_result,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': list(cluster_labels).count(-1),
            'cluster_stats': cluster_stats,
            'parameters': {'eps_km': eps_km, 'min_samples': min_samples}
        }
    
    def calculate_spatial_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula estatísticas espaciais
        
        Args:
            df: DataFrame com coordenadas
            
        Returns:
            Estatísticas espaciais
        """
        df_coords = df.dropna(subset=['LONGITUDE', 'LATITUDE']).copy()
        
        if len(df_coords) < 3:
            return {"error": "Dados insuficientes para análise espacial"}
        
        coords = df_coords[['LATITUDE', 'LONGITUDE']].values
        
        # Centro geográfico
        center_lat = coords[:, 0].mean()
        center_lon = coords[:, 1].mean()
        
        # Dispersão espacial
        distances_to_center = [
            self._haversine_distance(lat, lon, center_lat, center_lon)
            for lat, lon in coords
        ]
        
        # Densidade espacial
        area_convex_hull = self._calculate_convex_hull_area(coords)
        density = len(coords) / area_convex_hull if area_convex_hull > 0 else 0
        
        # Distribuição por quadrantes
        quadrants = self._classify_by_quadrants(coords, center_lat, center_lon)
        
        # Índice de dispersão
        dispersion_index = np.std(distances_to_center) / np.mean(distances_to_center)
        
        # Autocorrelação espacial (Moran's I simplificado)
        morans_i = self._calculate_simple_morans_i(df_coords)
        
        return {
            'geographic_center': {'latitude': center_lat, 'longitude': center_lon},
            'spatial_dispersion': {
                'mean_distance_to_center': np.mean(distances_to_center),
                'std_distance_to_center': np.std(distances_to_center),
                'max_distance_to_center': np.max(distances_to_center),
                'dispersion_index': dispersion_index
            },
            'density_metrics': {
                'points_per_km2': density,
                'convex_hull_area_km2': area_convex_hull,
                'total_points': len(coords)
            },
            'quadrant_distribution': quadrants,
            'spatial_autocorrelation': morans_i
        }
    
    def hotspot_analysis(self, df: pd.DataFrame, 
                        value_column: str = 'CFEM',
                        radius_km: float = 100.0) -> Dict[str, Any]:
        """
        Identifica hotspots de atividade minerária
        
        Args:
            df: DataFrame com dados
            value_column: Coluna para análise de hotspots
            radius_km: Raio para definição de vizinhança
            
        Returns:
            Análise de hotspots
        """
        df_coords = df.dropna(subset=['LONGITUDE', 'LATITUDE', value_column]).copy()
        
        if len(df_coords) < 5:
            return {"error": "Dados insuficientes para análise de hotspots"}
        
        # Calcular densidade local para cada ponto
        hotspot_scores = []
        
        for idx, row in df_coords.iterrows():
            lat, lon = row['LATITUDE'], row['LONGITUDE']
            
            # Encontrar pontos vizinhos dentro do raio
            distances = [
                self._haversine_distance(lat, lon, row2['LATITUDE'], row2['LONGITUDE'])
                for _, row2 in df_coords.iterrows()
            ]
            
            neighbors = df_coords[np.array(distances) <= radius_km]
            
            # Calcular score do hotspot
            if len(neighbors) > 1:
                local_sum = neighbors[value_column].sum()
                local_density = len(neighbors)
                hotspot_score = local_sum * np.log(local_density + 1)
            else:
                hotspot_score = 0
            
            hotspot_scores.append(hotspot_score)
        
        df_coords['Hotspot_Score'] = hotspot_scores
        
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
        
        df_coords['Hotspot_Category'] = df_coords['Hotspot_Score'].apply(classify_hotspot)
        
        # Estatísticas dos hotspots
        hotspot_stats = df_coords.groupby('Hotspot_Category').agg({
            value_column: ['count', 'sum', 'mean'],
            'TITULAR': 'nunique',
            'PRIMEIRODESUBS': 'nunique'
        }).round(2)
        
        return {
            'data_with_hotspots': df_coords,
            'hotspot_statistics': hotspot_stats,
            'parameters': {'radius_km': radius_km, 'value_column': value_column},
            'top_hotspots': df_coords.nlargest(10, 'Hotspot_Score')[
                ['TITULAR', 'MUNICIPIO(S)', 'ESTADO', 'Hotspot_Score', value_column]
            ]
        }
    
    def accessibility_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa acessibilidade das operações minerárias
        
        Args:
            df: DataFrame com coordenadas
            
        Returns:
            Análise de acessibilidade
        """
        df_coords = df.dropna(subset=['LONGITUDE', 'LATITUDE']).copy()
        
        # Distância para capitais estaduais
        capital_distances = self._calculate_distances_to_capitals(df_coords)
        df_coords['Distance_to_Capital'] = capital_distances
        
        # Classificação de acessibilidade
        def classify_accessibility(distance):
            if distance < 100:
                return 'Muito Acessível'
            elif distance < 300:
                return 'Acessível'
            elif distance < 600:
                return 'Moderadamente Acessível'
            else:
                return 'Remoto'
        
        df_coords['Accessibility_Category'] = df_coords['Distance_to_Capital'].apply(classify_accessibility)
        
        # Análise por categoria
        accessibility_stats = df_coords.groupby('Accessibility_Category').agg({
            'CFEM': ['count', 'sum', 'mean'],
            'TITULAR': 'nunique'
        }).round(2)
        
        # Correlação entre acessibilidade e valor CFEM
        correlation_access_value = stats.pearsonr(
            df_coords['Distance_to_Capital'], 
            df_coords['CFEM']
        )
        
        return {
            'data_with_accessibility': df_coords,
            'accessibility_stats': accessibility_stats,
            'correlation_analysis': {
                'pearson_r': correlation_access_value[0],
                'p_value': correlation_access_value[1],
                'interpretation': self._interpret_correlation(correlation_access_value[0])
            },
            'average_distance_to_capital': df_coords['Distance_to_Capital'].mean()
        }
    
    def territorial_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Análise territorial e administrativa
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Análise territorial
        """
        # Análise por estado
        state_analysis = df.groupby('ESTADO').agg({
            'CFEM': ['count', 'sum', 'mean'],
            'TITULAR': 'nunique',
            'PRIMEIRODESUBS': 'nunique',
            'MUNICIPIO(S)': 'nunique'
        }).reset_index()
        
        state_analysis.columns = ['ESTADO', 'NUM_OPERACOES', 'CFEM_TOTAL', 'CFEM_MEDIO',
                                'NUM_EMPRESAS', 'NUM_SUBSTANCIAS', 'NUM_MUNICIPIOS']
        
        # Análise por região
        df_with_region = df.copy()
        df_with_region['REGIAO'] = df_with_region['ESTADO'].map(self.region_mapping)
        
        region_analysis = df_with_region.groupby('REGIAO').agg({
            'CFEM': ['count', 'sum', 'mean'],
            'TITULAR': 'nunique',
            'ESTADO': 'nunique'
        }).round(2)
        
        # Índice de concentração territorial
        territorial_hhi = self._calculate_territorial_hhi(df)
        
        # Diversificação territorial por empresa
        company_territorial_div = df.groupby('TITULAR')['ESTADO'].nunique().reset_index()
        company_territorial_div.columns = ['TITULAR', 'NUM_ESTADOS_ATUACAO']
        
        avg_territorial_diversification = company_territorial_div['NUM_ESTADOS_ATUACAO'].mean()
        
        return {
            'state_analysis': state_analysis,
            'region_analysis': region_analysis,
            'territorial_concentration': territorial_hhi,
            'company_territorial_diversification': company_territorial_div,
            'avg_territorial_diversification': avg_territorial_diversification,
            'most_diversified_companies': company_territorial_div.nlargest(10, 'NUM_ESTADOS_ATUACAO')
        }
    
    def create_interactive_maps(self, df: pd.DataFrame) -> Dict[str, folium.Map]:
        """
        Cria mapas interativos
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Dicionário com mapas interativos
        """
        maps = {}
        
        # Mapa base
        center_lat, center_lon = -15.0, -50.0
        
        # 1. Mapa de operações com clusters
        maps['operations_map'] = self._create_operations_map(df, center_lat, center_lon)
        
        # 2. Mapa de densidade de valor
        maps['value_density_map'] = self._create_value_density_map(df, center_lat, center_lon)
        
        # 3. Mapa de análise territorial
        maps['territorial_map'] = self._create_territorial_map(df, center_lat, center_lon)
        
        return maps
    
    def _analyze_spatial_clusters(self, df: pd.DataFrame) -> Dict:
        """Analisa características dos clusters espaciais"""
        cluster_stats = {}
        
        for cluster_id in df[df['Spatial_Cluster'] != -1]['Spatial_Cluster'].unique():
            cluster_data = df[df['Spatial_Cluster'] == cluster_id]
            
            if len(cluster_data) > 0:
                cluster_stats[f'cluster_{cluster_id}'] = {
                    'size': len(cluster_data),
                    'total_cfem': cluster_data['CFEM'].sum(),
                    'avg_cfem': cluster_data['CFEM'].mean(),
                    'num_companies': cluster_data['TITULAR'].nunique(),
                    'dominant_substance': cluster_data['PRIMEIRODESUBS'].mode().iloc[0],
                    'geographic_center': {
                        'lat': cluster_data['LATITUDE'].mean(),
                        'lon': cluster_data['LONGITUDE'].mean()
                    },
                    'spatial_extent': {
                        'lat_range': cluster_data['LATITUDE'].max() - cluster_data['LATITUDE'].min(),
                        'lon_range': cluster_data['LONGITUDE'].max() - cluster_data['LONGITUDE'].min()
                    }
                }
        
        return cluster_stats
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distância haversine entre dois pontos"""
        R = 6371  # Raio da Terra em km
        
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _calculate_convex_hull_area(self, coords: np.ndarray) -> float:
        """Calcula área do convex hull em km²"""
        try:
            from scipy.spatial import ConvexHull
            
            # Converter para coordenadas aproximadamente métricas
            coords_metric = coords.copy()
            coords_metric[:, 0] *= 111  # lat para km
            coords_metric[:, 1] *= 111 * np.cos(np.radians(coords[:, 0].mean()))  # lon para km
            
            hull = ConvexHull(coords_metric)
            return hull.volume  # Em 2D, volume = área
        except:
            return 0
    
    def _classify_by_quadrants(self, coords: np.ndarray, center_lat: float, center_lon: float) -> Dict:
        """Classifica pontos por quadrantes"""
        quadrants = {'NE': 0, 'NW': 0, 'SE': 0, 'SW': 0}
        
        for lat, lon in coords:
            if lat >= center_lat and lon >= center_lon:
                quadrants['NE'] += 1
            elif lat >= center_lat and lon < center_lon:
                quadrants['NW'] += 1
            elif lat < center_lat and lon >= center_lon:
                quadrants['SE'] += 1
            else:
                quadrants['SW'] += 1
        
        return quadrants
    
    def _calculate_simple_morans_i(self, df: pd.DataFrame) -> Dict:
        """Calcula Moran's I simplificado para autocorrelação espacial"""
        coords = df[['LATITUDE', 'LONGITUDE']].values
        values = df['CFEM'].values
        
        n = len(coords)
        if n < 4:
            return {"morans_i": 0, "interpretation": "Dados insuficientes"}
        
        # Matriz de pesos espaciais simples (baseada em distância)
        distances = squareform(pdist(coords))
        weights = np.where(distances > 0, 1/distances, 0)
        
        # Normalizar pesos
        row_sums = weights.sum(axis=1)
        weights = np.divide(weights, row_sums[:, np.newaxis], 
                           out=np.zeros_like(weights), where=row_sums[:, np.newaxis]!=0)
        
        # Calcular Moran's I
        mean_value = values.mean()
        deviations = values - mean_value
        
        numerator = np.sum(weights * np.outer(deviations, deviations))
        denominator = np.sum(deviations**2)
        
        morans_i = (n / np.sum(weights)) * (numerator / denominator) if denominator != 0 else 0
        
        # Interpretação
        if morans_i > 0.1:
            interpretation = "Autocorrelação espacial positiva (clustering)"
        elif morans_i < -0.1:
            interpretation = "Autocorrelação espacial negativa (dispersão)"
        else:
            interpretation = "Distribuição espacial aleatória"
        
        return {"morans_i": morans_i, "interpretation": interpretation}
    
    def _calculate_distances_to_capitals(self, df: pd.DataFrame) -> List[float]:
        """Calcula distâncias para capitais estaduais"""
        distances = []
        
        for _, row in df.iterrows():
            state = row['ESTADO']
            lat, lon = row['LATITUDE'], row['LONGITUDE']
            
            if state in self.state_centroids:
                cap_lat, cap_lon = self.state_centroids[state]
                distance = self._haversine_distance(lat, lon, cap_lat, cap_lon)
            else:
                distance = np.nan
            
            distances.append(distance)
        
        return distances
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpreta coeficiente de correlação"""
        abs_corr = abs(correlation)
        direction = "positiva" if correlation > 0 else "negativa"
        
        if abs_corr < 0.1:
            strength = "muito fraca"
        elif abs_corr < 0.3:
            strength = "fraca"
        elif abs_corr < 0.5:
            strength = "moderada"
        elif abs_corr < 0.7:
            strength = "forte"
        else:
            strength = "muito forte"
        
        return f"Correlação {direction} {strength}"
    
    def _calculate_territorial_hhi(self, df: pd.DataFrame) -> Dict:
        """Calcula índice de concentração territorial"""
        # HHI por estado
        state_shares = df.groupby('ESTADO')['CFEM'].sum()
        total_cfem = state_shares.sum()
        state_proportions = (state_shares / total_cfem) ** 2
        hhi_states = state_proportions.sum()
        
        # HHI por região
        df_region = df.copy()
        df_region['REGIAO'] = df_region['ESTADO'].map(self.region_mapping)
        region_shares = df_region.groupby('REGIAO')['CFEM'].sum()
        total_cfem_region = region_shares.sum()
        region_proportions = (region_shares / total_cfem_region) ** 2
        hhi_regions = region_proportions.sum()
        
        return {
            'hhi_by_state': hhi_states,
            'hhi_by_region': hhi_regions,
            'interpretation_state': self._interpret_hhi(hhi_states),
            'interpretation_region': self._interpret_hhi(hhi_regions)
        }
    
    def _interpret_hhi(self, hhi: float) -> str:
        """Interpreta índice HHI"""
        if hhi < 0.15:
            return "Baixa concentração"
        elif hhi < 0.25:
            return "Concentração moderada"
        else:
            return "Alta concentração"
    
    def _create_operations_map(self, df: pd.DataFrame, center_lat: float, center_lon: float) -> folium.Map:
        """Cria mapa de operações"""
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Marker cluster para operações
        marker_cluster = plugins.MarkerCluster().add_to(m)
        
        df_coords = df.dropna(subset=['LONGITUDE', 'LATITUDE'])
        
        for _, row in df_coords.iterrows():
            # Definir cor baseada no valor CFEM
            if row['CFEM'] > df['CFEM'].quantile(0.9):
                color = 'red'
                size = 10
            elif row['CFEM'] > df['CFEM'].quantile(0.7):
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
            ).add_to(marker_cluster)
        
        # Adicionar legenda
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 150px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Legenda CFEM</b></p>
        <p><i class="fa fa-circle" style="color:red"></i> Alto (Top 10%)</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Médio (Top 30%)</p>
        <p><i class="fa fa-circle" style="color:blue"></i> Baixo</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def _create_value_density_map(self, df: pd.DataFrame, center_lat: float, center_lon: float) -> folium.Map:
        """Cria mapa de densidade de valor"""
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        df_coords = df.dropna(subset=['LONGITUDE', 'LATITUDE'])
        
        # Dados para mapa de calor com peso por valor CFEM
        heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['CFEM']] 
                    for _, row in df_coords.iterrows()]
        
        plugins.HeatMap(
            heat_data, 
            radius=20, 
            blur=15,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}
        ).add_to(m)
        
        return m
    
    def _create_territorial_map(self, df: pd.DataFrame, center_lat: float, center_lon: float) -> folium.Map:
        """Cria mapa de análise territorial"""
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Análise por estado
        state_stats = df.groupby('ESTADO').agg({
            'CFEM': 'sum',
            'TITULAR': 'nunique'
        }).reset_index()
        
        # Adicionar centroides dos estados com informações
        for _, row in state_stats.iterrows():
            state = row['ESTADO']
            if state in self.state_centroids:
                lat, lon = self.state_centroids[state]
                
                # Tamanho do círculo baseado no CFEM total
                max_cfem = state_stats['CFEM'].max()
                radius = 10 + (row['CFEM'] / max_cfem) * 40
                
                popup_text = f"""
                <b>Estado: {state}</b><br>
                CFEM Total: R$ {row['CFEM']:,.2f}<br>
                Número de Empresas: {row['TITULAR']}
                """
                
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=radius,
                    color='darkblue',
                    fillColor='lightblue',
                    fillOpacity=0.6,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=f"{state}: R$ {row['CFEM']:,.0f}"
                ).add_to(m)
        
        return m
    
    def _get_state_centroids(self) -> Dict[str, Tuple[float, float]]:
        """Retorna centroides aproximados dos estados brasileiros"""
        return {
            'AC': (-9.0238, -70.8120),   # Acre
            'AL': (-9.5713, -36.7820),   # Alagoas
            'AP': (1.4050, -51.7700),    # Amapá
            'AM': (-3.4168, -65.8561),   # Amazonas
            'BA': (-12.5797, -41.7007),  # Bahia
            'CE': (-5.4984, -39.3206),   # Ceará
            'DF': (-15.7998, -47.8645),  # Distrito Federal
            'ES': (-19.1834, -40.3089),  # Espírito Santo
            'GO': (-15.8270, -49.8362),  # Goiás
            'MA': (-4.9609, -45.2744),   # Maranhão
            'MT': (-12.6819, -56.9211),  # Mato Grosso
            'MS': (-20.7722, -54.7852),  # Mato Grosso do Sul
            'MG': (-18.5122, -44.5550),  # Minas Gerais
            'PA': (-3.9999, -51.9000),   # Pará
            'PB': (-7.2399, -36.7820),   # Paraíba
            'PR': (-24.8932, -51.4386),  # Paraná
            'PE': (-8.8137, -36.9541),   # Pernambuco
            'PI': (-8.5000, -42.9000),   # Piauí
            'RJ': (-22.3094, -42.4508),  # Rio de Janeiro
            'RN': (-5.4026, -36.9541),   # Rio Grande do Norte
            'RS': (-30.0346, -51.2177),  # Rio Grande do Sul
            'RO': (-10.9472, -61.9857),  # Rondônia
            'RR': (1.9981, -61.3300),    # Roraima
            'SC': (-27.2423, -50.2189),  # Santa Catarina
            'SP': (-23.6821, -46.8755),  # São Paulo
            'SE': (-10.5741, -37.3857),  # Sergipe
            'TO': (-10.1753, -48.2982)   # Tocantins
        }
    
    def _get_region_mapping(self) -> Dict[str, str]:
        """Retorna mapeamento de estados para regiões"""
        return {
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
    # Adicionando os métodos que faltam no geo_analysis.py

    def _create_timeline_chart(self, df: pd.DataFrame) -> go.Figure:
        """Cria gráfico de evolução temporal"""
        if 'DATA' not in df.columns:
            return None
            
        df_time = df.copy()
        df_time['DATA'] = pd.to_datetime(df_time['DATA'])
        df_time['ANO_MES'] = df_time['DATA'].dt.to_period('M')
        
        # Agrupar por período
        timeline_data = df_time.groupby('ANO_MES').agg({
            'CFEM': ['sum', 'count', 'mean']
        }).reset_index()
        
        timeline_data.columns = ['PERIODO', 'CFEM_TOTAL', 'NUM_OPERACOES', 'CFEM_MEDIO']
        timeline_data['PERIODO_STR'] = timeline_data['PERIODO'].astype(str)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Linha de CFEM total
        fig.add_trace(
            go.Scatter(x=timeline_data['PERIODO_STR'], y=timeline_data['CFEM_TOTAL'],
                      mode='lines+markers', name='CFEM Total',
                      line=dict(color='#1f4e79', width=3)),
            secondary_y=False
        )
        
        # Barras de número de operações
        fig.add_trace(
            go.Bar(x=timeline_data['PERIODO_STR'], y=timeline_data['NUM_OPERACOES'],
                  name='Número de Operações', opacity=0.7,
                  marker_color='#667eea'),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Período")
        fig.update_yaxes(title_text="CFEM Total (R$)", secondary_y=False)
        fig.update_yaxes(title_text="Número de Operações", secondary_y=True)
        
        fig.update_layout(
            title='Evolução Temporal do CFEM',
            height=500,
            template='plotly_white'
        )
        
        return fig

    def calculate_spatial_autocorrelation(self, df: pd.DataFrame, method: str = 'moran') -> Dict[str, Any]:
        """
        Calcula autocorrelação espacial usando diferentes métodos
        
        Args:
            df: DataFrame com coordenadas e valores
            method: Método de cálculo ('moran', 'geary')
            
        Returns:
            Resultados da análise de autocorrelação
        """
        df_coords = df.dropna(subset=['LONGITUDE', 'LATITUDE', 'CFEM']).copy()
        
        if len(df_coords) < 5:
            return {"error": "Dados insuficientes para análise de autocorrelação"}
        
        coords = df_coords[['LATITUDE', 'LONGITUDE']].values
        values = df_coords['CFEM'].values
        
        # Calcular matriz de pesos espaciais
        distances = squareform(pdist(coords))
        
        # Matriz de pesos baseada na distância inversa
        weights = np.where(distances > 0, 1/distances, 0)
        np.fill_diagonal(weights, 0)  # Diagonal zerada
        
        # Normalizar pesos por linha
        row_sums = weights.sum(axis=1)
        weights = np.divide(weights, row_sums[:, np.newaxis], 
                           out=np.zeros_like(weights), where=row_sums[:, np.newaxis]!=0)
        
        if method.lower() == 'moran':
            result = self._calculate_morans_i_detailed(values, weights)
        elif method.lower() == 'geary':
            result = self._calculate_gearys_c(values, weights)
        else:
            raise ValueError("Método deve ser 'moran' ou 'geary'")
        
        return result

    def _calculate_morans_i_detailed(self, values: np.ndarray, weights: np.ndarray) -> Dict:
        """Calcula Moran's I com detalhes estatísticos"""
        n = len(values)
        mean_value = values.mean()
        deviations = values - mean_value
        
        # Moran's I
        numerator = np.sum(weights * np.outer(deviations, deviations))
        denominator = np.sum(deviations**2)
        S0 = np.sum(weights)
        
        morans_i = (n / S0) * (numerator / denominator) if denominator != 0 and S0 != 0 else 0
        
        # Expectativa e variância (aproximadas)
        expected_i = -1 / (n - 1)
        
        # Calcular variância (simplificada)
        S1 = 0.5 * np.sum((weights + weights.T)**2)
        S2 = np.sum(np.sum(weights, axis=1)**2)
        
        b2 = n * np.sum(deviations**4) / (np.sum(deviations**2)**2)
        
        var_i = ((n*((n**2 - 3*n + 3)*S1 - n*S2 + 3*S0**2) - 
                 b2*((n**2 - n)*S1 - 2*n*S2 + 6*S0**2)) / 
                ((n-1)*(n-2)*(n-3)*S0**2) - expected_i**2)
        
        # Z-score
        z_score = (morans_i - expected_i) / np.sqrt(var_i) if var_i > 0 else 0
        
        # P-valor (aproximado)
        from scipy.stats import norm
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
        
        return {
            'morans_i': morans_i,
            'expected_i': expected_i,
            'variance': var_i,
            'z_score': z_score,
            'p_value': p_value,
            'interpretation': self._interpret_morans_i(morans_i, p_value)
        }

    def _calculate_gearys_c(self, values: np.ndarray, weights: np.ndarray) -> Dict:
        """Calcula Geary's C"""
        n = len(values)
        
        # Geary's C
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                if weights[i, j] > 0:
                    numerator += weights[i, j] * (values[i] - values[j])**2
                    denominator += weights[i, j]
        
        mean_value = values.mean()
        variance = np.sum((values - mean_value)**2)
        
        gearys_c = ((n - 1) * numerator) / (2 * denominator * variance) if denominator != 0 and variance != 0 else 1
        
        return {
            'gearys_c': gearys_c,
            'expected_c': 1.0,
            'interpretation': self._interpret_gearys_c(gearys_c)
        }

    def _interpret_morans_i(self, morans_i: float, p_value: float) -> str:
        """Interpreta resultado do Moran's I"""
        significance = "significativo" if p_value < 0.05 else "não significativo"
        
        if morans_i > 0.1:
            pattern = "clustering espacial positivo"
        elif morans_i < -0.1:
            pattern = "dispersão espacial"
        else:
            pattern = "distribuição espacial aleatória"
        
        return f"{pattern} ({significance}, p={p_value:.3f})"

    def _interpret_gearys_c(self, gearys_c: float) -> str:
        """Interpreta resultado do Geary's C"""
        if gearys_c < 1:
            return "Autocorrelação espacial positiva"
        elif gearys_c > 1:
            return "Autocorrelação espacial negativa"
        else:
            return "Sem autocorrelação espacial"

    def create_spatial_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Cria relatório completo de análise espacial
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Relatório espacial completo
        """
        report = {}
        
        # Estatísticas espaciais básicas
        report['basic_stats'] = self.calculate_spatial_statistics(df)
        
        # Clustering espacial
        report['clustering'] = self.spatial_clustering(df)
        
        # Análise de hotspots
        report['hotspots'] = self.hotspot_analysis(df)
        
        # Análise de acessibilidade
        report['accessibility'] = self.accessibility_analysis(df)
        
        # Análise territorial
        report['territorial'] = self.territorial_analysis(df)
        
        # Autocorrelação espacial
        report['autocorrelation'] = self.calculate_spatial_autocorrelation(df)
        
        # Resumo executivo
        report['executive_summary'] = self._create_spatial_executive_summary(report)
        
        return report

    def _create_spatial_executive_summary(self, spatial_results: Dict) -> Dict[str, str]:
        """Cria resumo executivo da análise espacial"""
        summary = {}
        
        # Distribuição espacial
        if 'basic_stats' in spatial_results and 'error' not in spatial_results['basic_stats']:
            dispersion = spatial_results['basic_stats']['spatial_dispersion']['dispersion_index']
            if dispersion > 1.5:
                summary['distribuicao'] = "Operações altamente dispersas geograficamente"
            elif dispersion > 1.0:
                summary['distribuicao'] = "Operações moderadamente dispersas"
            else:
                summary['distribuicao'] = "Operações geograficamente concentradas"
        
        # Clustering
        if 'clustering' in spatial_results and 'error' not in spatial_results['clustering']:
            n_clusters = spatial_results['clustering']['n_clusters']
            if n_clusters > 5:
                summary['clustering'] = f"Identificados {n_clusters} clusters espaciais distintos"
            elif n_clusters > 0:
                summary['clustering'] = f"Poucas concentrações espaciais ({n_clusters} clusters)"
            else:
                summary['clustering'] = "Sem padrões de agrupamento espacial"
        
        # Hotspots
        if 'hotspots' in spatial_results and 'error' not in spatial_results['hotspots']:
            hotspot_data = spatial_results['hotspots']['data_with_hotspots']
            hot_areas = len(hotspot_data[hotspot_data['Hotspot_Category'].isin(['Hot', 'Very Hot'])])
            total_areas = len(hotspot_data)
            hot_percentage = (hot_areas / total_areas * 100) if total_areas > 0 else 0
            summary['hotspots'] = f"{hot_percentage:.1f}% das áreas são hotspots de alta atividade"
        
        # Acessibilidade
        if 'accessibility' in spatial_results:
            avg_distance = spatial_results['accessibility']['average_distance_to_capital']
            if avg_distance > 500:
                summary['acessibilidade'] = "Operações predominantemente em áreas remotas"
            elif avg_distance > 200:
                summary['acessibilidade'] = "Operações em áreas de média acessibilidade"
            else:
                summary['acessibilidade'] = "Operações em áreas de boa acessibilidade"
        
        return summary

    def export_spatial_analysis(self, spatial_results: Dict, output_path: str = "spatial_analysis_results"):
        """
        Exporta resultados da análise espacial
        
        Args:
            spatial_results: Resultados da análise espacial
            output_path: Caminho base para exportação
        """
        import json
        import os
        
        # Criar diretório se não existir
        os.makedirs(output_path, exist_ok=True)
        
        # Exportar dados processados para CSV
        if 'clustering' in spatial_results and 'data_with_clusters' in spatial_results['clustering']:
            clustering_data = spatial_results['clustering']['data_with_clusters']
            clustering_data.to_csv(f"{output_path}/clustering_results.csv", index=False)
        
        if 'hotspots' in spatial_results and 'data_with_hotspots' in spatial_results['hotspots']:
            hotspot_data = spatial_results['hotspots']['data_with_hotspots']
            hotspot_data.to_csv(f"{output_path}/hotspot_analysis.csv", index=False)
        
        if 'accessibility' in spatial_results and 'data_with_accessibility' in spatial_results['accessibility']:
            accessibility_data = spatial_results['accessibility']['data_with_accessibility']
            accessibility_data.to_csv(f"{output_path}/accessibility_analysis.csv", index=False)
        
        # Exportar resumo executivo como JSON
        if 'executive_summary' in spatial_results:
            with open(f"{output_path}/executive_summary.json", 'w', encoding='utf-8') as f:
                json.dump(spatial_results['executive_summary'], f, ensure_ascii=False, indent=2)
        
        # Exportar estatísticas espaciais
        clean_stats = {}
        for key, value in spatial_results.items():
            if key not in ['clustering', 'hotspots', 'accessibility'] and 'data_with' not in str(key):
                clean_stats[key] = value
        
        with open(f"{output_path}/spatial_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(clean_stats, f, ensure_ascii=False, indent=2, default=str)