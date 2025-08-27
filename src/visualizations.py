import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import folium
from folium import plugins
import seaborn as sns
import matplotlib.pyplot as plt

class CFEMVisualizations:
    """
    Classe para criação de visualizações avançadas dos dados CFEM
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.theme_colors = {
            'primary': '#1f4e79',
            'secondary': '#667eea',
            'accent': '#764ba2',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545'
        }
    
    def create_executive_dashboard(self, df: pd.DataFrame, stats: Dict) -> Dict[str, go.Figure]:
        """
        Cria dashboard executivo com KPIs principais
        
        Args:
            df: DataFrame processado
            stats: Estatísticas calculadas
            
        Returns:
            Dicionário com figuras do dashboard
        """
        figures = {}
        
        # 1. Gráfico de KPIs principais
        figures['kpis'] = self._create_kpi_chart(stats)
        
        # 2. Evolução temporal (se houver dados de data)
        if 'DATA' in df.columns:
            figures['timeline'] = self._create_timeline_chart(df)
        
        # 3. Top empresas
        figures['top_empresas'] = self._create_top_companies_chart(df)
        
        # 4. Distribuição geográfica
        figures['distribuicao_geografica'] = self._create_geographic_distribution(df)
        
        # 5. Análise por substância
        figures['substancias'] = self._create_substance_analysis(df)
        
        return figures
    
    def create_geospatial_analysis(self, df: pd.DataFrame) -> Dict[str, object]:
        """
        Cria análises geoespaciais avançadas
        
        Args:
            df: DataFrame com dados geográficos
            
        Returns:
            Dicionário com mapas e análises
        """
        maps = {}
        
        # 1. Mapa de calor
        maps['heatmap'] = self._create_heatmap(df)
        
        # 2. Mapa de clusters
        maps['clusters'] = self._create_cluster_map(df)
        
        # 3. Mapa de densidade
        maps['density'] = self._create_density_map(df)
        
        # 4. Análise regional
        maps['regional'] = self._create_regional_analysis(df)
        
        return maps
    
    def create_statistical_analysis(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Cria análises estatísticas avançadas
        
        Args:
            df: DataFrame processado
            
        Returns:
            Dicionário com gráficos estatísticos
        """
        figures = {}
        
        # 1. Distribuição de valores CFEM
        figures['distribuicao'] = self._create_distribution_analysis(df)
        
        # 2. Análise de correlação
        figures['correlacao'] = self._create_correlation_analysis(df)
        
        # 3. Box plots por categoria
        figures['boxplots'] = self._create_categorical_boxplots(df)
        
        # 4. Análise de concentração
        figures['concentracao'] = self._create_concentration_analysis(df)
        
        return figures
    
    def create_interactive_scatter(self, df: pd.DataFrame, 
                                 x_col: str, y_col: str, 
                                 color_col: Optional[str] = None,
                                 size_col: Optional[str] = None) -> go.Figure:
        """
        Cria scatter plot interativo personalizado
        """
        fig = px.scatter(
            df, x=x_col, y=y_col, 
            color=color_col, size=size_col,
            hover_data=['TITULAR', 'MUNICIPIO(S)', 'ESTADO', 'CFEM'],
            title=f'Análise: {x_col} vs {y_col}',
            template='plotly_white'
        )
        
        fig.update_layout(
            font=dict(family="Arial, sans-serif", size=12),
            title_font_size=16,
            showlegend=True
        )
        
        return fig
    
    def _create_kpi_chart(self, stats: Dict) -> go.Figure:
        """Cria gráfico de KPIs principais"""
        fig = go.Figure()
        
        kpis = [
            ('Total CFEM', stats['cfem_total'], 'R$'),
            ('Empresas', stats['total_empresas'], ''),
            ('Estados', stats['total_estados'], ''),
            ('Substâncias', stats['total_substancias'], '')
        ]
        
        for i, (label, value, prefix) in enumerate(kpis):
            fig.add_trace(go.Indicator(
                mode="number+delta",
                value=value,
                title={"text": label},
                number={'prefix': prefix, 'font': {'size': 24}},
                domain={'row': i // 2, 'column': i % 2}
            ))
        
        fig.update_layout(
            grid={'rows': 2, 'columns': 2, 'pattern': "independent"},
            height=400,
            title="KPIs Principais"
        )
        
        return fig
    
    def _create_top_companies_chart(self, df: pd.DataFrame) -> go.Figure:
        """Cria gráfico das principais empresas"""
        top_companies = df.groupby('TITULAR')['CFEM'].sum().nlargest(15).reset_index()
        
        fig = px.bar(
            top_companies,
            x='CFEM',
            y='TITULAR',
            orientation='h',
            title='Top 15 Empresas por Valor CFEM',
            labels={'CFEM': 'Valor CFEM (R$)', 'TITULAR': 'Empresa'},
            color='CFEM',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white'
        )
        
        return fig
    
    def _create_geographic_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Cria gráfico de distribuição geográfica"""
        state_data = df.groupby(['ESTADO', 'REGIAO'])['CFEM'].agg(['sum', 'count']).reset_index()
        state_data.columns = ['ESTADO', 'REGIAO', 'CFEM_TOTAL', 'NUM_OPERACOES']
        
        fig = px.treemap(
            state_data,
            path=['REGIAO', 'ESTADO'],
            values='CFEM_TOTAL',
            title='Distribuição do CFEM por Região e Estado',
            color='NUM_OPERACOES',
            color_continuous_scale='RdYlBu'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def _create_substance_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Cria análise por substância"""
        substance_data = df.groupby('PRIMEIRODESUBS').agg({
            'CFEM': ['sum', 'count', 'mean'],
            'TITULAR': 'nunique'
        }).reset_index()
        
        substance_data.columns = ['SUBSTANCIA', 'CFEM_TOTAL', 'NUM_OPERACOES', 
                                'CFEM_MEDIO', 'NUM_EMPRESAS']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Valor Total por Substância', 'Número de Operações',
                          'Valor Médio', 'Número de Empresas'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Gráfico 1: Valor total
        fig.add_trace(
            go.Bar(x=substance_data['SUBSTANCIA'], y=substance_data['CFEM_TOTAL'],
                  name='Total CFEM', marker_color=self.theme_colors['primary']),
            row=1, col=1
        )
        
        # Gráfico 2: Número de operações
        fig.add_trace(
            go.Bar(x=substance_data['SUBSTANCIA'], y=substance_data['NUM_OPERACOES'],
                  name='Operações', marker_color=self.theme_colors['secondary']),
            row=1, col=2
        )
        
        # Gráfico 3: Valor médio
        fig.add_trace(
            go.Bar(x=substance_data['SUBSTANCIA'], y=substance_data['CFEM_MEDIO'],
                  name='CFEM Médio', marker_color=self.theme_colors['accent']),
            row=2, col=1
        )
        
        # Gráfico 4: Número de empresas
        fig.add_trace(
            go.Bar(x=substance_data['SUBSTANCIA'], y=substance_data['NUM_EMPRESAS'],
                  name='Empresas', marker_color=self.theme_colors['success']),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Análise por Substância")
        
        return fig
    
    def _create_heatmap(self, df: pd.DataFrame) -> folium.Map:
        """Cria mapa de calor das operações"""
        # Centro do Brasil
        center_lat, center_lon = -15.0, -50.0
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Dados para o mapa de calor
        heat_data = [[row['LATITUDE'], row['LONGITUDE'], row['CFEM']] 
                    for idx, row in df.iterrows() 
                    if pd.notna(row['LATITUDE']) and pd.notna(row['LONGITUDE'])]
        
        # Adicionar mapa de calor
        plugins.HeatMap(heat_data, radius=15, blur=10).add_to(m)
        
        return m
    
    def _create_cluster_map(self, df: pd.DataFrame) -> folium.Map:
        """Cria mapa com clusters de operações"""
        center_lat, center_lon = -15.0, -50.0
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Criar cluster
        marker_cluster = plugins.MarkerCluster().add_to(m)
        
        # Adicionar marcadores ao cluster
        for idx, row in df.iterrows():
            if pd.notna(row['LATITUDE']) and pd.notna(row['LONGITUDE']):
                popup_text = f"""
                <b>{row['TITULAR']}</b><br>
                Local: {row['MUNICIPIO(S)']} - {row['ESTADO']}<br>
                Substância: {row['PRIMEIRODESUBS']}<br>
                CFEM: R$ {row['CFEM']:,.2f}
                """
                
                folium.Marker(
                    location=[row['LATITUDE'], row['LONGITUDE']],
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=row['TITULAR']
                ).add_to(marker_cluster)
        
        return m
    
    def _create_density_map(self, df: pd.DataFrame) -> folium.Map:
        """Cria mapa de densidade populacional de operações"""
        center_lat, center_lon = -15.0, -50.0
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Criar dados de densidade
        coordinates = df[['LATITUDE', 'LONGITUDE']].dropna().values.tolist()
        
        if coordinates:
            plugins.HeatMap(coordinates, radius=20, gradient={
                0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'
            }).add_to(m)
        
        return m
    
    def _create_regional_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Cria análise regional"""
        regional_data = df.groupby('REGIAO').agg({
            'CFEM': ['sum', 'mean', 'count'],
            'TITULAR': 'nunique',
            'PRIMEIRODESUBS': 'nunique'
        }).reset_index()
        
        regional_data.columns = ['REGIAO', 'CFEM_TOTAL', 'CFEM_MEDIO', 'NUM_OPERACOES',
                               'NUM_EMPRESAS', 'NUM_SUBSTANCIAS']
        
        fig = go.Figure()
        
        # Gráfico de barras para CFEM total
        fig.add_trace(go.Bar(
            x=regional_data['REGIAO'],
            y=regional_data['CFEM_TOTAL'],
            name='CFEM Total',
            yaxis='y',
            offsetgroup=1
        ))
        
        # Gráfico de linha para número de operações
        fig.add_trace(go.Scatter(
            x=regional_data['REGIAO'],
            y=regional_data['NUM_OPERACOES'],
            mode='lines+markers',
            name='Número de Operações',
            yaxis='y2'
        ))
        
        # Layout com eixo duplo
        fig.update_layout(
            title='Análise Regional - CFEM vs Número de Operações',
            xaxis=dict(title='Região'),
            yaxis=dict(title='CFEM Total (R$)', side='left'),
            yaxis2=dict(title='Número de Operações', side='right', overlaying='y'),
            height=500
        )
        
        return fig
    
    def _create_distribution_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Cria análise de distribuição dos valores CFEM"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Histograma', 'Box Plot', 'Q-Q Plot', 'Distribuição Log'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histograma
        fig.add_trace(
            go.Histogram(x=df['CFEM'], nbinsx=30, name='Frequência',
                        marker_color=self.theme_colors['primary']),
            row=1, col=1
        )
        
        # Box Plot
        fig.add_trace(
            go.Box(y=df['CFEM'], name='CFEM', 
                  marker_color=self.theme_colors['secondary']),
            row=1, col=2
        )
        
        # Q-Q Plot aproximado
        sorted_data = np.sort(df['CFEM'])
        theoretical_quantiles = np.quantile(sorted_data, np.linspace(0, 1, len(sorted_data)))
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sorted_data,
                      mode='markers', name='Q-Q Plot',
                      marker_color=self.theme_colors['accent']),
            row=2, col=1
        )
        
        # Distribuição em escala log
        fig.add_trace(
            go.Histogram(x=np.log10(df['CFEM'] + 1), nbinsx=30, 
                        name='Log(CFEM)', marker_color=self.theme_colors['success']),
            row=2, col=2
        )
        
        fig.update_layout(height=600, title_text="Análise de Distribuição dos Valores CFEM")
        
        return fig
    
    def _create_correlation_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Cria análise de correlação entre variáveis numéricas"""
        # Selecionar apenas colunas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            # Criar dados dummy se não houver colunas suficientes
            correlation_data = pd.DataFrame({
                'CFEM': df['CFEM'],
                'LONGITUDE': df['LONGITUDE'],
                'LATITUDE': df['LATITUDE']
            }).corr()
        else:
            correlation_data = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_data.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title='Matriz de Correlação',
            height=500,
            width=500
        )
        
        return fig
    
    def _create_categorical_boxplots(self, df: pd.DataFrame) -> go.Figure:
        """Cria box plots por categorias"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Por Estado', 'Por Substância', 'Por Região', 'Por Porte da Empresa'),
        )
        
        # Box plot por Estado (top 10)
        top_states = df['ESTADO'].value_counts().head(10).index
        df_top_states = df[df['ESTADO'].isin(top_states)]
        
        for state in top_states:
            state_data = df_top_states[df_top_states['ESTADO'] == state]['CFEM']
            fig.add_trace(
                go.Box(y=state_data, name=state, showlegend=False),
                row=1, col=1
            )
        
        # Box plot por Substância
        substances = df['PRIMEIRODESUBS'].unique()
        for substance in substances[:8]:  # Limitar a 8 substâncias
            substance_data = df[df['PRIMEIRODESUBS'] == substance]['CFEM']
            fig.add_trace(
                go.Box(y=substance_data, name=substance, showlegend=False),
                row=1, col=2
            )
        
        # Box plot por Região
        if 'REGIAO' in df.columns:
            regions = df['REGIAO'].unique()
            for region in regions:
                region_data = df[df['REGIAO'] == region]['CFEM']
                fig.add_trace(
                    go.Box(y=region_data, name=region, showlegend=False),
                    row=2, col=1
                )
        
        # Box plot por Porte da Empresa
        if 'PORTE_EMPRESA' in df.columns:
            company_sizes = df['PORTE_EMPRESA'].unique()
            for size in company_sizes:
                size_data = df[df['PORTE_EMPRESA'] == size]['CFEM']
                fig.add_trace(
                    go.Box(y=size_data, name=size, showlegend=False),
                    row=2, col=2
                )
        
        fig.update_layout(height=600, title_text="Análise por Categorias")
        
        return fig
    
    def _create_concentration_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Cria análise de concentração de mercado"""
        # Análise de concentração por empresa
        company_cfem = df.groupby('TITULAR')['CFEM'].sum().sort_values(ascending=False)
        total_cfem = company_cfem.sum()
        
        # Calcular percentual acumulativo
        company_pct = (company_cfem / total_cfem * 100).reset_index()
        company_pct['cumulative'] = company_pct['CFEM'].cumsum()
        company_pct['rank'] = range(1, len(company_pct) + 1)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Curva de Lorenz - Concentração', 'Top 20 Empresas (% do Total)')
        )
        
        # Curva de Lorenz
        fig.add_trace(
            go.Scatter(
                x=company_pct['rank'],
                y=company_pct['cumulative'],
                mode='lines',
                name='Concentração Real',
                line=dict(color=self.theme_colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Linha de igualdade perfeita
        fig.add_trace(
            go.Scatter(
                x=[1, len(company_pct)],
                y=[0, 100],
                mode='lines',
                name='Distribuição Uniforme',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=1
        )
        
        # Top 20 empresas
        top_20 = company_pct.head(20)
        fig.add_trace(
            go.Bar(
                x=top_20.index,
                y=top_20['CFEM'],
                name='% do Total',
                marker_color=self.theme_colors['secondary']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            title_text="Análise de Concentração de Mercado"
        )
        
        return fig
    
    def create_advanced_analytics_dashboard(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Cria dashboard de analytics avançados"""
        figures = {}
        
        # 1. Análise de eficiência (CFEM vs número de operações)
        figures['eficiencia'] = self._create_efficiency_analysis(df)
        
        # 2. Análise temporal (se houver dados de data)
        if 'DATA' in df.columns:
            figures['temporal'] = self._create_temporal_analysis(df)
        
        # 3. Análise de portfólio (diversificação)
        figures['portfolio'] = self._create_portfolio_analysis(df)
        
        # 4. Análise de market share
        figures['market_share'] = self._create_market_share_analysis(df)
        
        # 5. Análise de densidade geográfica
        figures['densidade_geo'] = self._create_geographic_density_analysis(df)
        
        return figures
    
    def _create_efficiency_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Cria análise de eficiência operacional"""
        # Agrupar por empresa
        company_analysis = df.groupby('TITULAR').agg({
            'CFEM': ['sum', 'mean', 'count'],
            'PRIMEIRODESUBS': 'nunique',
            'ESTADO': 'nunique'
        }).reset_index()
        
        company_analysis.columns = ['EMPRESA', 'CFEM_TOTAL', 'CFEM_MEDIO', 'NUM_OPERACOES',
                                  'NUM_SUBSTANCIAS', 'NUM_ESTADOS']
        
        # Calcular eficiência (CFEM médio por operação)
        company_analysis['EFICIENCIA'] = company_analysis['CFEM_TOTAL'] / company_analysis['NUM_OPERACOES']
        
        fig = px.scatter(
            company_analysis,
            x='NUM_OPERACOES',
            y='CFEM_TOTAL',
            size='EFICIENCIA',
            color='NUM_SUBSTANCIAS',
            hover_data=['EMPRESA', 'NUM_ESTADOS'],
            title='Análise de Eficiência: Total CFEM vs Número de Operações',
            labels={
                'NUM_OPERACOES': 'Número de Operações',
                'CFEM_TOTAL': 'CFEM Total (R$)',
                'NUM_SUBSTANCIAS': 'Diversificação (Substâncias)'
            }
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def _create_portfolio_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Cria análise de portfólio de empresas"""
        # Calcular diversificação por empresa
        portfolio_data = df.groupby('TITULAR').agg({
            'PRIMEIRODESUBS': ['nunique', lambda x: list(x.unique())],
            'CFEM': 'sum',
            'ESTADO': 'nunique'
        }).reset_index()
        
        portfolio_data.columns = ['EMPRESA', 'DIVERSIFICACAO', 'SUBSTANCIAS', 'CFEM_TOTAL', 'NUM_ESTADOS']
        
        # Classificar nível de diversificação
        portfolio_data['NIVEL_DIVERSIFICACAO'] = pd.cut(
            portfolio_data['DIVERSIFICACAO'],
            bins=[0, 1, 3, 5, float('inf')],
            labels=['Especializada', 'Pouco Diversificada', 'Diversificada', 'Muito Diversificada']
        )
        
        fig = px.sunburst(
            portfolio_data,
            path=['NIVEL_DIVERSIFICACAO', 'EMPRESA'],
            values='CFEM_TOTAL',
            title='Análise de Portfólio - Diversificação das Empresas'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def _create_market_share_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Cria análise de market share"""
        # Market share por substância
        substance_totals = df.groupby('PRIMEIRODESUBS')['CFEM'].sum()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Market Share Geral', 'Top Empresas por Substância',
                          'Concentração HHI', 'Evolução Market Share'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Market share geral
        company_totals = df.groupby('TITULAR')['CFEM'].sum().nlargest(10)
        fig.add_trace(
            go.Pie(labels=company_totals.index, values=company_totals.values, name="Market Share"),
            row=1, col=1
        )
        
        # Top empresas por substância principal
        top_substance = substance_totals.idxmax()
        top_companies_in_substance = df[df['PRIMEIRODESUBS'] == top_substance].groupby('TITULAR')['CFEM'].sum().nlargest(10)
        
        fig.add_trace(
            go.Bar(x=top_companies_in_substance.index, y=top_companies_in_substance.values,
                  name=f"Top em {top_substance}"),
            row=1, col=2
        )
        
        # HHI por substância
        hhi_by_substance = []
        for substance in df['PRIMEIRODESUBS'].unique():
            substance_data = df[df['PRIMEIRODESUBS'] == substance]
            company_shares = substance_data.groupby('TITULAR')['CFEM'].sum()
            total = company_shares.sum()
            shares_squared = ((company_shares / total) ** 2).sum()
            hhi_by_substance.append({'Substancia': substance, 'HHI': shares_squared})
        
        hhi_df = pd.DataFrame(hhi_by_substance)
        fig.add_trace(
            go.Bar(x=hhi_df['Substancia'], y=hhi_df['HHI'], name="Índice HHI"),
            row=2, col=1
        )
        
        fig.update_layout(height=800, title_text="Análise de Market Share e Concentração")
        
        return fig
    
    def _create_geographic_density_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Cria análise de densidade geográfica"""
        # Análise por estado
        state_analysis = df.groupby(['ESTADO', 'REGIAO']).agg({
            'CFEM': 'sum',
            'TITULAR': 'nunique',
            'LONGITUDE': 'count'  # Número de operações
        }).reset_index()
        
        state_analysis.columns = ['ESTADO', 'REGIAO', 'CFEM_TOTAL', 'NUM_EMPRESAS', 'NUM_OPERACOES']
        state_analysis['DENSIDADE'] = state_analysis['NUM_OPERACOES']  # Simplificado
        
        fig = px.scatter(
            state_analysis,
            x='NUM_EMPRESAS',
            y='CFEM_TOTAL',
            size='DENSIDADE',
            color='REGIAO',
            text='ESTADO',
            title='Densidade Operacional por Estado',
            labels={
                'NUM_EMPRESAS': 'Número de Empresas',
                'CFEM_TOTAL': 'CFEM Total (R$)',
                'DENSIDADE': 'Densidade de Operações'
            }
        )
        
        fig.update_traces(textposition="middle center")
        fig.update_layout(height=500)
        
        return fig
    
    def export_charts_to_html(self, figures: Dict[str, go.Figure], output_dir: str = "reports"):
        """
        Exporta gráficos para arquivos HTML
        
        Args:
            figures: Dicionário com figuras
            output_dir: Diretório de saída
        """
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for name, fig in figures.items():
            if isinstance(fig, go.Figure):
                fig.write_html(f"{output_dir}/{name}.html")
    
    def create_executive_report_layout(self, df: pd.DataFrame, stats: Dict) -> str:
        """
        Cria layout HTML para relatório executivo
        
        Args:
            df: DataFrame processado
            stats: Estatísticas calculadas
            
        Returns:
            String com HTML do relatório
        """
        html_template = f"""
        <html>
        <head>
            <title>Relatório CFEM Analytics</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; color: #1f4e79; margin-bottom: 30px; }}
                .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 20px 0; }}
                .kpi-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                           color: white; padding: 20px; border-radius: 10px; text-align: center; }}
                .section {{ margin: 30px 0; }}
                .highlight {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #667eea; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>📊 CFEM Analytics - Relatório Executivo</h1>
                <p>Análise da Compensação Financeira pela Exploração de Recursos Minerais</p>
            </div>
            
            <div class="kpi-grid">
                <div class="kpi-card">
                    <h3>Total CFEM</h3>
                    <h2>R$ {stats['cfem_total']:,.0f}</h2>
                </div>
                <div class="kpi-card">
                    <h3>Empresas</h3>
                    <h2>{stats['total_empresas']}</h2>
                </div>
                <div class="kpi-card">
                    <h3>Estados</h3>
                    <h2>{stats['total_estados']}</h2>
                </div>
                <div class="kpi-card">
                    <h3>Substâncias</h3>
                    <h2>{stats['total_substancias']}</h2>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Principais Insights</h2>
                <div class="highlight">
                    <ul>
                        <li>Maior empresa: {max(stats['top_empresas'], key=stats['top_empresas'].get)}</li>
                        <li>Estado com maior arrecadação: {max(stats['top_estados'], key=stats['top_estados'].get)}</li>
                        <li>Principal substância: {max(stats['top_substancias'], key=stats['top_substancias'].get)}</li>
                        <li>Índice de concentração (HHI): {stats['hhi_empresas']:.4f}</li>
                    </ul>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template