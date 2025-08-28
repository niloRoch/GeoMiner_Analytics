import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from visualizations import CFEMVisualizations

# Configuração da página
st.set_page_config(
    page_title="Dashboard Executivo - CFEM Analytics",
    page_icon="📊",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 1.1rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .insight-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-card {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_executive_summary_chart(data):
    """Cria gráfico de resumo executivo"""
    # Top 10 empresas
    top_companies = data.groupby('TITULAR')['CFEM'].sum().nlargest(10).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=top_companies['CFEM'],
        y=top_companies['TITULAR'],
        orientation='h',
        marker=dict(
            color='rgba(102, 126, 234, 0.8)',
            line=dict(color='rgba(102, 126, 234, 1.0)', width=2)
        ),
        text=[f'R$ {val:,.0f}' for val in top_companies['CFEM']],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='🏆 Top 10 Empresas por Valor CFEM',
        xaxis_title='Valor CFEM (R$)',
        yaxis_title='Empresa',
        height=500,
        template='plotly_white',
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig

def create_regional_distribution(data):
    """Cria gráfico de distribuição regional"""
    if 'REGIAO' in data.columns:
        regional_data = data.groupby('REGIAO').agg({
            'CFEM': 'sum',
            'TITULAR': 'nunique'
        }).reset_index()
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Valor CFEM por Região', 'Número de Empresas por Região'),
            specs=[[{"type": "pie"}, {"type": "pie"}]]
        )
        
        # Gráfico de pizza - CFEM
        fig.add_trace(
            go.Pie(
                labels=regional_data['REGIAO'],
                values=regional_data['CFEM'],
                name="CFEM",
                marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
            ),
            row=1, col=1
        )
        
        # Gráfico de pizza - Empresas
        fig.add_trace(
            go.Pie(
                labels=regional_data['REGIAO'],
                values=regional_data['TITULAR'],
                name="Empresas",
                marker_colors=['#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43']
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=True,
            title_text="🌍 Distribuição Regional do CFEM"
        )
        
        return fig
    else:
        # Fallback para distribuição por estado
        state_data = data.groupby('ESTADO')['CFEM'].sum().nlargest(10).reset_index()
        
        fig = px.pie(
            state_data,
            values='CFEM',
            names='ESTADO',
            title='🌍 Top 10 Estados por Valor CFEM'
        )
        
        fig.update_layout(height=500)
        return fig

def create_substance_treemap(data):
    """Cria treemap das substâncias"""
    substance_data = data.groupby('PRIMEIRODESUBS').agg({
        'CFEM': 'sum',
        'TITULAR': 'nunique'
    }).reset_index()
    
    fig = px.treemap(
        substance_data,
        values='CFEM',
        path=['PRIMEIRODESUBS'],
        title='🔨 Composição do CFEM por Substância',
        color='TITULAR',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(height=500)
    return fig

def create_concentration_analysis(data):
    """Cria análise de concentração de mercado"""
    # Calcular participação das empresas
    company_cfem = data.groupby('TITULAR')['CFEM'].sum().sort_values(ascending=False)
    total_cfem = company_cfem.sum()
    
    # Top 10 e outros
    top_10 = company_cfem.head(10)
    others = company_cfem.iloc[10:].sum()
    
    # Calcular percentuais
    top_10_pct = (top_10 / total_cfem * 100)
    others_pct = (others / total_cfem * 100)
    
    # Preparar dados para o gráfico
    plot_data = pd.concat([top_10_pct, pd.Series([others_pct], index=['Outras Empresas'])])
    
    fig = go.Figure(data=[
        go.Bar(
            x=plot_data.index,
            y=plot_data.values,
            marker_color=['#1f77b4' if x != 'Outras Empresas' else '#ff7f0e' for x in plot_data.index],
            text=[f'{val:.1f}%' for val in plot_data.values],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title='📊 Concentração de Mercado - Participação das Empresas (%)',
        xaxis_title='Empresa',
        yaxis_title='Participação (%)',
        height=500,
        template='plotly_white',
        xaxis_tickangle=-45
    )
    
    return fig

def create_monthly_trend(data):
    """Cria gráfico de tendência temporal (se houver dados de data)"""
    if 'DATA' in data.columns:
        try:
            data['DATA'] = pd.to_datetime(data['DATA'])
            monthly_data = data.groupby(data['DATA'].dt.to_period('M')).agg({
                'CFEM': 'sum',
                'TITULAR': 'nunique'
            }).reset_index()
            
            monthly_data['DATA_STR'] = monthly_data['DATA'].astype(str)
            
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig.add_trace(
                go.Scatter(
                    x=monthly_data['DATA_STR'],
                    y=monthly_data['CFEM'],
                    mode='lines+markers',
                    name='CFEM Total',
                    line=dict(color='#1f77b4', width=3)
                ),
                secondary_y=False,
            )
            
            fig.add_trace(
                go.Bar(
                    x=monthly_data['DATA_STR'],
                    y=monthly_data['TITULAR'],
                    name='Empresas Ativas',
                    opacity=0.7,
                    marker_color='#ff7f0e'
                ),
                secondary_y=True,
            )
            
            fig.update_xaxes(title_text="Período")
            fig.update_yaxes(title_text="CFEM Total (R$)", secondary_y=False)
            fig.update_yaxes(title_text="Número de Empresas", secondary_y=True)
            
            fig.update_layout(
                title='📈 Evolução Temporal do CFEM e Empresas Ativas',
                height=500,
                template='plotly_white'
            )
            
            return fig
        except:
            return None
    else:
        return None

def generate_insights(data, stats):
    """Gera insights automáticos dos dados"""
    insights = []
    
    # Insight sobre concentração
    total_empresas = data['TITULAR'].nunique()
    top_5_cfem = data.groupby('TITULAR')['CFEM'].sum().nlargest(5).sum()
    total_cfem = data['CFEM'].sum()
    concentracao_top5 = (top_5_cfem / total_cfem * 100)
    
    if concentracao_top5 > 50:
        insights.append({
            'type': 'warning',
            'title': 'Alta Concentração de Mercado',
            'text': f'As top 5 empresas concentram {concentracao_top5:.1f}% do total de CFEM arrecadado, indicando alta concentração no setor.'
        })
    else:
        insights.append({
            'type': 'success',
            'title': 'Mercado Distribuído',
            'text': f'As top 5 empresas representam {concentracao_top5:.1f}% do CFEM, indicando boa distribuição no mercado.'
        })
    
    # Insight sobre diversificação geográfica
    if 'REGIAO' in data.columns:
        regiao_dominante = data.groupby('REGIAO')['CFEM'].sum().idxmax()
        cfem_regiao_dominante = data.groupby('REGIAO')['CFEM'].sum().max()
        pct_regiao_dominante = (cfem_regiao_dominante / total_cfem * 100)
        
        insights.append({
            'type': 'info',
            'title': 'Distribuição Regional',
            'text': f'A região {regiao_dominante} é dominante, representando {pct_regiao_dominante:.1f}% do total de CFEM arrecadado.'
        })
    
    # Insight sobre substâncias
    substancia_principal = data.groupby('PRIMEIRODESUBS')['CFEM'].sum().idxmax()
    cfem_substancia_principal = data.groupby('PRIMEIRODESUBS')['CFEM'].sum().max()
    pct_substancia_principal = (cfem_substancia_principal / total_cfem * 100)
    
    insights.append({
        'type': 'info',
        'title': 'Substância Principal',
        'text': f'{substancia_principal} é a principal substância, representando {pct_substancia_principal:.1f}% do CFEM total.'
    })
    
    # Insight sobre valor médio
    valor_medio = data['CFEM'].mean()
    mediana = data['CFEM'].median()
    
    if valor_medio > mediana * 2:
        insights.append({
            'type': 'warning',
            'title': 'Distribuição Assimétrica',
            'text': f'O valor médio (R$ {valor_medio:,.0f}) é significativamente maior que a mediana (R$ {mediana:,.0f}), indicando presença de outliers.'
        })
    
    return insights

def main():
    """Função principal da página"""
    
    st.title("📊 Dashboard Executivo")
    st.markdown("Visão executiva dos principais KPIs e indicadores do CFEM")
    
    # Verificar se os dados estão disponíveis
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        st.warning("⚠️ Nenhum dado carregado. Por favor, carregue os dados na página principal.")
        st.markdown("[← Voltar para a página principal](../)")
        return
    
    data = st.session_state.filtered_data
    stats = st.session_state.get('stats', {})
    
    # KPIs Principais
    st.markdown("## 🎯 KPIs Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">R$ {data['CFEM'].sum():,.0f}</div>
            <div class="metric-label">Valor Total CFEM</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['TITULAR'].nunique():,}</div>
            <div class="metric-label">Empresas Ativas</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['ESTADO'].nunique():,}</div>
            <div class="metric-label">Estados</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{data['PRIMEIRODESUBS'].nunique():,}</div>
            <div class="metric-label">Substâncias</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Linha adicional de KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        valor_medio = data['CFEM'].mean()
        st.metric(
            label="Valor Médio CFEM",
            value=f"R$ {valor_medio:,.2f}",
            delta=f"Mediana: R$ {data['CFEM'].median():,.2f}"
        )
    
    with col2:
        max_empresa = data.groupby('TITULAR')['CFEM'].sum().max()
        st.metric(
            label="Maior Empresa",
            value=f"R$ {max_empresa:,.0f}",
            delta=f"{(max_empresa/data['CFEM'].sum()*100):.1f}% do total"
        )
    
    with col3:
        operacoes_por_empresa = len(data) / data['TITULAR'].nunique()
        st.metric(
            label="Operações/Empresa",
            value=f"{operacoes_por_empresa:.1f}",
            delta="média por empresa"
        )
    
    with col4:
        diversificacao_media = data.groupby('TITULAR')['PRIMEIRODESUBS'].nunique().mean()
        st.metric(
            label="Diversificação Média",
            value=f"{diversificacao_media:.1f}",
            delta="substâncias por empresa"
        )
    
    # Insights Automáticos
    st.markdown("## 💡 Insights Automáticos")
    insights = generate_insights(data, stats)
    
    for insight in insights:
        if insight['type'] == 'success':
            st.markdown(f"""
            <div class="insight-card">
                <h4>✅ {insight['title']}</h4>
                <p>{insight['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        elif insight['type'] == 'warning':
            st.markdown(f"""
            <div class="warning-card">
                <h4>⚠️ {insight['title']}</h4>
                <p>{insight['text']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="insight-card">
                <h4>ℹ️ {insight['title']}</h4>
                <p>{insight['text']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Gráficos Principais
    st.markdown("## 📈 Análises Visuais")
    
    # Linha 1 de gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        fig_companies = create_executive_summary_chart(data)
        st.plotly_chart(fig_companies, use_container_width=True)
    
    with col2:
        fig_regional = create_regional_distribution(data)
        st.plotly_chart(fig_regional, use_container_width=True)
    
    # Linha 2 de gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        fig_substance = create_substance_treemap(data)
        st.plotly_chart(fig_substance, use_container_width=True)
    
    with col2:
        fig_concentration = create_concentration_analysis(data)
        st.plotly_chart(fig_concentration, use_container_width=True)
    
    # Gráfico temporal (se disponível)
    fig_temporal = create_monthly_trend(data)
    if fig_temporal:
        st.plotly_chart(fig_temporal, use_container_width=True)
    
    # Tabelas Resumo
    st.markdown("## 📋 Resumos Executivos")
    
    tab1, tab2, tab3 = st.tabs(["🏢 Top Empresas", "🌍 Análise Regional", "🔨 Análise por Substância"])
    
    with tab1:
        st.markdown("### Top 15 Empresas por Valor CFEM")
        top_companies = data.groupby('TITULAR').agg({
            'CFEM': ['sum', 'count', 'mean'],
            'PRIMEIRODESUBS': 'nunique',
            'ESTADO': 'nunique'
        }).reset_index()
        
        top_companies.columns = ['Empresa', 'CFEM_Total', 'Num_Operações', 'CFEM_Médio', 'Num_Substâncias', 'Num_Estados']
        top_companies = top_companies.sort_values('CFEM_Total', ascending=False).head(15)
        
        # Formatação
        top_companies['CFEM_Total'] = top_companies['CFEM_Total'].apply(lambda x: f"R$ {x:,.0f}")
        top_companies['CFEM_Médio'] = top_companies['CFEM_Médio'].apply(lambda x: f"R$ {x:,.0f}")
        
        st.dataframe(top_companies, use_container_width=True, hide_index=True)
    
    with tab2:
        if 'REGIAO' in data.columns:
            st.markdown("### Análise por Região")
            regional_summary = data.groupby('REGIAO').agg({
                'CFEM': ['sum', 'mean', 'count'],
                'TITULAR': 'nunique',
                'ESTADO': 'nunique',
                'PRIMEIRODESUBS': 'nunique'
            }).reset_index()
            
            regional_summary.columns = ['Região', 'CFEM_Total', 'CFEM_Médio', 'Num_Operações', 
                                      'Num_Empresas', 'Num_Estados', 'Num_Substâncias']
            regional_summary = regional_summary.sort_values('CFEM_Total', ascending=False)
            
            # Formatação
            regional_summary['CFEM_Total'] = regional_summary['CFEM_Total'].apply(lambda x: f"R$ {x:,.0f}")
            regional_summary['CFEM_Médio'] = regional_summary['CFEM_Médio'].apply(lambda x: f"R$ {x:,.0f}")
            
            st.dataframe(regional_summary, use_container_width=True, hide_index=True)
        else:
            st.markdown("### Análise por Estado")
            state_summary = data.groupby('ESTADO').agg({
                'CFEM': ['sum', 'mean', 'count'],
                'TITULAR': 'nunique',
                'PRIMEIRODESUBS': 'nunique'
            }).reset_index()
            
            state_summary.columns = ['Estado', 'CFEM_Total', 'CFEM_Médio', 'Num_Operações', 
                                   'Num_Empresas', 'Num_Substâncias']
            state_summary = state_summary.sort_values('CFEM_Total', ascending=False).head(15)
            
            # Formatação
            state_summary['CFEM_Total'] = state_summary['CFEM_Total'].apply(lambda x: f"R$ {x:,.0f}")
            state_summary['CFEM_Médio'] = state_summary['CFEM_Médio'].apply(lambda x: f"R$ {x:,.0f}")
            
            st.dataframe(state_summary, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown("### Análise por Substância")
        substance_summary = data.groupby('PRIMEIRODESUBS').agg({
            'CFEM': ['sum', 'mean', 'count'],
            'TITULAR': 'nunique',
            'ESTADO': 'nunique'
        }).reset_index()
        
        substance_summary.columns = ['Substância', 'CFEM_Total', 'CFEM_Médio', 'Num_Operações', 
                                   'Num_Empresas', 'Num_Estados']
        substance_summary = substance_summary.sort_values('CFEM_Total', ascending=False).head(15)
        
        # Formatação
        substance_summary['CFEM_Total'] = substance_summary['CFEM_Total'].apply(lambda x: f"R$ {x:,.0f}")
        substance_summary['CFEM_Médio'] = substance_summary['CFEM_Médio'].apply(lambda x: f"R$ {x:,.0f}")
        
        st.dataframe(substance_summary, use_container_width=True, hide_index=True)
    
    # Métricas de Performance
    st.markdown("## 📊 Métricas de Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Índice de Concentração HHI
        company_shares = data.groupby('TITULAR')['CFEM'].sum()
        total_cfem = company_shares.sum()
        market_shares = (company_shares / total_cfem) ** 2
        hhi = market_shares.sum()
        
        st.markdown("### Índice de Concentração (HHI)")
        if hhi > 0.25:
            concentration_status = "🔴 Alta Concentração"
            concentration_color = "#ff4757"
        elif hhi > 0.15:
            concentration_status = "🟡 Concentração Moderada"
            concentration_color = "#ffa502"
        else:
            concentration_status = "🟢 Baixa Concentração"
            concentration_color = "#2ed573"
        
        st.markdown(f"""
        <div style="background: {concentration_color}20; padding: 1rem; border-radius: 8px; border-left: 4px solid {concentration_color};">
            <h4>{concentration_status}</h4>
            <p><strong>HHI: {hhi:.4f}</strong></p>
            <p>Quanto mais próximo de 1, maior a concentração do mercado</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Eficiência Média
        st.markdown("### Eficiência Operacional")
        
        operations_per_company = len(data) / data['TITULAR'].nunique()
        avg_cfem_per_operation = data['CFEM'].mean()
        
        st.metric(
            label="Operações por Empresa",
            value=f"{operations_per_company:.1f}",
            delta="Média do setor"
        )
        
        st.metric(
            label="CFEM Médio por Operação",
            value=f"R$ {avg_cfem_per_operation:,.2f}",
            delta=f"±{data['CFEM'].std():,.0f} (desvio padrão)"
        )

if __name__ == "__main__":
    main()