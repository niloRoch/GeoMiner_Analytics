import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from src import CFEMVisualizations
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Análises Estatísticas - CFEM Analytics",
    page_icon="📈",
    layout="wide"
)

def create_distribution_analysis(data):
    """Cria análise completa de distribuição dos valores CFEM"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Histograma dos Valores CFEM', 'Box Plot', 
                       'Q-Q Plot (Normalidade)', 'Distribuição Log-Normal'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    cfem_values = data['CFEM']
    
    # Histograma
    fig.add_trace(
        go.Histogram(
            x=cfem_values, 
            nbinsx=50, 
            name='CFEM',
            marker_color='rgba(102, 126, 234, 0.7)'
        ),
        row=1, col=1
    )
    
    # Box Plot
    fig.add_trace(
        go.Box(
            y=cfem_values, 
            name='CFEM',
            marker_color='rgba(118, 75, 162, 0.7)'
        ),
        row=1, col=2
    )
    
    # Q-Q Plot
    sorted_data = np.sort(cfem_values)
    n = len(sorted_data)
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n+1) / (n+1))
    standardized_data = (sorted_data - np.mean(sorted_data)) / np.std(sorted_data)
    
    fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=standardized_data,
            mode='markers',
            name='Q-Q Plot',
            marker=dict(color='rgba(255, 107, 107, 0.7)', size=4)
        ),
        row=2, col=1
    )
    
    # Linha de referência para Q-Q Plot
    fig.add_trace(
        go.Scatter(
            x=[-3, 3],
            y=[-3, 3],
            mode='lines',
            name='Linha Teórica',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Distribuição Log-Normal
    log_cfem = np.log(cfem_values + 1)
    fig.add_trace(
        go.Histogram(
            x=log_cfem,
            nbinsx=50,
            name='Log(CFEM)',
            marker_color='rgba(76, 205, 196, 0.7)'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        title_text="Análise de Distribuição dos Valores CFEM",
        showlegend=False
    )
    
    return fig

def create_correlation_matrix(data):
    """Cria matriz de correlação das variáveis numéricas"""
    # Selecionar apenas colunas numéricas
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) < 2:
        return None
    
    correlation_matrix = data[numeric_columns].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(correlation_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverangles=0
    ))
    
    fig.update_layout(
        title='Matriz de Correlação das Variáveis Numéricas',
        height=500,
        width=500
    )
    
    return fig

def create_categorical_analysis(data):
    """Cria análise das variáveis categóricas"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Box Plot por Estado (Top 10)', 'Box Plot por Substância (Top 8)', 
                       'Violin Plot por Região', 'Distribuição por Faixa CFEM'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Box Plot por Estado (Top 10)
    top_states = data['ESTADO'].value_counts().head(10).index
    colors = px.colors.qualitative.Set3[:len(top_states)]
    
    for i, state in enumerate(top_states):
        state_data = data[data['ESTADO'] == state]['CFEM']
        fig.add_trace(
            go.Box(
                y=state_data,
                name=state,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Box Plot por Substância (Top 8)
    top_substances = data['PRIMEIRODESUBS'].value_counts().head(8).index
    
    for i, substance in enumerate(top_substances):
        substance_data = data[data['PRIMEIRODESUBS'] == substance]['CFEM']
        fig.add_trace(
            go.Box(
                y=substance_data,
                name=substance[:15] + "..." if len(substance) > 15 else substance,
                marker_color=colors[i % len(colors)],
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Violin Plot por Região
    if 'REGIAO' in data.columns:
        regions = data['REGIAO'].unique()
        for i, region in enumerate(regions):
            region_data = data[data['REGIAO'] == region]['CFEM']
            fig.add_trace(
                go.Violin(
                    y=region_data,
                    name=region,
                    marker_color=colors[i % len(colors)],
                    showlegend=False,
                    box_visible=True,
                    meanline_visible=True
                ),
                row=2, col=1
            )
    
    # Distribuição por Faixa CFEM
    if 'CFEM_FAIXA' in data.columns:
        faixa_counts = data['CFEM_FAIXA'].value_counts()
        fig.add_trace(
            go.Bar(
                x=faixa_counts.index,
                y=faixa_counts.values,
                name='Distribuição',
                marker_color='rgba(102, 126, 234, 0.7)',
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        title_text="Análise por Categorias",
        showlegend=False
    )
    
    return fig

def perform_statistical_tests(data):
    """Executa testes estatísticos básicos"""
    results = {}
    
    # Teste de normalidade (Shapiro-Wilk para amostras pequenas, Anderson-Darling para grandes)
    cfem_values = data['CFEM']
    
    if len(cfem_values) <= 5000:
        stat, p_value = stats.shapiro(cfem_values.sample(min(5000, len(cfem_values))))
        results['normalidade'] = {
            'teste': 'Shapiro-Wilk',
            'estatistica': stat,
            'p_valor': p_value,
            'interpretacao': 'Normal' if p_value > 0.05 else 'Não Normal'
        }
    else:
        stat, critical_values, sig_level = stats.anderson(cfem_values, dist='norm')
        results['normalidade'] = {
            'teste': 'Anderson-Darling',
            'estatistica': stat,
            'valores_criticos': critical_values,
            'interpretacao': 'Normal' if stat < critical_values[2] else 'Não Normal'
        }
    
    # Teste ANOVA para diferenças entre estados (se aplicável)
    if data['ESTADO'].nunique() > 2:
        top_states = data['ESTADO'].value_counts().head(5).index
        state_groups = [data[data['ESTADO'] == state]['CFEM'].values for state in top_states]
        
        try:
            f_stat, p_value = stats.f_oneway(*state_groups)
            results['anova_estados'] = {
                'f_estatistica': f_stat,
                'p_valor': p_value,
                'interpretacao': 'Diferenças significativas' if p_value < 0.05 else 'Sem diferenças significativas'
            }
        except:
            results['anova_estados'] = {'erro': 'Não foi possível executar o teste'}
    
    # Teste de Kruskal-Wallis (não-paramétrico) para substâncias
    if data['PRIMEIRODESUBS'].nunique() > 2:
        top_substances = data['PRIMEIRODESUBS'].value_counts().head(5).index
        substance_groups = [data[data['PRIMEIRODESUBS'] == sub]['CFEM'].values for sub in top_substances]
        
        try:
            h_stat, p_value = stats.kruskal(*substance_groups)
            results['kruskal_substancias'] = {
                'h_estatistica': h_stat,
                'p_valor': p_value,
                'interpretacao': 'Diferenças significativas' if p_value < 0.05 else 'Sem diferenças significativas'
            }
        except:
            results['kruskal_substancias'] = {'erro': 'Não foi possível executar o teste'}
    
    return results

def create_outlier_analysis(data):
    """Cria análise de outliers"""
    cfem_values = data['CFEM']
    
    # Método IQR
    Q1 = cfem_values.quantile(0.25)
    Q3 = cfem_values.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers_iqr = data[(cfem_values < lower_bound) | (cfem_values > upper_bound)]
    
    # Método Z-Score
    z_scores = np.abs(stats.zscore(cfem_values))
    outliers_zscore = data[z_scores > 3]
    
    # Método de percentis
    p1 = cfem_values.quantile(0.01)
    p99 = cfem_values.quantile(0.99)
    outliers_percentile = data[(cfem_values < p1) | (cfem_values > p99)]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Outliers por IQR', 'Outliers por Z-Score', 
                       'Outliers por Percentil', 'Comparação dos Métodos'),
    )
    
    # Gráfico 1: IQR
    fig.add_trace(
        go.Scatter(
            x=range(len(data)),
            y=cfem_values,
            mode='markers',
            marker=dict(
                color=['red' if i in outliers_iqr.index else 'blue' for i in data.index],
                size=4,
                opacity=0.6
            ),
            name='IQR',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Gráfico 2: Z-Score
    fig.add_trace(
        go.Scatter(
            x=range(len(data)),
            y=cfem_values,
            mode='markers',
            marker=dict(
                color=['red' if i in outliers_zscore.index else 'blue' for i in data.index],
                size=4,
                opacity=0.6
            ),
            name='Z-Score',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Gráfico 3: Percentil
    fig.add_trace(
        go.Scatter(
            x=range(len(data)),
            y=cfem_values,
            mode='markers',
            marker=dict(
                color=['red' if i in outliers_percentile.index else 'blue' for i in data.index],
                size=4,
                opacity=0.6
            ),
            name='Percentil',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Gráfico 4: Comparação
    methods = ['IQR', 'Z-Score', 'Percentil']
    counts = [len(outliers_iqr), len(outliers_zscore), len(outliers_percentile)]
    
    fig.add_trace(
        go.Bar(
            x=methods,
            y=counts,
            name='Outliers Detectados',
            marker_color='rgba(255, 107, 107, 0.7)',
            showlegend=False
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=700,
        title_text="Análise de Outliers - Diferentes Métodos"
    )
    
    return fig, {
        'iqr': len(outliers_iqr),
        'zscore': len(outliers_zscore),
        'percentile': len(outliers_percentile),
        'outliers_iqr': outliers_iqr,
        'outliers_zscore': outliers_zscore,
        'outliers_percentile': outliers_percentile
    }

def create_descriptive_statistics_table(data):
    """Cria tabela de estatísticas descritivas"""
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    
    stats_data = []
    for col in numeric_columns:
        if data[col].notna().sum() > 0:  # Verificar se há dados válidos
            stats_data.append({
                'Variável': col,
                'Contagem': data[col].count(),
                'Média': data[col].mean(),
                'Mediana': data[col].median(),
                'Desvio Padrão': data[col].std(),
                'Mínimo': data[col].min(),
                'Máximo': data[col].max(),
                'Q1 (25%)': data[col].quantile(0.25),
                'Q3 (75%)': data[col].quantile(0.75),
                'Assimetria': stats.skew(data[col].dropna()),
                'Curtose': stats.kurtosis(data[col].dropna())
            })
    
    return pd.DataFrame(stats_data)

def main():
    """Função principal da página"""
    
    st.title("📈 Análises Estatísticas")
    st.markdown("Análises estatísticas avançadas dos dados CFEM")
    
    # Verificar se os dados estão disponíveis
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        st.warning("⚠️ Nenhum dado carregado. Por favor, carregue os dados na página principal.")
        return
    
    data = st.session_state.filtered_data
    
    # Sidebar com opções de análise
    st.sidebar.markdown("## 📊 Opções de Análise")
    
    analysis_type = st.sidebar.selectbox(
        "Escolha o tipo de análise:",
        ["Estatísticas Descritivas", "Distribuições", "Correlações", 
         "Análise Categórica", "Outliers", "Testes Estatísticos"]
    )
    
    # Executar análise selecionada
    if analysis_type == "Estatísticas Descritivas":
        st.markdown("## 📊 Estatísticas Descritivas")
        
        # Tabela de estatísticas descritivas
        desc_stats = create_descriptive_statistics_table(data)
        
        if not desc_stats.empty:
            # Formatação da tabela
            formatted_stats = desc_stats.copy()
            
            numeric_cols = ['Média', 'Mediana', 'Desvio Padrão', 'Mínimo', 'Máximo', 'Q1 (25%)', 'Q3 (75%)']
            for col in numeric_cols:
                if col in formatted_stats.columns:
                    formatted_stats[col] = formatted_stats[col].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "N/A")
            
            formatted_stats['Assimetria'] = formatted_stats['Assimetria'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            formatted_stats['Curtose'] = formatted_stats['Curtose'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
            
            st.dataframe(formatted_stats, use_container_width=True, hide_index=True)
            
            # Interpretações
            st.markdown("### Interpretações:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Assimetria:")
                cfem_skew = stats.skew(data['CFEM'].dropna())
                if cfem_skew > 1:
                    st.info("🔄 Distribuição fortemente assimétrica à direita (valores extremos altos)")
                elif cfem_skew > 0.5:
                    st.info("📊 Distribuição moderadamente assimétrica à direita")
                elif cfem_skew < -1:
                    st.info("🔄 Distribuição fortemente assimétrica à esquerda")
                elif cfem_skew < -0.5:
                    st.info("📊 Distribuição moderadamente assimétrica à esquerda")
                else:
                    st.success("✅ Distribuição aproximadamente simétrica")
            
            with col2:
                st.markdown("#### Curtose:")
                cfem_kurt = stats.kurtosis(data['CFEM'].dropna())
                if cfem_kurt > 3:
                    st.info("📈 Distribuição leptocúrtica (caudas pesadas)")
                elif cfem_kurt < -3:
                    st.info("📉 Distribuição platicúrtica (caudas leves)")
                else:
                    st.success("✅ Distribuição mesocúrtica (próxima à normal)")
        else:
            st.error("Nenhuma variável numérica encontrada nos dados.")
    
    elif analysis_type == "Distribuições":
        st.markdown("## 📊 Análise de Distribuições")
        
        fig_dist = create_distribution_analysis(data)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Testes de normalidade
        st.markdown("### 🧪 Testes de Normalidade")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Shapiro-Wilk (para amostras pequenas)
            sample_size = min(5000, len(data['CFEM']))
            sample_data = data['CFEM'].sample(sample_size)
            shapiro_stat, shapiro_p = stats.shapiro(sample_data)
            
            st.markdown("#### Teste de Shapiro-Wilk")
            st.metric("Estatística", f"{shapiro_stat:.6f}")
            st.metric("P-valor", f"{shapiro_p:.2e}")
            
            if shapiro_p < 0.05:
                st.error("❌ Rejeita hipótese de normalidade")
            else:
                st.success("✅ Não rejeita hipótese de normalidade")
        
        with col2:
            # Kolmogorov-Smirnov
            ks_stat, ks_p = stats.kstest(data['CFEM'], 'norm', 
                                       args=(data['CFEM'].mean(), data['CFEM'].std()))
            
            st.markdown("#### Teste de Kolmogorov-Smirnov")
            st.metric("Estatística", f"{ks_stat:.6f}")
            st.metric("P-valor", f"{ks_p:.2e}")
            
            if ks_p < 0.05:
                st.error("❌ Rejeita hipótese de normalidade")
            else:
                st.success("✅ Não rejeita hipótese de normalidade")
    
    elif analysis_type == "Correlações":
        st.markdown("## 🔗 Análise de Correlações")
        
        fig_corr = create_correlation_matrix(data)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Tabela de correlações significativas
            numeric_data = data.select_dtypes(include=[np.number])
            corr_matrix = numeric_data.corr()
            
            # Encontrar correlações significativas
            significant_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:  # Correlação moderada ou forte
                        significant_corr.append({
                            'Variável 1': corr_matrix.columns[i],
                            'Variável 2': corr_matrix.columns[j],
                            'Correlação': corr_val,
                            'Força': 'Forte' if abs(corr_val) > 0.7 else 'Moderada',
                            'Direção': 'Positiva' if corr_val > 0 else 'Negativa'
                        })
            
            if significant_corr:
                st.markdown("### 📋 Correlações Significativas (|r| > 0.3)")
                corr_df = pd.DataFrame(significant_corr)
                corr_df['Correlação'] = corr_df['Correlação'].apply(lambda x: f"{x:.4f}")
                st.dataframe(corr_df, use_container_width=True, hide_index=True)
            else:
                st.info("Nenhuma correlação significativa encontrada entre as variáveis numéricas.")
        else:
            st.warning("Dados insuficientes para análise de correlação.")
    
    elif analysis_type == "Análise Categórica":
        st.markdown("## 📊 Análise por Categorias")
        
        fig_cat = create_categorical_analysis(data)
        st.plotly_chart(fig_cat, use_container_width=True)
        
        # Testes estatísticos para diferenças entre grupos
        st.markdown("### 🧪 Testes de Diferenças entre Grupos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ANOVA - Estados (Top 5)")
            top_states = data['ESTADO'].value_counts().head(5).index
            state_groups = [data[data['ESTADO'] == state]['CFEM'] for state in top_states]
            
            if len(state_groups) > 2 and all(len(group) > 1 for group in state_groups):
                f_stat, p_value = stats.f_oneway(*state_groups)
                
                st.metric("F-Estatística", f"{f_stat:.4f}")
                st.metric("P-valor", f"{p_value:.2e}")
                
                if p_value < 0.05:
                    st.error("❌ Diferenças significativas entre estados")
                else:
                    st.success("✅ Sem diferenças significativas entre estados")
            else:
                st.warning("Dados insuficientes para ANOVA")
        
        with col2:
            st.markdown("#### Kruskal-Wallis - Substâncias (Top 5)")
            top_substances = data['PRIMEIRODESUBS'].value_counts().head(5).index
            substance_groups = [data[data['PRIMEIRODESUBS'] == sub]['CFEM'] for sub in top_substances]
            
            if len(substance_groups) > 2 and all(len(group) > 1 for group in substance_groups):
                h_stat, p_value = stats.kruskal(*substance_groups)
                
                st.metric("H-Estatística", f"{h_stat:.4f}")
                st.metric("P-valor", f"{p_value:.2e}")
                
                if p_value < 0.05:
                    st.error("❌ Diferenças significativas entre substâncias")
                else:
                    st.success("✅ Sem diferenças significativas entre substâncias")
            else:
                st.warning("Dados insuficientes para Kruskal-Wallis")
    
    elif analysis_type == "Outliers":
        st.markdown("## 🎯 Análise de Outliers")
        
        fig_outliers, outlier_results = create_outlier_analysis(data)
        st.plotly_chart(fig_outliers, use_container_width=True)
        
        # Resumo dos outliers
        st.markdown("### 📊 Resumo dos Outliers Detectados")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Método IQR", f"{outlier_results['iqr']} outliers")
            st.metric("Percentual", f"{outlier_results['iqr']/len(data)*100:.2f}%")
        
        with col2:
            st.metric("Método Z-Score", f"{outlier_results['zscore']} outliers")
            st.metric("Percentual", f"{outlier_results['zscore']/len(data)*100:.2f}%")
        
        with col3:
            st.metric("Método Percentil", f"{outlier_results['percentile']} outliers")
            st.metric("Percentual", f"{outlier_results['percentile']/len(data)*100:.2f}%")
        
        # Tabela dos maiores outliers
        st.markdown("### 🔍 Maiores Outliers (Top 10)")
        
        outlier_method = st.selectbox("Escolha o método:", ["IQR", "Z-Score", "Percentil"])
        
        if outlier_method == "IQR":
            outliers_df = outlier_results['outliers_iqr']
        elif outlier_method == "Z-Score":
            outliers_df = outlier_results['outliers_zscore']
        else:
            outliers_df = outlier_results['outliers_percentile']
        
        if len(outliers_df) > 0:
            top_outliers = outliers_df.nlargest(10, 'CFEM')[['TITULAR', 'MUNICIPIO(S)', 'ESTADO', 'CFEM', 'PRIMEIRODESUBS']]
            st.dataframe(top_outliers, use_container_width=True)
        else:
            st.info("Nenhum outlier encontrado com este método.")
    
    elif analysis_type == "Testes Estatísticos":
        st.markdown("## 🧪 Testes Estatísticos")
        
        test_results = perform_statistical_tests(data)
        
        # Exibir resultados dos testes
        for test_name, results in test_results.items():
            if 'erro' not in results:
                st.markdown(f"### {test_name.replace('_', ' ').title()}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if test_name == 'normalidade':
                        st.metric("Teste", results['teste'])
                        st.metric("Estatística", f"{results['estatistica']:.6f}")
                        if 'p_valor' in results:
                            st.metric("P-valor", f"{results['p_valor']:.2e}")
                    elif 'f_estatistica' in results:
                        st.metric("F-Estatística", f"{results['f_estatistica']:.4f}")
                        st.metric("P-valor", f"{results['p_valor']:.2e}")
                    elif 'h_estatistica' in results:
                        st.metric("H-Estatística", f"{results['h_estatistica']:.4f}")
                        st.metric("P-valor", f"{results['p_valor']:.2e}")
                
                with col2:
                    interpretation = results.get('interpretacao', 'Sem interpretação disponível')
                    if 'significativas' in interpretation.lower() and 'sem' not in interpretation.lower():
                        st.error(f"❌ {interpretation}")
                    elif 'normal' in interpretation.lower() and 'não' not in interpretation.lower():
                        st.success(f"✅ {interpretation}")
                    else:
                        st.info(f"ℹ️ {interpretation}")
        
        # Resumo dos testes
        st.markdown("### 📋 Resumo dos Testes Estatísticos")
        
        summary_data = []
        for test_name, results in test_results.items():
            if 'erro' not in results:
                summary_data.append({
                    'Teste': test_name.replace('_', ' ').title(),
                    'Resultado': results.get('interpretacao', 'N/A'),
                    'P-valor': results.get('p_valor', 'N/A')
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":

    main()
