import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Machine Learning - CFEM Analytics",
    page_icon="🤖",
    layout="wide"
)

def prepare_ml_data(data):
    """Prepara dados para machine learning"""
    # Criar cópia dos dados
    ml_data = data.copy()
    
    # Selecionar features relevantes
    feature_columns = []
    
    # Features numéricas
    if 'LONGITUDE' in ml_data.columns and 'LATITUDE' in ml_data.columns:
        feature_columns.extend(['LONGITUDE', 'LATITUDE'])
    
    # Encoding de variáveis categóricas
    categorical_features = ['ESTADO', 'PRIMEIRODESUBS']
    label_encoders = {}
    
    for col in categorical_features:
        if col in ml_data.columns:
            le = LabelEncoder()
            ml_data[f'{col}_encoded'] = le.fit_transform(ml_data[col].astype(str))
            feature_columns.append(f'{col}_encoded')
            label_encoders[col] = le
    
    # Features derivadas
    if 'REGIAO' in ml_data.columns:
        le_region = LabelEncoder()
        ml_data['REGIAO_encoded'] = le_region.fit_transform(ml_data['REGIAO'].astype(str))
        feature_columns.append('REGIAO_encoded')
        label_encoders['REGIAO'] = le_region
    
    # Diversificação da empresa
    if 'DIVERSIFICACAO_EMPRESA' in ml_data.columns:
        feature_columns.append('DIVERSIFICACAO_EMPRESA')
    
    # Densidade do estado
    if 'DENSIDADE_ESTADO' in ml_data.columns:
        feature_columns.append('DENSIDADE_ESTADO')
    
    # Target variable
    target = 'CFEM'
    
    # Remover outliers extremos (opcional)
    Q1 = ml_data[target].quantile(0.25)
    Q3 = ml_data[target].quantile(0.75)
    IQR = Q3 - Q1
    
    # Definir limites mais permissivos para manter mais dados
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    ml_data_clean = ml_data[
        (ml_data[target] >= lower_bound) & 
        (ml_data[target] <= upper_bound)
    ].copy()
    
    # Verificar se temos features suficientes
    available_features = [col for col in feature_columns if col in ml_data_clean.columns]
    
    if len(available_features) == 0:
        return None, None, None, None
    
    # Preparar X e y
    X = ml_data_clean[available_features].fillna(0)  # Preencher NaN com 0
    y = ml_data_clean[target]
    
    return X, y, available_features, label_encoders

def train_regression_models(X, y):
    """Treina múltiplos modelos de regressão"""
    # Split dos dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Padronização
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Modelos
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'SVR': SVR(kernel='rbf', gamma='scale')
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            # Treinar modelo
            if name in ['Linear Regression', 'Ridge', 'Lasso', 'SVR']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                          scoring='neg_mean_absolute_error')
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                                          scoring='neg_mean_absolute_error')
            
            # Métricas
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'actual': y_test,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'cv_score_mean': -cv_scores.mean(),
                'cv_score_std': cv_scores.std()
            }
        except Exception as e:
            st.warning(f"Erro ao treinar {name}: {str(e)}")
            continue
    
    return results, X_test, y_test, scaler

def create_model_comparison_chart(results):
    """Cria gráfico de comparação dos modelos"""
    if not results:
        return None
    
    models = list(results.keys())
    r2_scores = [results[model]['r2'] for model in models]
    mae_scores = [results[model]['mae'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('R² Score (maior é melhor)', 'MAE (menor é melhor)'),
    )
    
    # R² Score
    fig.add_trace(
        go.Bar(
            x=models,
            y=r2_scores,
            name='R²',
            marker_color='rgba(102, 126, 234, 0.7)'
        ),
        row=1, col=1
    )
    
    # MAE
    fig.add_trace(
        go.Bar(
            x=models,
            y=mae_scores,
            name='MAE',
            marker_color='rgba(255, 107, 107, 0.7)'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title_text="Comparação de Performance dos Modelos",
        showlegend=False
    )
    
    return fig

def create_prediction_vs_actual_chart(results, best_model_name):
    """Cria gráfico de predições vs valores reais"""
    if best_model_name not in results:
        return None
    
    best_result = results[best_model_name]
    y_pred = best_result['predictions']
    y_test = best_result['actual']
    
    fig = go.Figure()
    
    # Scatter plot predições vs real
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Predições',
            marker=dict(
                color='rgba(102, 126, 234, 0.6)',
                size=8,
                opacity=0.6
            ),
            text=[f'Real: {real:,.0f}<br>Pred: {pred:,.0f}' for real, pred in zip(y_test, y_pred)],
            hovertemplate='%{text}<extra></extra>'
        )
    )
    
    # Linha de referência perfeita
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Linha Perfeita',
            line=dict(color='red', dash='dash'),
            showlegend=True
        )
    )
    
    fig.update_layout(
        title=f'Predições vs Valores Reais - {best_model_name}',
        xaxis_title='Valores Reais (CFEM)',
        yaxis_title='Predições (CFEM)',
        height=500,
        template='plotly_white'
    )
    
    return fig

def perform_clustering_analysis(data):
    """Executa análise de clustering nos dados"""
    # Preparar dados para clustering
    clustering_features = []
    
    if 'CFEM' in data.columns:
        clustering_features.append('CFEM')
    
    if 'LONGITUDE' in data.columns and 'LATITUDE' in data.columns:
        clustering_features.extend(['LONGITUDE', 'LATITUDE'])
    
    if 'DIVERSIFICACAO_EMPRESA' in data.columns:
        clustering_features.append('DIVERSIFICACAO_EMPRESA')
    
    if len(clustering_features) < 2:
        return None, None
    
    # Selecionar dados válidos
    clustering_data = data[clustering_features].dropna()
    
    if len(clustering_data) < 10:
        return None, None
    
    # Padronização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clustering_data)
    
    # Determinar número ótimo de clusters usando método do cotovelo
    inertias = []
    k_range = range(2, min(11, len(clustering_data)//2))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # Executar K-means com número ótimo (heurística: k=5 ou menor se poucos dados)
    optimal_k = min(5, len(k_range))
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Adicionar labels aos dados originais
    clustering_result = clustering_data.copy()
    clustering_result['cluster'] = cluster_labels
    
    # Análise dos clusters
    cluster_analysis = clustering_result.groupby('cluster').agg({
        'CFEM': ['mean', 'sum', 'count'],
        'LONGITUDE': 'mean' if 'LONGITUDE' in clustering_features else lambda x: None,
        'LATITUDE': 'mean' if 'LATITUDE' in clustering_features else lambda x: None,
        'DIVERSIFICACAO_EMPRESA': 'mean' if 'DIVERSIFICACAO_EMPRESA' in clustering_features else lambda x: None
    }).round(2)
    
    return clustering_result, cluster_analysis

def create_clustering_visualization(clustering_result):
    """Cria visualização dos clusters"""
    if clustering_result is None:
        return None
    
    # Determinar quais features usar para visualização
    if 'LONGITUDE' in clustering_result.columns and 'LATITUDE' in clustering_result.columns:
        # Visualização geográfica
        fig = px.scatter(
            clustering_result,
            x='LONGITUDE',
            y='LATITUDE',
            color='cluster',
            size='CFEM',
            title='Clusters Geográficos',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=500)
        return fig
    
    elif len(clustering_result.columns) >= 3:  # CFEM + pelo menos 2 outras features
        # PCA para redução dimensional
        features_for_pca = [col for col in clustering_result.columns if col != 'cluster']
        
        if len(features_for_pca) >= 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(clustering_result[features_for_pca])
            
            fig = go.Figure()
            
            colors = px.colors.qualitative.Set1
            
            for cluster_id in clustering_result['cluster'].unique():
                cluster_data = pca_result[clustering_result['cluster'] == cluster_id]
                
                fig.add_trace(
                    go.Scatter(
                        x=cluster_data[:, 0],
                        y=cluster_data[:, 1],
                        mode='markers',
                        name=f'Cluster {cluster_id}',
                        marker=dict(
                            color=colors[cluster_id % len(colors)],
                            size=8,
                            opacity=0.7
                        )
                    )
                )
            
            fig.update_layout(
                title='Clusters (PCA - 2 Componentes)',
                xaxis_title='Primeira Componente Principal',
                yaxis_title='Segunda Componente Principal',
                height=500
            )
            
            return fig
    
    return None

def feature_importance_analysis(results, feature_names):
    """Analisa importância das features"""
    importance_data = {}
    
    # Random Forest e Gradient Boosting têm feature importance
    for model_name in ['Random Forest', 'Gradient Boosting']:
        if model_name in results:
            model = results[model_name]['model']
            if hasattr(model, 'feature_importances_'):
                importance_data[model_name] = model.feature_importances_
    
    if not importance_data:
        return None
    
    # Criar gráfico de importância
    fig = go.Figure()
    
    for model_name, importances in importance_data.items():
        fig.add_trace(
            go.Bar(
                x=feature_names,
                y=importances,
                name=model_name,
                opacity=0.7
            )
        )
    
    fig.update_layout(
        title='Importância das Features',
        xaxis_title='Features',
        yaxis_title='Importância',
        height=400,
        barmode='group'
    )
    
    return fig

def main():
    """Função principal da página"""
    
    st.title("🤖 Machine Learning")
    st.markdown("Análises preditivas e modelos de machine learning para dados CFEM")
    
    # Verificar se os dados estão disponíveis
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        st.warning("⚠️ Nenhum dado carregado. Por favor, carregue os dados na página principal.")
        return
    
    data = st.session_state.filtered_data
    
    # Sidebar com opções
    st.sidebar.markdown("## 🎛️ Opções de ML")
    
    analysis_type = st.sidebar.selectbox(
        "Escolha o tipo de análise:",
        ["Previsão de CFEM", "Clustering de Operações", "Análise de Features", "Validação de Modelos"]
    )
    
    if analysis_type == "Previsão de CFEM":
        st.markdown("## 📊 Previsão de Valores CFEM")
        
        # Preparar dados
        with st.spinner("Preparando dados para machine learning..."):
            X, y, feature_names, label_encoders = prepare_ml_data(data)
        
        if X is None:
            st.error("❌ Não foi possível preparar os dados para machine learning. Verifique se há features suficientes.")
            return
        
        st.success(f"✅ Dados preparados: {len(X)} amostras, {len(feature_names)} features")
        
        # Mostrar features utilizadas
        st.markdown("### Features Utilizadas:")
        feature_info = pd.DataFrame({
            'Feature': feature_names,
            'Tipo': ['Numérica' if 'encoded' not in f else 'Categórica (Encoded)' for f in feature_names]
        })
        st.dataframe(feature_info, use_container_width=True, hide_index=True)
        
        # Treinar modelos
        with st.spinner("Treinando modelos de machine learning..."):
            results, X_test, y_test, scaler = train_regression_models(X, y)
        
        if not results:
            st.error("❌ Nenhum modelo foi treinado com sucesso.")
            return
        
        # Comparação de modelos
        st.markdown("### 📊 Comparação de Modelos")
        
        comparison_fig = create_model_comparison_chart(results)
        if comparison_fig:
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Tabela de métricas
        metrics_data = []
        for model_name, result in results.items():
            metrics_data.append({
                'Modelo': model_name,
                'R²': f"{result['r2']:.4f}",
                'MAE': f"{result['mae']:,.2f}",
                'RMSE': f"{result['rmse']:,.2f}",
                'CV Score (MAE)': f"{result['cv_score_mean']:,.2f} ± {result['cv_score_std']:.2f}"
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Melhor modelo
        best_model = min(results.keys(), key=lambda x: results[x]['mae'])
        st.success(f"🏆 Melhor modelo: **{best_model}** (menor MAE)")
        
        # Gráfico de predições vs real
        pred_fig = create_prediction_vs_actual_chart(results, best_model)
        if pred_fig:
            st.plotly_chart(pred_fig, use_container_width=True)
        
        # Análise de erros
        st.markdown("### 🔍 Análise de Erros")
        
        best_result = results[best_model]
        errors = best_result['actual'] - best_result['predictions']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma dos erros
            fig_errors = go.Figure(data=[
                go.Histogram(x=errors, nbinsx=30, name='Distribuição dos Erros')
            ])
            fig_errors.update_layout(
                title='Distribuição dos Erros de Previsão',
                xaxis_title='Erro (Real - Predição)',
                yaxis_title='Frequência',
                height=400
            )
            st.plotly_chart(fig_errors, use_container_width=True)
        
        with col2:
            # Estatísticas dos erros
            st.markdown("#### Estatísticas dos Erros")
            st.metric("Erro Médio", f"{errors.mean():,.2f}")
            st.metric("Desvio Padrão dos Erros", f"{errors.std():,.2f}")
            st.metric("Erro Máximo", f"{errors.max():,.2f}")
            st.metric("Erro Mínimo", f"{errors.min():,.2f}")
    
    elif analysis_type == "Clustering de Operações":
        st.markdown("## 🎯 Clustering de Operações")
        
        with st.spinner("Executando análise de clustering..."):
            clustering_result, cluster_analysis = perform_clustering_analysis(data)
        
        if clustering_result is None:
            st.error("❌ Não foi possível executar o clustering. Dados insuficientes.")
            return
        
        # Visualização dos clusters
        cluster_viz = create_clustering_visualization(clustering_result)
        if cluster_viz:
            st.plotly_chart(cluster_viz, use_container_width=True)
        
        # Análise dos clusters
        st.markdown("### 📊 Análise dos Clusters")
        
        if cluster_analysis is not None:
            # Reformatar dados da análise
            cluster_summary = cluster_analysis.reset_index()
            cluster_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in cluster_summary.columns]
            
            st.dataframe(cluster_summary, use_container_width=True)
        
        # Distribuição por cluster
        cluster_counts = clustering_result['cluster'].value_counts().sort_index()
        
        fig_cluster_dist = px.bar(
            x=cluster_counts.index,
            y=cluster_counts.values,
            title='Distribuição das Operações por Cluster',
            labels={'x': 'Cluster', 'y': 'Número de Operações'}
        )
        
        st.plotly_chart(fig_cluster_dist, use_container_width=True)
        
        # Características dos clusters
        st.markdown("### 🔍 Características dos Clusters")
        
        for cluster_id in sorted(clustering_result['cluster'].unique()):
            cluster_data = clustering_result[clustering_result['cluster'] == cluster_id]
            
            with st.expander(f"Cluster {cluster_id} ({len(cluster_data)} operações)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("CFEM Médio", f"R$ {cluster_data['CFEM'].mean():,.2f}")
                    st.metric("CFEM Total", f"R$ {cluster_data['CFEM'].sum():,.2f}")
                
                with col2:
                    if 'LONGITUDE' in cluster_data.columns:
                        st.metric("Longitude Média", f"{cluster_data['LONGITUDE'].mean():.4f}")
                    if 'LATITUDE' in cluster_data.columns:
                        st.metric("Latitude Média", f"{cluster_data['LATITUDE'].mean():.4f}")
    
    elif analysis_type == "Análise de Features":
        st.markdown("## 🔍 Análise de Features")
        
        # Preparar dados
        X, y, feature_names, label_encoders = prepare_ml_data(data)
        
        if X is None:
            st.error("❌ Não foi possível preparar os dados para análise de features.")
            return
        
        # Treinar modelos para análise de importância
        with st.spinner("Analisando importância das features..."):
            results, _, _, _ = train_regression_models(X, y)
        
        # Importância das features
        importance_fig = feature_importance_analysis(results, feature_names)
        if importance_fig:
            st.plotly_chart(importance_fig, use_container_width=True)
        
        # Correlação entre features e target
        st.markdown("### 📊 Correlação com CFEM")
        
        correlation_data = X.copy()
        correlation_data['CFEM'] = y
        
        correlations = correlation_data.corr()['CFEM'].drop('CFEM').sort_values(key=abs, ascending=False)
        
        fig_corr = go.Figure(data=[
            go.Bar(
                x=correlations.index,
                y=correlations.values,
                marker_color=['red' if x < 0 else 'blue' for x in correlations.values]
            )
        ])
        
        fig_corr.update_layout(
            title='Correlação das Features com CFEM',
            xaxis_title='Features',
            yaxis_title='Correlação',
            height=400
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Tabela de correlações
        corr_df = pd.DataFrame({
            'Feature': correlations.index,
            'Correlação': correlations.values,
            'Correlação Abs': abs(correlations.values)
        }).sort_values('Correlação Abs', ascending=False)
        
        st.dataframe(corr_df, use_container_width=True, hide_index=True)
    
    elif analysis_type == "Validação de Modelos":
        st.markdown("## ✅ Validação de Modelos")
        
        # Preparar dados
        X, y, feature_names, label_encoders = prepare_ml_data(data)
        
        if X is None:
            st.error("❌ Não foi possível preparar os dados para validação.")
            return
        
        # Parâmetros de validação
        st.sidebar.markdown("### Parâmetros de Validação")
        test_size = st.sidebar.slider("Tamanho do conjunto de teste:", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.sidebar.slider("Número de folds (CV):", 3, 10, 5)
        
        # Executar validação
        with st.spinner("Executando validação cruzada..."):
            # Split personalizado
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Modelos para validação
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression()
            }
            
            validation_results = {}
            
            for name, model in models.items():
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, 
                                          scoring='neg_mean_absolute_error')
                
                # Treinar no conjunto completo de treino e testar
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                test_mae = mean_absolute_error(y_test, y_pred)
                test_r2 = r2_score(y_test, y_pred)
                
                validation_results[name] = {
                    'cv_scores': -cv_scores,
                    'cv_mean': -cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_mae': test_mae,
                    'test_r2': test_r2
                }
        
        # Resultados da validação
        st.markdown("### 📊 Resultados da Validação Cruzada")
        
        val_data = []
        for model_name, results in validation_results.items():
            val_data.append({
                'Modelo': model_name,
                'CV MAE (média)': f"{results['cv_mean']:,.2f}",
                'CV MAE (desvio)': f"{results['cv_std']:.2f}",
                'Test MAE': f"{results['test_mae']:,.2f}",
                'Test R²': f"{results['test_r2']:.4f}"
            })
        
        val_df = pd.DataFrame(val_data)
        st.dataframe(val_df, use_container_width=True, hide_index=True)
        
        # Gráfico de box plot dos CV scores
        fig_cv = go.Figure()
        
        for model_name, results in validation_results.items():
            fig_cv.add_trace(
                go.Box(
                    y=results['cv_scores'],
                    name=model_name,
                    boxpoints='all'
                )
            )
        
        fig_cv.update_layout(
            title='Distribuição dos Scores de Validação Cruzada (MAE)',
            yaxis_title='MAE',
            height=400
        )
        
        st.plotly_chart(fig_cv, use_container_width=True)
        
        # Interpretação dos resultados
        st.markdown("### 💡 Interpretação dos Resultados")
        
        best_cv_model = min(validation_results.keys(), 
                           key=lambda x: validation_results[x]['cv_mean'])
        
        st.success(f"🏆 Melhor modelo (CV): **{best_cv_model}**")
        
        # Verificar overfitting
        for model_name, results in validation_results.items():
            cv_score = results['cv_mean']
            test_score = results['test_mae']
            
            if abs(cv_score - test_score) / cv_score > 0.2:  # Diferença > 20%
                st.warning(f"⚠️ {model_name}: Possível overfitting detectado (CV: {cv_score:.2f}, Test: {test_score:.2f})")
            else:
                st.info(f"✅ {model_name}: Performance consistente entre CV e teste")

if __name__ == "__main__":
    main()