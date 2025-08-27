import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CFEMAnalytics:
    """
    Classe para análises avançadas e machine learning dos dados CFEM
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        
    def perform_clustering_analysis(self, df: pd.DataFrame, 
                                  features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Realiza análise de clustering nas operações minerárias
        
        Args:
            df: DataFrame com dados processados
            features: Lista de features para clustering (opcional)
            
        Returns:
            Resultados do clustering
        """
        if features is None:
            features = ['CFEM', 'LONGITUDE', 'LATITUDE']
        
        # Preparar dados
        df_cluster = df[features + ['TITULAR', 'ESTADO', 'PRIMEIRODESUBS']].copy()
        df_cluster = df_cluster.dropna()
        
        # Normalizar features numéricas
        numeric_features = df_cluster[features].select_dtypes(include=[np.number]).columns
        X_scaled = self.scaler.fit_transform(df_cluster[numeric_features])
        
        # K-Means Clustering
        kmeans_results = self._perform_kmeans_clustering(X_scaled, df_cluster)
        
        # DBSCAN Clustering
        dbscan_results = self._perform_dbscan_clustering(X_scaled, df_cluster)
        
        # Análise de componentes principais
        pca_results = self._perform_pca_analysis(X_scaled, df_cluster)
        
        return {
            'kmeans': kmeans_results,
            'dbscan': dbscan_results,
            'pca': pca_results,
            'features_used': features,
            'data_shape': df_cluster.shape
        }
    
    def _perform_kmeans_clustering(self, X: np.ndarray, df: pd.DataFrame) -> Dict:
        """Executa K-Means clustering"""
        results = {}
        
        # Encontrar número ótimo de clusters
        k_range = range(2, min(11, len(X)//2))
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            if k < len(X):
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)
                inertias.append(kmeans.inertia_)
                if len(set(labels)) > 1:
                    silhouette_scores.append(silhouette_score(X, labels))
                else:
                    silhouette_scores.append(0)
        
        # Usar cotovelo para determinar K ótimo
        optimal_k = self._find_elbow_point(inertias) + 2  # +2 porque k_range começa em 2
        
        # Executar clustering com K ótimo
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        
        # Adicionar labels ao dataframe
        df_result = df.copy()
        df_result['Cluster_KMeans'] = cluster_labels
        
        # Analisar clusters
        cluster_analysis = df_result.groupby('Cluster_KMeans').agg({
            'CFEM': ['count', 'mean', 'sum'],
            'TITULAR': 'nunique',
            'ESTADO': 'nunique',
            'PRIMEIRODESUBS': 'nunique'
        }).round(2)
        
        results = {
            'optimal_k': optimal_k,
            'labels': cluster_labels,
            'centroids': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_score(X, cluster_labels) if len(set(cluster_labels)) > 1 else 0,
            'cluster_analysis': cluster_analysis,
            'data_with_clusters': df_result
        }
        
        return results
    
    def _perform_dbscan_clustering(self, X: np.ndarray, df: pd.DataFrame) -> Dict:
        """Executa DBSCAN clustering"""
        # Testar diferentes valores de eps
        eps_values = np.arange(0.1, 2.0, 0.1)
        best_eps = 0.5
        best_score = -1
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=3)
            labels = dbscan.fit_predict(X)
            if len(set(labels)) > 1 and -1 not in labels:
                score = silhouette_score(X, labels)
                if score > best_score:
                    best_score = score
                    best_eps = eps
        
        # Executar DBSCAN com melhor eps
        dbscan = DBSCAN(eps=best_eps, min_samples=3)
        cluster_labels = dbscan.fit_predict(X)
        
        # Adicionar labels ao dataframe
        df_result = df.copy()
        df_result['Cluster_DBSCAN'] = cluster_labels
        
        # Analisar clusters
        cluster_analysis = df_result[df_result['Cluster_DBSCAN'] != -1].groupby('Cluster_DBSCAN').agg({
            'CFEM': ['count', 'mean', 'sum'],
            'TITULAR': 'nunique',
            'ESTADO': 'nunique',
            'PRIMEIRODESUBS': 'nunique'
        }).round(2)
        
        results = {
            'eps': best_eps,
            'labels': cluster_labels,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_noise': list(cluster_labels).count(-1),
            'silhouette_score': best_score,
            'cluster_analysis': cluster_analysis,
            'data_with_clusters': df_result
        }
        
        return results
    
    def _perform_pca_analysis(self, X: np.ndarray, df: pd.DataFrame) -> Dict:
        """Executa análise de componentes principais"""
        pca = PCA()
        X_pca = pca.fit_transform(X)
        
        # Calcular variância explicada
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        
        # Determinar número de componentes para 95% da variância
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        results = {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components_95': n_components_95,
            'components': pca.components_,
            'transformed_data': X_pca,
            'original_features': df.columns.tolist()
        }
        
        return results
    
    def detect_anomalies(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Detecta anomalias nos dados usando Isolation Forest
        
        Args:
            df: DataFrame com dados
            features: Features para detecção (opcional)
            
        Returns:
            Resultados da detecção de anomalias
        """
        if features is None:
            features = ['CFEM', 'LONGITUDE', 'LATITUDE']
        
        # Preparar dados
        df_anomaly = df[features].copy()
        df_anomaly = df_anomaly.dropna()
        
        # Normalizar dados
        X_scaled = self.scaler.fit_transform(df_anomaly)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        anomaly_scores = iso_forest.score_samples(X_scaled)
        
        # Adicionar resultados ao dataframe
        df_result = df.iloc[df_anomaly.index].copy()
        df_result['Anomaly_Label'] = anomaly_labels
        df_result['Anomaly_Score'] = anomaly_scores
        df_result['Is_Anomaly'] = anomaly_labels == -1
        
        # Analisar anomalias
        anomalies = df_result[df_result['Is_Anomaly']]
        normal_points = df_result[~df_result['Is_Anomaly']]
        
        anomaly_analysis = {
            'total_anomalies': len(anomalies),
            'anomaly_percentage': (len(anomalies) / len(df_result)) * 100,
            'anomaly_stats': {
                'cfem_mean_anomaly': anomalies['CFEM'].mean() if len(anomalies) > 0 else 0,
                'cfem_mean_normal': normal_points['CFEM'].mean(),
                'top_anomalous_companies': anomalies.nlargest(5, 'CFEM')['TITULAR'].tolist() if len(anomalies) > 0 else []
            }
        }
        
        return {
            'model': iso_forest,
            'data_with_anomalies': df_result,
            'anomaly_analysis': anomaly_analysis,
            'features_used': features
        }
    
    def build_predictive_model(self, df: pd.DataFrame, 
                             target: str = 'CFEM',
                             features: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Constrói modelo preditivo para valores CFEM
        
        Args:
            df: DataFrame com dados
            target: Variável alvo
            features: Features para o modelo
            
        Returns:
            Resultados do modelo preditivo
        """
        if features is None:
            features = ['LONGITUDE', 'LATITUDE', 'ESTADO', 'PRIMEIRODESUBS']
        
        # Preparar dados
        df_model = df[features + [target]].copy()
        df_model = df_model.dropna()
        
        # Encoding de variáveis categóricas
        categorical_features = df_model.select_dtypes(include=['object']).columns
        df_encoded = df_model.copy()
        
        for col in categorical_features:
            if col != target:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
        
        # Separar features e target
        X = df_encoded.drop(columns=[target])
        y = df_encoded[target]
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Treinar modelo Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Predições
        y_pred_train = rf_model.predict(X_train)
        y_pred_test = rf_model.predict(X_test)
        
        # Métricas
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Salvar modelo
        self.models['cfem_predictor'] = rf_model
        
        results = {
            'model': rf_model,
            'metrics': {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            },
            'feature_importance': feature_importance,
            'predictions': {
                'y_test': y_test,
                'y_pred_test': y_pred_test
            },
            'data_split': {
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        }
        
        return results
    
    def perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Realiza testes estatísticos nos dados
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Resultados dos testes estatísticos
        """
        results = {}
        
        # Teste de normalidade (Shapiro-Wilk para amostras pequenas, D'Agostino para grandes)
        if len(df) < 5000:
            stat, p_value = stats.shapiro(df['CFEM'])
            test_name = 'Shapiro-Wilk'
        else:
            stat, p_value = stats.normaltest(df['CFEM'])
            test_name = "D'Agostino"
        
        results['normality_test'] = {
            'test': test_name,
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
        
        # Teste ANOVA entre estados
        estados_cfem = [group['CFEM'].values for name, group in df.groupby('Estado') if len(group) > 1]
        if len(estados_cfem) > 1:
            f_stat, p_value = stats.f_oneway(*estados_cfem)
            results['anova_estados'] = {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant_difference': p_value < 0.05
            }
        
        # Correlação entre variáveis numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            results['correlation_analysis'] = correlation_matrix
        
        # Teste de Kruskal-Wallis para substâncias (não paramétrico)
        substancias_cfem = [group['CFEM'].values for name, group in df.groupby('PRIMEIRODESUBS') if len(group) > 1]
        if len(substancias_cfem) > 1:
            h_stat, p_value = stats.kruskal(*substancias_cfem)
            results['kruskal_substancias'] = {
                'h_statistic': h_stat,
                'p_value': p_value,
                'significant_difference': p_value < 0.05
            }
        
        return results
    
    def calculate_market_concentration(self, df: pd.DataFrame, 
                                     groupby_col: str = 'TITULAR') -> Dict[str, float]:
        """
        Calcula índices de concentração de mercado
        
        Args:
            df: DataFrame com dados
            groupby_col: Coluna para agrupar (empresa, estado, etc.)
            
        Returns:
            Índices de concentração
        """
        # Calcular shares
        group_totals = df.groupby(groupby_col)['CFEM'].sum().sort_values(ascending=False)
        total_market = group_totals.sum()
        market_shares = group_totals / total_market
        
        # Índice Herfindahl-Hirschman (HHI)
        hhi = (market_shares ** 2).sum()
        
        # Razão de concentração CR4 e CR8
        cr4 = market_shares.head(4).sum() if len(market_shares) >= 4 else market_shares.sum()
        cr8 = market_shares.head(8).sum() if len(market_shares) >= 8 else market_shares.sum()
        
        # Índice de Theil
        theil_index = -(market_shares * np.log(market_shares)).sum()
        
        # Coeficiente de Gini
        gini = self._calculate_gini_coefficient(group_totals.values)
        
        return {
            'hhi': hhi,
            'cr4': cr4,
            'cr8': cr8,
            'theil_index': theil_index,
            'gini_coefficient': gini,
            'market_interpretation': self._interpret_market_concentration(hhi)
        }
    
    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calcula o coeficiente de Gini"""
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
    
    def _interpret_market_concentration(self, hhi: float) -> str:
        """Interpreta o índice HHI"""
        if hhi < 0.15:
            return "Mercado não concentrado"
        elif hhi < 0.25:
            return "Mercado moderadamente concentrado"
        else:
            return "Mercado altamente concentrado"
    
    def perform_time_series_analysis(self, df: pd.DataFrame, 
                                   date_col: str = 'DATA') -> Dict[str, Any]:
        """
        Realiza análise de séries temporais (se dados de data disponíveis)
        
        Args:
            df: DataFrame com dados
            date_col: Coluna com datas
            
        Returns:
            Análise temporal
        """
        if date_col not in df.columns:
            return {"error": "Coluna de data não encontrada"}
        
        # Converter para datetime
        df_ts = df.copy()
        df_ts[date_col] = pd.to_datetime(df_ts[date_col])
        df_ts = df_ts.sort_values(date_col)
        
        # Agrupar por período
        monthly_data = df_ts.groupby(df_ts[date_col].dt.to_period('M'))['CFEM'].agg(['sum', 'count', 'mean']).reset_index()
        monthly_data['period'] = monthly_data[date_col].astype(str)
        
        # Calcular tendências
        monthly_cfem = monthly_data['sum'].values
        trend_slope = np.polyfit(range(len(monthly_cfem)), monthly_cfem, 1)[0]
        
        # Sazonalidade (simplificada)
        df_ts['month'] = df_ts[date_col].dt.month
        seasonal_pattern = df_ts.groupby('month')['CFEM'].mean()
        
        # Growth rate
        monthly_data['growth_rate'] = monthly_data['sum'].pct_change() * 100
        
        results = {
            'monthly_data': monthly_data,
            'trend_slope': trend_slope,
            'seasonal_pattern': seasonal_pattern.to_dict(),
            'average_growth_rate': monthly_data['growth_rate'].mean(),
            'volatility': monthly_data['sum'].std() / monthly_data['sum'].mean()
        }
        
        return results
    
    def generate_insights_report(self, df: pd.DataFrame, 
                               clustering_results: Dict,
                               anomaly_results: Dict,
                               prediction_results: Dict,
                               concentration_results: Dict) -> Dict[str, Any]:
        """
        Gera relatório de insights baseado nas análises
        
        Args:
            df: DataFrame original
            clustering_results: Resultados do clustering
            anomaly_results: Resultados de detecção de anomalias
            prediction_results: Resultados do modelo preditivo
            concentration_results: Resultados de concentração
            
        Returns:
            Relatório de insights
        """
        insights = {}
        
        # Insights de Clustering
        if 'kmeans' in clustering_results:
            kmeans_data = clustering_results['kmeans']['data_with_clusters']
            dominant_cluster = kmeans_data['Cluster_KMeans'].mode().iloc[0]
            cluster_analysis = clustering_results['kmeans']['cluster_analysis']
            
            insights['clustering'] = {
                'optimal_clusters': clustering_results['kmeans']['optimal_k'],
                'dominant_cluster': dominant_cluster,
                'cluster_characteristics': self._analyze_cluster_characteristics(kmeans_data)
            }
        
        # Insights de Anomalias
        anomaly_stats = anomaly_results['anomaly_analysis']
        insights['anomalies'] = {
            'anomaly_rate': anomaly_stats['anomaly_percentage'],
            'high_value_anomalies': anomaly_stats['anomaly_stats']['top_anomalous_companies'],
            'anomaly_impact': self._assess_anomaly_impact(anomaly_results['data_with_anomalies'])
        }
        
        # Insights do Modelo Preditivo
        insights['predictions'] = {
            'model_accuracy': prediction_results['metrics']['test_r2'],
            'most_important_features': prediction_results['feature_importance'].head(3)['feature'].tolist(),
            'prediction_reliability': self._assess_prediction_reliability(prediction_results['metrics'])
        }
        
        # Insights de Concentração
        insights['market_structure'] = {
            'concentration_level': concentration_results['market_interpretation'],
            'top_player_dominance': concentration_results['cr4'],
            'market_inequality': concentration_results['gini_coefficient'],
            'competitive_assessment': self._assess_competition_level(concentration_results)
        }
        
        # Insights Geográficos
        insights['geographic'] = self._generate_geographic_insights(df)
        
        # Insights por Substância
        insights['substances'] = self._generate_substance_insights(df)
        
        # Recomendações Estratégicas
        insights['recommendations'] = self._generate_strategic_recommendations(insights)
        
        return insights
    
    def _analyze_cluster_characteristics(self, df_clusters: pd.DataFrame) -> Dict:
        """Analisa características dos clusters"""
        characteristics = {}
        
        for cluster_id in df_clusters['Cluster_KMeans'].unique():
            cluster_data = df_clusters[df_clusters['Cluster_KMeans'] == cluster_id]
            
            characteristics[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_cfem': cluster_data['CFEM'].mean(),
                'dominant_state': cluster_data['ESTADO'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                'dominant_substance': cluster_data['PRIMEIRODESUBS'].mode().iloc[0] if len(cluster_data) > 0 else 'N/A',
                'geographic_spread': cluster_data['ESTADO'].nunique()
            }
        
        return characteristics
    
    def _assess_anomaly_impact(self, df_anomalies: pd.DataFrame) -> str:
        """Avalia o impacto das anomalias"""
        anomalies = df_anomalies[df_anomalies['Is_Anomaly']]
        normal = df_anomalies[~df_anomalies['Is_Anomaly']]
        
        if len(anomalies) == 0:
            return "Baixo impacto - poucas anomalias detectadas"
        
        anomaly_cfem_share = anomalies['CFEM'].sum() / df_anomalies['CFEM'].sum()
        
        if anomaly_cfem_share > 0.5:
            return "Alto impacto - anomalias representam mais de 50% do CFEM total"
        elif anomaly_cfem_share > 0.2:
            return "Médio impacto - anomalias representam 20-50% do CFEM total"
        else:
            return "Baixo impacto - anomalias representam menos de 20% do CFEM total"
    
    def _assess_prediction_reliability(self, metrics: Dict) -> str:
        """Avalia confiabilidade do modelo preditivo"""
        test_r2 = metrics['test_r2']
        
        if test_r2 > 0.8:
            return "Alta confiabilidade"
        elif test_r2 > 0.6:
            return "Média confiabilidade"
        else:
            return "Baixa confiabilidade - modelo precisa ser refinado"
    
    def _assess_competition_level(self, concentration: Dict) -> str:
        """Avalia nível de competição do mercado"""
        hhi = concentration['hhi']
        cr4 = concentration['cr4']
        
        if hhi > 0.25 and cr4 > 0.6:
            return "Mercado oligopolizado - poucas empresas dominam"
        elif hhi > 0.15 or cr4 > 0.4:
            return "Mercado moderadamente concentrado"
        else:
            return "Mercado competitivo com boa distribuição"
    
    def _generate_geographic_insights(self, df: pd.DataFrame) -> Dict:
        """Gera insights geográficos"""
        state_analysis = df.groupby('ESTADO').agg({
            'CFEM': ['sum', 'count', 'mean'],
            'TITULAR': 'nunique',
            'PRIMEIRODESUBS': 'nunique'
        })
        
        top_state = state_analysis['CFEM']['sum'].idxmax()
        most_diverse_state = state_analysis['PRIMEIRODESUBS']['nunique'].idxmax()
        
        return {
            'leading_state': top_state,
            'most_diverse_state': most_diverse_state,
            'geographic_concentration': len(df['ESTADO'].unique()),
            'regional_dominance': state_analysis['CFEM']['sum'].max() / state_analysis['CFEM']['sum'].sum()
        }
    
    def _generate_substance_insights(self, df: pd.DataFrame) -> Dict:
        """Gera insights por substância"""
        substance_analysis = df.groupby('PRIMEIRODESUBS').agg({
            'CFEM': ['sum', 'mean'],
            'TITULAR': 'nunique'
        })
        
        top_substance = substance_analysis['CFEM']['sum'].idxmax()
        highest_value_substance = substance_analysis['CFEM']['mean'].idxmax()
        
        return {
            'most_valuable_substance': top_substance,
            'highest_average_value': highest_value_substance,
            'substance_diversity': len(df['PRIMEIRODESUBS'].unique()),
            'concentration_by_substance': substance_analysis['CFEM']['sum'].max() / substance_analysis['CFEM']['sum'].sum()
        }
    
    def _generate_strategic_recommendations(self, insights: Dict) -> List[str]:
        """Gera recomendações estratégicas baseadas nos insights"""
        recommendations = []
        
        # Recomendações baseadas na concentração
        if insights['market_structure']['concentration_level'] == "Mercado altamente concentrado":
            recommendations.append("Considerar políticas antitruste para promover maior competição")
        
        # Recomendações baseadas em anomalias
        if insights['anomalies']['anomaly_rate'] > 10:
            recommendations.append("Investigar operações anômalas para garantir compliance")
        
        # Recomendações geográficas
        regional_dominance = insights['geographic']['regional_dominance']
        if regional_dominance > 0.5:
            recommendations.append("Diversificar geograficamente para reduzir dependência regional")
        
        # Recomendações de clustering
        if insights['clustering']['optimal_clusters'] > 5:
            recommendations.append("Segmentar estratégias por clusters identificados")
        
        return recommendations
    
    def _find_elbow_point(self, inertias: List[float]) -> int:
        """Encontra o ponto do cotovelo para determinar número ótimo de clusters"""
        if len(inertias) < 3:
            return 0
        
        # Calcular a segunda derivada
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_derivative = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_derivative)
        
        # Encontrar o ponto com maior segunda derivada (cotovelo)
        if second_derivatives:
            return np.argmax(second_derivatives) + 1
        else:
            return len(inertias) // 2
    
    def export_analysis_results(self, results: Dict, output_path: str = "analysis_results.json"):
        """
        Exporta resultados das análises para arquivo JSON
        
        Args:
            results: Dicionário com resultados
            output_path: Caminho para salvar arquivo
        """
        import json
        
        # Converter numpy arrays para listas para serialização JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return obj
        
        # Limpar resultados para JSON
        cleaned_results = {}
        for key, value in results.items():
            if key not in ['model', 'models']:  # Excluir objetos modelo
                cleaned_results[key] = self._clean_for_json(value, convert_numpy)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_results, f, indent=2, ensure_ascii=False, default=convert_numpy)
    
    def _clean_for_json(self, obj, converter):
        """Limpa objetos recursivamente para serialização JSON"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v, converter) for k, v in obj.items() 
                   if k not in ['model', 'models']}
        elif isinstance(obj, (list, tuple)):
            return [self._clean_for_json(item, converter) for item in obj]
        else:
            return converter(obj)