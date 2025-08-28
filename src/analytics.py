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

    # ======================================================
    # CLUSTERING
    # ======================================================
        
    def perform_clustering_analysis(
        self, df: pd.DataFrame, features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Executa KMeans, DBSCAN e PCA nos dados"""

        if features is None:
            features = ["CFEM", "LONGITUDE", "LATITUDE"]

        available_features = [f for f in features if f in df.columns]
        if not available_features:
            return {"error": "Nenhuma feature válida encontrada"}

        df_cluster = df[available_features + ["TITULAR", "ESTADO", "PRIMEIRODESUBS"]].dropna()
        if len(df_cluster) < 5:
            return {"error": "Dados insuficientes para clustering (mín. 5 registros)"}

        # Normalização
        numeric_features = df_cluster[available_features].select_dtypes(include=[np.number]).columns
        if len(numeric_features) == 0:
            return {"error": "Nenhuma feature numérica disponível"}

        X_scaled = self.scaler.fit_transform(df_cluster[numeric_features])

        return {
            "kmeans": self._perform_kmeans(X_scaled, df_cluster),
            "dbscan": self._perform_dbscan(X_scaled, df_cluster),
            "pca": self._perform_pca(X_scaled, df_cluster, numeric_features),
            "features_used": available_features,
            "data_shape": df_cluster.shape,
        }
    
    def _perform_kmeans(self, X: np.ndarray, df: pd.DataFrame) -> Dict:
        """Executa KMeans clustering com seleção automática de K"""

        max_k = min(10, len(X) - 1)
        if max_k < 2:
            return {"error": "Poucos registros para KMeans"}

        inertias, silhouette_scores = [], []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(
                silhouette_score(X, labels) if len(set(labels)) > 1 else 0
            )

        # Define k ótimo pelo "cotovelo"
        optimal_k = self._find_elbow_point(inertias) + 2
        optimal_k = min(optimal_k, max_k)

        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        df_result = df.copy()
        df_result["Cluster_KMeans"] = labels

        return {
            "optimal_k": optimal_k,
            "labels": labels,
            "silhouette_score": silhouette_score(X, labels) if len(set(labels)) > 1 else 0,
            "data_with_clusters": df_result,
        }
    
    def _perform_dbscan(self, X: np.ndarray, df: pd.DataFrame) -> Dict:
        """Executa DBSCAN clustering com ajuste automático de eps"""

        eps_values = np.arange(0.1, 2.0, 0.1)
        best_eps, best_score = 0.5, -1

        for eps in eps_values:
            labels = DBSCAN(eps=eps, min_samples=3).fit_predict(X)
            if len(set(labels)) > 1 and -1 not in labels:
                try:
                    score = silhouette_score(X, labels)
                    if score > best_score:
                        best_score, best_eps = score, eps
                except Exception:
                    continue

        dbscan = DBSCAN(eps=best_eps, min_samples=3)
        labels = dbscan.fit_predict(X)

        df_result = df.copy()
        df_result["Cluster_DBSCAN"] = labels

        return {
            "eps": best_eps,
            "labels": labels,
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "n_noise": list(labels).count(-1),
            "silhouette_score": best_score if best_score > -1 else 0,
            "data_with_clusters": df_result,
        }
    
    def _perform_pca(self, X: np.ndarray, df: pd.DataFrame, feature_names: List[str]) -> Dict:
        """Executa PCA para redução de dimensionalidade"""
        pca = PCA()
        X_pca = pca.fit_transform(X)

        return {
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
            "n_components_95": np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1,
            "components": pca.components_,
            "transformed_data": X_pca,
            "feature_names": feature_names,
        }
    
    def _find_elbow_point(self, inertias: List[float]) -> int:
        """Heurística simples para ponto de cotovelo"""
        if len(inertias) < 2:
            return 0
        diffs = np.diff(inertias)
        return int(np.argmin(diffs))
    

    # ======================================================
    # DETECÇÃO DE ANOMALIAS
    # ======================================================

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
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            return {"error": "Nenhuma feature válida encontrada"}
        
        df_anomaly = df[available_features].copy()
        df_anomaly = df_anomaly.dropna()
        
        if len(df_anomaly) < 5:
            return {"error": "Dados insuficientes para detecção de anomalias"}
        
        # Normalizar dados
        X_scaled = self.scaler.fit_transform(df_anomaly)
        
        # Isolation Forest
        contamination = min(0.1, 50/len(df_anomaly))  # Ajustar contaminação para datasets pequenos
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
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
                'cfem_mean_anomaly': anomalies['CFEM'].mean() if len(anomalies) > 0 and 'CFEM' in anomalies.columns else 0,
                'cfem_mean_normal': normal_points['CFEM'].mean() if 'CFEM' in normal_points.columns else 0,
                'top_anomalous_companies': anomalies.nlargest(5, 'CFEM')['TITULAR'].tolist() if len(anomalies) > 0 and 'CFEM' in anomalies.columns else []
            }
        }
        
        return {
            'model': iso_forest,
            'data_with_anomalies': df_result,
            'anomaly_analysis': anomaly_analysis,
            'features_used': available_features
        }
    
    # ======================================================
    # MODELO PREDITIVO
    # ======================================================

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
        if target not in df.columns:
            return {"error": f"Variável alvo '{target}' não encontrada"}
        
        if features is None:
            features = ['LONGITUDE', 'LATITUDE', 'ESTADO', 'PRIMEIRODESUBS']
        
        # Filtrar features disponíveis
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            return {"error": "Nenhuma feature válida encontrada"}
        
        # Preparar dados
        df_model = df[available_features + [target]].copy()
        df_model = df_model.dropna()
        
        if len(df_model) < 10:
            return {"error": "Dados insuficientes para modelagem"}
        
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
        test_size = min(0.3, 0.8)  # Ajustar para datasets pequenos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
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
    
    # ======================================================
    # TESTES ESTATÍSTICOS
    # ======================================================

    def perform_statistical_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Realiza testes estatísticos nos dados
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Resultados dos testes estatísticos
        """
        results = {}
        
        if 'CFEM' not in df.columns:
            return {"error": "Coluna CFEM não encontrada"}
        
        cfem_data = df['CFEM'].dropna()
        if len(cfem_data) < 3:
            return {"error": "Dados insuficientes para testes estatísticos"}
        
        # Teste de normalidade
        if len(cfem_data) < 5000:
            try:
                stat, p_value = stats.shapiro(cfem_data.sample(min(5000, len(cfem_data))))
                test_name = 'Shapiro-Wilk'
            except:
                stat, p_value = stats.normaltest(cfem_data)
                test_name = "D'Agostino"
        else:
            stat, p_value = stats.normaltest(cfem_data)
            test_name = "D'Agostino"
        
        results['normality_test'] = {
            'test': test_name,
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
        
        # Teste ANOVA entre estados
        if 'ESTADO' in df.columns:
            estados_cfem = [group['CFEM'].values for name, group in df.groupby('ESTADO') if len(group) > 1]
            if len(estados_cfem) > 1:
                try:
                    f_stat, p_value = stats.f_oneway(*estados_cfem)
                    results['anova_estados'] = {
                        'f_statistic': f_stat,
                        'p_value': p_value,
                        'significant_difference': p_value < 0.05
                    }
                except:
                    results['anova_estados'] = {"error": "Erro no teste ANOVA"}
        
        # Correlação entre variáveis numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            try:
                correlation_matrix = df[numeric_cols].corr()
                results['correlation_analysis'] = correlation_matrix
            except:
                results['correlation_analysis'] = {"error": "Erro no cálculo de correlação"}
        
        # Teste de Kruskal-Wallis para substâncias
        if 'PRIMEIRODESUBS' in df.columns:
            substancias_cfem = [group['CFEM'].values for name, group in df.groupby('PRIMEIRODESUBS') if len(group) > 1]
            if len(substancias_cfem) > 1:
                try:
                    h_stat, p_value = stats.kruskal(*substancias_cfem)
                    results['kruskal_substancias'] = {
                        'h_statistic': h_stat,
                        'p_value': p_value,
                        'significant_difference': p_value < 0.05
                    }
                except:
                    results['kruskal_substancias'] = {"error": "Erro no teste Kruskal-Wallis"}
        
        return results
    
    # ======================================================
    # CONCENTRAÇÃO DE MERCADO
    # ======================================================

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
        if groupby_col not in df.columns or 'CFEM' not in df.columns:
            return {"error": f"Coluna '{groupby_col}' ou 'CFEM' não encontrada"}
        
        # Calcular shares
        group_totals = df.groupby(groupby_col)['CFEM'].sum().sort_values(ascending=False)
        total_market = group_totals.sum()
        
        if total_market == 0:
            return {"error": "Total de mercado é zero"}
        
        market_shares = group_totals / total_market
        
        # Índice Herfindahl-Hirschman (HHI)
        hhi = (market_shares ** 2).sum()
        
        # Razão de concentração CR4 e CR8
        cr4 = market_shares.head(4).sum() if len(market_shares) >= 4 else market_shares.sum()
        cr8 = market_shares.head(8).sum() if len(market_shares) >= 8 else market_shares.sum()
        
        # Índice de Theil
        theil_index = -(market_shares * np.log(market_shares + 1e-10)).sum()
        
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
        if len(values) == 0:
            return 0.0
        
        values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(values)
        
        if cumsum[-1] == 0:
            return 0.0
        
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
            return {"error": f"Coluna de data '{date_col}' não encontrada"}
        
        if 'CFEM' not in df.columns:
            return {"error": "Coluna CFEM não encontrada"}
        
        try:
            # Converter para datetime
            df_ts = df.copy()
            df_ts[date_col] = pd.to_datetime(df_ts[date_col])
            df_ts = df_ts.sort_values(date_col)
            
            # Remover dados inválidos
            df_ts = df_ts.dropna(subset=[date_col, 'CFEM'])
            
            if len(df_ts) < 2:
                return {"error": "Dados insuficientes para análise temporal"}
            
            # Agrupar por período
            monthly_data = df_ts.groupby(df_ts[date_col].dt.to_period('M'))['CFEM'].agg(['sum', 'count', 'mean']).reset_index()
            monthly_data['period'] = monthly_data[date_col].astype(str)
            
            # Calcular tendências
            monthly_cfem = monthly_data['sum'].values
            if len(monthly_cfem) > 1:
                trend_slope = np.polyfit(range(len(monthly_cfem)), monthly_cfem, 1)[0]
            else:
                trend_slope = 0
            
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
                'volatility': monthly_data['sum'].std() / monthly_data['sum'].mean() if monthly_data['sum'].mean() > 0 else 0
            }
            
            return results
        
        except Exception as e:
            return {"error": f"Erro na análise temporal: {str(e)}"}
    
    def generate_insights_report(self, df: pd.DataFrame, 
                               clustering_results: Dict,
                               anomaly_results: Dict,
                               prediction_results: Dict,
                               concentration_results: Dict) -> Dict[str, Any]:
        """
        Gera relatório de insights baseado nas análises
        """
        insights = {}
        
        # Insights de Clustering
        if 'kmeans' in clustering_results and 'error' not in clustering_results['kmeans']:
            kmeans_data = clustering_results['kmeans']['data_with_clusters']
            dominant_cluster = kmeans_data['Cluster_KMeans'].mode().iloc[0] if len(kmeans_data) > 0 else 0
            
            insights['clustering'] = {
                'optimal_clusters': clustering_results['kmeans']['optimal_k'],
                'dominant_cluster': dominant_cluster,
                'cluster_characteristics': self._analyze_cluster_characteristics(kmeans_data)
            }
        
        # Insights de Anomalias
        if 'anomaly_analysis' in anomaly_results:
            anomaly_stats = anomaly_results['anomaly_analysis']
            insights['anomalies'] = {
                'anomaly_rate': anomaly_stats['anomaly_percentage'],
                'high_value_anomalies': anomaly_stats['anomaly_stats']['top_anomalous_companies'],
                'anomaly_impact': self._assess_anomaly_impact(anomaly_results.get('data_with_anomalies', pd.DataFrame()))
            }
        
        # Insights do Modelo Preditivo
        if 'metrics' in prediction_results:
            insights['predictions'] = {
                'model_accuracy': prediction_results['metrics']['test_r2'],
                'most_important_features': prediction_results['feature_importance'].head(3)['feature'].tolist() if 'feature_importance' in prediction_results else [],
                'prediction_reliability': self._assess_prediction_reliability(prediction_results['metrics'])
            }
        
        # Insights de Concentração
        if 'market_interpretation' in concentration_results:
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
        
        if 'Cluster_KMeans' not in df_clusters.columns:
            return characteristics
        
        for cluster_id in df_clusters['Cluster_KMeans'].unique():
            cluster_data = df_clusters[df_clusters['Cluster_KMeans'] == cluster_id]
            
            characteristics[f'cluster_{cluster_id}'] = {
                'size': len(cluster_data),
                'avg_cfem': cluster_data['CFEM'].mean() if 'CFEM' in cluster_data.columns else 0,
                'dominant_state': cluster_data['ESTADO'].mode().iloc[0] if 'ESTADO' in cluster_data.columns and len(cluster_data) > 0 else 'N/A',
                'dominant_substance': cluster_data['PRIMEIRODESUBS'].mode().iloc[0] if 'PRIMEIRODESUBS' in cluster_data.columns and len(cluster_data) > 0 else 'N/A',
                'geographic_spread': cluster_data['ESTADO'].nunique() if 'ESTADO' in cluster_data.columns else 0
            }
        
        return characteristics
    
    def _assess_anomaly_impact(self, df_anomalies: pd.DataFrame) -> str:
        """Avalia o impacto das anomalias"""
        if df_anomalies.empty or 'Is_Anomaly' not in df_anomalies.columns or 'CFEM' not in df_anomalies.columns:
            return "Não foi possível avaliar impacto"
        
        anomalies = df_anomalies[df_anomalies['Is_Anomaly']]
        
        if len(anomalies) == 0:
            return "Baixo impacto - poucas anomalias detectadas"
        
        total_cfem = df_anomalies['CFEM'].sum()
        if total_cfem == 0:
            return "Não foi possível calcular impacto"
        
        anomaly_cfem_share = anomalies['CFEM'].sum() / total_cfem
        
        if anomaly_cfem_share > 0.5:
            return "Alto impacto - anomalias representam mais de 50% do CFEM total"
        elif anomaly_cfem_share > 0.2:
            return "Médio impacto - anomalias representam 20-50% do CFEM total"
        else:
            return "Baixo impacto - anomalias representam menos de 20% do CFEM total"
    
    def _assess_prediction_reliability(self, metrics: Dict) -> str:
        """Avalia confiabilidade do modelo preditivo"""
        test_r2 = metrics.get('test_r2', 0)
        
        if test_r2 > 0.8:
            return "Alta confiabilidade"
        elif test_r2 > 0.6:
            return "Média confiabilidade"
        else:
            return "Baixa confiabilidade - modelo precisa ser refinado"
    
    def _assess_competition_level(self, concentration: Dict) -> str:
        """Avalia nível de competição do mercado"""
        hhi = concentration.get('hhi', 0)
        cr4 = concentration.get('cr4', 0)
        
        if hhi > 0.25 and cr4 > 0.6:
            return "Mercado oligopolizado - poucas empresas dominam"
        elif hhi > 0.15 or cr4 > 0.4:
            return "Mercado moderadamente concentrado"
        else:
            return "Mercado competitivo com boa distribuição"
    
    def _generate_geographic_insights(self, df: pd.DataFrame) -> Dict:
        """Gera insights geográficos"""
        if 'ESTADO' not in df.columns or 'CFEM' not in df.columns:
            return {"error": "Colunas necessárias não encontradas"}
        
        state_analysis = df.groupby('ESTADO').agg({
            'CFEM': ['sum', 'count', 'mean'],
            'TITULAR': 'nunique',
            'PRIMEIRODESUBS': 'nunique'
        })
        
        if state_analysis.empty:
            return {"error": "Não foi possível gerar insights geográficos"}
        
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
        if 'PRIMEIRODESUBS' not in df.columns or 'CFEM' not in df.columns:
            return {"error": "Colunas necessárias não encontradas"}
        
        substance_analysis = df.groupby('PRIMEIRODESUBS').agg({
            'CFEM': ['sum', 'mean'],
            'TITULAR': 'nunique'
        })
        
        if substance_analysis.empty:
            return {"error": "Não foi possível gerar insights por substância"}
        
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
        market_structure = insights.get('market_structure', {})
        if market_structure.get('concentration_level') == "Mercado altamente concentrado":
            recommendations.append("Considerar políticas antitruste para promover maior competição")
        
        # Recomendações baseadas em anomalias
        anomalies = insights.get('anomalies', {})
        if anomalies.get('anomaly_rate', 0) > 10:
            recommendations.append("Investigar operações anômalas para garantir compliance")
        
        # Recomendações geográficas
        geographic = insights.get('geographic', {})
        regional_dominance = geographic.get('regional_dominance', 0)
        if regional_dominance > 0.5:
            recommendations.append("Diversificar geograficamente para reduzir dependência regional")
        
        # Recomendações de clustering
        clustering = insights.get('clustering', {})
        if clustering.get('optimal_clusters', 0) > 5:
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
        import os
        
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
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
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
    
    def create_comprehensive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Cria análise abrangente dos dados CFEM
        
        Args:
            df: DataFrame com dados processados
            
        Returns:
            Dicionário com todas as análises
        """
        comprehensive_results = {}
        
        try:
            # 1. Análise de clustering
            comprehensive_results['clustering'] = self.perform_clustering_analysis(df)
            
            # 2. Detecção de anomalias
            comprehensive_results['anomalies'] = self.detect_anomalies(df)
            
            # 3. Modelo preditivo
            comprehensive_results['predictions'] = self.build_predictive_model(df)
            
            # 4. Concentração de mercado
            comprehensive_results['market_concentration'] = self.calculate_market_concentration(df)
            
            # 5. Testes estatísticos
            comprehensive_results['statistical_tests'] = self.perform_statistical_tests(df)
            
            # 6. Análise temporal (se dados disponíveis)
            if 'DATA' in df.columns:
                comprehensive_results['time_series'] = self.perform_time_series_analysis(df)
            
            # 7. Relatório de insights
            comprehensive_results['insights'] = self.generate_insights_report(
                df,
                comprehensive_results.get('clustering', {}),
                comprehensive_results.get('anomalies', {}),
                comprehensive_results.get('predictions', {}),
                comprehensive_results.get('market_concentration', {})
            )
            
        except Exception as e:
            comprehensive_results['error'] = f"Erro na análise abrangente: {str(e)}"
        
        return comprehensive_results
    
    def calculate_efficiency_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula métricas de eficiência das operações
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Métricas de eficiência
        """
        if 'CFEM' not in df.columns or 'TITULAR' not in df.columns:
            return {"error": "Colunas necessárias não encontradas"}
        
        # Eficiência por empresa
        company_efficiency = df.groupby('TITULAR').agg({
            'CFEM': ['sum', 'count', 'mean'],
            'PRIMEIRODESUBS': 'nunique',
            'ESTADO': 'nunique'
        }).reset_index()
        
        company_efficiency.columns = ['EMPRESA', 'CFEM_TOTAL', 'NUM_OPERACOES', 
                                     'CFEM_MEDIO', 'NUM_SUBSTANCIAS', 'NUM_ESTADOS']
        
        # Calcular índices de eficiência
        company_efficiency['EFICIENCIA_OPERACIONAL'] = (
            company_efficiency['CFEM_TOTAL'] / company_efficiency['NUM_OPERACOES']
        )
        
        company_efficiency['DIVERSIFICACAO_GEOGRAFICA'] = (
            company_efficiency['NUM_ESTADOS'] / company_efficiency['NUM_OPERACOES']
        )
        
        company_efficiency['DIVERSIFICACAO_SUBSTANCIAS'] = (
            company_efficiency['NUM_SUBSTANCIAS'] / company_efficiency['NUM_OPERACOES']
        )
        
        # Rankings
        top_by_efficiency = company_efficiency.nlargest(10, 'EFICIENCIA_OPERACIONAL')
        most_diversified_geo = company_efficiency.nlargest(10, 'DIVERSIFICACAO_GEOGRAFICA')
        most_diversified_sub = company_efficiency.nlargest(10, 'DIVERSIFICACAO_SUBSTANCIAS')
        
        return {
            'company_efficiency': company_efficiency,
            'top_by_efficiency': top_by_efficiency,
            'most_diversified_geographic': most_diversified_geo,
            'most_diversified_substances': most_diversified_sub,
            'efficiency_stats': {
                'avg_efficiency': company_efficiency['EFICIENCIA_OPERACIONAL'].mean(),
                'median_efficiency': company_efficiency['EFICIENCIA_OPERACIONAL'].median(),
                'efficiency_std': company_efficiency['EFICIENCIA_OPERACIONAL'].std()
            }
        }
    
    def analyze_competitive_positioning(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa posicionamento competitivo das empresas
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Análise de posicionamento competitivo
        """
        if 'TITULAR' not in df.columns or 'CFEM' not in df.columns:
            return {"error": "Colunas necessárias não encontradas"}
        
        # Análise por empresa
        company_analysis = df.groupby('TITULAR').agg({
            'CFEM': ['sum', 'count', 'mean'],
            'PRIMEIRODESUBS': ['nunique', lambda x: list(x.unique())],
            'ESTADO': ['nunique', lambda x: list(x.unique())]
        }).reset_index()
        
        company_analysis.columns = ['EMPRESA', 'CFEM_TOTAL', 'NUM_OPERACOES', 'CFEM_MEDIO',
                                   'NUM_SUBSTANCIAS', 'SUBSTANCIAS', 'NUM_ESTADOS', 'ESTADOS']
        
        # Calcular market share
        total_market = company_analysis['CFEM_TOTAL'].sum()
        company_analysis['MARKET_SHARE'] = (company_analysis['CFEM_TOTAL'] / total_market) * 100
        
        # Classificar empresas por porte
        def classify_company_size(market_share):
            if market_share >= 10:
                return 'Líder'
            elif market_share >= 5:
                return 'Grande'
            elif market_share >= 1:
                return 'Média'
            else:
                return 'Pequena'
        
        company_analysis['CATEGORIA_PORTE'] = company_analysis['MARKET_SHARE'].apply(classify_company_size)
        
        # Análise competitiva
        competitive_matrix = {
            'leaders': company_analysis[company_analysis['CATEGORIA_PORTE'] == 'Líder'],
            'challengers': company_analysis[company_analysis['CATEGORIA_PORTE'] == 'Grande'],
            'followers': company_analysis[company_analysis['CATEGORIA_PORTE'] == 'Média'],
            'nichers': company_analysis[company_analysis['CATEGORIA_PORTE'] == 'Pequena']
        }
        
        return {
            'company_analysis': company_analysis,
            'competitive_matrix': competitive_matrix,
            'market_structure': {
                'total_companies': len(company_analysis),
                'leaders': len(competitive_matrix['leaders']),
                'challengers': len(competitive_matrix['challengers']),
                'followers': len(competitive_matrix['followers']),
                'nichers': len(competitive_matrix['nichers'])
            }
        }
    
    def calculate_sustainability_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calcula métricas de sustentabilidade das operações
        
        Args:
            df: DataFrame com dados
            
        Returns:
            Métricas de sustentabilidade
        """
        sustainability_metrics = {}
        
        # Diversificação geográfica como proxy para sustentabilidade
        if 'ESTADO' in df.columns and 'TITULAR' in df.columns:
            geographic_diversification = df.groupby('TITULAR')['ESTADO'].nunique()
            sustainability_metrics['geographic_diversification'] = {
                'avg_states_per_company': geographic_diversification.mean(),
                'companies_single_state': (geographic_diversification == 1).sum(),
                'companies_multi_state': (geographic_diversification > 1).sum()
            }
        
        # Diversificação de substâncias
        if 'PRIMEIRODESUBS' in df.columns and 'TITULAR' in df.columns:
            substance_diversification = df.groupby('TITULAR')['PRIMEIRODESUBS'].nunique()
            sustainability_metrics['substance_diversification'] = {
                'avg_substances_per_company': substance_diversification.mean(),
                'companies_single_substance': (substance_diversification == 1).sum(),
                'companies_multi_substance': (substance_diversification > 1).sum()
            }
        
        # Distribuição de risco por região
        if 'REGIAO' in df.columns or 'ESTADO' in df.columns:
            region_col = 'REGIAO' if 'REGIAO' in df.columns else 'ESTADO'
            regional_distribution = df.groupby(region_col)['CFEM'].sum()
            total_cfem = regional_distribution.sum()
            regional_risk = (regional_distribution / total_cfem) ** 2
            
            sustainability_metrics['regional_risk'] = {
                'concentration_index': regional_risk.sum(),
                'risk_level': 'Alto' if regional_risk.sum() > 0.5 else 'Médio' if regional_risk.sum() > 0.25 else 'Baixo'
            }
        
        return sustainability_metrics
    
    def benchmark_companies(self, df: pd.DataFrame, target_company: str) -> Dict[str, Any]:
        """
        Realiza benchmarking de uma empresa específica
        
        Args:
            df: DataFrame com dados
            target_company: Nome da empresa alvo
            
        Returns:
            Análise de benchmarking
        """
        if 'TITULAR' not in df.columns:
            return {"error": "Coluna TITULAR não encontrada"}
        
        if target_company not in df['TITULAR'].values:
            return {"error": f"Empresa '{target_company}' não encontrada nos dados"}
        
        # Dados da empresa alvo
        target_data = df[df['TITULAR'] == target_company]
        
        # Métricas da empresa alvo
        target_metrics = {
            'cfem_total': target_data['CFEM'].sum() if 'CFEM' in df.columns else 0,
            'num_operations': len(target_data),
            'avg_cfem': target_data['CFEM'].mean() if 'CFEM' in df.columns else 0,
            'num_states': target_data['ESTADO'].nunique() if 'ESTADO' in df.columns else 0,
            'num_substances': target_data['PRIMEIRODESUBS'].nunique() if 'PRIMEIRODESUBS' in df.columns else 0
        }
        
        # Métricas do mercado
        market_metrics = {
            'cfem_total': df['CFEM'].sum() if 'CFEM' in df.columns else 0,
            'avg_cfem_per_company': df.groupby('TITULAR')['CFEM'].sum().mean() if 'CFEM' in df.columns else 0,
            'avg_operations_per_company': df.groupby('TITULAR').size().mean(),
            'avg_states_per_company': df.groupby('TITULAR')['ESTADO'].nunique().mean() if 'ESTADO' in df.columns else 0,
            'avg_substances_per_company': df.groupby('TITULAR')['PRIMEIRODESUBS'].nunique().mean() if 'PRIMEIRODESUBS' in df.columns else 0
        }
        
        # Cálculo de percentis
        company_totals = df.groupby('TITULAR')['CFEM'].sum() if 'CFEM' in df.columns else pd.Series()
        
        benchmarks = {}
        if not company_totals.empty:
            target_percentile = (company_totals < target_metrics['cfem_total']).mean() * 100
            benchmarks['cfem_percentile'] = target_percentile
        
        return {
            'target_company': target_company,
            'target_metrics': target_metrics,
            'market_metrics': market_metrics,
            'benchmarks': benchmarks,
            'performance_vs_market': {
                'cfem_vs_avg': (target_metrics['cfem_total'] / market_metrics['avg_cfem_per_company']) if market_metrics['avg_cfem_per_company'] > 0 else 0,
                'operations_vs_avg': target_metrics['num_operations'] / market_metrics['avg_operations_per_company'] if market_metrics['avg_operations_per_company'] > 0 else 0
            }
        }
