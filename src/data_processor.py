import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import re
import logging

class CFEMDataProcessor:
    """
    Classe para processamento e limpeza de dados CFEM
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def load_excel_data(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Carrega dados de arquivo Excel
        
        Args:
            file_path: Caminho para o arquivo Excel
            sheet_name: Nome da planilha (opcional)
            
        Returns:
            DataFrame com os dados carregados
        """
        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
                
            self.logger.info(f"Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")
            return df
            
        except Exception as e:
            self.logger.error(f"Erro ao carregar dados: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        # 1) Remove linhas totalmente vazias
        df_clean = df_clean.dropna(how='all')

        # 2) Padroniza nomes
        df_clean.columns = df_clean.columns.str.strip().str.upper()

        # 3) Aliases de colunas comuns (garante nomes do schema)
        alias = {
            'UF': 'ESTADO',
            'ESTADO(S)': 'ESTADO',
            'MUNICIPIO': 'MUNICIPIO(S)',
            'MUNICÍPIO': 'MUNICIPIO(S)',
            'PRIMEIRO DE SUBS': 'PRIMEIRODESUBS',
            'SUBSTÂNCIA': 'PRIMEIRODESUBS',
            'SUBSTANCIA': 'PRIMEIRODESUBS'
        }
        df_clean = df_clean.rename(columns=alias)

        # 4) Texto → maiúsculas/coerência
        for col in ['TITULAR', 'MUNICIPIO(S)', 'ESTADO', 'PRIMEIRODESUBS']:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.upper()

        # 5) Trata vírgula decimal antes do to_numeric
        for col in ['LONGITUDE', 'LATITUDE', 'CFEM']:
            if col in df_clean.columns:
                df_clean[col] = (
                    df_clean[col]
                    .astype(str)
                    .str.replace(r'\.', '', regex=True)  # remove separador de milhar (se existir)
                    .str.replace(',', '.', regex=False)  # vírgula → ponto
                )
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

        # 6) Demais validações já existentes
        df_clean = self._clean_coordinates(df_clean)   # mantém
        df_clean = self._clean_cfem_values(df_clean)   # mantém
        df_clean = self._standardize_states(df_clean)  # mantém
        df_clean = self._standardize_companies(df_clean)  # mantém

        self.logger.info(f"Limpeza concluída: {df_clean.shape[0]} registros mantidos")
        return df_clean
        
        # Validar colunas essenciais
        required = ['TITULAR', 'MUNICIPIO(S)', 'ESTADO', 'PRIMEIRODESUBS', 'CFEM']
        missing = [c for c in required if c not in df_clean.columns]
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {missing}. Colunas encontradas: {df_clean.columns.tolist()}")
    
        # Limpar e padronizar dados de texto
        text_columns = ['TITULAR', 'MUNICIPIO(S)', 'ESTADO', 'PRIMEIRODESUBS']
        for col in text_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.strip().str.upper()
    
        # Validar e limpar coordenadas
        df_clean = self._clean_coordinates(df_clean)
    
        # Validar e limpar valores CFEM
        df_clean = self._clean_cfem_values(df_clean)
    
        # Padronizar nomes de estados
        df_clean = self._standardize_states(df_clean)
    
        # Padronizar nomes de empresas
        df_clean = self._standardize_companies(df_clean)
    
        self.logger.info(f"Limpeza concluída: {df_clean.shape[0]} registros mantidos")
        return df_clean    
        
    def _clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e valida coordenadas geográficas"""
        df_clean = df.copy()
        
        if 'LONGITUDE' in df_clean.columns and 'LATITUDE' in df_clean.columns:
            # Converter para numérico
            df_clean['LONGITUDE'] = pd.to_numeric(df_clean['LONGITUDE'], errors='coerce')
            df_clean['LATITUDE'] = pd.to_numeric(df_clean['LATITUDE'], errors='coerce')
            
            # Validar range de coordenadas para o Brasil
            valid_coords = (
                (df_clean['LONGITUDE'] >= -74) & (df_clean['LONGITUDE'] <= -32) &
                (df_clean['LATITUDE'] >= -34) & (df_clean['LATITUDE'] <= 6)
            )
            
            invalid_coords = ~valid_coords
            if invalid_coords.sum() > 0:
                self.logger.warning(f"Removendo {invalid_coords.sum()} registros com coordenadas inválidas")
                df_clean = df_clean[valid_coords]
        
        return df_clean
    
    def _clean_cfem_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e valida valores de CFEM"""
        df_clean = df.copy()
        
        if 'CFEM' in df_clean.columns:
            # Converter para numérico
            df_clean['CFEM'] = pd.to_numeric(df_clean['CFEM'], errors='coerce')
            
            # Remover valores negativos ou zero
            df_clean = df_clean[df_clean['CFEM'] > 0]
            
            # Identificar outliers (valores extremos)
            Q1 = df_clean['CFEM'].quantile(0.25)
            Q3 = df_clean['CFEM'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Flagging outliers (não removendo, apenas marcando)
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            df_clean['CFEM_OUTLIER'] = (
                (df_clean['CFEM'] < lower_bound) | 
                (df_clean['CFEM'] > upper_bound)
            )
        
        return df_clean
    
    def _standardize_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza siglas de estados"""
        df_clean = df.copy()
        
        if 'ESTADO' in df_clean.columns:
            # Dicionário de padronização de estados
            state_mapping = {
                'ACRE': 'AC', 'ALAGOAS': 'AL', 'AMAPÁ': 'AP', 'AMAZONAS': 'AM',
                'BAHIA': 'BA', 'CEARÁ': 'CE', 'DISTRITO FEDERAL': 'DF',
                'ESPÍRITO SANTO': 'ES', 'GOIÁS': 'GO', 'MARANHÃO': 'MA',
                'MATO GROSSO': 'MT', 'MATO GROSSO DO SUL': 'MS', 'MINAS GERAIS': 'MG',
                'PARÁ': 'PA', 'PARAÍBA': 'PB', 'PARANÁ': 'PR', 'PERNAMBUCO': 'PE',
                'PIAUÍ': 'PI', 'RIO DE JANEIRO': 'RJ', 'RIO GRANDE DO NORTE': 'RN',
                'RIO GRANDE DO SUL': 'RS', 'RONDÔNIA': 'RO', 'RORAIMA': 'RR',
                'SANTA CATARINA': 'SC', 'SÃO PAULO': 'SP', 'SERGIPE': 'SE',
                'TOCANTINS': 'TO'
            }
            
            # Aplicar mapeamento
            df_clean['ESTADO'] = df_clean['ESTADO'].replace(state_mapping)
            
            # Garantir que sejam siglas válidas
            valid_states = set(state_mapping.values())
            df_clean = df_clean[df_clean['ESTADO'].isin(valid_states)]
        
        return df_clean
    
    def _standardize_companies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Padroniza nomes de empresas"""
        df_clean = df.copy()
        
        if 'TITULAR' in df_clean.columns:
            # Remover caracteres especiais extras
            df_clean['TITULAR'] = df_clean['TITULAR'].str.replace(r'[^\w\s.-]', '', regex=True)
            
            # Padronizar sufixos empresariais
            suffixes = {
                r'\bLTDA\.?\b': 'LTDA',
                r'\bS\.?A\.?\b': 'S.A.',
                r'\bS\.?A\.?\s+LTDA\.?\b': 'S.A.',
                r'\bME\b': 'ME',
                r'\bEPP\b': 'EPP',
                r'\bEIRELI\b': 'EIRELI'
            }
            
            for pattern, replacement in suffixes.items():
                df_clean['TITULAR'] = df_clean['TITULAR'].str.replace(
                    pattern, replacement, regex=True
                )
        
        return df_clean
    
    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enriquece dados com informações calculadas
        
        Args:
            df: DataFrame limpo
            
        Returns:
            DataFrame enriquecido
        """
        df_enriched = df.copy()
        
        # Calcular faixas de valor CFEM
        df_enriched['CFEM_FAIXA'] = pd.cut(
            df_enriched['CFEM'],
            bins=[0, 10000, 100000, 1000000, 10000000, float('inf')],
            labels=['Até 10K', '10K-100K', '100K-1M', '1M-10M', 'Acima 10M']
        )
        
        # Calcular região geográfica
        df_enriched['REGIAO'] = df_enriched['ESTADO'].map(self._get_region_mapping())
        
        # Classificar porte da empresa baseado no CFEM
        df_enriched['PORTE_EMPRESA'] = self._classify_company_size(df_enriched['CFEM'])
        
        # Calcular densidade de operações por estado
        state_counts = df_enriched['ESTADO'].value_counts()
        df_enriched['DENSIDADE_ESTADO'] = df_enriched['ESTADO'].map(state_counts)
        
        # Diversificação por empresa (número de substâncias diferentes)
        company_diversity = df_enriched.groupby('TITULAR')['PRIMEIRODESUBS'].nunique()
        df_enriched['DIVERSIFICACAO_EMPRESA'] = df_enriched['TITULAR'].map(company_diversity)
        
        return df_enriched
    
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
    
    def _classify_company_size(self, cfem_values: pd.Series) -> pd.Series:
        """Classifica porte da empresa baseado no valor CFEM"""
        conditions = [
            cfem_values <= 50000,
            (cfem_values > 50000) & (cfem_values <= 500000),
            (cfem_values > 500000) & (cfem_values <= 5000000),
            cfem_values > 5000000
        ]
        
        choices = ['Pequena', 'Média', 'Grande', 'Muito Grande']
        
        return pd.Series(np.select(conditions, choices, default='Não Classificada'), 
                        index=cfem_values.index)
    
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calcula estatísticas descritivas dos dados
        
        Args:
            df: DataFrame processado
            
        Returns:
            Dicionário com estatísticas
        """
        stats = {}
        
        # Estatísticas gerais
        stats['total_registros'] = len(df)
        stats['total_empresas'] = df['TITULAR'].nunique()
        stats['total_estados'] = df['ESTADO'].nunique()
        stats['total_municipios'] = df['MUNICIPIO(S)'].nunique()
        stats['total_substancias'] = df['PRIMEIRODESUBS'].nunique()
        
        # Estatísticas CFEM
        stats['cfem_total'] = df['CFEM'].sum()
        stats['cfem_medio'] = df['CFEM'].mean()
        stats['cfem_mediano'] = df['CFEM'].median()
        stats['cfem_std'] = df['CFEM'].std()
        stats['cfem_min'] = df['CFEM'].min()
        stats['cfem_max'] = df['CFEM'].max()
        
        # Top empresas
        stats['top_empresas'] = df.groupby('TITULAR')['CFEM'].sum().nlargest(10).to_dict()
        
        # Top estados
        stats['top_estados'] = df.groupby('ESTADO')['CFEM'].sum().nlargest(10).to_dict()
        
        # Top substâncias
        stats['top_substancias'] = df.groupby('PRIMEIRODESUBS')['CFEM'].sum().nlargest(10).to_dict()
        
        # Concentração de mercado (HHI - Herfindahl-Hirschman Index)
        company_shares = df.groupby('TITULAR')['CFEM'].sum()
        total_cfem = company_shares.sum()
        market_shares = (company_shares / total_cfem) ** 2
        stats['hhi_empresas'] = market_shares.sum()
        
        return stats
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Valida qualidade dos dados
        
        Args:
            df: DataFrame a ser validado
            
        Returns:
            Relatório de qualidade dos dados
        """
        quality_report = {}
        
        # Completude dos dados
        quality_report['completude'] = {
            col: (df[col].notna().sum() / len(df)) * 100 
            for col in df.columns
        }
        
        # Duplicatas
        quality_report['duplicatas'] = {
            'total_duplicatas': df.duplicated().sum(),
            'duplicatas_por_empresa': df.duplicated(subset=['TITULAR']).sum(),
            'duplicatas_por_localizacao': df.duplicated(subset=['LONGITUDE', 'LATITUDE']).sum()
        }
        
        # Outliers
        if 'CFEM_OUTLIER' in df.columns:
            quality_report['outliers'] = {
                'total_outliers': df['CFEM_OUTLIER'].sum(),
                'percentual_outliers': (df['CFEM_OUTLIER'].sum() / len(df)) * 100
            }
        
        # Consistência geográfica
        if all(col in df.columns for col in ['LONGITUDE', 'LATITUDE', 'ESTADO']):
            quality_report['consistencia_geografica'] = self._validate_geographic_consistency(df)
        
        return quality_report
    
    def _validate_geographic_consistency(self, df: pd.DataFrame) -> Dict:
        """Valida consistência entre coordenadas e estados"""
        # Ranges aproximados de coordenadas por região
        region_bounds = {
            'Norte': {'lat_min': -9, 'lat_max': 5, 'lon_min': -74, 'lon_max': -44},
            'Nordeste': {'lat_min': -18, 'lat_max': -1, 'lon_min': -48, 'lon_max': -35},
            'Centro-Oeste': {'lat_min': -24, 'lat_max': -7, 'lon_min': -61, 'lon_max': -47},
            'Sudeste': {'lat_min': -25, 'lat_max': -14, 'lon_min': -50, 'lon_max': -39},
            'Sul': {'lat_min': -34, 'lat_max': -22, 'lon_min': -58, 'lon_max': -48}
        }
        
        inconsistencies = 0
        total_with_coords = 0
        
        for _, row in df.iterrows():
            if pd.notna(row['LONGITUDE']) and pd.notna(row['LATITUDE']):
                total_with_coords += 1
                region = self._get_region_mapping().get(row['ESTADO'])
                
                if region and region in region_bounds:
                    bounds = region_bounds[region]
                    if not (bounds['lat_min'] <= row['LATITUDE'] <= bounds['lat_max'] and
                            bounds['lon_min'] <= row['LONGITUDE'] <= bounds['lon_max']):
                        inconsistencies += 1
        
        return {
            'inconsistencias_geograficas': inconsistencies,
            'total_com_coordenadas': total_with_coords,
            'percentual_inconsistencias': (inconsistencies / total_with_coords * 100) if total_with_coords > 0 else 0

        }

