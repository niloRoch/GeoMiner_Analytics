import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import base64
from src import CFEMDataProcessor
from src import CFEMVisualizations
import json
import zipfile
import os

# Configuração da página
st.set_page_config(
    page_title="Configurações - CFEM Analytics",
    page_icon="⚙️",
    layout="wide"
)

def generate_pdf_report():
    """Gera relatório em PDF (simulação com HTML)"""
    if 'filtered_data' not in st.session_state:
        return None
    
    data = st.session_state.filtered_data
    stats = st.session_state.get('stats', {})
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Relatório CFEM Analytics</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ text-align: center; color: #1f4e79; margin-bottom: 30px; }}
            .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .section {{ margin: 30px 0; }}
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .footer {{ margin-top: 50px; text-align: center; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Relatório CFEM Analytics</h1>
            <p>Relatório gerado em {datetime.now().strftime('%d/%m/%Y às %H:%M')}</p>
        </div>
        
        <div class="section">
            <h2>Resumo Executivo</h2>
            <div class="metric">
                <h3>Total CFEM</h3>
                <p>R$ {data['CFEM'].sum():,.2f}</p>
            </div>
            <div class="metric">
                <h3>Empresas</h3>
                <p>{data['TITULAR'].nunique():,}</p>
            </div>
            <div class="metric">
                <h3>Estados</h3>
                <p>{data['ESTADO'].nunique():,}</p>
            </div>
            <div class="metric">
                <h3>Substâncias</h3>
                <p>{data['PRIMEIRODESUBS'].nunique():,}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Top 10 Empresas</h2>
            <table>
                <tr>
                    <th>Empresa</th>
                    <th>CFEM Total (R$)</th>
                    <th>Número de Operações</th>
                </tr>
    """
    
    # Top empresas
    top_empresas = data.groupby('TITULAR').agg({
        'CFEM': ['sum', 'count']
    }).nlargest(10, ('CFEM', 'sum'))
    
    for idx, row in top_empresas.iterrows():
        cfem_total = row[('CFEM', 'sum')]
        num_ops = row[('CFEM', 'count')]
        html_content += f"""
                <tr>
                    <td>{idx}</td>
                    <td>R$ {cfem_total:,.2f}</td>
                    <td>{num_ops}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>Distribuição por Estado</h2>
            <table>
                <tr>
                    <th>Estado</th>
                    <th>CFEM Total (R$)</th>
                    <th>Número de Empresas</th>
                </tr>
    """
    
    # Por estado
    por_estado = data.groupby('ESTADO').agg({
        'CFEM': 'sum',
        'TITULAR': 'nunique'
    }).nlargest(15, 'CFEM')
    
    for estado, row in por_estado.iterrows():
        html_content += f"""
                <tr>
                    <td>{estado}</td>
                    <td>R$ {row['CFEM']:,.2f}</td>
                    <td>{row['TITULAR']}</td>
                </tr>
        """
    
    html_content += """
            </table>
        </div>
        
        <div class="footer">
            <p>Relatório gerado pelo CFEM Analytics Dashboard</p>
        </div>
    </body>
    </html>
    """
    
    return html_content

def create_download_link(content, filename, text="Download"):
    """Cria link de download para conteúdo"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}">{text}</a>'
    return href

def export_data_to_excel():
    """Exporta dados para Excel"""
    if 'filtered_data' not in st.session_state:
        return None
    
    data = st.session_state.filtered_data
    stats = st.session_state.get('stats', {})
    
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Dados principais
        data.to_excel(writer, sheet_name='Dados_Principais', index=False)
        
        # Resumo por empresa
        resumo_empresa = data.groupby('TITULAR').agg({
            'CFEM': ['sum', 'mean', 'count'],
            'PRIMEIRODESUBS': 'nunique',
            'ESTADO': 'nunique'
        }).round(2)
        resumo_empresa.to_excel(writer, sheet_name='Resumo_Empresas')
        
        # Resumo por estado
        resumo_estado = data.groupby('ESTADO').agg({
            'CFEM': ['sum', 'mean', 'count'],
            'TITULAR': 'nunique',
            'PRIMEIRODESUBS': 'nunique'
        }).round(2)
        resumo_estado.to_excel(writer, sheet_name='Resumo_Estados')
        
        # Resumo por substância
        resumo_substancia = data.groupby('PRIMEIRODESUBS').agg({
            'CFEM': ['sum', 'mean', 'count'],
            'TITULAR': 'nunique',
            'ESTADO': 'nunique'
        }).round(2)
        resumo_substancia.to_excel(writer, sheet_name='Resumo_Substancias')
        
        # Estatísticas gerais
        if stats:
            stats_df = pd.DataFrame([stats])
            stats_df.to_excel(writer, sheet_name='Estatisticas_Gerais', index=False)
    
    return output.getvalue()

def create_data_dictionary():
    """Cria dicionário de dados"""
    dictionary = {
        "CFEM Analytics - Dicionário de Dados": {
            "TITULAR": "Nome da empresa titular da operação minerária",
            "MUNICIPIO(S)": "Município(s) onde a operação está localizada",
            "ESTADO": "Estado brasileiro (sigla de 2 letras)",
            "LONGITUDE": "Coordenada de longitude (graus decimais)",
            "LATITUDE": "Coordenada de latitude (graus decimais)",
            "CFEM": "Valor da Compensação Financeira pela Exploração de Recursos Minerais (R$)",
            "PRIMEIRODESUBS": "Principal substância mineral explorada",
            "REGIAO": "Região geográfica brasileira (Norte, Nordeste, Centro-Oeste, Sudeste, Sul)",
            "CFEM_FAIXA": "Faixa de valor CFEM categorizada",
            "PORTE_EMPRESA": "Classificação do porte da empresa baseada no valor CFEM",
            "DENSIDADE_ESTADO": "Número de operações no estado",
            "DIVERSIFICACAO_EMPRESA": "Número de substâncias diferentes exploradas pela empresa"
        },
        "Faixas de Valor CFEM": {
            "Até 10K": "Operações com CFEM até R$ 10.000",
            "10K-100K": "Operações com CFEM entre R$ 10.001 e R$ 100.000",
            "100K-1M": "Operações com CFEM entre R$ 100.001 e R$ 1.000.000",
            "1M-10M": "Operações com CFEM entre R$ 1.000.001 e R$ 10.000.000",
            "Acima 10M": "Operações com CFEM acima de R$ 10.000.000"
        },
        "Porte das Empresas": {
            "Pequena": "CFEM até R$ 50.000",
            "Média": "CFEM entre R$ 50.001 e R$ 500.000",
            "Grande": "CFEM entre R$ 500.001 e R$ 5.000.000",
            "Muito Grande": "CFEM acima de R$ 5.000.000"
        }
    }
    
    return dictionary

def main():
    """Função principal da página"""
    
    st.title("⚙️ Configurações e Exportação")
    st.markdown("Configurações do sistema, exportação de dados e relatórios")
    
    # Verificar se os dados estão disponíveis
    if 'filtered_data' not in st.session_state or st.session_state.filtered_data is None:
        st.warning("⚠️ Nenhum dado carregado. Por favor, carregue os dados na página principal.")
        return
    
    data = st.session_state.filtered_data
    stats = st.session_state.get('stats', {})
    
    # Tabs principais
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Exportação de Dados", 
        "📋 Relatórios", 
        "🔧 Configurações", 
        "📖 Documentação",
        "ℹ️ Informações do Sistema"
    ])
    
    with tab1:
        st.markdown("## 📊 Exportação de Dados")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Exportar Dados Principais")
            
            # Opções de formato
            export_format = st.selectbox(
                "Formato de exportação:",
                ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"]
            )
            
            # Opções de filtros para exportação
            st.markdown("#### Opções de Exportação:")
            
            include_raw = st.checkbox("Incluir dados brutos originais", value=False)
            include_processed = st.checkbox("Incluir dados processados", value=True)
            include_summaries = st.checkbox("Incluir resumos por categoria", value=True)
            
            # Botão de exportação
            if st.button("🎪 Gerar Arquivo de Exportação", type="primary"):
                with st.spinner("Preparando exportação..."):
                    if export_format == "Excel (.xlsx)":
                        excel_data = export_data_to_excel()
                        if excel_data:
                            st.download_button(
                                label="📥 Download Excel",
                                data=excel_data,
                                file_name=f"cfem_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            st.success("✅ Arquivo Excel gerado com sucesso!")
                    
                    elif export_format == "CSV (.csv)":
                        csv_data = data.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_data,
                            file_name=f"cfem_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        st.success("✅ Arquivo CSV gerado com sucesso!")
                    
                    elif export_format == "JSON (.json)":
                        json_data = data.to_json(orient='records', indent=2, force_ascii=False)
                        st.download_button(
                            label="📥 Download JSON",
                            data=json_data,
                            file_name=f"cfem_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        st.success("✅ Arquivo JSON gerado com sucesso!")
        
        with col2:
            st.markdown("### Estatísticas da Exportação")
            
            st.metric("Total de Registros", f"{len(data):,}")
            st.metric("Colunas Disponíveis", f"{len(data.columns):,}")
            st.metric("Tamanho Estimado", f"{data.memory_usage().sum() / 1024 / 1024:.2f} MB")
            
            # Prévia dos dados a serem exportados
            st.markdown("#### Prévia dos Dados:")
            st.dataframe(data.head(), use_container_width=True)
    
    with tab2:
        st.markdown("## 📋 Relatórios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Relatório Executivo")
            st.markdown("Relatório completo com KPIs, análises e insights principais.")
            
            if st.button("📄 Gerar Relatório HTML", type="primary"):
                with st.spinner("Gerando relatório..."):
                    html_report = generate_pdf_report()
                    if html_report:
                        st.download_button(
                            label="📥 Download Relatório HTML",
                            data=html_report,
                            file_name=f"relatorio_cfem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                            mime="text/html"
                        )
                        st.success("✅ Relatório HTML gerado com sucesso!")
            
            st.markdown("### Relatório de Qualidade dos Dados")
            
            if st.button("🔍 Gerar Relatório de Qualidade"):
                quality_report = st.session_state.get('quality_report', {})
                if quality_report:
                    quality_html = f"""
                    <h2>Relatório de Qualidade dos Dados</h2>
                    <h3>Completude dos Dados</h3>
                    <ul>
                    """
                    
                    for col, completude in quality_report.get('completude', {}).items():
                        quality_html += f"<li>{col}: {completude:.1f}%</li>"
                    
                    quality_html += """
                    </ul>
                    <h3>Duplicatas</h3>
                    """
                    
                    duplicatas = quality_report.get('duplicatas', {})
                    quality_html += f"<p>Total de duplicatas: {duplicatas.get('total_duplicatas', 0)}</p>"
                    
                    st.download_button(
                        label="📥 Download Relatório de Qualidade",
                        data=quality_html,
                        file_name=f"qualidade_dados_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
        
        with col2:
            st.markdown("### Tipos de Relatório Disponíveis")
            
            report_types = [
                ("📊 Dashboard Executivo", "KPIs principais e métricas de performance"),
                ("📈 Análise Temporal", "Evolução dos dados ao longo do tempo"),
                ("🌍 Análise Geográfica", "Distribuição espacial das operações"),
                ("🏢 Análise por Empresa", "Performance e ranking das empresas"),
                ("🔨 Análise por Substância", "Distribuição e valor por substância mineral"),
                ("🔍 Relatório de Qualidade", "Avaliação da qualidade dos dados"),
                ("🤖 Resultados de ML", "Resultados dos modelos de machine learning")
            ]
            
            for titulo, descricao in report_types:
                st.markdown(f"**{titulo}**")
                st.markdown(f"*{descricao}*")
                st.markdown("---")
    
    with tab3:
        st.markdown("## 🔧 Configurações do Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Configurações de Visualização")
            
            # Configurações de tema
            theme = st.selectbox(
                "Tema dos gráficos:",
                ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
            )
            
            # Configurações de cor
            color_palette = st.selectbox(
                "Paleta de cores:",
                ["Viridis", "Plasma", "Inferno", "Set3", "Pastel1", "Dark2"]
            )
            
            # Configurações de performance
            st.markdown("### Configurações de Performance")
            
            max_points_map = st.slider(
                "Máximo de pontos em mapas:",
                min_value=100,
                max_value=10000,
                value=5000,
                step=500
            )
            
            cache_timeout = st.slider(
                "Timeout do cache (minutos):",
                min_value=5,
                max_value=60,
                value=30,
                step=5
            )
            
            # Salvar configurações
            if st.button("💾 Salvar Configurações"):
                config = {
                    'theme': theme,
                    'color_palette': color_palette,
                    'max_points_map': max_points_map,
                    'cache_timeout': cache_timeout,
                    'saved_at': datetime.now().isoformat()
                }
                
                st.session_state.config = config
                st.success("✅ Configurações salvas com sucesso!")
        
        with col2:
            st.markdown("### Configurações Avançadas")
            
            # Configurações de análise
            st.markdown("#### Parâmetros de Análise")
            
            outlier_method = st.selectbox(
                "Método de detecção de outliers:",
                ["IQR", "Z-Score", "Percentil", "Isolation Forest"]
            )
            
            clustering_algorithm = st.selectbox(
                "Algoritmo de clustering:",
                ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"]
            )
            
            # Configurações de ML
            st.markdown("#### Machine Learning")
            
            test_size = st.slider(
                "Tamanho do conjunto de teste (%):",
                min_value=10,
                max_value=40,
                value=20,
                step=5
            )
            
            cv_folds = st.slider(
                "Número de folds (validação cruzada):",
                min_value=3,
                max_value=10,
                value=5
            )
            
            # Reset de dados
            st.markdown("### Gerenciamento de Dados")
            
            if st.button("🔄 Limpar Cache", type="secondary"):
                # Limpar cache do Streamlit
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("✅ Cache limpo com sucesso!")
            
            if st.button("❌ Remover Dados da Sessão", type="secondary"):
                # Limpar dados da sessão
                keys_to_remove = ['data', 'filtered_data', 'stats', 'quality_report']
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.success("✅ Dados removidos da sessão!")
                st.rerun()
    
    with tab4:
        st.markdown("## 📖 Documentação")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dicionário de Dados")
            
            dictionary = create_data_dictionary()
            
            for section, items in dictionary.items():
                st.markdown(f"#### {section}")
                
                if isinstance(items, dict):
                    for key, value in items.items():
                        st.markdown(f"**{key}**: {value}")
                else:
                    st.markdown(items)
                
                st.markdown("---")
            
            # Download do dicionário
            dict_json = json.dumps(dictionary, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download Dicionário (JSON)",
                data=dict_json,
                file_name="dicionario_dados_cfem.json",
                mime="application/json"
            )
        
        with col2:
            st.markdown("### Guia de Uso")
            
            usage_guide = """
            #### Como usar o CFEM Analytics
            
            1. **Carregamento de Dados**
               - Carregue um arquivo Excel na página principal
               - Aguarde o processamento automático
               - Verifique a qualidade dos dados
            
            2. **Dashboard Executivo**
               - Visualize KPIs principais
               - Analise top empresas e distribuições
               - Obtenha insights automáticos
            
            3. **Análises Estatísticas**
               - Explore distribuições e correlações
               - Execute testes estatísticos
               - Identifique outliers
            
            4. **Análises Geoespaciais**
               - Visualize mapas interativos
               - Execute clustering espacial
               - Analise hotspots e acessibilidade
            
            5. **Machine Learning**
               - Treine modelos preditivos
               - Execute clustering de operações
               - Analise importância das features
            
            6. **Exportação**
               - Exporte dados em múltiplos formatos
               - Gere relatórios executivos
               - Configure o sistema
            """
            
            st.markdown(usage_guide)
            
            st.markdown("### FAQ - Perguntas Frequentes")
            
            with st.expander("Como interpretar os valores de CFEM?"):
                st.markdown("""
                O CFEM (Compensação Financeira pela Exploração de Recursos Minerais) 
                é um tributo devido pelos mineradores ao governo brasileiro. 
                Valores maiores indicam maior atividade minerária ou maior valor 
                dos recursos extraídos.
                """)
            
            with st.expander("O que significam os clusters espaciais?"):
                st.markdown("""
                Clusters espaciais são agrupamentos de operações minerárias 
                geograficamente próximas. Eles podem indicar regiões de alta 
                atividade minerária, jazidas importantes ou infraestrutura compartilhada.
                """)
            
            with st.expander("Como interpretar os modelos de ML?"):
                st.markdown("""
                Os modelos de machine learning tentam prever valores de CFEM 
                baseados em características como localização, empresa e substância. 
                O R² indica a qualidade do modelo (quanto mais próximo de 1, melhor).
                """)
    
    with tab5:
        st.markdown("## ℹ️ Informações do Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Status do Sistema")
            
            # Informações da sessão
            if 'filtered_data' in st.session_state:
                data_info = st.session_state.filtered_data
                st.success(f"✅ Dados carregados: {len(data_info):,} registros")
                st.info(f"📊 Colunas disponíveis: {len(data_info.columns)}")
                
                memory_usage = data_info.memory_usage(deep=True).sum() / 1024 / 1024
                st.info(f"💾 Uso de memória: {memory_usage:.2f} MB")
            else:
                st.warning("⚠️ Nenhum dado carregado")
            
            # Status do cache
            st.markdown("### Cache do Sistema")
            st.info("🔄 Cache ativo para melhor performance")
            
            # Configurações atuais
            if 'config' in st.session_state:
                config = st.session_state.config
                st.markdown("### Configurações Atuais")
                st.json(config)
            else:
                st.info("⚙️ Usando configurações padrão")
        
        with col2:
            st.markdown("### Informações Técnicas")
            
            tech_info = {
                "Versão": "1.0.0",
                "Framework": "Streamlit",
                "Análise Geoespacial": "Folium, GeoPandas",
                "Machine Learning": "Scikit-learn",
                "Visualizações": "Plotly, Matplotlib",
                "Processamento": "Pandas, NumPy"
            }
            
            for key, value in tech_info.items():
                st.metric(key, value)
            
            st.markdown("### Contato e Suporte")
            st.markdown("""
            - 📧 **Email**: support@cfemanalytics.com
            - 🌐 **Website**: www.cfemanalytics.com
            - 📚 **Documentação**: docs.cfemanalytics.com
            - 🐛 **Reportar Bugs**: github.com/cfem-analytics/issues
            """)
            
            st.markdown("### Licença e Termos")
            st.markdown("""
            Este software é fornecido sob licença MIT. 
            Os dados CFEM são de domínio público, 
            conforme legislação brasileira.
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px; margin-top: 20px;'>
        CFEM Analytics Dashboard v1.0 | Desenvolvido para análise de dados minerários brasileiros
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
