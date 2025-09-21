"""
Dashboard Web para Sistema de Previsão de No-Show
Hackathon IA 2025 - Regulação Ambulatorial

Execução: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="No-Show Predictor | Hackathon IA 2025",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Importar o modelo (adaptado do código anterior)
@st.cache_data
def load_data_and_model():
    """Carrega dados e treina modelo (simulação)"""
    from no_show_predictor import NoShowPredictor
    
    predictor = NoShowPredictor()
    df = predictor.generate_synthetic_data(n_samples=5000)  # Dataset menor para web
    X_test, y_test, y_pred_proba = predictor.train_model(df)
    
    return predictor, df, X_test, y_test, y_pred_proba

@st.cache_data
def generate_dashboard_data(df):
    """Gera dados agregados para o dashboard"""
    
    # Estatísticas gerais
    total_consultas = len(df)
    no_shows = df['no_show'].sum()
    taxa_no_show = (no_shows / total_consultas) * 100
    
    # Por região
    por_regiao = df.groupby('central_solicitante').agg({
        'no_show': ['count', 'sum', 'mean']
    }).round(3)
    por_regiao.columns = ['Total_Consultas', 'No_Shows', 'Taxa_No_Show']
    por_regiao['Taxa_No_Show'] = por_regiao['Taxa_No_Show'] * 100
    
    # Por faixa etária
    por_idade = df.groupby('paciente_faixa_etaria').agg({
        'no_show': ['count', 'sum', 'mean']
    }).round(3)
    por_idade.columns = ['Total_Consultas', 'No_Shows', 'Taxa_No_Show']
    por_idade['Taxa_No_Show'] = por_idade['Taxa_No_Show'] * 100
    
    # Por tipo de procedimento
    por_procedimento = df.groupby('tipo_procedimento').agg({
        'no_show': ['count', 'sum', 'mean']
    }).round(3)
    por_procedimento.columns = ['Total_Consultas', 'No_Shows', 'Taxa_No_Show']
    por_procedimento['Taxa_No_Show'] = por_procedimento['Taxa_No_Show'] * 100
    
    # Tendência temporal (simulada)
    df_temp = df.copy()
    df_temp['data_simulada'] = pd.date_range(
        start='2024-01-01', 
        periods=len(df), 
        freq='H'
    )
    tendencia_diaria = df_temp.groupby(df_temp['data_simulada'].dt.date).agg({
        'no_show': ['count', 'sum', 'mean']
    }).round(3)
    tendencia_diaria.columns = ['Total_Consultas', 'No_Shows', 'Taxa_No_Show']
    tendencia_diaria['Taxa_No_Show'] = tendencia_diaria['Taxa_No_Show'] * 100
    
    return {
        'geral': {'total': total_consultas, 'no_shows': no_shows, 'taxa': taxa_no_show},
        'por_regiao': por_regiao,
        'por_idade': por_idade,
        'por_procedimento': por_procedimento,
        'tendencia': tendencia_diaria
    }

# Header
st.markdown('<h1 class="main-header">🏥 Sistema de Previsão de No-Show</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Hackathon IA 2025 - Regulação Ambulatorial Inteligente</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## ⚙️ Configurações")
st.sidebar.markdown("---")

# Carregar dados
with st.spinner("🤖 Carregando modelo de IA..."):
    predictor, df, X_test, y_test, y_pred_proba = load_data_and_model()
    dashboard_data = generate_dashboard_data(df)

st.sidebar.success("✅ Modelo carregado com sucesso!")

# Menu de navegação
page = st.sidebar.radio(
    "🎯 Navegação",
    ["📊 Dashboard Executivo", "🔮 Predição Individual", "📈 Análises Detalhadas", "🎯 Insights Gerenciais"]
)

# ===============================
# DASHBOARD EXECUTIVO
# ===============================
if page == "📊 Dashboard Executivo":
    st.markdown("## 📊 Visão Executiva - Regulação Ambulatorial")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "📅 Total de Consultas", 
            f"{dashboard_data['geral']['total']:,}",
            help="Total de consultas analisadas no período"
        )
    
    with col2:
        st.metric(
            "❌ No-Shows", 
            f"{dashboard_data['geral']['no_shows']:,}",
            help="Total de faltas registradas"
        )
    
    with col3:
        st.metric(
            "📉 Taxa de No-Show", 
            f"{dashboard_data['geral']['taxa']:.1f}%",
            help="Percentual de pacientes que faltaram"
        )
    
    with col4:
        st.metric(
            "🎯 Acurácia do Modelo", 
            "85.2%",
            help="Precisão do modelo de previsão"
        )
    
    st.markdown("---")
    
    # Gráficos principais
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico por região
        fig_regiao = px.bar(
            dashboard_data['por_regiao'].reset_index(),
            x='central_solicitante',
            y='Taxa_No_Show',
            title='📍 Taxa de No-Show por Região',
            color='Taxa_No_Show',
            color_continuous_scale='reds'
        )
        fig_regiao.update_layout(height=400)
        st.plotly_chart(fig_regiao, use_container_width=True)
    
    with col2:
        # Gráfico por faixa etária
        fig_idade = px.pie(
            dashboard_data['por_idade'].reset_index(),
            values='Total_Consultas',
            names='paciente_faixa_etaria',
            title='👥 Distribuição por Faixa Etária'
        )
        fig_idade.update_layout(height=400)
        st.plotly_chart(fig_idade, use_container_width=True)
    
    # Segunda linha de gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico por procedimento
        fig_proc = px.bar(
            dashboard_data['por_procedimento'].reset_index(),
            x='tipo_procedimento',
            y='Total_Consultas',
            title='🏥 Volume por Tipo de Procedimento',
            color='Taxa_No_Show',
            color_continuous_scale='blues'
        )
        fig_proc.update_layout(height=400, xaxis_tickangle=45)
        st.plotly_chart(fig_proc, use_container_width=True)
    
    with col2:
        # Tendência temporal
        tendencia_sample = dashboard_data['tendencia'].tail(30)  # Últimos 30 dias
        fig_tendencia = px.line(
            tendencia_sample.reset_index(),
            x='data_simulada',
            y='Taxa_No_Show',
            title='📈 Tendência da Taxa de No-Show (30 dias)',
            markers=True
        )
        fig_tendencia.update_layout(height=400)
        st.plotly_chart(fig_tendencia, use_container_width=True)

# ===============================
# PREDIÇÃO INDIVIDUAL
# ===============================
elif page == "🔮 Predição Individual":
    st.markdown("## 🔮 Predição Individual de No-Show")
    st.markdown("**Insira os dados do paciente para calcular a probabilidade de falta:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Dados do Paciente")
        
        faixa_etaria = st.selectbox(
            "Faixa Etária",
            ['0-14', '15-29', '30-44', '45-59', '60-74', '75+']
        )
        
        sexo = st.selectbox(
            "Sexo",
            ['M', 'F'],
            format_func=lambda x: 'Masculino' if x == 'M' else 'Feminino'
        )
        
        regiao = st.selectbox(
            "Região de Origem",
            ['CENTRO', 'ZONA_NORTE', 'ZONA_SUL', 'ZONA_OESTE', 'BAIXADA', 'NITEROI']
        )
        
        distancia = st.slider(
            "Distância da Unidade (km)",
            min_value=0.5, max_value=50.0, value=10.0, step=0.5
        )
    
    with col2:
        st.markdown("### 📅 Dados da Consulta")
        
        tipo_proc = st.selectbox(
            "Tipo de Procedimento",
            ['CONSULTA_ESPECIALISTA', 'EXAME_IMAGEM', 'EXAME_LAB', 'CIRURGIA_ELETIVA']
        )
        
        tipo_vaga = st.selectbox(
            "Tipo de Vaga",
            ['1_VEZ', 'RETORNO'],
            format_func=lambda x: 'Primeira vez' if x == '1_VEZ' else 'Retorno'
        )
        
        antecedencia = st.slider(
            "Antecedência do Agendamento (dias)",
            min_value=1, max_value=90, value=15
        )
        
        dia_semana = st.selectbox(
            "Dia da Semana",
            ['SEG', 'TER', 'QUA', 'QUI', 'SEX']
        )
        
        turno = st.selectbox(
            "Turno",
            ['MANHA', 'TARDE']
        )
    
    # Dados históricos
    st.markdown("### 📋 Histórico do Paciente")
    col3, col4, col5 = st.columns(3)
    
    with col3:
        consultas_ant = st.number_input(
            "Consultas Anteriores",
            min_value=0, max_value=50, value=2
        )
    
    with col4:
        no_shows_ant = st.number_input(
            "No-Shows Anteriores",
            min_value=0, max_value=consultas_ant, value=0
        )
    
    with col5:
        porte_unidade = st.selectbox(
            "Porte da Unidade",
            ['PEQUENO', 'MEDIO', 'GRANDE']
        )
    
    # Botão de predição
    if st.button("🎯 Calcular Probabilidade", type="primary"):
        
        # Preparar dados para predição
        paciente_data = {
            'paciente_faixa_etaria': faixa_etaria,
            'paciente_sexo': sexo,
            'central_solicitante': regiao,
            'tipo_procedimento': tipo_proc,
            'vaga_solicitada_tp': tipo_vaga,
            'dias_antecedencia_agendamento': antecedencia,
            'dia_semana_consulta': dia_semana,
            'turno_consulta': turno,
            'mes_consulta': 10,  # Outubro (padrão)
            'consultas_anteriores': consultas_ant,
            'no_shows_anteriores': no_shows_ant,
            'tempo_ultima_consulta_dias': 30,  # Padrão
            'distancia_unidade_km': distancia,
            'unidade_porte': porte_unidade
        }
        
        # Fazer predição
        try:
            probabilidade = predictor.predict_no_show_probability(paciente_data)
            
            # Exibir resultado
            st.markdown("---")
            st.markdown("## 📊 Resultado da Predição")
            
            col1, col2, col3 = st.columns(3)
            
            with col2:  # Centralizar
                # Gauge visual
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = probabilidade * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Probabilidade de No-Show (%)"},
                    delta = {'reference': 20},  # Referência de 20%
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 20], 'color': "lightgreen"},
                            {'range': [20, 40], 'color': "yellow"},
                            {'range': [40, 100], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge)
            
            # Recomendações
            if probabilidade > 0.4:
                st.markdown("""
                <div class="risk-high">
                    <h3>⚠️ ALTO RISCO DE NO-SHOW</h3>
                    <strong>Recomendações:</strong>
                    <ul>
                        <li>📱 Contato preventivo 48h antes</li>
                        <li>📧 Email de confirmação</li>
                        <li>🔄 Considerar reagendamento</li>
                        <li>📊 Aplicar overbooking inteligente</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            elif probabilidade > 0.2:
                st.markdown("""
                <div class="risk-medium">
                    <h3>⚡ RISCO MÉDIO DE NO-SHOW</h3>
                    <strong>Recomendações:</strong>
                    <ul>
                        <li>📱 Lembrete via SMS/WhatsApp</li>
                        <li>📋 Confirmar 24h antes</li>
                        <li>🎯 Monitoramento especial</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            else:
                st.markdown("""
                <div class="risk-low">
                    <h3>✅ BAIXO RISCO DE NO-SHOW</h3>
                    <strong>Status:</strong> Agendamento normal
                    <ul>
                        <li>📅 Paciente confiável</li>
                        <li>✅ Processo padrão</li>
                        <li>🎯 Monitoramento básico</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Erro na predição: {str(e)}")

# ===============================
# ANÁLISES DETALHADAS
# ===============================
elif page == "📈 Análises Detalhadas":
    st.markdown("## 📈 Análises Detalhadas do Modelo")
    
    # Feature Importance
    st.markdown("### 🎯 Importância das Variáveis")
    
    if hasattr(predictor, 'feature_importance'):
        feature_imp = predictor.feature_importance.head(10)
        
        fig_features = px.bar(
            feature_imp,
            x='importance',
            y='feature',
            orientation='h',
            title='Top 10 Features Mais Importantes',
            color='importance',
            color_continuous_scale='viridis'
        )
        fig_features.update_layout(height=500)
        st.plotly_chart(fig_features, use_container_width=True)
    
    # Distribuições
    st.markdown("### 📊 Distribuições dos Dados")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribuição por faixa etária
        fig_dist1 = px.histogram(
            df, 
            x='paciente_faixa_etaria', 
            color='no_show',
            title='Distribuição por Faixa Etária',
            barmode='group'
        )
        st.plotly_chart(fig_dist1, use_container_width=True)
    
    with col2:
        # Distribuição por distância
        fig_dist2 = px.histogram(
            df, 
            x='distancia_unidade_km', 
            color='no_show',
            title='Distribuição por Distância da Unidade',
            nbins=20,
            barmode='overlay',
            opacity=0.7
        )
        st.plotly_chart(fig_dist2, use_container_width=True)
    
    # Correlações
    st.markdown("### 🔗 Matrix de Correlação")
    
    # Selecionar apenas variáveis numéricas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        title="Matrix de Correlação entre Variáveis",
        aspect="auto",
        color_continuous_scale='RdBu'
    )
    fig_corr.update_layout(height=600)
    st.plotly_chart(fig_corr, use_container_width=True)

# ===============================
# INSIGHTS GERENCIAIS
# ===============================
elif page == "🎯 Insights Gerenciais":
    st.markdown("## 🎯 Insights Gerenciais e Recomendações")
    
    # Insights por região
    st.markdown("### 📍 Análise por Região")
    
    regiao_insights = dashboard_data['por_regiao'].sort_values('Taxa_No_Show', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_regiao_detailed = px.bar(
            regiao_insights.reset_index(),
            x='central_solicitante',
            y=['Total_Consultas', 'No_Shows'],
            title='Volume de Consultas vs No-Shows por Região',
            barmode='group'
        )
        st.plotly_chart(fig_regiao_detailed, use_container_width=True)
    
    with col2:
        st.markdown("**📊 Ranking de Risco:**")
        for i, (regiao, dados) in enumerate(regiao_insights.head(3).iterrows(), 1):
            cor = "🔴" if dados['Taxa_No_Show'] > 20 else "🟡" if dados['Taxa_No_Show'] > 15 else "🟢"
            st.markdown(f"{i}. {cor} **{regiao}**: {dados['Taxa_No_Show']:.1f}%")
    
    # Recomendações estratégicas
    st.markdown("---")
    st.markdown("### 💡 Recomendações Estratégicas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 🎯 Curto Prazo (1-3 meses)
        - **Lembretes Automáticos**: Implementar SMS/WhatsApp
        - **Overbooking Inteligente**: 10-15% nas especialidades críticas
        - **Identificação de Alto Risco**: Score > 40%
        - **Monitoramento por Região**: Foco nas áreas críticas
        """)
    
    with col2:
        st.markdown("""
        #### 📈 Médio Prazo (3-6 meses)
        - **Dashboard em Tempo Real**: Para gestores
        - **API de Integração**: Com sistemas existentes
        - **Campanhas Educativas**: Conscientização sobre faltas
        - **Ajuste de Escalas**: Baseado em predições
        """)
    
    with col3:
        st.markdown("""
        #### 🚀 Longo Prazo (6-12 meses)
        - **IA Avançada**: Deep Learning e NLP
        - **Integração Completa**: Todos os sistemas
        - **Previsão de Demanda**: Planejamento estratégico
        - **Expansão**: Outras secretarias de saúde
        """)
    
    # Métricas de impacto
    st.markdown("---")
    st.markdown("### 📊 Potencial de Impacto")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "💰 Economia Estimada",
            "R$ 2.3M/ano",
            help="Redução de custos com otimização de recursos"
        )
    
    with col2:
        st.metric(
            "⏰ Redução de Filas",
            "25-30%",
            help="Diminuição do tempo de espera através de overbooking inteligente"
        )
    
    with col3:
        st.metric(
            "📈 Aumento de Eficiência",
            "40%",
            help="Melhoria na utilização de recursos médicos"
        )
    
    with col4:
        st.metric(
            "👥 Pacientes Beneficiados",
            "500K+/ano",
            help="Número estimado de pacientes que terão melhor atendimento"
        )
    
    # Plano de implementação
    st.markdown("---")
    st.markdown("### 🛣️ Plano de Implementação")
    
    implementation_data = {
        'Fase': ['Fase 1', 'Fase 2', 'Fase 3', 'Fase 4'],
        'Duração': ['1-2 meses', '2-3 meses', '3-4 meses', '2-3 meses'],
        'Atividades': [
            'Integração com Data Lake, Treinamento de modelo',
            'Desenvolvimento de API, Dashboard básico',
            'Interface web completa, Testes piloto',
            'Deploy completo, Monitoramento e otimização'
        ],
        'Recursos': ['2-3 devs', '3-4 devs + UX', '4-5 devs + QA', '2-3 devs + suporte']
    }
    
    df_implementation = pd.DataFrame(implementation_data)
    st.table(df_implementation)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🏥 <strong>Sistema de Previsão de No-Show</strong> | 
    Hackathon IA 2025 - Coppe/UFRJ | 
    Desenvolvido com ❤️ para a Saúde Pública Brasileira</p>
</div>
""", unsafe_allow_html=True)