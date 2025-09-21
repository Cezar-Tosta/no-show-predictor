"""
Sistema de Previsão de No-Show em Consultas Ambulatoriais
Projeto para Hackathon IA 2025 - Regulação em Saúde Pública

Este sistema utiliza Machine Learning para prever a probabilidade de pacientes 
faltarem às consultas agendadas, permitindo otimização de recursos e redução de filas.

Autor: [Seu Nome]
Data: Setembro 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

class NoShowPredictor:
    """
    Sistema de previsão de no-show para consultas ambulatoriais
    Baseado nos desafios do Hackathon IA 2025 - Coppe/UFRJ
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def generate_synthetic_data(self, n_samples=10000):
        """
        Gera dados sintéticos baseados na estrutura do Data Lake Rio de Janeiro
        Simula as tabelas: marcacao, solicitacao, e dados demográficos
        """
        np.random.seed(42)
        
        # Baseado na tabela 'marcacao' do Data Lake
        data = {
            # Demografia do paciente
            'paciente_faixa_etaria': np.random.choice(['0-14', '15-29', '30-44', '45-59', '60-74', '75+'], 
                                                    n_samples, p=[0.15, 0.20, 0.25, 0.20, 0.15, 0.05]),
            'paciente_sexo': np.random.choice(['M', 'F'], n_samples, p=[0.45, 0.55]),
            
            # Características da solicitação
            'central_solicitante': np.random.choice(['CENTRO', 'ZONA_NORTE', 'ZONA_SUL', 'ZONA_OESTE', 
                                                   'BAIXADA', 'NITEROI'], n_samples),
            'tipo_procedimento': np.random.choice(['CONSULTA_ESPECIALISTA', 'EXAME_IMAGEM', 
                                                 'EXAME_LAB', 'CIRURGIA_ELETIVA'], n_samples),
            'vaga_solicitada_tp': np.random.choice(['1_VEZ', 'RETORNO'], n_samples, p=[0.6, 0.4]),
            
            # Temporal
            'dias_antecedencia_agendamento': np.random.exponential(scale=15, size=n_samples).astype(int),
            'dia_semana_consulta': np.random.choice(['SEG', 'TER', 'QUA', 'QUI', 'SEX'], n_samples),
            'turno_consulta': np.random.choice(['MANHA', 'TARDE'], n_samples, p=[0.6, 0.4]),
            'mes_consulta': np.random.randint(1, 13, n_samples),
            
            # Histórico do paciente
            'consultas_anteriores': np.random.poisson(lam=2, size=n_samples),
            'no_shows_anteriores': np.random.poisson(lam=0.5, size=n_samples),
            'tempo_ultima_consulta_dias': np.random.exponential(scale=60, size=n_samples).astype(int),
            
            # Características da unidade
            'distancia_unidade_km': np.random.exponential(scale=8, size=n_samples),
            'unidade_porte': np.random.choice(['PEQUENO', 'MEDIO', 'GRANDE'], n_samples, p=[0.4, 0.4, 0.2]),
        }
        
        df = pd.DataFrame(data)
        
        # Criar variável target baseada em regras realistas
        no_show_prob = 0.15  # Taxa base de no-show (~15%)
        
        # Fatores que aumentam probabilidade de no-show
        prob_adjustments = np.zeros(n_samples)
        
        # Faixa etária (jovens adultos faltam mais)
        prob_adjustments += np.where(df['paciente_faixa_etaria'].isin(['15-29', '30-44']), 0.05, 0)
        
        # Antecedência (muito cedo ou muito em cima da hora)
        prob_adjustments += np.where(df['dias_antecedencia_agendamento'] > 30, 0.08, 0)
        prob_adjustments += np.where(df['dias_antecedencia_agendamento'] < 3, 0.06, 0)
        
        # Histórico de no-shows
        prob_adjustments += df['no_shows_anteriores'] * 0.15
        
        # Distância da unidade
        prob_adjustments += np.where(df['distancia_unidade_km'] > 15, 0.07, 0)
        
        # Primeira vez vs retorno
        prob_adjustments += np.where(df['vaga_solicitada_tp'] == '1_VEZ', 0.04, -0.02)
        
        # Segunda-feira e sexta-feira
        prob_adjustments += np.where(df['dia_semana_consulta'].isin(['SEG']), 0.03, 0)
        
        # Gerar target
        final_probs = np.clip(no_show_prob + prob_adjustments, 0.01, 0.95)
        df['no_show'] = np.random.binomial(1, final_probs)
        
        return df
    
    def preprocess_data(self, df):
        """Preprocessa os dados para o modelo"""
        df_processed = df.copy()
        
        # Encoding de variáveis categóricas
        categorical_cols = ['paciente_faixa_etaria', 'paciente_sexo', 'central_solicitante', 
                           'tipo_procedimento', 'vaga_solicitada_tp', 'dia_semana_consulta', 
                           'turno_consulta', 'unidade_porte']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        # Features derivadas
        df_processed['taxa_no_show_historica'] = np.where(
            df_processed['consultas_anteriores'] > 0,
            df_processed['no_shows_anteriores'] / df_processed['consultas_anteriores'],
            0
        )
        
        df_processed['paciente_novo'] = (df_processed['consultas_anteriores'] == 0).astype(int)
        df_processed['distancia_categoria'] = pd.cut(df_processed['distancia_unidade_km'], 
                                                   bins=[0, 5, 15, 50], labels=['PERTO', 'MEDIO', 'LONGE'])
        df_processed['distancia_categoria'] = LabelEncoder().fit_transform(df_processed['distancia_categoria'])
        
        return df_processed
    
    def train_model(self, df):
        """Treina o modelo de previsão"""
        df_processed = self.preprocess_data(df)
        
        # Separar features e target
        feature_cols = [col for col in df_processed.columns if col not in ['no_show']]
        X = df_processed[feature_cols]
        y = df_processed['no_show']
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Normalizar features numéricas
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
        
        # Treinar modelo ensemble
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Avaliar modelo
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        print("=== AVALIAÇÃO DO MODELO ===")
        print(f"AUC-ROC Score: {roc_auc_score(y_test, y_pred_proba):.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return X_test_scaled, y_test, y_pred_proba
    
    def predict_no_show_probability(self, patient_data):
        """Prediz probabilidade de no-show para um paciente específico"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado ainda!")
        
        df_patient = pd.DataFrame([patient_data])
        df_processed = self.preprocess_data(df_patient)
        
        # Aplicar mesma normalização
        feature_cols = [col for col in df_processed.columns if col not in ['no_show']]
        X_patient = df_processed[feature_cols]
        
        numeric_cols = X_patient.select_dtypes(include=[np.number]).columns
        X_patient_scaled = X_patient.copy()
        X_patient_scaled[numeric_cols] = self.scaler.transform(X_patient[numeric_cols])
        
        probability = self.model.predict_proba(X_patient_scaled)[0, 1]
        return probability
    
    def generate_insights(self):
        """Gera insights para gestão hospitalar"""
        if self.feature_importance is None:
            return "Modelo não treinado ainda."
        
        insights = []
        top_features = self.feature_importance.head(5)
        
        insights.append("=== INSIGHTS PARA GESTÃO AMBULATORIAL ===\n")
        insights.append("🎯 FATORES MAIS IMPORTANTES PARA NO-SHOW:")
        for idx, row in top_features.iterrows():
            insights.append(f"   • {row['feature']}: {row['importance']:.3f}")
        
        insights.append(f"\n💡 RECOMENDAÇÕES ESTRATÉGICAS:")
        insights.append("   • Implementar lembretes via SMS/WhatsApp 2-3 dias antes")
        insights.append("   • Criar overbooking inteligente baseado no score de risco")
        insights.append("   • Priorizar pacientes com baixo risco para horários críticos")
        insights.append("   • Monitorar padrões por região e especialidade")
        insights.append("   • Desenvolver campanhas de conscientização para grupos de alto risco")
        
        return "\n".join(insights)
    
    def plot_analysis(self, X_test, y_test, y_pred_proba):
        """Gera visualizações para análise"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análise do Sistema de Previsão de No-Show', fontsize=16, fontweight='bold')
        
        # 1. Feature Importance
        axes[0,0].barh(self.feature_importance.head(8)['feature'], 
                       self.feature_importance.head(8)['importance'])
        axes[0,0].set_title('Importância das Features')
        axes[0,0].set_xlabel('Importância')
        
        # 2. Distribuição de probabilidades
        axes[0,1].hist(y_pred_proba[y_test==0], alpha=0.5, label='Compareceu', bins=30)
        axes[0,1].hist(y_pred_proba[y_test==1], alpha=0.5, label='No-Show', bins=30)
        axes[0,1].set_title('Distribuição de Probabilidades Preditas')
        axes[0,1].set_xlabel('Probabilidade de No-Show')
        axes[0,1].legend()
        
        # 3. Confusion Matrix
        from sklearn.metrics import confusion_matrix
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred_binary)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1,0], cmap='Blues')
        axes[1,0].set_title('Matriz de Confusão')
        axes[1,0].set_ylabel('Real')
        axes[1,0].set_xlabel('Predito')
        
        # 4. ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        axes[1,1].plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
        axes[1,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[1,1].set_title('Curva ROC')
        axes[1,1].set_xlabel('Taxa de Falso Positivo')
        axes[1,1].set_ylabel('Taxa de Verdadeiro Positivo')
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.show()

def main():
    """Função principal - Demonstração do sistema"""
    print("🏥 SISTEMA DE PREVISÃO DE NO-SHOW - REGULAÇÃO AMBULATORIAL")
    print("=" * 60)
    print("Projeto para Hackathon IA 2025 - Coppe/UFRJ")
    print("Foco: Otimização de recursos em saúde pública\n")
    
    # Inicializar sistema
    predictor = NoShowPredictor()
    
    # Gerar dados sintéticos baseados no Data Lake Rio
    print("📊 Gerando dados sintéticos baseados no Data Lake da Saúde RJ...")
    df = predictor.generate_synthetic_data(n_samples=15000)
    
    print(f"✅ Dataset criado: {len(df)} registros")
    print(f"   Taxa de No-Show: {df['no_show'].mean():.1%}")
    print(f"   Colunas: {', '.join(df.columns[:8])}...\n")
    
    # Treinar modelo
    print("🤖 Treinando modelo de Machine Learning...")
    X_test, y_test, y_pred_proba = predictor.train_model(df)
    
    # Exemplo de predição
    print("\n" + "="*60)
    print("🔍 EXEMPLO DE PREDIÇÃO INDIVIDUAL:")
    
    paciente_exemplo = {
        'paciente_faixa_etaria': '30-44',
        'paciente_sexo': 'F',
        'central_solicitante': 'ZONA_NORTE',
        'tipo_procedimento': 'CONSULTA_ESPECIALISTA',
        'vaga_solicitada_tp': '1_VEZ',
        'dias_antecedencia_agendamento': 45,
        'dia_semana_consulta': 'SEG',
        'turno_consulta': 'MANHA',
        'mes_consulta': 10,
        'consultas_anteriores': 0,
        'no_shows_anteriores': 0,
        'tempo_ultima_consulta_dias': 0,
        'distancia_unidade_km': 22.5,
        'unidade_porte': 'MEDIO'
    }
    
    prob_no_show = predictor.predict_no_show_probability(paciente_exemplo)
    
    print(f"Paciente: Mulher, 30-44 anos, primeira consulta")
    print(f"Agendamento: 45 dias de antecedência, segunda-feira manhã")
    print(f"📈 Probabilidade de No-Show: {prob_no_show:.1%}")
    
    if prob_no_show > 0.4:
        print("⚠️  ALTO RISCO - Recomendação: Contato preventivo + overbooking")
    elif prob_no_show > 0.2:
        print("⚡ RISCO MÉDIO - Recomendação: Lembrete via SMS")
    else:
        print("✅ BAIXO RISCO - Agendamento normal")
    
    # Insights
    print("\n" + predictor.generate_insights())
    
    # Visualizações
    print("\n📊 Gerando análises visuais...")
    predictor.plot_analysis(X_test, y_test, y_pred_proba)
    
    print("\n🎯 PRÓXIMOS PASSOS PARA O HACKATHON:")
    print("   1. Integrar com dados reais do Data Lake Rio")
    print("   2. Desenvolver API REST para integração")
    print("   3. Criar dashboard em tempo real")
    print("   4. Implementar sistema de overbooking inteligente")
    print("   5. Adicionar notificações automáticas")

if __name__ == "__main__":
    main()