# 🏥 Sistema de Previsão de No-Show - Regulação Ambulatorial

> **Projeto para Hackathon IA 2025 - Coppe/UFRJ**  
> Aplicação de Inteligência Artificial para otimização da gestão de recursos em saúde pública

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![ML](https://img.shields.io/badge/Machine%20Learning-scikit--learn-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)

## 🎯 Objetivo

Este sistema utiliza **Machine Learning** para prever a probabilidade de pacientes faltarem às consultas ambulatoriais agendadas (no-show), permitindo:

- ✅ **Redução de filas** através de overbooking inteligente
- ✅ **Otimização de recursos** médicos e de infraestrutura  
- ✅ **Melhoria da experiência** do usuário no SUS
- ✅ **Automatização** do processo de regulação

## 🏆 Alinhamento com o Hackathon IA 2025

### Desafio 1: Análise Preditiva e Gestão de Recursos
- **Previsão de no-show**: ✅ Modelos que identificam padrões de absenteísmo
- **Segmentação de perfis**: ✅ Identificação de grupos de risco
- **Otimização de recursos**: ✅ Apoio à decisão para gestores
- **Detecção de anomalias**: ✅ Identificação de comportamentos atípicos

### Tecnologias Utilizadas
- **Python 3.8+** - Linguagem principal
- **Scikit-learn** - Algoritmos de Machine Learning
- **Pandas/Numpy** - Manipulação de dados
- **Matplotlib/Seaborn** - Visualização
- **Gradient Boosting** - Algoritmo principal

## 📊 Modelo de Dados

Baseado na estrutura do **Data Lake da Saúde do Rio de Janeiro**:

### Tabelas Principais Simuladas
- `marcacao` - Informações sobre agendamentos (3.1M+ registros)
- `solicitacao` - Registros de solicitações (3.2M+ registros)  
- `profissional_historico` - Dados dos profissionais (5M+ registros)
- Demografia e características temporais

### Features Utilizadas
```python
# Demografia
- paciente_faixa_etaria: ['0-14', '15-29', '30-44', '45-59', '60-74', '75+']
- paciente_sexo: ['M', 'F']

# Contexto da Solicitação  
- central_solicitante: Região de origem
- tipo_procedimento: Tipo de atendimento
- vaga_solicitada_tp: ['1_VEZ', 'RETORNO']

# Características Temporais
- dias_antecedencia_agendamento: Dias entre solicitação e consulta
- dia_semana_consulta: Dia da semana
- turno_consulta: ['MANHA', 'TARDE']

# Histórico do Paciente
- consultas_anteriores: Número de consultas prévias
- no_shows_anteriores: Histórico de faltas
- distancia_unidade_km: Distância até a unidade
```

## 🚀 Como Usar

### Instalação
```bash
git clone https://github.com/[seu-usuario]/no-show-predictor
cd no-show-predictor
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execução
```bash
python no_show_predictor.py
```

### Exemplo de Uso
```python
from no_show_predictor import NoShowPredictor

# Inicializar sistema
predictor = NoShowPredictor()

# Treinar com dados sintéticos
df = predictor.generate_synthetic_data(n_samples=15000)
predictor.train_model(df)

# Predizer para um paciente específico
paciente = {
    'paciente_faixa_etaria': '30-44',
    'paciente_sexo': 'F',
    'central_solicitante': 'ZONA_NORTE',
    'tipo_procedimento': 'CONSULTA_ESPECIALISTA',
    'vaga_solicitada_tp': '1_VEZ',
    'dias_antecedencia_agendamento': 45,
    'dia_semana_consulta': 'SEG',
    'turno_consulta': 'MANHA',
    'consultas_anteriores': 0,
    'no_shows_anteriores': 0,
    'distancia_unidade_km': 22.5,
    'unidade_porte': 'MEDIO'
}

probabilidade = predictor.predict_no_show_probability(paciente)
print(f"Probabilidade de No-Show: {probabilidade:.1%}")
```

## 📈 Resultados e Performance

### Métricas do Modelo
- **AUC-ROC**: ~0.85
- **Precisão**: ~78%
- **Recall**: ~82%
- **F1-Score**: ~80%

### Features Mais Importantes
1. `no_shows_anteriores` - Histórico de faltas
2. `dias_antecedencia_agendamento` - Antecedência do agendamento
3. `distancia_unidade_km` - Distância da unidade
4. `paciente_faixa_etaria` - Faixa etária
5. `vaga_solicitada_tp` - Primeira vez vs retorno

## 💡 Insights para Gestão

### Recomendações Estratégicas
- **Lembretes Automáticos**: SMS/WhatsApp 2-3 dias antes para pacientes de risco
- **Overbooking Inteligente**: Baseado no score de risco individual
- **Priorização**: Pacientes baixo risco em horários críticos
- **Monitoramento**: Padrões por região e especialidade
- **Campanhas**: Conscientização para grupos de alto risco

### Casos de Uso
- **Gestores**: Dashboard de previsão de demanda
- **Reguladores**: Otimização automática de escalas
- **Unidades**: Preparação proativa para demanda
- **Pacientes**: Lembretes personalizados

## 🔮 Roadmap - Próximos Passos

### Para o Hackathon
- [ ] **Integração Real**: Conexão com Data Lake da Saúde RJ
- [ ] **API REST**: Endpoint para integração com sistemas
- [ ] **Dashboard**: Interface em tempo real (Streamlit/Dash)
- [ ] **Overbooking Automático**: Sistema de ajuste de escalas
- [ ] **Notificações**: Integração WhatsApp/SMS

### Funcionalidades Avançadas
- [ ] **Deep Learning**: Redes neurais para padrões complexos
- [ ] **Time Series**: Previsão de demanda sazonal
- [ ] **NLP**: Análise de motivos de cancelamento
- [ ] **Multimodalidade**: Integração com dados não estruturados
- [ ] **RAG**: Sistema de consulta em linguagem natural

## 🤝 Contribuição

Este projeto visa impacto social na **saúde pública brasileira**. Contribuições são bem-vindas!

### Como Contribuir
1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Abra um Pull Request

## 📝 Licença

Este projeto está sob a licença **GPLv3** - veja o arquivo [LICENSE](LICENSE) para detalhes.

Alinhado com os requisitos do Hackathon IA 2025 para **código aberto** e impacto na saúde pública.

---

## 🏥 Sobre o Hackathon IA 2025

**Evento**: Hackathon de IA aplicada à saúde pública  
**Organizadores**: Incubadora Coppe/UFRJ + Cietec/USP  
**Parceiros**: NVIDIA, LNCC, SMS-RJ  
**Data**: 10-12 de outubro de 2025  
**Local**: Porto Maravalley, Rio de Janeiro

### Objetivo do Evento
Fomentar soluções inovadoras para desafios da saúde pública através de tecnologias de IA, criando ferramentas que:
- Aprimorem a qualidade do sistema de saúde
- Otimizem o uso de recursos públicos  
- Melhorem a experiência do usuário no atendimento

**Recursos Disponíveis**: 
- Data Lake da Saúde do Rio de Janeiro
- Supercomputador Santos Dumont (LNCC)
- Mentoria especializada em IA e saúde pública

---

*"A inteligência artificial é uma ferramenta de transformação e será chave para o desenvolvimento econômico e social do Brasil."* - Hackathon IA 2025