import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns





# Função para carregar e treinar o modelo SARIMA
@st.cache_data
def train_sarima_model():
    df = pd.read_csv('dataset/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
    model = sm.tsa.statespace.SARIMAX(df['#Passengers'], 
                                      order=(1, 1, 1),
                                      seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    return results, df

# Função para gerar previsões
def generate_forecast(start_date, end_date):
    forecast_steps = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days // 30
    forecast = results.get_forecast(steps=forecast_steps)
    conf_int = forecast.conf_int()
    forecast_df = forecast.predicted_mean
    forecast_df.index = pd.date_range(start=start_date, periods=forecast_steps, freq='M')
    return forecast_df, conf_int

# Configuração do Streamlit
st.set_page_config(page_title='📈 Dashboard de Previsão de Passageiros Aéreos ✈️', layout='wide')
st.title('📈 Dashboard de Previsão de Passageiros Aéreos ✨')

#aplicar estilos de css a pagina
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


st.markdown("""
## 🚀 Visão Geral do Projeto

Este dashboard utiliza um modelo **SARIMA** para prever o número de passageiros aéreos nos próximos meses. O modelo foi treinado com dados históricos de passageiros para oferecer previsões precisas e úteis para o planejamento futuro.

### 🗂️ Sobre o Dataset

**AirPassengers.csv** contém dados mensais sobre o número de passageiros aéreos, com duas colunas principais:
- **Month:** Data do registro.
- **#Passengers:** Número de passageiros registrados.

### 🔍 O Modelo SARIMA

**SARIMA** (Seasonal AutoRegressive Integrated Moving Average) é um modelo de séries temporais que:
- **Captura Tendências:** Identifica padrões gerais de aumento ou diminuição.
- **Reconhece Sazonalidade:** Considera padrões que se repetem em intervalos regulares (como um ano).

#### **Como Funciona:**
1. **Treinamento:** O modelo é ajustado aos dados históricos para identificar padrões e sazonalidade.
2. **Previsão:** Baseado nesses padrões, o modelo prevê valores futuros.
3. **Intervalo de Confiança:** Indica a faixa onde os valores futuros provavelmente se situarão, ajudando a entender a incerteza das previsões.

""")

# Carregar modelo e dados
results, df = train_sarima_model()

# Sidebar com imagem e configurações
with st.sidebar:
    st.image("imagens/logo.jpg")
    st.header('🔧 Configurações de Previsão 📅')
    start_date = st.date_input('Data de Início 🗓️', value=pd.to_datetime(df.index.max()) + pd.DateOffset(months=1))
    end_date = st.date_input('Data de Término 📅', value=pd.to_datetime(df.index.max()) + pd.DateOffset(months=12))

    if end_date <= start_date:
        st.sidebar.error('⚠️ A data de término deve ser posterior à data de início.')

# Botão para gerar previsões
if st.sidebar.button('Gerar Previsões 📊'):
    if end_date > start_date:
        # Gerar previsões
        forecast_df, conf_int = generate_forecast(start_date, end_date)

        # Gráfico de Previsão
        st.markdown("### 📊 Gráfico de Previsão ✈️")
        st.write("""
        O gráfico abaixo ilustra:
        - **Dados Observados (em Azul):** O número real de passageiros ao longo do tempo.
        - **Previsão (em Laranja):** O número esperado de passageiros para o período selecionado.
        - **Intervalo de Confiança:** A faixa sombreada que representa a incerteza nas previsões.
        """)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=df, x=df.index, y='#Passengers', label='Dados Observados', color='blue', linewidth=2)
        sns.lineplot(x=forecast_df.index, y=forecast_df, label='Previsão', color='orange', linewidth=2)
        ax.fill_between(forecast_df.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='orange', alpha=0.3)
        ax.set_xlabel('Data', fontsize=12)
        ax.set_ylabel('Número de Passageiros', fontsize=12)
        ax.legend()
        plt.title('Previsão de Passageiros Aéreos', fontsize=16)
        st.pyplot(fig)

        # Explicação do Gráfico
        st.markdown("### ✨ Interpretação do Gráfico 🧐")
        st.write("""
        - **Dados Observados (em Azul):** Representam o número real de passageiros em cada mês.
        - **Previsão (em Laranja):** Mostra a tendência esperada de passageiros para o futuro.
        - **Intervalo de Confiança:** Indica a faixa onde o verdadeiro número de passageiros pode estar. A área sombreada representa a incerteza nas previsões.

        **Interpretação:**
        - 📈 **Aumento:** Se a linha de previsão sobe, indica um crescimento na demanda de passageiros.
        - 📉 **Queda:** Se a linha de previsão desce, sugere uma diminuição na demanda.
        - ⚖️ **Estabilidade:** Se a linha de previsão é estável, a demanda deverá permanecer constante.

        💡 **Dica:** Use essas previsões para ajustar suas estratégias de marketing e operações para melhor atender à demanda futura.
        """)

        # Mostrar os dados de previsão
        st.markdown("### 📅 Tabela de Previsões 📉")
        st.write("""
        Abaixo estão os valores previstos para o período selecionado, junto com o intervalo de confiança para cada previsão. Isso oferece uma visão detalhada das expectativas futuras.
        """)
        st.dataframe(pd.DataFrame({
            'Data': forecast_df.index,
            'Previsão': forecast_df.values,
            'Intervalo Inferior': conf_int.iloc[:, 0].values,
            'Intervalo Superior': conf_int.iloc[:, 1].values
        }))

        # Conclusão
        st.markdown("""
        ### 🌟 Conclusão Final 🌟

        As previsões fornecidas oferecem uma visão valiosa sobre a demanda futura de passageiros aéreos. Com base nessas previsões, você pode planejar melhor e ajustar suas estratégias de acordo com as expectativas.

        **Obrigado por utilizar nosso dashboard!** Esperamos que as análises e previsões ajudem no planejamento e tomada de decisões. ✨
        """)
    else:
        st.sidebar.error('⚠️ Por favor, selecione um intervalo de datas válido.')
