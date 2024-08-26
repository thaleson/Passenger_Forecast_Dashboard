# 📈 Dashboard de Previsão de Passageiros Aéreos ✈️

Bem-vindo ao projeto **Dashboard de Previsão de Passageiros Aéreos**! Este dashboard foi desenvolvido para prever o número de passageiros aéreos nos próximos meses usando um modelo SARIMA. Abaixo, você encontrará uma visão geral do projeto, instruções de instalação e uso, e detalhes sobre o modelo.

## 🚀 Visão Geral

Este dashboard utiliza um modelo **SARIMA** para prever o número de passageiros aéreos para os próximos meses. O modelo é treinado com dados históricos e permite gerar previsões para um intervalo de datas especificado. 

### 🗂️ Sobre o Dataset

O dataset utilizado é o **AirPassengers.csv** e contém informações mensais sobre o número de passageiros aéreos. O dataset possui as seguintes colunas:

- **Month:** Data do registro.
- **#Passengers:** Número de passageiros registrados.

### 🔍 O Modelo SARIMA

**SARIMA** (Seasonal AutoRegressive Integrated Moving Average) é um modelo de séries temporais que:

- **Captura Tendências:** Identifica padrões gerais de aumento ou diminuição.
- **Reconhece Sazonalidade:** Considera padrões que se repetem em intervalos regulares, como anos.

#### **Como Funciona:**
1. **Treinamento:** O modelo é ajustado aos dados históricos para identificar padrões e sazonalidade.
2. **Previsão:** Com base nesses padrões, o modelo prevê valores futuros.
3. **Intervalo de Confiança:** Indica a faixa onde os valores futuros provavelmente se situarão, ajudando a entender a incerteza das previsões.

## 🛠️ Instalação

Para rodar este projeto, siga os passos abaixo:

1. **Clone o Repositório**

   ```sh
   git clone https://github.com/thaleson/Passenger_Forecast_Dashboard.git
   ```

2. **Navegue até o Diretório do Projeto**

   ```sh
   cd Passenger_Forecast_Dashboard
   ```

3. **Crie e Ative um Ambiente Virtual**

   ```sh
   python -m venv .venv
   source .venv/bin/activate  # Para Linux/macOS
   .venv\Scripts\activate  # Para Windows
   ```

4. **Instale as Dependências**

   ```sh
   pip install -r requirements.txt
   ```

## 🚀 Como Usar

1. **Execute o Aplicativo Streamlit**

   ```sh
   streamlit run main.py
   ```

2. **Acesse o Dashboard**

   Abra o navegador e vá para `http://localhost:8501` para visualizar o dashboard.

### 📊 Funcionalidades

- **Selecione o Período de Previsão:** Defina a data de início e término para obter previsões.
- **Visualize o Gráfico de Previsão:** Veja a série temporal dos passageiros observados e as previsões futuras.
- **Tabela de Previsões:** Examine as previsões detalhadas junto com o intervalo de confiança.

## 📂 Estrutura do Projeto

- `main.py`: Código principal do aplicativo Streamlit.
- `AirPassengers.csv`: Dataset com dados de passageiros aéreos.
- `style.css`: Arquivo de estilos CSS para personalização do dashboard.

## 📑 Exemplo de Código

Aqui está um exemplo de como o modelo SARIMA é treinado e utilizado para gerar previsões:

```python
import pandas as pd
import statsmodels.api as sm

# Carregar os dados
df = pd.read_csv('AirPassengers.csv', parse_dates=['Month'], index_col='Month')

# Treinar o modelo SARIMA
model = sm.tsa.statespace.SARIMAX(df['#Passengers'], 
                                  order=(1, 1, 1),
                                  seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Gerar previsões
forecast = results.get_forecast(steps=12)
forecast_df = forecast.predicted_mean
```

## 📝 Contribuições

Se você quiser contribuir para o projeto, fique à vontade para abrir issues ou pull requests. Toda contribuição é bem-vinda!

## 📜 Licença

Este projeto está licenciado sob a [MIT License](LICENSE).



---

Obrigado por usar o **Dashboard de Previsão de Passageiros Aéreos**! Esperamos que ele seja útil para o seu planejamento de demanda. 🚀✨
