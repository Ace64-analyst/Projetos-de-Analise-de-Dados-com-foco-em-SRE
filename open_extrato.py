from pathlib import Path 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Carregam bibliotecas essenciais: pandas e numpy para manipulação de dados; matplotlib e seaborn para visualização.

# DF (Data Frame)
df = pd.read_csv("logs_jornada_extrato.csv.csv")
# Carrega e inspeciona a estrutura do CSV. Essa etapa é crucial para entender a granularidade dos dados e os tipos.

# 5 primeiras linhas do DF
print(df.head())

# Info do DF
print(df.info())

# Colunas do DF
print(df.columns)

# Converte a coluna 'timestamp' em um formato datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Cria uma nova coluna 'dia' com apenas as datas
df["dia"] = df["timestamp"].dt.date
print(df.head())

# Total de requisições
total_requisicoes = len(df)

# Número de sucessos
sucessos = (df['status'] == 'sucesso').sum() # - Em uma série de bool (==) a função .sum() soma os valores 
                                             # como se fossem True = 1 e False = 0

# Número de erros
erros = (df['status'] != 'sucesso').sum()

# Calcula a disponibilidade em porcentagem
disponibilidade = (sucessos / total_requisicoes) * 100

# Calcula a taxa de erro
taxa_erro = (erros / total_requisicoes) * 100

# Calcula a latência média
latencia_media = df['latencia_ms'].mean()

# Agrupa todas as linhas que tem o mesmo valor em dia
reqs_por_dia = df.groupby("dia").size()

# Agrupa os dados por dias e calcula as médias em porcentagens
agrupado = df.groupby("dia").agg({
    "status": lambda x: (x == "sucesso").mean() * 100,  # Porcentagem
    "latencia_ms": "mean",  # Latência média
    "timestamp": "count"   # Quantidade de requisições
}).reset_index()

# Deixa as colunas mais claras
agrupado.columns = ["Dia", "Disponibilidade (%)", "Latência Média (ms)", "Quantidade de Requisições"]

print(agrupado)

# Contagem das ocorrências de cada status
erro_counts = df["status"].value_counts()

# Define o limite de orçamento de erro
limite_erro = 300

# Calcula a porcentagem consumida do orçamento
porcentagem_consumida = (erros / limite_erro) * 100


# Resultados
print(f"Total de Requisições: {total_requisicoes}")
print(f"Número de Sucessos: {sucessos}")
print(f"Taxa de Erro: {taxa_erro:.2f}%")
print(f"Disponibilidade: {disponibilidade:.2f}%")
print(f"Latência Média: {latencia_media:.2f} ms")
print("Status:") # Um pequeno esforço para ficar com uma formatação legal
for status, count in erro_counts.items():
    print(f"{status}:{count}")
print(f"Porcentagem Consumida do Orçamento: {porcentagem_consumida:.0f}%")

# Formatação para o Power BI
agrupado['Dia_num'] = pd.to_datetime(agrupado['Dia']).map(pd.Timestamp.toordinal)
z = np.polyfit(agrupado['Dia_num'], agrupado['Latência Média (ms)'], 1)
p = np.poly1d(z)

# Calcula a média de requisições diárias
media_requisicoes = reqs_por_dia.mean()
print(f"Média de requisições por dia: {media_requisicoes:.0f}")

# Gráfico das requisições diárias
dates = pd.date_range(start="2025-06-01", end="2025-06-29", freq='D')
reqs_por_dia = pd.Series([170, 200, 165, 175, 180, 185, 190, 195, 150, 160, 185, 190, 195, 
                        200, 175, 180, 185, 150, 160, 195, 210, 180, 185, 190, 205, 210, 175, 180, 170], index=dates)

plt.figure(figsize=(12, 6))
sns.lineplot(x=reqs_por_dia.index, y=reqs_por_dia.values, marker="o", label="Requisições por Dia")
plt.axhline(media_requisicoes, color="red", linestyle="--", label=f"Média: {media_requisicoes:.1f} req/dia")

plt.title("Quantidade de Requisições por Dia")
plt.xlabel("Dia")
plt.ylabel("Número de Requisições")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Distribuição das Latências
plt.figure(figsize=(10, 6))
sns.boxplot(y=df['latencia_ms'])
plt.title('Distribuição das Latências')
plt.show()

plt.figure(figsize=(12, 8))
sns.histplot(df['latencia_ms'], bins=50, kde=True)
plt.title('Histograma da Distribuição das Latências')
plt.xlabel('Latência (ms)')
plt.ylabel('Frequência')
plt.show()

# Análise Temporal das Latências
plt.figure(figsize=(12, 6))
sns.lineplot(data=agrupado, x='Dia', y='Latência Média (ms)')
plt.plot(agrupado['Dia'], p(agrupado['Dia_num']), "r--", label="Tendência")
plt.title('Latências Médias por Dia')
plt.xlabel('Data')
plt.ylabel('Latência Média (ms)')
plt.xticks(rotation=45)
plt.legend()
plt.show()

correlacao = agrupado['Dia_num'].corr(agrupado['Latência Média (ms)'])
print(f"Correlação entre data e latência média: {correlacao:.2f}")

# Relação Latência x Status
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='latencia_ms', y='status', hue='status')
plt.title('Relação entre Latência e Status de Requisição')
plt.xlabel('Latência (ms)')
plt.ylabel('Status')
plt.legend(title='Status')
plt.show()

# Frequência de Tipos de Erro
plt.figure(figsize=(10, 6))
sns.barplot(x=erro_counts.index, y=erro_counts.values)
plt.title('Frequência de Tipos de Erro')
plt.xlabel('Tipo de Erro')
plt.ylabel('Frequência')
plt.xticks(rotation=45)
plt.show()

# Frequência dos erros ao longo do dia, para identificar horários mais congestionados
df["hora"] = df["timestamp"].dt.hour
erros_por_hora = df[df["status"] != "sucesso"].groupby("hora").size()
sns.barplot(x=erros_por_hora.index, y=erros_por_hora.values)
plt.title("Erros por Hora do Dia")
plt.xlabel("Hora")
plt.ylabel("Erros")
plt.show()

# Matriz de Correlação
correlation_matrix = agrupado.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlação')
plt.show()
print("Correlação entre Latência Média e Disponibilidade (%):",
      agrupado['Latência Média (ms)'].corr(agrupado['Disponibilidade (%)']))

# Orçamento de Erro
consumo = 166  
limite = 100  

cores = ['green' if consumo <= 100 else 'red']

plt.figure(figsize=(8, 2))
plt.barh(['Orçamento de Erro'], [consumo], color=cores, height=0.4)
plt.axvline(x=limite, color='black', linestyle='--', label='Limite de 100%')
plt.xlim(0, max(180, consumo + 10))  # espaço extra
plt.title('Consumo do Orçamento de Erro')
plt.xlabel('Porcentagem (%)')
plt.legend()
plt.tight_layout()
plt.show()


# SLO alvo
SLO = 90
agrupado['SLO Atingido'] = agrupado['Disponibilidade (%)'] >= SLO

cores_slo = ['green' if atingido else 'red' for atingido in agrupado['SLO Atingido']]

plt.figure(figsize=(12, 6))
sns.barplot(x='Dia', y='Disponibilidade (%)', data=agrupado, palette=cores_slo)
plt.axhline(y=SLO, color='black', linestyle='--', label=f'SLO Alvo ({SLO}%)')
plt.title('Disponibilidade por Dia e SLO Atingido')
plt.xlabel('Data')
plt.ylabel('Disponibilidade (%)')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Percentil 95 de latência
p95 = df.groupby('dia')['latencia_ms'].quantile(0.95).reset_index(name='Latência p95 (ms)')
agrupado = agrupado.merge(p95, left_on='Dia', right_on='dia', how='left').drop(columns=['dia'])

p95_global = df['latencia_ms'].quantile(0.95)
print(f'Latência P95 global: {p95_global:.0f} ms')

agrupado['Dia_num'] = pd.to_datetime(agrupado['Dia']).map(pd.Timestamp.toordinal)
z = np.polyfit(agrupado['Dia_num'], agrupado['Latência p95 (ms)'], 1)
p = np.poly1d(z)

plt.figure(figsize=(12, 6))
sns.lineplot(x='Dia', y='Latência p95 (ms)', data=agrupado, marker='o', label='Latência p95')
plt.plot(agrupado['Dia'], p(agrupado['Dia_num']), "r--", label='Tendência Linear')
plt.title('Latência p95 por Dia')
plt.xlabel('Data')
plt.ylabel('Latência p95 (ms)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Cálculo da latência P95 por dia
p95_por_dia = (
    df
    .groupby('dia')['latencia_ms']
    .quantile(0.95)
    .reset_index(name='P95 (ms)')
)

# Identifica o dia de pico e o valor
idx_pico = p95_por_dia['P95 (ms)'].idxmax()
pico = p95_por_dia.loc[idx_pico]

print(f"Pico de Latência P95: {pico['P95 (ms)']:.0f} ms em {pico['dia']}")

# ─── Correlação Latência p95 x Disponibilidade ───

# Calcula a correlação
corr_p95_dispo = agrupado['Latência p95 (ms)'].corr(agrupado['Disponibilidade (%)'])
print(f'Correlação Latência p95 x Disponibilidade: {corr_p95_dispo:.2f}')

# Matriz de correlação (heatmap)
plt.figure(figsize=(6, 4))
sns.heatmap(
    agrupado[['Latência p95 (ms)', 'Disponibilidade (%)']].corr(),
    annot=True, cmap='coolwarm', vmin=-1, vmax=1
)
plt.title('Matriz de Correlação p95 x Disponibilidade')
plt.tight_layout()
plt.show()




agrupado['Disponibilidade (%)'] = agrupado['Disponibilidade (%)'].astype(float)
agrupado['Latência Média (ms)'] = agrupado['Latência Média (ms)'].astype(float)
agrupado['Latência p95 (ms)'] = agrupado['Latência p95 (ms)'].astype(float)


agrupado = agrupado.drop('Dia_num', axis=1) # Escondendo a coluna dia_num já que não precisa ser mostrada no Power BI
agrupado.to_csv(
    "analise_pronta.csv",
    index=False,
    encoding="utf-8",
    sep=";",
    decimal=",",
    float_format="%.2f"  # tentando salvar direito para o Power BI
)



