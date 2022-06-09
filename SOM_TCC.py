# -*- coding: utf-8 -*-
"""
Created on Mon May 23 20:25:40 2022

@author: André Leme

Programa exemplo/genérico para tratamento de dados (nan, labelencoder, onehot)
Finaliza separando dataframes de teste e treinamento (se preferir arquivos csv)
"""
#print("\x1b[2J") # limpar console

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def analisarNaN(dataFrameDados):
    flag = False
    analisado = dataFrameDados
    cabec = analisado.columns
    for atributo in range(len(cabec)):
        extraiNaN = analisado.loc[pd.isnull(analisado[cabec[atributo]])]
        linhasComNaN = extraiNaN.shape[0]
        if linhasComNaN > 0:
            print('Atributo: ' + cabec[atributo] + ' possui ' + str(linhasComNaN) + ' registros com NaN')
            flag = True
    return(flag)

def preencherNaN(dataFrameDados, metodo):
    analisado = dataFrameDados
    cabec = analisado.columns
    for atributo in range(len(cabec)):
        extraiNaN = analisado.loc[pd.isnull(analisado[cabec[atributo]])]
        linhasComNaN = extraiNaN.shape[0]
        if linhasComNaN > 0:
            objImputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
            analisado = objImputer.fit_transform(analisado)
            analisado = pd.DataFrame(analisado, columns=cabec)
    return(analisado)

def mais_proximo(data, t, pesos, m_rows, m_cols):
  result = (0,0)
  small_dist = 1.0e20
  for i in range(m_rows):
    for j in range(m_cols):
      ed = distancia_euclidiana(pesos[i][j], data[t])
      if ed < small_dist:
        small_dist = ed
        result = (i, j)
  return result

def distancia_euclidiana(v1, v2):
  return np.linalg.norm(v1 - v2) 

def distancia_manhattan(r1, c1, r2, c2):
  return np.abs(r1-r2) + np.abs(c1-c2)


# Inicio: importação do arquivo - observar separador e encoding
#arquivo = 'heart_2020_cleaned.csv'
# arquivo = 'ensaio01_formados.csv'
arquivo = 'ForEva.csv'
dados = pd.read_csv(arquivo, sep=';', encoding='WINDOWS-1252')
cabecDados = dados.columns

'''
['id', 'ano_conclusao_ensino_anterior', 'ano_ingresso',
       'diferenca_ingresso_ensino_anterior', 'bairro', 'cidade', 'grade',
       'idade_ingresso', 'data_conclusao_curso', 'tempo_conclusa',
       'data_nascimento', 'idade_conclusao', 'deficiencia', 'estado',
       'estado_civil', 'forma_ingresso', 'frequencia_periodo',
       'instituicao_ensino_anterior', 'tipo_escola', 'municipio_residencia',
       'naturalidade', 'necessidade_especial', 'nivel_ensino_anterior',
       'percentual_progresso', 'percentual_progresso_faixa', 'sexo',
       'situacao_curso', 'tipo_escola_origem', 'turno', 'rendimento_boletim']
'''

# Trata e mostra atributos com NaN
if analisarNaN(dados):
    # Coloca dados mais frequentes em NaN na base toda
    # Se preferir média usar strategy='mean'
    # Para variar um e outro conforme a coluna, precisa rever função
    dados = preencherNaN(dados,'most_frequent')

# Extrai as features e a classe
# Extrai as features e a classe
previsores = dados.loc[:,['diferenca_ingresso_ensino_anterior',
                          'tempo_conclusa',
                          'idade_ingresso',
                          'forma_ingresso',
                          'tipo_escola',
                          'percentual_progresso_faixa'
                          ]
                       ]
classe = pd.DataFrame(dados.loc[:,'situacao_curso'])

cabecPrevisores = previsores.columns
cabecClasse = classe.columns

# labelencoder conforme o formato do atributo
le = LabelEncoder()
for coluna in range(len(cabecPrevisores)):
    if previsores.loc[:,cabecPrevisores[coluna]].dtype == 'object':
        print('Atributo ' + cabecPrevisores[coluna] + ' Tipo ' + str(previsores.loc[:,cabecPrevisores[coluna]].dtype))
        previsores.loc[:,cabecPrevisores[coluna]] = le.fit_transform(previsores.loc[:,cabecPrevisores[coluna]])

classe.loc[:,cabecClasse[0]] = le.fit_transform(classe.loc[:,cabecClasse[0]])
   
# Normalização, se necessário
if 1 == 1:
    ss = StandardScaler()
    previsores = pd.DataFrame(ss.fit_transform(previsores))
    #previsores.to_csv('previsores_treinamento.csv', sep=',', index=False)
    #classe.to_csv('classe_treinamento.csv', sep=',', index=False)

x = previsores #colocar as colunas de dados a serem extraidas
y = classe # coloca a coluna onde estão as classes

atributos = x.to_numpy()
classe = y.to_numpy()

np.random.seed(1)
 # dimensoes do S.O.M.
Rows = 20
Cols = 20

RangeMax = Rows + Cols
taxa_aprendizagem = 0.5 # taxa de aprendizagem inicial
iteracoes = 7000 # quantidade de iterações

quantidadeLinhasAtributos = len(atributos)
quantidadeColunasAtributos = len(atributos[0])

Dim = quantidadeColunasAtributos

# Verificar se os dados não precisam ser tratados (NaN, categoricos, normalização)

# 2. Construção do Mapa (pesos)
pesos = np.random.random_sample(size=(Rows,Cols,Dim))

for s in range(iteracoes):
  if s % (iteracoes/10) == 0: print("step = ", str(s))
  pct_left = 1.0 - ((s * 1.0) / iteracoes)
  curr_range = (int)(pct_left * RangeMax)
  curr_rate = pct_left * taxa_aprendizagem

  t = np.random.randint(len(atributos))
  (bmu_row, bmu_col) = mais_proximo(atributos, t, pesos, Rows, Cols)
  for i in range(Rows):
    for j in range(Cols):
      if distancia_manhattan(bmu_row, bmu_col, i, j) < curr_range:
         pesos[i][j] = pesos[i][j] + curr_rate * (atributos[t] - pesos[i][j])

# 3. Construção da Matriz de Distâncias
u_matrix = np.zeros(shape=(Rows,Cols), dtype=np.float64)
for i in range(Rows):
  for j in range(Cols):
    v = pesos[i][j]  # a vector 
    sum_dists = 0.0; ct = 0
   
    if i-1 >= 0:    # above
      sum_dists += distancia_euclidiana(v, pesos[i-1][j]); ct += 1
    if i+1 <= Rows-1:   # below
      sum_dists += distancia_euclidiana(v, pesos[i+1][j]); ct += 1
    if j-1 >= 0:   # left
      sum_dists += distancia_euclidiana(v, pesos[i][j-1]); ct += 1
    if j+1 <= Cols-1:   # right
      sum_dists += distancia_euclidiana(v, pesos[i][j+1]); ct += 1
    
    u_matrix[i][j] = sum_dists / ct

# Exibição da Matriz de Distâncias
plt.title('Distâncias')
plt.imshow(u_matrix, cmap='gray')  # black = close = clusters
plt.show()

# 4. Associa os labels da classe (y) com os quadrantes do mapa
grafico = np.empty(shape=(Rows,Cols), dtype=object)
for i in range(Rows):
  for j in range(Cols):
    grafico[i][j] = []

for t in range(len(atributos)):
  (m_row, m_col) = mais_proximo(atributos, t, pesos, Rows, Cols)
  #grafico[m_row][m_col].append(classe[t])
  grafico[m_row][m_col].append(t)

# Preenche os quadrantes com o label escolhido
arrayLabels = np.empty(shape=(Rows,Cols), dtype=object)
tamanho = len(grafico)
linha = 0
coluna = 0
for linha in range(tamanho):
    for coluna in range(tamanho):
        arrayLabels[linha][coluna] = len(grafico[linha][coluna])

#plt.figure(figsize=(30,30))

# Escreve labels nos quadrantes do gráfico
for i in range(len(arrayLabels)):
    for j in range(len(arrayLabels)):
        cortexto = 'black'
        if arrayLabels[i, j] == 0:
            legenda = ''
        else:
           #legenda = (arrayLabels[i, j]/237)*100
           legenda = arrayLabels[i, j]
           #legenda = '%.3f' % legenda
           
        text = plt.text(j, i, legenda,
                       ha="center",
                       va="center",
                       fontsize=6,
                       color=cortexto)

plt.imshow(u_matrix, cmap=plt.cm.get_cmap('rainbow_r', 13))
plt.colorbar()
plt.show()
