# -*- coding: utf-8 -*-
"""
Created on Tue May 31 19:52:34 2022

@author: andre
"""
# -*- coding: utf-8 -*-
#importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plotagem

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


# Inicio: importação do arquivo - observar separador e encoding
arquivo = 'ForEva.csv'
dados = pd.read_csv(arquivo, sep=';', encoding='WINDOWS-1252')

#dados = dados.drop(columns=['id','grade','turno'])

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
previsores = dados.loc[:,['diferenca_ingresso_ensino_anterior',
                          'tempo_conclusa',
                          'idade_ingresso',
                          'forma_ingresso',
                          'tipo_escola',
                          'percentual_progresso_faixa'
                          ]
                       ]

cabecPrevisores = previsores.columns

linhas, colunas = previsores.shape

# labelencoder conforme o formato do atributo
le = LabelEncoder()
for coluna in range(len(cabecPrevisores)):
    if previsores.loc[:,cabecPrevisores[coluna]].dtype == 'object':
        print('Atributo ' + cabecPrevisores[coluna] + ' Tipo ' + str(previsores.loc[:,cabecPrevisores[coluna]].dtype))
        previsores.loc[:,cabecPrevisores[coluna]] = le.fit_transform(previsores.loc[:,cabecPrevisores[coluna]])

# Normalização, se necessário
if 1 == 1:
    ss = StandardScaler()
    previsores = pd.DataFrame(ss.fit_transform(previsores))
    #previsores.to_csv('previsores_treinamento.csv', sep=',', index=False)
    #classe.to_csv('classe_treinamento.csv', sep=',', index=False)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5,
                init='k-means++',
                algorithm='auto',
                max_iter=5000,
                n_init=100,
                random_state=0)

kmeans.fit_predict(previsores)


#Getting the Centroids
centroides = kmeans.cluster_centers_
estudantes_cluster = kmeans.labels_
estudantes_cluster = pd.DataFrame(estudantes_cluster)

#Gera arquivo com os clusters e identificação doe studante
cabecCluster = ['matricula','nome','situacao','cluster']
conteudoCluster = pd.DataFrame(columns=cabecCluster)

for linhaCluster in range(len(estudantes_cluster)):
    conteudoCluster.loc[len(conteudoCluster.index)] = [dados.iloc[linhaCluster,1],
                                                       dados.iloc[linhaCluster,2],
                                                       dados.iloc[linhaCluster,28],
                                                       estudantes_cluster.iloc[linhaCluster,0]
                                                       ]
conteudoCluster.to_csv('conteudoCluster.csv', sep=',', index=False, encoding='WINDOWS-1252')
