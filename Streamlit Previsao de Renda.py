import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
import patsy 
import statsmodels.api as sm
import emoji

st.set_page_config(page_title='Previsão de Renda', page_icon=emoji.emojize(":money_with_wings:"))


#### - Def´s ########
# Verificação de outliers
def det_outliers(df_1, campo, met='quartil'):
    q1=df_1[campo].quantile(0.25)
    q3=df_1[campo].quantile(0.75)
    print(f'primeiro quartil = {round(q1,2)} e terceiro = {round(q3,2)}')
    if met == 'quartil':
        #Método do Quartil
        iqr = q3-q1
        lim_inf= q1 - (iqr * 1.5)
        lim_sup= q1 + (iqr * 1.5)
        print('Método Quartil')
        print(f'Limite inferior = {round(lim_inf,2)} e limite superior = {round(lim_sup,2)}')
        return lim_inf, lim_sup
    else:
        #método do desvio padrão
        media = np.mean(df_1[campo])
        desvio = np.std(df_1[campo])
        cut_off = desvio * 3
        lim_inf_dev = media - cut_off
        lim_sup_dev = media + cut_off
        print('Método Desvio Padrão')
        print(f'Limite inferior = {round(lim_inf_dev,2)} e limite superior = {round(lim_sup_dev,2)}')
        return lim_inf_dev, lim_sup_dev 
   

####Executando o modelo proposto de stepwise
def stepwise_selection(X, y, 
                        initial_list=[], 
                        threshold_in=0.01, 
                        threshold_out = 0.05, 
                        verbose=True):
        """ Perform a forward-backward feature selection 
        based on p-value from statsmodels.api.OLS
        Arguments:
            X - pandas.DataFrame with candidate features
            y - list-like with the target
            initial_list - list of features to start with (column names of X)
            threshold_in - include a feature if its p-value < threshold_in
            threshold_out - exclude a feature if its p-value > threshold_out
            verbose - whether to print the sequence of inclusions and exclusions
        Returns: list of selected features 
        Always set threshold_in < threshold_out to avoid infinite looping.
        See https://en.wikipedia.org/wiki/Stepwise_regression for the details
        """
        # deixei alguns comando prints na rotina somente para não perder os IF´s de alerta que tenho no programa.
        #Mas para título deste exercício o que me interessa é o resultado com as variáveis mais importantes para o modelo.
        included = list(initial_list)
        #print("included")
        #print(included)
       
        while True:
            changed=False
            # forward step
            excluded = list(set(X.columns)-set(included))
            new_pval = pd.Series(index=excluded)
            
            for new_column in excluded:
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()                
                #print("modelo 1")
                #print(model.summary())
                new_pval[new_column] = model.pvalues[new_column]
                #print("new_pval")
                #print(new_pval)
                
            best_pval = new_pval.min()
            #print("best_pval")
            #print(best_pval)      
            if best_pval < threshold_in:
                #print("best_pval")
                best_feature = new_pval.idxmin()
                included.append(best_feature)
                changed=True
                if verbose:
                    print("Entrou")
                    #st.write('Entrou na rotina de cáclulo de melhor variáveis')
                    #print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))
              
            
            #sys.exit()
            # backward step
            #model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
            #print("modelo 2")
            #st.write("modelo 2")
            #print(model.summary())
            #st.write(model.summary())
            # use all coefs except intercept
            pvalues = model.pvalues.iloc[1:]
            worst_pval = pvalues.max() # null if pvalues is empty
            #print("worst_pval")
            #print(worst_pval) 
            if worst_pval > threshold_out:
                changed=True
                worst_feature = pvalues.idxmax()
                included.remove(worst_feature)
                if verbose:
                    print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))
                    #st.write(' ')
            if not changed:
                break
        return included

#### - Def´s Fim ########

sns.set(context='talk', style='ticks')

st.write('# Análise exploratória da previsão de renda')

renda = pd.read_csv('./input/previsao_de_renda.csv')





#plots
fig, ax = plt.subplots(8,1,figsize=(10,70))
renda[['posse_de_imovel','renda']].plot(kind='hist', ax=ax[0])
st.write('## Gráficos ao longo do tempo')
sns.lineplot(x='data_ref',y='renda', hue='posse_de_imovel',data=renda, ax=ax[1])
ax[1].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='posse_de_veiculo',data=renda, ax=ax[2])
ax[2].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='qtd_filhos',data=renda, ax=ax[3])
ax[3].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_renda',data=renda, ax=ax[4])
ax[4].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='educacao',data=renda, ax=ax[5])
ax[5].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='estado_civil',data=renda, ax=ax[6])
ax[6].tick_params(axis='x', rotation=45)
sns.lineplot(x='data_ref',y='renda', hue='tipo_residencia',data=renda, ax=ax[7])
ax[7].tick_params(axis='x', rotation=45)
sns.despine()
st.pyplot(plt)

st.write('## Gráficos bivariada')
fig, ax = plt.subplots(7,1,figsize=(10,50))
sns.barplot(x='posse_de_imovel',y='renda',data=renda, ax=ax[0])
sns.barplot(x='posse_de_veiculo',y='renda',data=renda, ax=ax[1])
sns.barplot(x='qtd_filhos',y='renda',data=renda, ax=ax[2])
sns.barplot(x='tipo_renda',y='renda',data=renda, ax=ax[3])
sns.barplot(x='educacao',y='renda',data=renda, ax=ax[4])
sns.barplot(x='estado_civil',y='renda',data=renda, ax=ax[5])
sns.barplot(x='tipo_residencia',y='renda',data=renda, ax=ax[6])
sns.despine()
st.pyplot(plt)


#############preparação da Base eliminação dos outliers
# Eliminação de colunas que não serão utilizados no modelo e registros com dados faltantes. 

st.write("# Arquivo Base para nosso trabalho eliminando campos não importantes e campos em branco")

renda.drop(columns=['Unnamed: 0','data_ref','id_cliente'],inplace=True)
renda = renda.dropna()

st.subheader('Tabela de Dados')
st.dataframe(renda)


#Verificação da distribuição da variável renda.
renda_aux = renda['renda']
plt.hist(renda_aux, density=True, bins=50,rwidth=0.7) 
plt.show()

#Histograma
st.write('----')
st.subheader('Histograma Renda Original')
hist_values = np.histogram(
    renda['renda'], bins=50, range=(0,500000))[0]
  #  DATE_COLUMN
st.bar_chart(hist_values)

st.write('----')

# Verificação e eliminação de outliers
limites= det_outliers(renda,'renda',met='x')
renda = renda[(renda['renda'] >= limites[0]) & (renda['renda'] <= limites[1])]
#print(renda.shape)

# Novo Gráfico
st.subheader('Histograma Renda eliminando outliers')
hist_values = np.histogram(
    renda['renda'], bins=50, range=(0,500000))[0]
  #  DATE_COLUMN
st.bar_chart(hist_values)

st.write('----')
st.write(' ')
st.subheader("Criamos variáveis dummies para os campos 'sexo', 'posse_de_veiculo', 'posse_de_imovel','tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia'")
st.write(' ')
st.write('----')


#Criação de variáveis dummies
df_dummies_renda = pd.get_dummies(renda, columns=['sexo', 'posse_de_veiculo', 'posse_de_imovel','tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia'])

#eliminando colunas dos dummies que não serão utilizadas no modelo e organizando o dataframe
df_dummies_renda.drop(columns=['sexo_M', 'posse_de_veiculo_True','posse_de_imovel_False','tipo_renda_Bolsista'
                               , 'educacao_Pós graduação','estado_civil_Viúvo', 'tipo_residencia_Estúdio'], inplace=True)

#Organizando
df_dummies_renda= df_dummies_renda[['qtd_filhos', 'idade', 'tempo_emprego', 'qt_pessoas_residencia'
                                       , 'sexo_F', 'posse_de_veiculo_False', 'posse_de_imovel_True'
                                       ,'tipo_renda_Assalariado', 'tipo_renda_Empresário'
                                       ,'tipo_renda_Pensionista', 'tipo_renda_Servidor público'
                                       ,'educacao_Primário', 'educacao_Secundário'
                                       ,'educacao_Superior completo', 'educacao_Superior incompleto'
                                       ,'estado_civil_Casado', 'estado_civil_Separado', 'estado_civil_Solteiro'
                                       ,'estado_civil_União', 'tipo_residencia_Aluguel', 'tipo_residencia_Casa'
                                       ,'tipo_residencia_Com os pais', 'tipo_residencia_Comunitário'
                                       ,'tipo_residencia_Governamental','renda']]



#separação dos dados em treino e teste
y= df_dummies_renda['renda']
x = df_dummies_renda.drop('renda',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.25, random_state=100)

##### Falar que o modelo Stepwise selecionou algumas variáveis
#Rodando modelo stepwise
