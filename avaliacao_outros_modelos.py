import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit,RandomizedSearchCV
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from scipy.stats import uniform
#from sklearn.neighbors import KNeighborsRegressor


#Dados Iris
iris=load_iris().data
classe=load_iris().target
classe=np.array([str(i) for i in classe]).reshape(-1,1)
Transf=OneHotEncoder(sparse=False)
classe=Transf.fit_transform(classe)
x=np.column_stack((iris[:,1:4],classe))
y=iris[:,0]


#Dados mpg
dados=pd.read_csv("auto_mpg_ajeitado.csv")
#pd.isnull(dados)
#np.isnan(dados)
#b=np.setdiff1d(np.arange(398),np.array([ 32, 126, 330, 336, 354, 374]))
#dados=dados.values[b,:]
y=dados.values[:,6]
x=dados.values[:,[1,2,3,4,5,7,8]]
#x[0,:]

#Dados slump test
dados=pd.read_csv("slump_test.csv")
y=dados.values[:,10]
x=dados.values[:,np.arange(1,10)]

#Dados parkinson
dados=pd.read_csv("parkinsons_updrs.csv")
y=dados.values[:,4]
x=dados.values[:,np.setdiff1d(np.arange(22),4)]

#Dados qualidade do vinho
dados=pd.read_csv("winequality-red.csv")
y=dados.values[:,11]
x=dados.values[:,np.setdiff1d(np.arange(12),11)]

#Dados preço das casas(Real estate valuation)
dados=pd.read_csv("Real_estate_valuation_data_set.csv")
y=dados.values[:,7]
x=dados.values[:,np.setdiff1d(np.arange(8),[0,1,7])]


tabela_media=[]
desvio=[]
hiper_algoritmos=[]
k=ShuffleSplit(n_splits=20, test_size=0.3, random_state=0)

#Modelos
#Regressão Linear
modelo=LinearRegression()
erro=[]

for treino_index, teste_index in k.split(x):
    xtreino=x[treino_index,:] 
    xteste=x[teste_index,:]
    ytreino= y[treino_index] 
    yteste=y[teste_index]
    modelo.fit(xtreino,ytreino)
    erro_medio_absoluto=mean_absolute_error(yteste,modelo.predict(xteste))
    erro.append(erro_medio_absoluto)
print(np.mean(erro))
tabela_media.append(np.mean(erro))
desvio.append(np.std(erro))
hiper_algoritmos.append("Nenhum")


#Regressão Ridge
modelo=Ridge()
#uniform(loc=0, scale=4).rvs(size=10)
hiperp = {"alpha":uniform(loc=0,scale=7)}
Otimizacao = RandomizedSearchCV(modelo, hiperp,cv=k, n_iter = 30
                                ,scoring="neg_mean_absolute_error", random_state=0)
Otimizacao.fit(x, y)
Otimizacao.best_params_
erro=-Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(-Otimizacao.cv_results_["mean_test_score"])[0]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_algoritmos.append(Otimizacao.best_params_)

#Regressão Lasso
modelo=Lasso()
hiperp = {"alpha":uniform(loc=0,scale=7)}
Otimizacao = RandomizedSearchCV(modelo, hiperp,cv=k, n_iter = 30
                                ,scoring="neg_mean_absolute_error", random_state=0)
Otimizacao.fit(x, y)
Otimizacao.best_params_
erro=-Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(-Otimizacao.cv_results_["mean_test_score"])[0]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_algoritmos.append(Otimizacao.best_params_)

#Floresta Aleatória
modelo=RandomForestRegressor()
hiperp = {"n_estimators":[100,200,250,300,400],"max_depth":[2,3,4],
                 "min_samples_split":[10,15,20,25,30]
                 ,"min_samples_leaf":[10,15,8,20]}
Otimizacao = RandomizedSearchCV(modelo, hiperp,cv=k, n_iter = 30
                                ,scoring="neg_mean_absolute_error", random_state=0)
Otimizacao.fit(x, y)
Otimizacao.best_params_
erro=-Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(-Otimizacao.cv_results_["mean_test_score"])[0]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_algoritmos.append(Otimizacao.best_params_)

#Árvore de Regressão
modelo=DecisionTreeRegressor()
hiperp = {"max_depth":[2,3,4],
                 "min_samples_split":[10,15,20,25,30]
                 ,"min_samples_leaf":[10,15,8,20]}
Otimizacao = RandomizedSearchCV(modelo, hiperp,cv=k, n_iter = 30
                                ,scoring="neg_mean_absolute_error", random_state=0)
Otimizacao.fit(x, y)
Otimizacao.best_params_
erro=-Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(-Otimizacao.cv_results_["mean_test_score"])[0]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_algoritmos.append(Otimizacao.best_params_)

#MultiLayerPerceptron
modelo=MLPRegressor(max_iter=500)
hiperp={"activation":["identity", "logistic", "tanh","relu"],
"hidden_layer_sizes":[20,50,80,(20,80),10,8,(100,50),100,300,(250,100,50)]
,"learning_rate_init":uniform(loc=0,scale=0.4)}
Otimizacao = RandomizedSearchCV(modelo, hiperp,cv=k, n_iter = 30
                                ,scoring="neg_mean_absolute_error", random_state=0)
Otimizacao.fit(x, y)
Otimizacao.best_params_
erro=-Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(-Otimizacao.cv_results_["mean_test_score"])[0]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_algoritmos.append(Otimizacao.best_params_)

#SVM
modelo=SVR()
hiperp={"kernel":["linear", "poly", "rbf", "sigmoid"],
       "C":uniform(loc=0,scale=5)}
Otimizacao = RandomizedSearchCV(modelo, hiperp,cv=k, n_iter = 30
                                ,scoring="neg_mean_absolute_error", random_state=0)
Otimizacao.fit(x, y)
Otimizacao.best_params_
erro=-Otimizacao.best_score_
tabela_media.append(erro)
ind_menor_erro=np.argsort(-Otimizacao.cv_results_["mean_test_score"])[0]
dp=Otimizacao.cv_results_["std_test_score"][ind_menor_erro]
desvio.append(dp)
hiper_algoritmos.append(Otimizacao.best_params_)

tabela_media
desvio
hiper_algoritmos

tabela_algoritmos_python=pd.DataFrame(np.column_stack((tabela_media,desvio,
                                        hiper_algoritmos)),index=[
        "Regressao Linear","Regressao Ridge",
"Regressao Lasso","Floresta Aleatoria","Arvore de Regressao"
,"MLP","SVM"],columns=["Erro Medio Absoluto","Desvio Padrao","Hiperparametros"])
tabela_algoritmos_python
tabela_algoritmos_python.to_csv(path_or_buf="\\Users\\Alberto\\Desktop\\Trabalhos\\Monografia\\resultados_erros_python_iris.csv")
tabela_algoritmos_python.to_csv(path_or_buf="\\Users\\Alberto\\Desktop\\Trabalhos\\Monografia\\resultados_erros_python_auto_mpg.csv")
tabela_algoritmos_python.to_csv(path_or_buf="\\Users\\Alberto\\Desktop\\Trabalhos\\Monografia\\resultados_erros_python_slump_test.csv")
tabela_algoritmos_python.to_csv(path_or_buf="\\Users\\Alberto\\Desktop\\Trabalhos\\Monografia\\resultados_erros_python_parkinsons.csv")
tabela_algoritmos_python.to_csv(path_or_buf="\\Users\\Alberto\\Desktop\\Trabalhos\\Monografia\\resultados_erros_python_real_estate_valuation.csv")

#Exportação dados auto-mpg
pd.DataFrame(dados).to_csv(path_or_buf="C:\\Users\\Alberto\\Desktop\\Trabalhos\\Monografia\\conferencia_bahia\\auto_mpg_ajeitado.csv")