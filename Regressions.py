from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, validation_curve
from bokeh.models import ColumnDataSource
from bokeh.models import Div
from bokeh.plotting import figure
from bokeh.models.widgets import DataTable, TableColumn
from time import time
import warnings
warnings.filterwarnings("ignore")


class Regression:

    def __init__(self, dataframe, explicatives, cible, taille, kv, kvoisin, NbTreeForRandomForest):
        var_quanti = [var_quanti for var_quanti in explicatives if dataframe[str(var_quanti)].dtypes!=object]
        self.taille = (100-taille)/100
        self.dataframe=dataframe
        self.Y=dataframe[cible]
        if len(var_quanti)>0:
            self.X = dataframe[var_quanti]
        else:
            self.X = dataframe[explicatives]
        print(self.X)
        self.cv = kv # for cross validation
        self.k = kvoisin
        self.NbTreeForRandomForest = NbTreeForRandomForest 
        #self.X = pandas.get_dummies(data=self.X, drop_first=True)
        self.XTrain, self.XTest, self.YTrain, self.YTest = train_test_split(self.X, self.Y, test_size=self.taille, random_state = 1)

    def afficher(self):
        print("dim: ",self.XTrain.shape)
        print("dim: ",self.YTrain.shape)
        print("dim: ",self.XTest.shape)
        print("dim: ",self.YTest.shape)
        print(self.taille)

    
    def regression_lineaire_multiple(self):
        # Creation du modéle
        lm = LinearRegression()
        lm.fit(self.XTrain,self.YTrain)
        # Les coefficient du modele
        self.title_intercept = Div(text= "<h4>Intercept_ :</h4>")
        self.intercept = Div(text=str(lm.intercept_)) 
        coeff_df=pd.DataFrame({"variables":self.X.columns,"coefficient":lm.coef_})
        columns=[TableColumn(field=Ci, title=Ci) for Ci in coeff_df.columns] 
        self.title_for_coeff=Div(text="<h4>Coefficients du modéle de regression </h4></br>")
        self.coefficients=DataTable(source=ColumnDataSource(coeff_df),columns=columns,width=400, height=400)
        # Phase de predictions sur l'echantillon de test
        predictions = lm.predict(self.XTest)
        # Mesure de performances
        self.title_performance = Div(text= "<h4>Indicateurs de performance</h4></br>")
        mae = metrics.mean_absolute_error(self.YTest, predictions)
        mse = metrics.mean_squared_error(self.YTest, predictions)
        rmse = np.sqrt(metrics.mean_squared_error(self.YTest, predictions))
        r2=metrics.r2_score(self.YTest, predictions)
        metrcs = ["MAE","MSE","RMSE","R2"]
        valeurs = [round(mae,4),round(mse,4),round(rmse,4),round(r2,4)]
        performances = pd.DataFrame(list(zip(metrcs,valeurs)), columns = ["Mesure","Valeurs"])
        columns_metrics=[TableColumn(field=Ci, title=Ci) for Ci in performances.columns] 
        self.metrics=DataTable(source=ColumnDataSource(performances),columns=columns_metrics)
        # Visualisation
        self.title_fig=Div(text= "<h4>Data Visualization</h4></br>")
        self.fig= figure(title="Y reel en vert et Y prédit en rouge", plot_height = 400, plot_width = 800)
        self.fig.circle(range(len(self.YTest)), predictions[np.argsort(self.YTest)], color="red", size=8)
        self.fig.line(range(len(self.YTest)), np.sort(self.YTest), color = "green", line_width=2)
        start = time()
        cross_validation = cross_val_score(lm, self.X, self.Y, cv=self.cv)
        end=time()
        self.duree_val_cv = Div(text="<h4>Temps d'éxecution : </h4>" + str(end-start))
        print(self.duree_val_cv)
        print(type(self.duree_val_cv))
        listCrosVal=["n°: "+str(i) for i in range(1,self.cv+1)]
        # resultats validation croisée
        dfTmp=pd.DataFrame({"n° cross validation":listCrosVal,"res":cross_validation})
        columns_vc=[TableColumn(field=col, title=col) for col in dfTmp.columns] 
        self.cross_validation=DataTable(source=ColumnDataSource(dfTmp),columns=columns_vc, width=400, height=400)
        self.title_vc=Div(text="<h4>Tableau validation croisée</h4></br>")
        self.title_moy_vc=Div(text="<h4>Moyenne des validations croisées: </h4>")
        self.mean_val_croisee=Div(text=str(mean(cross_validation)))
        return self
    
    def knn_reg(self):
        # creation du modele
        knn_reg = KNeighborsRegressor(self.k)
        knn_reg.fit(self.XTrain, self.YTrain)
        # prediction sur l'echantillon de test
        predictions = knn_reg.predict(self.XTest)
        # Mesures de performances
        self.title_performance = Div(text= "<h4>Indicateurs de performance</h4></br>")
        mae = metrics.mean_absolute_error(self.YTest, predictions)
        mse = metrics.mean_squared_error(self.YTest, predictions)
        rmse = np.sqrt(metrics.mean_squared_error(self.YTest, predictions))
        r2=metrics.r2_score(self.YTest, predictions)
        metrcs = ["MAE","MSE","RMSE","R2"]
        valeurs = [round(mae,4),round(mse,4),round(rmse,4),round(r2,4)]
        performances = pd.DataFrame(list(zip(metrcs,valeurs)), columns = ["Mesure","Valeurs"])
        columns_metrics=[TableColumn(field=col, title=col) for col in performances.columns] 
        self.metrics=DataTable(source=ColumnDataSource(performances),columns=columns_metrics)
        # visualisation des resultats de predictions
        self.title_fig=Div(text= "<h4>Visualization des resultats de predictions</h4></br>")
        self.fig= figure(title="Y reel en blue et Y prédit en rouge", plot_height = 400, plot_width = 800)
        self.fig.circle(range(len(self.YTest)), predictions[np.argsort(self.YTest)], color="red", size=8)
        self.fig.line(range(len(self.YTest)), np.sort(self.YTest), color = "blue", line_width=2)
        # Cross validation
        start = time()
        cross_validation = cross_val_score(knn_reg, self.X, self.Y, cv=self.cv)
        end=time()
        self.duree_val_cv = Div(text="<h4>Temps d'éxecution : </h4>" + str(end-start))
        listCrosVal = ["n°: "+str(i) for i in range(1,self.cv+1)]
        dfTmp=pd.DataFrame({"n° cross validation":listCrosVal,"res":cross_validation})
        columns_vc=[TableColumn(field=Ci, title=Ci) for Ci in dfTmp.columns] 
        self.cross_validation=DataTable(source=ColumnDataSource(dfTmp),columns=columns_vc, width=400, height=400)
        self.title_vc=Div(text="<h4>Tableau validation croisée</h4></br>")
        self.title_moy_vc=Div(text="<h4>Moyenne des validations croisées: </h4>")
        self.mean_val_croisee=Div(text=str(mean(cross_validation)))
        # courbe de cuve des validations croisées
        k = np.arange(1,50)
        train_score, val_score = validation_curve(knn_reg, self.X, self.Y, 'n_neighbors', k, cv = self.cv)
        #self.title_fig=Div(text= "<h4>Courbe de cuve des validations croisées:</h4></br>")
        self.figcurve= figure(title="Courbe de cuve des validations croisées", plot_height = 400, plot_width = 500)
        self.figcurve.line(k, val_score.mean(axis=1))
        return self
    
    def random_forest_reg(self):
        # creation du modele
        rf_reg = RandomForestRegressor(n_estimators = self.NbTreeForRandomForest, random_state = 42)
        rf_reg.fit(self.XTrain, self.YTrain)
        # prediction sur l'echantillon de test
        predictions = rf_reg.predict(self.XTest)
         # Mesures de performances
        self.title_performance = Div(text= "<h4>Indicateurs de performance</h4></br>")
        mae = metrics.mean_absolute_error(self.YTest, predictions)
        mse = metrics.mean_squared_error(self.YTest, predictions)
        rmse = np.sqrt(metrics.mean_squared_error(self.YTest, predictions))
        r2=metrics.r2_score(self.YTest, predictions)
        metrcs = ["MAE","MSE","RMSE","R2"]
        valeurs = [round(mae,4),round(mse,4),round(rmse,4),round(r2,4)]
        performances = pd.DataFrame(list(zip(metrcs,valeurs)), columns = ["Mesure","Valeurs"])
        columns_metrics=[TableColumn(field=col, title=col) for col in performances.columns] 
        self.metrics=DataTable(source=ColumnDataSource(performances),columns=columns_metrics)
        # visualisation des resultats de predictions
        self.title_fig=Div(text= "<h4>Visualization des resultats de predictions</h4></br>")
        self.fig= figure(title="Y reel en blue et Y prédit en rouge", plot_height = 400, plot_width = 800)
        self.fig.circle(range(len(self.YTest)), predictions[np.argsort(self.YTest)], color="red", size=8)
        self.fig.line(range(len(self.YTest)), np.sort(self.YTest), color = "blue", line_width=2)
        # Cross validation
        start = time()
        cross_validation = cross_val_score(rf_reg, self.X, self.Y, cv=self.cv)
        end=time()
        self.duree_val_cv = Div(text="<h4>Temps d'éxecution : </h4>" + str(end-start))
        listCrosVal = ["n°: "+str(i) for i in range(1,self.cv+1)]
        dfTmp=pd.DataFrame({"n° cross validation":listCrosVal,"res":cross_validation})
        columns_vc=[TableColumn(field=Ci, title=Ci) for Ci in dfTmp.columns] 
        self.cross_validation=DataTable(source=ColumnDataSource(dfTmp),columns=columns_vc, width=400, height=400)
        self.title_vc=Div(text="<h4>Tableau validation croisée</h4></br>")
        self.title_moy_vc=Div(text="<h4>Moyenne des validations croisées: </h4>")
        self.mean_val_croisee=Div(text=str(mean(cross_validation)))
        return self


        






        