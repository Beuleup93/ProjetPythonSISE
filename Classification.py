#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 17:09:39 2021

@author: macbookair
"""
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn import model_selection
import statsmodels as sm
from statsmodels.tools import add_constant
import matplotlib.pyplot as plt
import numpy as np
import pandas
from time import time

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure
from bokeh.models.widgets import DataTable,TableColumn,StringFormatter, DataCube,GroupingInfo
from bokeh.models import  Div


class Classification:
    
    def __init__(self, dataframe, explicatives, cible, taille, cv,nb_feuille):
        var_quali = [var_quali for var_quali in explicatives if dataframe[str(var_quali)].dtypes==object]
        self.taille = (100-taille)/100
        self.dataframe=dataframe
        self.Y=dataframe[cible]
        if len(var_quali)>0:
            self.X = pandas.get_dummies(dataframe[explicatives], columns=var_quali) 
        else:
            self.X = dataframe[explicatives]
        print(self.X)
        self.cible = cible
        self.cv = cv
        self.nb_feuille = nb_feuille
        self.explicatives = explicatives
        self.XTrain, self.XTest, self.YTrain, self.YTest = train_test_split(self.X, self.Y, test_size=self.taille, random_state = 1)
        


    def arbre_decision_class(self):
        
        # creation du modéle
        dtree = DecisionTreeClassifier(max_depth = self.nb_feuille)
        dtree.fit(self.XTrain,self.YTrain)
        self.title_algo=Div(text="<h4><center>ARBRE DE DECISION</center></h4>")
        print(dtree)
        '''
            Affichage de l'arbre avec toutes les informations
            Affichage sous formes d'un ensemble de régles
            On stocke le plot sous forme d'image dans le rep courant
            puis on l'affiche dans l'interfache graphique
        ''' 
        tree_rules = export_graphviz(dtree,feature_names = list(self.XTrain.columns))
        self.reglesT=Div(text="<h4>Régles de décision </h4>")
        self.regles=Div(text=str(tree_rules))
        self.treeT=Div(text="<h4>Arbre de décision : </h4>")
        plt.figure()
        plot_tree(dtree,feature_names = list(self.XTrain.columns),filled=True)
        plt.savefig('tree.jpg', dpi=95)
        self.tree= Div(text="""<img src="tree.jpg", alt="'tree.jpg' est enregistrée dans le dossier courant">""", width=150, height=150)
        
        # importance des variables
        imp = {"VarName":self.X.columns,"Importance":dtree.feature_importances_}
        self.title_impVariables=Div(text="<h4>Importance des variables :</h4>")
        tmp=pandas.DataFrame(imp).sort_values(by="Importance",ascending=False)
        columns=[TableColumn(field=col, title=col) for col in tmp.columns] 
        self.importance_variables=DataTable(source=ColumnDataSource(tmp),columns=columns)
        
        # Prediction sur l'echantillon de test               
        predictions = dtree.predict(self.XTest)                 
        
        #distribution des classes prédites
        self.distribprediction1=Div(text="Classe de prédiction : <br/>")
        temp=pandas.DataFrame({"var":np.unique(predictions,return_counts=True)[0],"distribution":np.unique(predictions,return_counts=True)[1]})
        columns=[TableColumn(field=col, title=col) for col in temp.columns] 
        self.distribpredictions2=DataTable(source=ColumnDataSource(temp),columns=columns)
    
        #matrice de confusion
        conf_matrix = metrics.confusion_matrix(self.YTest,predictions)
        self.title_matrice_confusion=Div(text="</br><h4>Matrice de confusion :</h4></br>")
        d = dict()
        d["affichage"]= []
        d["var"]=self.dataframe[str(self.cible)].unique()
        for i in range(len(d["var"])):
            d[d["var"][i]]=list(conf_matrix[i])
            d["affichage"].append("")
        
        source = ColumnDataSource(data=d)
        target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
        formatter = StringFormatter(font_style='bold')
        columns=[TableColumn(field='var', title=str(self.cible), width=40, sortable=False, formatter=formatter)]
        columns[1:(len(self.dataframe[str(self.cible)].unique()))]=[TableColumn(field=str(modalite), title=str(modalite), width=40, sortable=False) for modalite in self.dataframe[str(self.cible)].unique()]
        grouping = [GroupingInfo(getter='affichage'),]
        self.cube = DataCube(source=source, columns=columns, grouping=grouping, target=target)
        #Taux de reconnaissance
        acc_score = metrics.accuracy_score(self.YTest,predictions)
        self.taux_reconnaissance=Div(text="<h4>Taux de reconnaissance :</h4>" + str(round(acc_score,4)))
        #Taux d'erreur
        self.taux_erreur=Div(text="<h4>Taux d'erreur :</h4>" + str(round(1.0-metrics.accuracy_score(self.YTest,predictions),4)))
        #rappel par classe
        self.rapel_class=Div(text="<h4>Rappel par classe :</h4>" +str(metrics.recall_score(self.YTest,predictions,average=None)))
        #precision par classe
        self.precclasse=Div(text="<h4>Précision par classe : </h4>" + str(metrics.precision_score(self.YTest,predictions,average=None)))
        #rapport général
        rapport_general = metrics.classification_report(self.YTest,predictions)
        self.title_rapport=Div(text="<h4> Rapport sur la qualité de prédiction :</h4>")
        self.rapport=Div(text=str(rapport_general))
        # Evaluation du modéle en validation croisée
        start=time()
        succes = model_selection.cross_val_score(dtree,self.X,self.Y,cv=self.cv,scoring='accuracy')
        end=time()
        self.duree_val_cv = Div(text="<h4>Temps d'éxecution : </h4>" + str(end-start))
        self.int_succes=Div(text="<h4>Succès de la validation croisée :</h4>"+ str(succes))
        #moyenne des taux de succès = estimation du taux de succès en CV
        self.moy_succes=Div(text="<h4>Moyenne des succès :</h4>" + str(round(succes.mean(),4))) 
        return self  
        
    
    def analyse_discriminante_class(self):
        # creation du modéle
        lda = LinearDiscriminantAnalysis()
        lda.fit(self.XTrain,self.YTrain)
        # cas d'une variable y binaire
        if len(self.Y.unique())==2 : 
            self.title_coefficients=Div(text="<h4>Coefficient du modéle : </h4>")
            tmpDf=pandas.DataFrame({"var":self.XTrain.columns,"positif":lda.coef_[0]})
            columns=[TableColumn(field=col, title=col) for col in tmpDf.columns] 
            self.coefficients=DataTable(source=ColumnDataSource(tmpDf),columns=columns)
        else:
        # cas d'une variable y multiclass
            tmpDf= pandas.DataFrame(lda.coef_.transpose(), columns=lda.classes_, index=self.XTrain.columns)
            tmp={tmpDf.columns[0] : [lda.intercept_[0]]}
            for i in range(1,len(tmpDf.columns)): 
                tmp.update({tmpDf.columns[i] : [lda.intercept_[i]]})
                
            tmp=pandas.DataFrame(tmp)
            tmp=tmp.rename(index={0 : "Intercept_"})
            final=pandas.concat([tmp,tmpDf])
            self.title_coefficients=Div(text="<h4>Table des coefficients et des intercepts : </h4>")
            d = dict()
            d["affichage"]= ['','','','','']
            d["var"]=['Constante', 'area','parameter','compactness',"len_kernel"]
            for i in (range(len(final.columns))):
                d[final.columns[i]]=final.iloc[:,i]
                
            source = ColumnDataSource(data=d)
            target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
            formatter = StringFormatter(font_style='bold')
            columns=[TableColumn(field='var', title="", width=40, sortable=False, formatter=formatter)]
            columns[1:(len(final.columns))]=[TableColumn(field=str(NomMod), title=str(NomMod), width=40, sortable=False) for NomMod in final.columns]
            grouping = [GroupingInfo(getter='affichage'),]
            self.coefficients = DataCube(source=source, columns=columns, grouping=grouping, target=target)
            
        ''' Phase de prediction'''
        predictions = lda.predict(self.XTest)
        '''matrice de confusion'''
        matrix_confusion=pandas.crosstab(self.YTest,predictions)
        matrix_confusion_val = matrix_confusion.values
        self.title_matrice_confusion=Div(text="</br><h4>Matrice de confusion :</h4>")
        data_cube = dict()
        data_cube["affichage"]= []
        data_cube["var"]=self.dataframe[str(self.cible)].unique()
        for i in range(len(data_cube["var"])):
            data_cube[data_cube["var"][i]]=matrix_confusion_val[i]
            data_cube["affichage"].append("")
            
        source = ColumnDataSource(data=data_cube)
        target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
        formatter = StringFormatter(font_style='bold')
        columns=[TableColumn(field='var', title=str(self.cible), width=40, sortable=False, formatter=formatter)]
        columns[1:(len(self.dataframe[str(self.cible)].unique()))]=[TableColumn(field=str(modalite), title=str(modalite), width=40, sortable=False) for modalite in self.dataframe[str(self.cible)].unique()]
        grouping = [GroupingInfo(getter='affichage'),]
        self.cube = DataCube(source=source, columns=columns, grouping=grouping, target=target)
        '''taux de reconnaissance '''
        sumDiagonal = np.sum(np.diagonal(matrix_confusion_val))/np.sum(matrix_confusion_val)
        self.taux_reconnaissance=Div(text="<h4>Taux de reconnaissance :</h4> " + str(round(sumDiagonal,4)))
        '''calcul du taux d'erreur'''
        self.taux_erreur=Div(text="<h4>Taux d'erreur :</h4><br/>" + str(round(1.0-metrics.accuracy_score(self.YTest,predictions),4)))
        '''sensibilité(rappel) et précision par classe'''
        self.rapport_qualite=Div(text="<h4>Rapport sur la qualité de prédiction : </h4>"+str(metrics.classification_report(self.YTest,predictions)))
        '''Validation croisée by defalt cv = 5'''
        start = time()
        cv_score = model_selection.cross_val_score(lda,self.X,self.Y,cv=self.cv,scoring='accuracy')
        end=time()
        self.duree_val_cv = Div(text="<h4>Temps d'éxecution : </h4>" + str(end-start))
        self.int_succes=Div(text="<h4>Succès de la validation croisée :</h4>"+ str(cv_score))
        '''estimation du taux de succès en CV'''
        self.moy_succes_rate=Div(text="<h4>Moyenne des accuracy :</h4>" + str(round(cv_score.mean(),4))) 
        return self
        
    def logistic_reg_class(self):
        
        ''' ICI NOUS DEVONS TESTER LE NOMBRE DE MODALITÉ DE LA VARIABLE CIBLE: 
            REGRESSION LOGISTIQUE BINAIRE OU REGRESSION MULTINOMIALE
        ''' 
        if len(self.Y.unique())==2:
            modlite1 = self.y.unique()[0]
            modalite2 = self.y.unique()[1]
            ''' Recodage des modalité en 0 et 1'''
            for i in range(0, len(self.dataframe[self.cible])):
                if self.dataframe[self.cible][i] == modlite1 : 
                    self.dataframe[self.cible][i] = 0
                if self.dataframe[self.cible][i] == modalite2 : 
                    self.dataframe[self.cible][i] = 1
            
            self.Y = self.dataframe[self.cible]
            self.X = self.dataframe[self.explicatives]
            '''changeons le type cible en int'''
            self.dataframe[str(self.var_cible)] = self.dataframe[str(self.var_cible)].astype('int')
            self.XTrain, self.XTest, self.YTrain, self.YTest = train_test_split(self.X, self.Y, test_size=self.taille, random_state = 1, stratify=self.dataframe[self.cible])

        ''' Instanciation de l'objet logistique selon la regression multi ou binary'''
        isBinary = len(self.dataframe[self.cible].unique())==0
        
        if isBinary:
            regLog = LogisticRegression(penalty='none')
        else:
            regLog = LogisticRegression(penalty='none', multi_class='multinomial')
        
        '''Ramenons les variables sur les unité: Standardisation'''
        standz = preprocessing.StandardScaler()
        '''Colonne de biais'''
        XTrain = sm.tools.add_constant(self.XTrain)
        '''Centrage reduction'''
        XTBrain = standz.fit_transform(XTrain)
        '''Création du modéle logistique'''
        regLog.fit(XTBrain,self.YTrain)
        '''Normalisation des coefficients du modéle avec les ecartype du modéle'''
        coefsStd = regLog.coef_[0] / standz.scale_
        self.coefficients=Div(text="<h4>Coefficients du modéles : </h4>")
        tmpCoef=pandas.DataFrame({"var":XTrain.columns,"coef":coefsStd})
        columns=[TableColumn(field=cols, title=cols) for cols in tmpCoef.columns] 
        self.coefDatable=DataTable(source=ColumnDataSource(tmpCoef),columns=columns)
        '''Intercept'''
        intercept = regLog.intercept_ + np.sum(regLog.coef_[0]*(-standz.mean_/standz.scale_))
        self.intercept=Div(text= "<h4>Intercepts :</h4>"+str(intercept))
        
        if isBinary:
            '''Recuperons les probabilité'''
            probalitesPred = regLog.predict_proba(XTrain)[:,1]
            logLikelihood = np.sum((self.YTrain * np.log(probalitesPred))+((1.0-self.YTrain)*np.log(1.0-probalitesPred)))
            self.log_vraisemblance=Div(text="<h4>La log-vraisemblance vaut : </h4>"+str(round(logLikelihood,4)))
        else :
            self.log_vraisemblance=Div(text="")
            
        '''Prediction sur l'echantillon de teste'''
        XTest = add_constant(self.XTest)
        XBTest = standz.transform(XTest)
        
        if isBinary:
            predProbas = regLog.predict_proba(XBTest)
            ''' 1 est notre modalité positive'''
            predClass = np.where(predProbas[:,1] > 0.5, 1, 0)
            '''Matrice de confusion'''
            mc=pandas.crosstab(self.YTest,predClass)
            self.matrice_confusion=Div(text="</br><h4>Matrice de confusion :</h4>")
            source = ColumnDataSource(data=dict(
                affichage=["",""],
                var=['positif','negatif'],
                positif=[mc[0][0],mc[0][1]],
                negatif=[mc[1][0],mc[1][1]]
            ))
            target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
            formatter = StringFormatter(font_style='bold')
            columns = [
                TableColumn(field='var', title=str(self.cible), width=40, sortable=False, formatter=formatter),
                TableColumn(field='positif', title='positif', width=40, sortable=False),
                TableColumn(field='negatif', title='negatif', width=40, sortable=False),
            ]
            grouping = [
                GroupingInfo(getter='affichage'),
            ]
            self.cube = DataCube(source=source, columns=columns, grouping=grouping, target=target)
            
            matrix_numpy = mc.values
            '''Taux de reconnaissance'''
            accuracy = np.sum(np.diagonal(matrix_numpy))/np.sum(matrix_numpy)
            self.accuracy=Div(text="<h4>Taux de reconnaissance : </h4>" + str(round(accuracy,4)))
            '''Taux d'error'''
            error_rate = 1.0 - accuracy
            self.error_rate=Div(text="<h4>Taux d'erreur : </h4><br/>"+str(round(error_rate,4)))
            '''rapport sur la qualité de prédictions'''
            self.rapport_qualite=Div(text="<h4>Rapport sur la qualité de prédiction : </h4>" + str(metrics.classification_report(self.YTest,predClass)))
            '''Courbe ROC et Accuracy sur l'echantillon de test'''
            fprSm, tprSm, _ = metrics.roc_curve(self.YTest,predProbas[:,1],pos_label=1)
            '''Courbe ROC'''
            self.fig_roc= figure(title="Courbe ROC")
            self.fig_roc.multi_line(xs=[fprSm,np.arange(0,1.1,0.1)],ys=[tprSm,np.arange(0,1.1,0.1)], color=['green','blue'])
            
            '''Taux de reconnaissance sur l'echantillon de teste'''
            accuracy_roc = metrics.roc_auc_score(self.YTest,predClass)
            self.accuracy_roc=Div(text=" AUC : " +str(round(accuracy_roc,4)))
        
        else:
            '''Prediction sur l'echantillon de teste'''
            predClass = regLog.predict(XBTest)
            '''Matrice de confusion'''
            self.matrice_confusion=Div(text="</br><h4>Matrice de confusion :</h4>")
            mc=pandas.crosstab(self.YTest,predClass)
            matrix_numpy = mc.values
            data_cube = dict()
            data_cube["affichage"]= []
            data_cube["var"]=self.dataframe[self.cible].unique()
            for i in range(len(data_cube["var"])):
                data_cube[data_cube["var"][i]] = matrix_numpy[i]
                data_cube["affichage"].append("")
                
            source = ColumnDataSource(data=data_cube)
            target = ColumnDataSource(data=dict(row_indices=[], labels=[]))
            formatter = StringFormatter(font_style='bold')
            columns=[TableColumn(field='var', title=self.cible, width=40, sortable=False, formatter=formatter)]
            columns[1:(len(self.dataframe[self.cible].unique()))]=[TableColumn(field=str(modalite), title=str(modalite), width=40, sortable=False) 
                                                                   for modalite in self.dataframe[self.cible].unique()]
            grouping = [GroupingInfo(getter='affichage'),]
            self.cube = DataCube(source=source, columns=columns, grouping=grouping, target=target)
        
            '''Taux de reconnaissance'''
            accuracy = np.sum(np.diagonal(matrix_numpy))/np.sum(matrix_numpy)
            self.accuracy=Div(text="<h4>Taux de reconnaissance :</h4> " + str(round(accuracy,4)))
            '''Taux d'erreur'''
            error_rate = 1.0 - accuracy
            self.error_rate=Div(text="<h4>Taux d'erreur : </h4>"+str(round(error_rate,4)))
            '''rapport sur la qualité de prédictions'''
            self.rapport_qualite=Div(text="<h4>Rapport sur la qualité de prédiction : </h4>" + str(metrics.classification_report(self.YTest,predClass)))
            self.fig_roc= Div(text="")
            self.accuracy_roc=Div(text="")
            
        '''Evaluation du modéle en validation'''
        start = time()
        succes = model_selection.cross_val_score(regLog,self.X,self.Y,cv=self.cv,scoring='accuracy')
        end=time()
        self.duree_val_cv = Div(text="<h4>Temps d'éxecution : </h4>" + str(end-start))
        self.int_succes=Div(text="<h4>Succès de la validation croisée :</h4>"+ str(succes))
        '''Estimation des taux de succes: Moyenne des succes'''
        self.moy_succes=Div(text="<h4>Moyenne des succès :</h4>" + str(round(succes.mean(),4))) 
        return self
