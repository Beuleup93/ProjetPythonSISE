#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:25:22 2021

@author: macbookair
"""
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import column,row, layout
from bokeh.models import DataTable, TableColumn, ColumnDataSource, Slider, Panel, Tabs, NumberFormatter,Select, Button,MultiSelect
from bokeh.models import CategoricalColorMapper
from bokeh.models import FileInput
import base64
import io
import pandas as pd
import numpy as np
from bokeh.models.widgets import Paragraph, Div
import warnings
warnings.filterwarnings("ignore")
# MY personnal module
from Regressions import Regression
from Classification import Classification

# DATAFRAME À CHARGER À PARTIR DU FILEINPUT
data = pd.DataFrame()
source = ColumnDataSource(data = data) 

# STATISTIQUE DESCRIPTIVE
source_summary = ColumnDataSource(data = data)

# TABLEAU POUR LA CORRELATION
source_correlation = ColumnDataSource(data = data)

# TABLEAU COEFFICIENT REGRESSION
source_coef = ColumnDataSource(data =data)

# TABLEAU COEFFICIENT REGRESSION
source_prediction = ColumnDataSource(data =data)

# SOURCE POUR LA CORRELATION CORRELATION
source_cor_plot = ColumnDataSource(data = data)

# OPTIONS POUR LES ALGORITHMES D'APPRENTISSAGES SUPPERVISÉS
algo_reg = ["Regression simple/multiple", "KNN","Random Forest"]
algo_clas = ["Regression Logistique","Analyse discriminante","Arbre de decision"]
algos = ["Select Algorithm"] + algo_reg + algo_clas

# DATATABLE COLUMN
columns=[TableColumn(field=col, title=col) for col in data.columns]
columns_summary=[TableColumn(field=col, title=col) for col in data.columns]
columns_correlation = [TableColumn(field=col, title=col) for col in data.columns]
columns_coef = [TableColumn(field=col, title=col) for col in data.columns]
columns_prediction = [TableColumn(field=col, title=col) for col in data.columns]
# FOR LOADING DATA
file_input = FileInput(accept=".csv, .xlsx", width=500)

# VERIFIER LE TYPE DE VARIABLES
def verifier_type_var(df,nom_va):
    if (np.issubdtype(df[nom_va].dtype, np.number)==True):
        return 'Numeric'
    else:
        return 'String'

# FONCTION CALLBACK À APPELLER 
def callback_select(attr, old, new):
    data = pd.read_csv(io.BytesIO(base64.b64decode(file_input.value)))
    # COURANTE VALEUR DANS LE SELECT
    x = select_x.value
    y = select_y.value
    color_var = select_color.value
    
    # CORRELATION ENTRE X ET Y
    if str(x)!= "select variable x" and str(y) != "select variable y":
        # LABEL CORRELATION PLOT
        corr_plot.xaxis.axis_label = x
        corr_plot.yaxis.axis_label = y
        # METTRE À JOUR LA SOURCE
        source_cor_plot.data =  {
            'x': data[x],
            'y': data[y]
        }
        # DELIMITONS LES AXES CORRELATIONS
        corr_plot.x_range.start = min(data[x])
        corr_plot.x_range.end = max(data[x])
        corr_plot.y_range.start = min(data[y])
        corr_plot.y_range.end = max(data[y])
        # AJOUTER UN TITRE SUR LE PLOT CORRELATION
        corr_plot.title.text = 'Corelation entre {} et {}'.format(x,y)
        
        # PROJECTON DANS LE PLAN EN COLORIANT SELON LA VARIABLE CIBLE
        if(str(color_var)!="select variable color"):
            print(str(color_var))
            # METTRE À JOUR LA SOURCE AVEC LA VARIABLE DE COULEUR
            source_cor_plot.data =  {
            'x': data[x],
            'y': data[y],
            'color_var': data[str(color_var)]}
            # ONGLET PROJECTION DANS LE PLAN
            species_list = data[str(color_var)].unique().tolist()
            # MAKE A COLOR MAPPER: COLOR_MAPPER
            color_mapper = CategoricalColorMapper(factors=species_list, palette=["red","purple", "blue"])
            # LABEL AXES OF PLOT PROJECTION
            plot_proj.xaxis.axis_label = x
            plot_proj.yaxis.axis_label = y
            # DELIITONS LES AXES PROJECTIONS
            plot_proj.x_range.start = min(data[x])
            plot_proj.x_range.end = max(data[x])
            plot_proj.y_range.start = min(data[y])
            plot_proj.y_range.end = max(data[y])
            # AJOUTER UN TITRE SUR LE PLOT PROJECTION
            plot_proj.title.text = 'Projection dans le plan par rapport à {} et {}'.format(x,y)
            # ADD THE COLOR MAPPER TO THE CIRCLE GLYPH
            plot_proj.circle(x='x', y='y', 
                            fill_alpha=0.8, 
                            source=source_cor_plot,
                            color=dict(field='color_var', transform=color_mapper), 
                            legend='color_var')
            plot_proj.title.text_color="green"
            plot_proj.title.text_font_style="bold"
            plot_proj.title_location="above"
    else:
        corr_plot.title.text = '{} et {} doivent etre quantitative(Numeric)'.format(x,y)
        plot_proj.title.text = '{} et {} doivent etre des variables Numeric pour la projection dans le plan'.format(x,y)
   

# SELECT OPTIONS FOR CORRELATION AND PROJECTION
select_x = Select(title="variable_xaxix", value="select variable x", options = ["select variable x"])
select_y = Select(title="variable_yaxix", value="select variable y", options = ["select variable y"])
select_color = Select(title="Color variable(Projection)", value="select variable color", options = ["select variable color"])

def getChoiceForLearning(attr, old, new):
    data = pd.read_csv(io.BytesIO(base64.b64decode(file_input.value)))
    if verifier_type_var(data,str(select_cible.value)) == "Numeric":
        select_algorithm.options = ["select regression algorithm"] + algo_reg
    elif verifier_type_var(data,str(select_cible.value)) == "String":
        select_algorithm.options = ["select classification algorithm"] + algo_clas

def get_cible_and_remove_in_explicatives(attr, old, new):
    # Ici on supprime la variable cible sur la liste des variables explicatives
    data = pd.read_csv(io.BytesIO(base64.b64decode(file_input.value)))
    multi_select_explicative.options = list(data.columns)
    variable_cible = select_cible.value
    if variable_cible in multi_select_explicative.options:
        multi_select_explicative.options.remove(variable_cible)

'''Dans cette fonction, nous recuperons toutes les entrées de 
    l'utilisateur pour lancer l'apprentisage'''
def apprentissage_suppervise():
    data = pd.read_csv(io.BytesIO(base64.b64decode(file_input.value)))
    taille_train = slider.value
    kv = slider_kv.value
    kvoisin = slider_kvoisin.value
    nb_feuille = slider_profondeur.value
    NbTreeForRF = slider_nbArbreRF.value
    variables_explicative = multi_select_explicative.value
    variable_cible = select_cible.value
    algo = select_algorithm.value
    
    if len(variables_explicative)>0 and variable_cible != "select variable cible" and algo != "Select Algorithm":
        regression = Regression(data,variables_explicative,variable_cible,taille_train, kv,kvoisin,NbTreeForRF)
        classification = Classification(data,variables_explicative,variable_cible,taille_train,kv,nb_feuille)
        if algo == 'Regression Logistique':
            print("Regression logistique")
            obj_regLog = classification.logistic_reg_class();
            refresh_regression_logistique(obj_regLog)
        elif algo == "Analyse discriminante":
            print("LDA")
            obj_lda = classification.analyse_discriminante_class();
            refresh_analyse_discriminante(obj_lda)
        elif algo == "Arbre de decision":
            print("Arbre")
            obj_tree = classification.arbre_decision_class();
            refresh_decision_tree(obj_tree)
        elif algo == "KNN":
            print("KNN")
            objet_knn = regression.knn_reg()
            refresh_reg_knn(objet_knn)
        elif algo == "Regression simple/multiple":
            print("Regression Lineaire Simple/Multiple")
            objet_reg = regression.regression_lineaire_multiple()
            refresh_reg_multi(objet_reg)
            
        elif algo == "Random Forest":
            print("Random Forest")
            obj_rf = regression.random_forest_reg();
            refresh_random_forest(obj_rf)
            
        else:
            print("Vous devez choisir l'algorithme pour lancer l'apprentissage")
    else:
        print("Veuillez bien parametrer l'algorithme avant de lancer l'apprentissage")
        

# SELECT FOR SUPERVISED LEARNING
select_cible = Select(title="Variable Cible", value="select variable cible", options = ["select variable cible"])
multi_select_explicative = MultiSelect(title="Variable explicative", value=[], options = [])
select_algorithm = Select(title="Algorithme d'apprentisage", value="Select Algorithm", options = ["Select Algorithm"])
select_algorithm.options = algos

# Echantillonage du jeux de données
slider = Slider(start = 50, end=80, step=10, value=70, title="Taille Train Set")
slider_kv = Slider(start = 0, end=30, step=1, value=5, title="K-fold")
slider_kvoisin = Slider(start = 0, end=10, step=1, value=5, title="K-Voisins")
slider_profondeur = Slider(start = 2, end=10, step=1, value=2, title="Profondeur Arbre Decision")
slider_nbArbreRF = Slider(start = 50, end=500, step=50, value=100, title="Nombre Arbre for Random Forest")
lancer_app = Button(label = "LANCER APPRENTISSAGE", button_type = "success", width=70, height=40)
lancer_app.on_click(apprentissage_suppervise)

select_cible.on_change('value',get_cible_and_remove_in_explicatives)
select_cible.on_change('value',getChoiceForLearning)
#multi_select_explicative.on_change('value',getChoiceForLearning)
#select_algorithm.on_change('value',getChoiceForLearning)

# EVENMENTS DECLENCHÉ PAR LE USERS 
select_x.on_change('value', callback_select)
select_y.on_change('value', callback_select)
select_color.on_change('value', callback_select)

# CALLBACK CHARGEMENT
def load_data(attr, old, new):
    select_x.options = ["select variable x"]
    select_y.options = ["select variable y"]
    select_color.options = ["select variable color"]
    select_algorithm.options = ["Select Algorithm"]
    select_cible.options = ["select variable cible"]
    multi_select_explicative.options = []
    # CREATION DE LA CONTENUE
    decoded=base64.b64decode(new)
    f=io.BytesIO(decoded)
    data= pd.read_csv(f, sep='\n|;|,')
    source.data=data
    data_table.columns=[TableColumn(field=col, title=col) for col in data.columns]
    select_color.options =(select_color.options)+([option for option in list(data.columns) if verifier_type_var(data,option)=="String"])
    select_x.options =(select_x.options)+([option for option in list(data.columns) if verifier_type_var(data,option)=="Numeric"])
    select_y.options =(select_y.options)+([option for option in list(data.columns) if verifier_type_var(data,option)=="Numeric"])
    # SELECT APPRENTISSAGE
    select_cible.options =(select_cible.options)+([option for option in list(data.columns)])
    multi_select_explicative.options =[option for option in list(data.columns)]
    # STATISTIQUE DESCRIPTIVES 
    summary = data.describe()
    data_table_summary.columns=[TableColumn(field=col, title=col, formatter=NumberFormatter(format="0.0000000")) for col in summary.columns ]
    summary.insert(0,'statistiques descriptives',summary.index)
    source_summary.data=summary
    data_table_summary.columns.insert(0,TableColumn(field='statistiques descriptives', title='statistiques descriptives'))
    # CORRELATION TABLEAU 
    corr_tab = data.corr()
    data_table_corr.columns=[TableColumn(field=col, title=col, formatter=NumberFormatter(format="0.0000000")) for col in corr_tab.columns ]
    corr_tab.insert(0,'Corrélation variables',corr_tab.index)
    source_correlation.data=corr_tab
    data_table_corr.columns.insert(0,TableColumn(field='Corrélation variables', title='Corrélation variables'))
    # INITIATION DU PLOT CORRELATION
    source_cor_plot.data = {
        'x': [np.random.uniform(low=0, high=1, size=20)],
        'y': [np.random.uniform(low=0, high=1, size=20)]}
    
''' REFRESH AREGRESSION ALGORITHME'''
def refresh_reg_multi(obj):
    #child_reg1.children[0]=obj.msg
    child_reg1.children[0]=saut_ligne
    child_reg1.children[1]=obj.title_intercept
    child_reg1.children[2]=obj.intercept
    child_reg1.children[3]=saut_ligne
    child_reg1.children[4]=obj.title_for_coeff
    child_reg1.children[5]=obj.coefficients
    
    child_reg2.children[0]=saut_ligne
    child_reg2.children[1]=obj.title_moy_vc
    child_reg2.children[2]=obj.mean_val_croisee
    child_reg2.children[3]=saut_ligne
    child_reg2.children[4]=obj.title_vc
    child_reg2.children[5]=obj.cross_validation
    child_reg2.children[6]=saut_ligne
    
    child_reg3.children[0]=saut_ligne
    child_reg3.children[1]=obj.title_fig 
    child_reg3.children[2]=obj.fig

    
    child_reg4.children[0]=saut_ligne
    child_reg4.children[1]=obj.title_performance 
    child_reg4.children[2]=obj.metrics
    child_reg4.children[3]= saut_ligne
    child_reg4.children[4]= saut_ligne
    child_reg4.children[5]= saut_ligne
    child_reg4.children[6]= saut_ligne
    child_perf.children[0]= saut_ligne
    child_perf.children[1]= saut_ligne
    child_perf.children[2]= saut_ligne
    child_perf.children[3]= saut_ligne
    child_perf.children[4]= saut_ligne
    child_perf.children[5]= saut_ligne
    child_perf.children[6]= saut_ligne
    
    child_var.children[0] = saut_ligne
    child_var.children[1] = saut_ligne
    
    
    #child_reg.on_change('value', lambda attr, old, new: update_vc(new_df))
def refresh_reg_knn(obj):
    
    child_reg1.children[0]=saut_ligne
    child_reg1.children[1]=obj.title_moy_vc
    child_reg1.children[2]=obj.mean_val_croisee
    child_reg1.children[3]=saut_ligne
    child_reg1.children[4]=obj.title_vc
    child_reg1.children[5]=obj.cross_validation
    
    child_reg2.children[0]=saut_ligne
    child_reg2.children[1]=obj.figcurve
    child_reg2.children[2]=saut_ligne
    child_reg2.children[3]=saut_ligne
    child_reg2.children[4]=saut_ligne
    child_reg2.children[5]=saut_ligne
    child_reg2.children[6]=saut_ligne
    
    child_reg3.children[0]=saut_ligne
    child_reg3.children[1]=obj.title_fig 
    child_reg3.children[2]=obj.fig

    
    child_reg4.children[0] = saut_ligne
    child_reg4.children[1] = obj.title_performance 
    child_reg4.children[2] = obj.metrics
    child_reg4.children[3]= saut_ligne
    child_reg4.children[4]= saut_ligne
    child_reg4.children[5]= saut_ligne
    child_reg4.children[6]= saut_ligne
    child_perf.children[0]= saut_ligne
    child_perf.children[1]= saut_ligne
    child_perf.children[2]= saut_ligne
    child_perf.children[3]= saut_ligne
    child_perf.children[4]= saut_ligne
    child_perf.children[5]= saut_ligne
    child_perf.children[6]= saut_ligne
    
    child_var.children[0] = saut_ligne
    child_var.children[1] = saut_ligne

def refresh_random_forest(obj):
    
    child_reg1.children[0]= obj.title_moy_vc
    child_reg1.children[1]=obj.mean_val_croisee
    child_reg1.children[2]= saut_ligne
    child_reg1.children[3]=obj.title_vc
    child_reg1.children[4]=obj.cross_validation
    child_reg1.children[5]=saut_ligne
    
    child_reg2.children[0]=saut_ligne
    child_reg2.children[1]=saut_ligne
    child_reg2.children[2]=saut_ligne
    child_reg2.children[3]=saut_ligne
    child_reg2.children[4]=saut_ligne
    child_reg2.children[5]=saut_ligne
    child_reg2.children[6]=saut_ligne
    
    child_reg3.children[0]=saut_ligne
    child_reg3.children[1]=obj.title_fig 
    child_reg3.children[2]=obj.fig
    
    child_reg4.children[0] = saut_ligne
    child_reg4.children[1] = obj.title_performance 
    child_reg4.children[2] = obj.metrics
    child_reg4.children[3]= saut_ligne
    child_reg4.children[4]= saut_ligne
    child_reg4.children[5]= saut_ligne
    child_reg4.children[6]= saut_ligne
    child_perf.children[0]= saut_ligne
    child_perf.children[1]= saut_ligne
    child_perf.children[2]= saut_ligne
    child_perf.children[3]= saut_ligne
    child_perf.children[4]= saut_ligne
    child_perf.children[5]= saut_ligne
    child_perf.children[6]= saut_ligne
    
    child_var.children[0] = saut_ligne
    child_var.children[1] = saut_ligne
    
 
    
    
'''CLASSIFICATION ALGORITHME'''
    
def refresh_decision_tree(obj):
    
    child_reg1.children[0]=saut_ligne
    child_reg1.children[1]=obj.distribprediction1 
    child_reg1.children[2]=obj.distribpredictions2 
    child_reg1.children[3]=saut_ligne
    child_reg1.children[4]=saut_ligne
    child_reg1.children[5]=saut_ligne
    #child_reg1.children[4]=obj.treeT
    #child_reg1.children[5]=obj.tree
    
    child_reg2.children[0]=saut_ligne
    child_reg2.children[1]=obj.reglesT
    child_reg2.children[2]=obj.regles
    child_reg2.children[3]=saut_ligne
    child_reg2.children[4]=saut_ligne
    child_reg2.children[5]=saut_ligne
    child_reg2.children[6]=saut_ligne
    
    child_reg3.children[0]=obj.title_matrice_confusion
    child_reg3.children[1]=obj.cube
    child_reg3.children[2]=saut_ligne

    
    child_reg4.children[0]= obj.taux_reconnaissance
    child_reg4.children[1]= saut_ligne
    child_reg4.children[2]= obj.taux_erreur
    child_reg4.children[3]= saut_ligne
    child_reg4.children[4]= obj.rapel_class
    child_reg4.children[5]= saut_ligne
    child_reg4.children[6]= obj.precclasse
    child_perf.children[0]= obj.moy_succes
    child_perf.children[1]= saut_ligne
    child_perf.children[2]= obj.int_succes
    child_perf.children[3]= saut_ligne
    child_perf.children[4]= obj.title_rapport 
    child_perf.children[5]= obj.rapport
    child_perf.children[6]= saut_ligne
    
    child_var.children[0] = obj.title_impVariables
    child_var.children[1] = obj.importance_variables

def refresh_analyse_discriminante(obj):
    
    child_reg1.children[0]= obj.title_coefficients
    child_reg1.children[1]= obj.coefficients 
    child_reg1.children[2]= saut_ligne 
    child_reg1.children[3]= saut_ligne
    child_reg1.children[4]= saut_ligne
    child_reg1.children[5]= saut_ligne
    
    child_reg2.children[0]=saut_ligne
    child_reg2.children[1]=saut_ligne
    child_reg2.children[2]=saut_ligne
    child_reg2.children[3]=saut_ligne
    child_reg2.children[4]=saut_ligne
    child_reg2.children[5]=saut_ligne
    child_reg2.children[6]=saut_ligne
     
    child_reg3.children[0]=obj.title_matrice_confusion
    child_reg3.children[1]=obj.cube
    child_reg3.children[2]=saut_ligne
    
    child_reg4.children[0]= obj.taux_reconnaissance
    child_reg4.children[1]= saut_ligne
    child_reg4.children[2]= obj.taux_erreur
    child_reg4.children[3]= saut_ligne
    child_reg4.children[4]= obj.rapport_qualite
    child_reg4.children[5]= saut_ligne
    child_reg4.children[6]= saut_ligne
    child_perf.children[0]= obj.moy_succes_rate
    child_perf.children[1]= saut_ligne
    child_perf.children[2]= obj.int_succes
    child_perf.children[3]= saut_ligne
    child_perf.children[4]= saut_ligne 
    child_perf.children[5]= saut_ligne
    child_perf.children[6]= saut_ligne
    
    child_var.children[0] = saut_ligne
    child_var.children[1] = saut_ligne
    
    

def refresh_regression_logistique(obj):
    child_reg1.children[0]= obj.intercept
    child_reg1.children[1]= saut_ligne 
    child_reg1.children[2]= obj.coefficients 
    child_reg1.children[3]= obj.coefDatable
    child_reg1.children[4]= saut_ligne
    child_reg1.children[5]= obj.log_vraisemblance
    
    child_reg2.children[0]=obj.int_succes
    child_reg2.children[1]=saut_ligne
    child_reg2.children[2]=obj.moy_succes
    child_reg2.children[3]=saut_ligne
    child_reg2.children[4]=saut_ligne
    child_reg2.children[5]=saut_ligne
    child_reg2.children[6]=saut_ligne
    
    child_reg3.children[0]=obj.matrice_confusion
    child_reg3.children[1]=obj.cube
    child_reg3.children[2]=saut_ligne
    
    child_reg4.children[0]= obj.accuracy
    child_reg4.children[1]= saut_ligne
    child_reg4.children[2]= obj.error_rate
    child_reg4.children[3]= saut_ligne
    child_reg4.children[4]= obj.rapport_qualite
    child_reg4.children[5]= saut_ligne
    child_reg4.children[6]= saut_ligne
    child_perf.children[0]= saut_ligne
    child_perf.children[1]= saut_ligne
    child_perf.children[2]= saut_ligne
    child_perf.children[3]= saut_ligne
    child_perf.children[4]= saut_ligne 
    child_perf.children[5]= saut_ligne
    child_perf.children[6]= saut_ligne
    
    child_var.children[0] = obj.fig_roc
    child_var.children[1] = obj.accuracy_roc
    
    


# WIDGET FOR LOADING DATA
file_input.on_change('value', load_data)


# DATATBLE  POUR AFFICHAGE SOUS FORME TABLEAU
data_table = DataTable(source=source, columns=columns, width=900, height=350)
data_table_summary = DataTable(source=source_summary, columns=columns_summary, width=800, height=400)
data_table_corr = DataTable(source=source_correlation, columns=columns_correlation, width=700, height=400)
# datable algo suppervisé
# data_table_coef = DataTable(source=source_coef, columns=columns_coef, width=400, height=400)

# PLOT CORRELATION
corr_plot = figure(title="Corrélation entre variable",plot_height = 320)
corr_plot.circle(x='x',y='y', source = source_cor_plot, color = "purple")
corr_plot.title.text_color="green"
corr_plot.title.text_font_style="bold"
corr_plot.title_location="above"

# PLAN PROJECTION INSTANCE
plot_proj = figure(title="Projection dans le plan par rapport à sepal_length et sepal_width",plot_height = 350)

# TABS PANEL SECTION 
title = Div(text="<center><h3>INTERFACE D'ANALYSE DE DONNÉES</h3></center>", height=2)
p1 = Div(text="<center><h4>Statistique Descriptive et exploration des données</h4></center>")
line1 = Paragraph(text="_________________________________________________________")
p2  = Div(text="<center><h4>Apprentissage suppervisé</h4></center>",height=2)
p3  = Div(text="<center><h4>Regularisation et Optimisation</h4></center>",height=2)

# CREATION SECTION 
section1 = column(title,p1,line1)
section2 = column(p2,line1)
section3 = column(p3,line1)

header_coefficient = Div(text="")
header_intercept = Div(text="")
parag_intercept = Paragraph(text="")
constante = column(header_intercept,parag_intercept)

panel1 = Panel(child=data_table, title="Jeu de données")
panel2= Panel(child=data_table_summary, title="Statistiques descriptives")
panel3= Panel(child=row(corr_plot,data_table_corr), title="Corrélation entre Variables quantitatives")
panel4= Panel(child=plot_proj, title="Projection des points dans le plan")

# AFFICHAGE APPRENTISSAGE SUPPERVISÉ
#creation du layout correspondant
line=Div(text="__________________________________________________________________________")
Previsualisation_data=Div(text="<center><h2 >Prévisualisation des données</h2></center>")
saut_ligne=Div(text="<br/>")
title_reg=Div(text="<center><h3>Sortie apprentissage suppervisé</h3></center>")

child_reg1=layout([],[],[],[],[],[])
child_reg2=layout([],[],[],[],[],[],[])
child_reg3=layout([],[],[])
child_reg4=layout([],[],[],[],[],[],[])
child_perf = layout([],[],[],[],[],[],[])
child_var=layout([],[])

panel_regression = Panel(child=row(child_reg1,child_reg2), title="Sortie du modéle")
panel_reg_pred = Panel(child=child_reg3, title="Prédiction sur l'echantillon de test")
panel_reg_perf = Panel(child=row(child_reg4,child_perf), title="Mesures des performances")
panel_var = Panel(child=child_var, title="Variables pertinentes")

layout = row(column(section1,
                    file_input, 
                    select_x, 
                    select_y,
                    select_color,
                    section2,
                    select_cible,
                    multi_select_explicative,
                    select_algorithm,
                    slider,
                    slider_kv,
                    slider_kvoisin,
                    slider_profondeur,
                    slider_nbArbreRF,
                    lancer_app,
                    section3), 
            column(Paragraph(text="     ")),
            column(Tabs(tabs=[panel1, panel2, panel3, panel4]),
                   Tabs(tabs=[panel_regression, panel_reg_pred, panel_reg_perf, panel_var])))
curdoc().add_root(layout)






