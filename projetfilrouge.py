
from pyexpat import model
from xml.sax.handler import feature_string_interning
import pandas as pd 
import time
import shap


import numpy as np 
import streamlit as st 
import seaborn as sns 
import matplotlib.pyplot as plt
import plotly.express as px #librairie de DataVizualization interactive
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from bokeh.plotting import figure
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.models import LogTicker
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category10
import streamlit as st
import matplotlib.cm as cm
from wordcloud import WordCloud
import nltk
nltk.download('stopwords')
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
import plotly.express as px
import joblib
from joblib import load, dump 
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from PIL import Image
from collections import OrderedDict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score




st.sidebar.title("Fund Success Predictor")

data = pd.read_csv("/Users/isabellebareyre/Documents/Kickstarter Campaigns DataSet.csv", index_col=0)
columns={"usd_pledged": "Pledged",
          "backers_count": "Backers",
         "goal_usd": "Goal",
         "sub_category": "Category",
         "main_category": "sub_category"
        }
data.rename(columns=columns, inplace=True)
data = data.drop_duplicates()
data = data.drop(['slug','name', 'creator_id','deadline','blurb','id'], axis = 1)
data= data.loc[(data['status']=='successful') | (data['status']=='failed')]
from datetime import datetime

data['launched_at'] = pd.to_datetime(data['launched_at'], format='%Y-%m-%d %H:%M')

import calendar
data['month'] = data['launched_at'].dt.month
data['year'] = data['launched_at'].dt.year.astype('string')
# Remplacer les mois numériques par les noms des mois
data['month'] = data['month'].apply(lambda x: calendar.month_name[x])
data['day_of_week'] = data['launched_at'].dt.strftime('%A')

pages=["Projet","Jeux de Données", "Visualisation des Données", "Preprocessing et modélisation", "Fund Success Predictor"]
page=st.sidebar.radio('Sélectionnez une partie :', pages)
if page == pages[0] :
      st.title('Prédire le Succès d\'une Campagne de Financement Participatif')
      
      st.image("Kickstarter.jpg")

      st.write('<h4>Introduction</h4>', unsafe_allow_html=True)
      st.write("<p style='text-align:justify'> Ce projet a été réalisé dans le cadre de notre formation : Data Analyst via l'organisme Datascientest. L'objectif principal est de prédire le succès d'une campagne de financement en utilisant des données disponibles sur la plateforme de financement participatif Kickstarter. </p>", unsafe_allow_html=True)
      
      
      st.write("<p style='text-align:justify'>Ce Streamlit présente notre démarche, depuis l'acquisition des données jusqu'à la création des variables explicatives pertinentes. Nous avons explorés les résultats finaux générés par nos modèles de prédiction . Nous avons également réservé une section dédiée à l'apprentissage automatique, où vous pouvez expérimenter par vous-même les variables que nous avons élaborées sur différents algorithmes de prédiction.</p>", unsafe_allow_html=True)
      st.write('<h4>Objectif </h4>', unsafe_allow_html=True)
      st.write("<p style='text-align:justify'>Nous avons utilisé des techniques de traitement de données avancées, de feature engineering et de machine learning pour élaborer notre modèle de prédiction. Notre objectif est d'aider les porteurs de projets à estimer leurs chances de succès lorsqu'ils lancent une campagne de financement participatif. En utilisant ce Streamlit, vous pourrez explorer notre méthodologie et tester la prédiction du succès de votre propre projet de financement.</p>", unsafe_allow_html=True)
     
      st.subheader("Isabelle BAREYRE et Khadija BENHAMOU")
elif page == pages[1] :
      st.write("### Jeux de Données")   
      tab1, tab2, tab3= st.tabs(["Choix du jeu de données", "Nettoyage des Données", "Exploration statistique"])
      
      with tab1 :
        col1, col2 = st.columns(2)  # Diviser l'écran en deux colonnes
        with col1:
          st.write('<h5 style="color:blue; font-weight: bold;"> Source : </h5>', unsafe_allow_html=True)
          st.write("<p> L'ensemble de données  de plus de 200 000 projets Kickstarter provient de la plateforme Kaggle. ", unsafe_allow_html=True)
        with col2:
          st.write('<h5 style="color:blue; font-weight: bold;"> Periode : </h5>', unsafe_allow_html=True)
          st.write("<p > Les données contiennent des informations sur les projets lancés sur Kickstarter depuis sa création en 2009 jusqu'à 2020 .", unsafe_allow_html=True)

        st.write('<h5 style="color:blue; font-weight: bold;"> Description des variables : </h5>', unsafe_allow_html=True)
        st.write("<p style='text-align:justify '> Notre DataFrame comporte 18 colonnes :", unsafe_allow_html=True)
      

        donnees = {
        'Nom de la colonne': [
            'id', 'name', 'currency', 'launched_at', 'Backers', 'blurb', 'country','Deadline', 'Slug', 'Status', 'Pledged', 'Category', 'sub_category','creator_id', 'blurb_length', 'Goal', 'City', 'Duration'],
        'Description': [ 
            'Identifiant unique de la campagne', 'Nom de la campagne',
            'Devise financement',
            'Date et heure du lancement de la campagne',
            'Nombre de contributeurs à la campagne',
            'Description de la campagne',
            'Pays d\'origine de la campagne',
            'Date et heure limite pour atteindre l\'objectif de financement.',
            'Version simplifiée du nom de la campagne',
            'Statut actuel de la campagne',
            'Montant total de fonds collectés',
            'Catégorie principale de la campagne',
            'Sous-catégorie de la campagne',
            'Identifiant du créateur ',
            'Longueur en caractères de la description',
            'Montant de l\'objectif de financement',
            'Ville d\'origine du projet',
            'Durée de la campagne'],
        'Type informatique': [
            'int64', 'string', 'string', 'datetime64', 'int64', 'string', 'string','string', 'string', 'string', 'float64', 'string', 'string', 'int64','int64', 'float64', 'string', 'float64']
        }
          # Créer un DataFrame à partir des données
        variables= pd.DataFrame(donnees)
           # Afficher le DataFrame dans Streamlit
        st.dataframe(variables)  # Masquer les en-têtes pour économiser de l'espace
      
        st.write('<h5 style="color:blue; font-weight: bold;"> Taille Initiale du DataFrame : </h5>', unsafe_allow_html=True)
        @st.cache_data
        def load_data():
           
            df = pd.read_csv("/Users/isabellebareyre/Documents/Kickstarter Campaigns DataSet.csv", index_col=0)
            return df

        df = load_data()
        
        st.write(f"{df.shape[0]:,} lignes x {df.shape[1]} colonnes")

      with tab2 : 
        st.write('<h5 style="color:blue; font-weight: bold;"> Suppression des variables unitiles : </h5>', unsafe_allow_html=True)
      
        st.write("<p style='text-align:justify'>Les variables <strong>id</strong>, <strong>name</strong>, <strong>Deadline</strong>, <strong>Slug</strong> et <strong>creator_id</strong> sont des variables avec des valeurs uniques pour chaque ligne et qui n'ont pas d'utilité significative dans l'analyse. Par conséquent, nous avons décidé de les supprimer.</p>", unsafe_allow_html=True) 
        st.write('<h5 style="color:blue; font-weight: bold;"> Suppression des campagnes ‘live’ est ‘canceled’: </h5>', unsafe_allow_html=True)
        
        st.write("<p style='text-align:justify'>Les campagnes incluses dans notre jeu de données peuvent être classées en l'une des quatre catégories suivantes : échec, réussite, en cours ou annulée.</p>", unsafe_allow_html=True)
        


        state_counts = df["status"].value_counts()
        fig = px.pie(state_counts, values=state_counts.values, names=state_counts.index, hole=0.4)
        fig.update_traces(textinfo='percent+label')
        fig.update_layout(title="Répartition des projets par statut")
        st.plotly_chart(fig)
        st.write("<p style='text-align:justify'>Nous avons exclu les campagnes 'live' et 'canceled' de notre analyse pour éliminer les biais potentiels, nous nous concentrons sur des campagnes terminées, assurant des données fiables pour notre modélisation prédictive.</p>", unsafe_allow_html=True)
     

      
        df = df.drop(['slug','name', 'creator_id','deadline','blurb','id'], axis = 1)
      
        st.write("<p style='text-align:justify'> <strong> Aucune valeur n'est manquante dans notre jeu de donnée.</strong>", unsafe_allow_html=True)
        st.write('<h5 style="color:blue; font-weight: bold;"> Suppression des doublons: </h5>', unsafe_allow_html=True)
     
        df= df.loc[(df['status']=='successful') | (df['status']=='failed')]
        # Utiliser la police définie dans la checkbox
        if st.checkbox("Afficher le nombre de doublons"):
          nombre_de_doublons = df.duplicated().sum()
          st.write("<p style='text-align:justify'> Le nombre des doublons s\'élève à :", unsafe_allow_html=True)
          texte_formate = f"<span style='font-size: 20px; font-weight: bold; color: blue;'>{nombre_de_doublons}</span>"
          st.markdown(texte_formate, unsafe_allow_html=True)
          st.write("<p style='text-align:justify'> Nous avons procédés à la suppression de tous les doublons", unsafe_allow_html=True)
          
        st.write('<h5 style="color:blue; font-weight: bold;"> Création de nouvelles variables : </h5>', unsafe_allow_html=True)

        
        st.dataframe(data[['launched_at','year', 'month','day_of_week']].head())
        
        data = data.drop('launched_at', axis = 1 )
        if st.checkbox("Afficher le nouveau DataFrame"):
          st.dataframe(data.head())
          st.write('La taille du nouveau Dataframe est : ' )
        
          st.write(f"{data.shape[0]:,} lignes x {data.shape[1]} colonnes")
      
        st.set_option('deprecation.showfileUploaderEncoding', False)
      with tab3 :  
       
        st.write("<p style='text-align:justify'>Le tableau présente des statistiques relatives aux projets Kickstarter, notamment le nombre de projets réussis et échoués, les montants d'objectifs, les montants promisla durée des projets et le nombre de contributeurs ...", unsafe_allow_html=True)
        st.image("describe.png")
    

elif page == pages[2] : 
    st.write("### Visualisation des Données")
    

    st.write("<p style='text-align:justify'> L\'exploration des données est cruciale pour prédire le succès des campagnes Kickstarter. Dans cette partie, nous cherchons à comprendre les facteurs influençant les résultats sur Kickstarter, en examinant les catégories les plus performantes, l\'impact des objectifs financiers , la durée de la campagne et les tendances temporelles.", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Répartion des Projets", "Analyse des caractéristiques","Analyse du Text"])
    with tab1 : 
    
      # Compter le nombre de projets par pays
      country_counts = data['country'].value_counts()

      # Sélectionner les quatre premiers pays et regrouper les autres
      top_countries = country_counts[:4]
      other_countries = country_counts[4:]
      other_total = other_countries.sum()

      # Créer une nouvelle série avec les quatre premiers pays et "Autres"
      top_and_other = top_countries.append(pd.Series([other_total], index=['Autres']))

      # Définir des couleurs personnalisées
      colors = ['#DAB22F', '#598D9B', '#635D83', '#CCC2D6']

      # Créer le graphique en secteurs (pie chart) avec Plotly
      fig = px.pie(names=top_and_other.index, values=top_and_other, title='Répartition des projets par pays:',
                 hole=0.4,
                 color=top_and_other.index, color_discrete_map={k: v for k, v in zip(top_and_other.index, colors)},
                 labels={'labels': 'Pays'})
      fig.update_layout(width=600, height=500)
      # Afficher le graphique
      st.plotly_chart(fig)

      # Compter le nombre de projets par catégorie
      category_counts = data['Category'].value_counts()

      # Extraire les couleurs utilisées dans le graphique de répartition par statut
      colors = [
      '#DAB22F', '#598D9B', '#635D83', '#CCC2D6', '#FF5733', 
      '#6B4226', '#A2AB58', '#8E7A6A', '#E73A32', '#3794E7',
      '#A8436D', '#57C798', '#FDE725', '#F57F53', '#AB83A1'
      ]

      # Créer le graphique de répartition par catégorie avec les mêmes couleurs
      fig = px.pie(names=category_counts.index, values=category_counts, title='Répartition des projets par catégorie : ',
             hole=0.4, color=category_counts.index, color_discrete_sequence=colors)
      fig.update_layout(width=600, height=500)
      # Afficher le graphique
      st.plotly_chart(fig)

    with tab2 :
    
      st.write('<h4 style="color:blue; font-weight: bold;"> Existe-t-il une évolution temporelle du nombre des campagnes et de leur taux de réussite ?</h4>', unsafe_allow_html=True)


    
      import plotly.graph_objects as go
      df3 = data
      df3['launched_at'] = pd.to_datetime(df3['launched_at'])
    
      df3['success'] = df3['status'] == 'successful'

      df_grouped = df3.groupby(pd.Grouper(key='launched_at', freq='M')).agg({'success': ['sum', 'count']})
      df_grouped.columns = ['success_count', 'total_count']
      df_grouped['success_percentage'] = df_grouped['success_count'] / df_grouped['total_count'] * 100
      df_grouped['month_year'] = df_grouped.index.strftime('%B %Y')
      df_grouped['failed_count'] = df_grouped['total_count'] - df_grouped['success_count']
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=df_grouped.index, y=df_grouped['success_count'], mode='lines',
                         name='Successful', line=dict(color='green')))
      fig.add_trace(go.Scatter(x=df_grouped.index, y=df_grouped['failed_count'], mode='lines',
                         name='Failed', line=dict(color='red')))

      # Ajoutez des informations au survol
      fig.update_traces(hoverinfo='x+y', line=dict(width=2))
      fig.update_layout( title='le nombe de campagnes par statut et année',
                      xaxis_title='Timeline',
                      yaxis_title='Count',
                      legend_title='Status',
                      width=1000,  # Largeur personnalisée en pixels
                      height=400,  # Hauteur personnalisée en pixels
                      )

       # Affichez le graphique Plotly dans Streamlit
      st.plotly_chart(fig, use_container_width=True)


      st.write('<h4 style="color:blue; font-weight: bold;""> La durée du projet influence t\'elle le succés du projet? </h4>', unsafe_allow_html=True)
   
      from plotly.subplots import make_subplots
      # Créez des catégories pour la durée des projets
      bins = [20, 29, 39, 49, 60]
      labels = ['[20-29]', '[30-39]', '[40-49]', '[50-60]']
      data['duration_category'] = pd.cut(data['duration'], bins=bins, labels=labels)

       # Utilisez pd.crosstab() pour obtenir le décompte par catégorie et par statut
      table = pd.crosstab(data['duration_category'], data['status'])

      # Réorganisez les données pour les barres côte à côte
      table_stacked = table.stack().reset_index().rename(columns={0: 'count'})

      # Réorganisez les données pour les pourcentages
      table_percent = table.div(table.sum(axis=1), axis=0) * 100
      table_percent = table_percent.stack().reset_index().rename(columns={0: 'percentage'})
    

       # Créez le premier graphique
      fig1 = go.Figure()

      for status in ['successful', 'failed']:
         table1 = table_stacked[table_stacked['status'] == status]
         fig1.add_trace(go.Bar(x=table1['duration_category'], y=table1['count'], name=status))
         fig1.update_layout(
            xaxis_title='Durée de la campagne (en jours)',
            yaxis_title='Nombre de Projets',
            title='Nombre de campagnes par durée et statut',
            )

      # Créez le deuxième graphique
      fig2 = go.Figure()

      for status in ['successful', 'failed']:
        table1 = table_percent[table_percent['status'] == status]
        fig2.add_trace(go.Bar(x=table1['duration_category'], y=table1['percentage'], name=status))
      fig2.update_layout(
        xaxis_title='Durée de la campagne (en jours)',
        yaxis_title='Pourcentage de campagnes',
        title='Répartition des campagnes par durée et statut (%)',)

      # Affichez les graphiques dans Streamlit
      st.plotly_chart(fig1, use_container_width=True)
      st.plotly_chart(fig2, use_container_width=True)
      st.write('<h4 style="color:blue; font-weight: bold;"> Quel impact a l\'objectif du financement sur le succés d\'une campagne? </h4>', unsafe_allow_html=True)
    
        # Créez des catégories pour les montants objectifs
      bins = [0, 100, 1000, 10000, 100000, 1000000]  # Définir les intervalles
      labels = ['[0-100]', '[100-1K]', '[1K-10K]', '[10K-100K]', '[100K-1M]']  # Définir les libellés des catégories
      data['goal_category'] = pd.cut(data['Goal'], bins=bins, labels=labels)

      # Utiliser pd.crosstab() pour obtenir le décompte par catégorie et par statut
      table = pd.crosstab(data['goal_category'], data['status'])
    
      # Réorganiser les données pour les barres côte à côte
      table_stacked =  table.stack().reset_index().rename(columns={0: 'count'})

       # Utilisez pd.crosstab() pour obtenir le décompte par catégorie et par statut pour les pourcentages
      table_percent = table.div(table.sum(axis=1), axis=0) * 100
      table_percent =table_percent.stack().reset_index().rename(columns={0: 'percentage'})
      # Créez le premier graphique pour le nombre de projets
      fig1 = go.Figure()
      for status in ['successful', 'failed']:
        table2= table_stacked[table_stacked['status'] == status]
        fig1.add_trace(go.Bar(x=table2['goal_category'], y=table2['count'], name=status))

      fig1.update_layout(
        xaxis_title='Objectif de financement',
        yaxis_title='Nombre de campagnes',
        title='Nombre de campagnes par objectif de fiancement et statut',)
  
    
      # Créez le deuxième graphique pour les pourcentages
      fig2 = go.Figure()

      for status in ['successful', 'failed']:
          table2 = table_percent[table_percent['status'] == status]
          fig2.add_trace(go.Bar(x=table2['goal_category'], y=table2['percentage'], name=status))
    
      fig2.update_layout(
        xaxis_title='Montant Objectif',
        yaxis_title='Pourcentage de Projets',
        title='Pourcentage de campanges par Montant Objectif et Statut',)

       # Créez une disposition en sous-tracés
      st.plotly_chart(fig1, use_container_width=True)
      st.plotly_chart(fig2, use_container_width=True)
      st.write('<h4 style="color:blue; font-weight: bold;"> Quelles sont les principales sous catégories qui réussissent le mieux à atteindre l\'objectif ?</h4>', unsafe_allow_html=True)
        
      # Filtrer les données pour n'inclure que les projets avec status == successful
      filtered_data = data[data['status'] == 'successful']

     # Calculer le nombre total de projets par catégorie
      total_projects = data.groupby('Category').size().reset_index(name='total_count')

     # Calculer le nombre de projets réussis par catégorie
      successful_projects = filtered_data.groupby('Category').size().reset_index(name='successful_count')

     # Fusionner les deux DataFrames pour obtenir le nombre total et le nombre de projets réussis par catégorie
      table3 = total_projects.merge(successful_projects, on='Category', how='left')

     # Calculer le pourcentage de projets réussis par rapport au nombre total de projets par catégorie
      table3['percentage'] = (table3['successful_count'] / table3['total_count']) * 100
      # Trier les données par pourcentage décroissant
      table3 = table3.sort_values('percentage', ascending=False)

      fig = px.bar(table3, x='Category', y='percentage', title='Pourcentage de projets réussis par catégorie', color='Category')
      fig.update_traces(
        marker=dict(
            line=dict( color=colors )))
      fig.update_layout(xaxis_title='Catégorie', yaxis_title='Pourcentage (%)')
      st.plotly_chart(fig)
    
      data['launched_at'] = pd.to_datetime(data['launched_at'], errors='coerce')
      st.write('<h4 style="color:blue; font-weight: bold;"> Quel impact a le mois et le jour de lancement sur le succès d’une campagne ?</h4>', unsafe_allow_html=True)
      # Vérifier les valeurs de date invalides
      # Extraire le jour de la semaine
      data['day_of_week'] = data['launched_at'].dt.strftime('%A')
      grouped_data = data.groupby(['day_of_week', 'status']).size().reset_index(name='count')
      order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
      total_per_day = data.groupby('day_of_week').size()

      # Divisez les données en deux ensembles : successful et failed
      successful_data = grouped_data[grouped_data['status'] == 'successful']
      failed_data = grouped_data[grouped_data['status'] == 'failed']
      # Créez un graphique en barres pour chaque ensemble de données

      # Divisez les données en deux ensembles : successful et failed
    
      data['month'] = data['launched_at'].dt.month


      import calendar


      # Utilisez pd.crosstab() pour obtenir le décompte par catégorie et par statut
    
      table = data.groupby(['month', 'status']).size().reset_index(name='count')
      colors = ['#90e0ef','#03045e','#0081a7','#00afb9','#fed9b7', '#f07167', '#fed9b7'] 
      total_per_month = data.groupby('month').size()
      successful_table = table[table['status'] == 'successful']
      failed_table = table[table['status'] == 'failed']
      fig4 = go.Figure()
      fig4.add_trace(go.Bar(x=successful_table['month'],
                   y=successful_table['count'],
                   name='Successful',
                   marker_color = '#90e0ef'
                   )
                   )
            
      fig4.add_trace(go.Bar(x=failed_table['month'],
                      y=failed_table['count'],
                      name='Failed',
                      marker_color='#03045e'
                      )
             )
      fig4.update_layout(
        title='Nombre de campagnes par mois',
        xaxis_title='Mois',
        yaxis_title='Nombre de campagnes',
        legend_title='Statut')
      st.plotly_chart(fig4)
      fig3 = go.Figure()

      fig3.add_trace(go.Bar(x=successful_data['day_of_week'],
                      y=successful_data['count'],
                      name='Successful',
                      marker_color = '#90e0ef'
                      )
             )

      fig3.add_trace(go.Bar(x=failed_data['day_of_week'],
                      y=failed_data['count'],
                      name='Failed',
                      marker_color = '#03045e'
                      )
             )
    
      fig3.update_xaxes(categoryorder='array', categoryarray=order)
      fig3.update_layout(
        title='Nombre de campagnes par jour de la Semaine',
        xaxis_title='Jour de la Semaine',
        yaxis_title='Nombre de campagnes',
        legend_title='Statut')
      st.plotly_chart(fig3)
    with tab3 : 
      

      st.write("<p style='text-align:justify'> Afin d'approfondir notre analyse du jeu de données, nous avons élaboré une analyse textuelle du contenu des descriptions de projet (dans la colonne Blurb), afin de voir si celle-ci à une incidence sur le succès d’un projet.", unsafe_allow_html=True)
      
      st.write('<h4 style="color:blue; font-weight: bold;"> Analyse quantitative nombre de caractère et nombre de mots:</h4>', unsafe_allow_html=True)
      st.image("nombre de caracteres.JPG")
     
      st.write('<h4 style="color:blue; font-weight: bold;"> Nuage des mots les plus utilisés pour les projets successful :</h4>', unsafe_allow_html=True)
      st.image("word_cloud.jpg")

  
      st.write('<h4 style="color:blue; font-weight: bold;""> Analyse des représentations des mots les plus utilisés pour les descriptions des projets: "Fréquence sur 3_grams</h4>', unsafe_allow_html=True)
      st.image("Fréquence_3_grams.jpg")
      st.write('<h4 style="color:blue; font-weight: bold;""> Topic Modeling :</h4>', unsafe_allow_html=True)

      st.write("<p style='text-align:justify'> Nous utiliserons LDA pour extraire les thèmes clés des descriptions de projets réussis, limités à deux sujets en raison de la taille de la base de données.", unsafe_allow_html=True)
      st.image("topic_modeling.jpg")
elif page == pages[3] :

      tab1, tab2= st.tabs(["Préprocessing", "Modélisation"])
      
      with tab1 :
              
         st.write('<h4 style="color:blue; font-weight: bold;"> Encodage des Variables Catégorielles : </h4>', unsafe_allow_html=True)
         st.write('<h5> Encodage de la Variable "City" : </h5>', unsafe_allow_html=True)
         st.write("<p>  Étant donné qu'il y avait environ 13 000 modalités pour la variable 'city', nous avons restreint l’encodage 'OneHot' aux 10 villes les plus importantes.  </p>", unsafe_allow_html=True)
         st.image("top10city.png")
         st.write('<h5 > Encodage des autres variables : </h5>', unsafe_allow_html=True)
      
         st.write("<p >  Nous avons appliqué l'encodage one-hot à toutes les autres variables catégorielles, sans restriction de catégories. </p>", unsafe_allow_html=True)
         @st.cache_data
     
         def load_data() : 
             var_cat = pd.read_csv('var_cat_encod.csv')
             return var_cat
         var_cat = load_data()
        # Ajoutez une case à cocher pour contrôler l'affichage du DataFrame
         show_var_cat = st.checkbox('Afficher un aperçu des variables catégorielles encodées :')

         # Si la case à cocher est cochée, affichez le DataFrame
         if show_var_cat:
            st.dataframe(var_cat.head(3))
      

         st.write('<h4 style="color:blue; font-weight: bold;"> Standardisation des variables numériques : </h4>', unsafe_allow_html=True)
         st.write("<p >  Afin de rendre les variables numériques comparables, nous avons employé la méthode StandarScaler pour effectuer leur mise à l'échelle. </p>", unsafe_allow_html=True)
         @st.cache_data
         def load_data() : 
            X_train= pd.read_csv('X_train.csv')
            return X_train
         X_train = load_data()

              # Ajoutez une case à cocher pour contrôler l'affichage du DataFrame
         show_X_train = st.checkbox('Afficher un aperçu des données après preprocessing :')

         # Si la case à cocher est cochée, affichez le DataFrame
         if show_X_train:
            st.dataframe(X_train.head(3))

      with tab2 :
         st.write('<h4 style="color:blue; font-weight: bold;"> Entrainement du modèle choisi: </h4>', unsafe_allow_html=True)
         @st.cache_data
         def load_data() : 
            X_test= pd.read_csv('X_test.csv')
            return X_test
         X_test = load_data()
         @st.cache_data
         def load_data() : 
            y_train= pd.read_csv('y_train.csv')
            return y_train
         y_train = load_data()

         @st.cache_data
         def load_data() : 
            y_test= pd.read_csv('y_test.csv')
            return y_test
         y_test = load_data()      

         RFC = joblib.load("RFC.joblib")
         ABC = joblib.load("ABC.joblib")
         LR = joblib.load("LR.joblib")
         KNN = joblib.load("KNN.joblib")
         y_pred_rfc = RFC.predict(X_test)
         y_pred_abc = ABC.predict(X_test)
         y_pred_lr = LR.predict(X_test)
         y_pred_knn = KNN.predict(X_test)

  
          # Define a function to train the model and calculate metrics
         def train_model(classifier, y_pred):
          # Simulate a time-consuming task
           progress_bar = st.progress(0)

           for i in range(101):
             time.sleep(0.1)  # Wait for a short period of time
             progress_bar.progress(i)  # Update the progress bar

         # Calculate performance metrics
           accuracy = accuracy_score(y_test, y_pred)
           precision = precision_score(y_test, y_pred)
           recall = recall_score(y_test, y_pred)
           f1 = f1_score(y_test, y_pred)

           return accuracy, precision, recall, f1

         # User selects the classifier
         classifier = st.selectbox(
           "Choisir un Modèle :",
           ["Random Forest", "Logistic Regression", "AdaBoost Classifier", "KNN"]
           )
     
         # Call the train_model function for the selected classifier
         if classifier == "Random Forest":
           accuracy, precision, recall, f1 = train_model(RFC, y_pred_rfc)
         elif classifier == "Logistic Regression":
           accuracy, precision, recall, f1 = train_model(LR, y_pred_lr)
         elif classifier == "AdaBoost Classifier":
           accuracy, precision, recall, f1 = train_model(ABC, y_pred_abc)
         elif classifier == "KNN":
           accuracy, precision, recall, f1 = train_model(KNN, y_pred_knn)

        # Display the metrics
         st.write("Accuracy:", accuracy.round(3))
         st.write("Précision:", precision.round(3))
         st.write("Recall:", recall.round(3))
         st.write("F1_score:", f1.round(3))
         st.write('<h4 style="color:blue; font-weight: bold;"> Amélioration des performances du modèle RandomForest : </h4>', unsafe_allow_html=True)
        
        # Display a message when the task is completed
         st.image("Tableau_récapitulatif.JPG")

if page == pages[4] :
        st.title('Fund Success Predictor')
        @st.cache_data
        def load_data():
            X_train = pd.read_csv("X_train_no_scaled.csv")
            y_train = pd.read_csv('y_train.csv')
            return X_train , y_train
            
        X_train , y_train = load_data()
        top_10_cities = data['city'].value_counts().head(10).index.tolist()

       # Ajoutez une option "Top 10 Cities" à la liste des villes
        city_options = ["Top 10 Cities"] + top_10_cities      
        st.sidebar.header('Specify Input Parameters')


        def user_input_features() :
           currency = st.sidebar.selectbox("Currency", data["currency"].unique())
           country = st.sidebar.selectbox("Country", data["country"].unique())
           city = st.sidebar.selectbox("City", city_options)
           blurb_length = st.sidebar.slider("blurb_length", int(data["blurb_length"].min()),int(data["blurb_length"].max()))
           Category = st.sidebar.selectbox("Category", data["Category"].unique())
           sub_category = st.sidebar.selectbox("Sub Category", data["sub_category"].unique())
           day_of_week = st.sidebar.selectbox("Day of the week", data["day_of_week"].unique())
           month = st.sidebar.selectbox("Month", data["month"].unique())
           year = st.sidebar.slider("Year", 2009, 2023)
           goal_usd = st.sidebar.slider("Goal (in USD)", int(data["Goal"].min()), 1000000)
           duration = st.sidebar.slider("Duration", int(data["duration"].min()), int(data["duration"].max()))
           input_data = { 'Goal': [goal_usd],
                          'duration': [duration],
                          'blurb_length':[blurb_length],
                          'currency': [currency],
                          'country': [country],
                          'Category': [Category],
                          'sub_category': [sub_category],
                          'month': [month],
                          'day_of_week': [day_of_week],
                          'year': [year],
                          'city': [city]
                          }
           features = pd.DataFrame(input_data)
           return features
        progress_bar = st.progress(0)
        df = user_input_features()
        for i in range(101):
              time.sleep(0.1)  # Wait for a short period of time
              progress_bar.progress(i)  # Update the progress bar

        cat_feat = ['currency','country','Category','sub_category','month','day_of_week', 'city','year']
        df_encoded = pd.get_dummies(df, columns = cat_feat)
        missing_cols = set(X_train.columns) - set(df_encoded.columns)
       
        # Créez un dataframe vide avec les colonnes attendues
        empty_df = pd.DataFrame(columns=X_train.columns)

        # Fusionnez le dataframe vide avec df_encoded
        merged_df = pd.concat([empty_df, df_encoded], axis=0)

         # Assurez-vous que les colonnes sont dans le même ordre que dans X_train
        merged_df = merged_df[X_train.columns]
        feats = merged_df.fillna(0)
        feats.columns = X_train.columns
        
          # Affichez les premières lignes pour vérification
        
        model_trained = False
        st.dataframe(df.head())
        if st.button("Entrainer les données", key="train_model"):
           
           

           
           model = joblib.load("model_rdf.joblib")   

           model_trained = True  # Mettez la variable à True pour indiquer que le modèle a été entraîné

         # Vérifiez si le modèle a été entraîné avant de faire des prédictions
        if model_trained:
            # Apply Model to Make Prediction
           prediction = model.predict(feats)
           st.write('### Prédiction de la campagne')
           #st.write("Successful" if prediction[0] == 1 else "Failed")
           if prediction[0] == 1:
              st.success("**Successful**")
           else:
              st.error("**Failed**")
           st.write('---')
 
        st.set_option('deprecation.showPyplotGlobalUse', False)
        
        
       

        if st.button('Afficher les SHAP values'):
           st.write(' ### Visualisation des valeurs SHAP')
    
    
           col1, col2 = st.columns(2)  # Diviser l'écran en deux colonnes
        
           with col1 :
           
              
              st.write("#### Valeurs SHAP pour cette observation :")
              regressor = joblib.load('regressor.joblib')
              explainer = shap.Explainer(regressor.predict, X_train[0:100])
              shap_values_new = explainer(feats)
              shap.plots.bar(shap_values_new[0])
              st.pyplot()
           with col2:
             
              st.write("#### Valeurs SHAP Globale :")
              shap_values = joblib.load('shap_values_train.pkl')
              # Générer le graphique en abeilles (beeswarm plot)
              shap.summary_plot(shap_values, features=X_train[0:1000])  
              # Afficher le graphique à l'aide de st.pyplot
              st.pyplot() 