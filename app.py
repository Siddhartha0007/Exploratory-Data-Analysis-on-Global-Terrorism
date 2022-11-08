# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 08:30:23 2022

@author: Siddhartha-PC
"""

###  Import Libreries  
import streamlit as st
from streamlit_option_menu import option_menu
st.set_option('deprecation.showPyplotGlobalUse', False)
import contractions 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image 
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
from collections import  Counter
import inflect
from bs4 import BeautifulSoup
import streamlit.components.v1 as components
import os
#for model-building
from sklearn.model_selection import train_test_split
import string
from tqdm import tqdm
#for model accuracy
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from numpy import mean
from numpy import absolute
from numpy import sqrt
#for visualization
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
init_notebook_mode(connected=True)
import plotly.express as px
import plotly.graph_objects as go
import pickle
from joblib import dump, load
import joblib
# Utils
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
import sys
from sklearn.metrics import r2_score,accuracy_score,mean_absolute_error,mean_squared_error,mean_absolute_percentage_error 
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from plotly import tools
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold
from sklearn.preprocessing import StandardScaler,LabelEncoder
import plotly.figure_factory as ff
import cufflinks as cf
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import scipy
import re
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats.mstats import winsorize
from scipy.stats import boxcox, probplot, norm
from scipy.special import inv_boxcox
import random
import datetime
import math
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
## Hyperopt modules
#from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
#from functools import partial
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
sns.set_palette("hls")
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import scipy
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
from yellowbrick.classifier import PrecisionRecallCurve
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import plot_importance, to_graphviz
from xgboost import plot_tree
#For Model Acuracy
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import plot_importance, to_graphviz #
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import plotly.graph_objs as go
import plotly.tools as tls
import cufflinks as cf
import plotly.figure_factory as ff
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.metrics import silhouette_score
from sklearn.metrics import roc_auc_score
import matplotlib.gridspec as gridspec
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import re
import sys
#preprocessing libraries
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import plot_importance, to_graphviz
from xgboost import plot_tree
#For Model Acuracy
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from xgboost import plot_importance, to_graphviz #
from sklearn.metrics import classification_report
from yellowbrick.classifier import PrecisionRecallCurve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,roc_auc_score,accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.metrics import log_loss
# EDA Pkgs
import pandas as pd 
import codecs
from pandas_profiling import ProfileReport 
# Components Pkgs
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report

# Custome Component Fxn
import sweetviz as sv 
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#lottie animations
import time
import requests
import json
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner

#nltk libreries
import io
import requests
import json
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
#from surprise import SVD,Reader,Dataset
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import matplotlib.patches as mpatches
import plotly.tools as tls
from mpl_toolkits.basemap import Basemap
import folium
import folium.plugins
from matplotlib import animation,rc
import io
import base64
from IPython.display import HTML, display

from streamlit_folium import folium_static
import folium
###############################################Data Processing###########################
terror_data=pd.read_csv("./globalterrorism.csv",encoding="latin-1")

def clean_dataframe(terror_data):
    terror_data.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','targsubtype1_txt':'Targetsubtype','natlty1_txt':'Nationality','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
    terror_data1=terror_data[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Targetsubtype','Nationality','Weapon_type','Motive']]
    terror_data1['casualities']=terror_data['Killed']+terror_data['Wounded']
    data=terror_data1.copy()
    data = data[pd.isna(data.Country)==False]
    data = data[pd.isna(data['Target'])==False]
    data = data[pd.isna(data['longitude'])==False]
    data = data[pd.isna(data['latitude'])==False]
    data = data[pd.isna(data['city'])==False]
    data = data[pd.isna(data['Nationality'])==False]
    data = data[pd.isna(data['Targetsubtype'])==False]
    data["Killed"]=data["Killed"].fillna(0)
    data["Wounded"]=data["Wounded"].fillna(0)
    data["Casualties"]=data["Killed"]+data["Wounded"]
    data["Casualties"]=data["Casualties"].fillna(0)
    data['Country'] = data['Country'].replace('South Vietnam','Vietnam', regex=True)
    data['Weapon_type'] = data['Weapon_type'].replace('Vehicle .*','Vehicle', regex=True)
    data['AttackType'] = data['AttackType'].replace('Hostage Taking .*','Hostage Taking', regex=True)
    data["Motive"]=data["Motive"].fillna('Unknown')
    data["Summary"]=data["Summary"].fillna('Others') #casualities
    data=data.drop(columns=["casualities"],axis=1)
    data.drop_duplicates(inplace=True)
    return data



def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
lottie_url_hello = "https://assets5.lottiefiles.com/packages/lf20_ev1cfn9h.json"
lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
lottie_hello = load_lottieurl(lottie_url_hello)
project_url_1="https://assets9.lottiefiles.com/packages/lf20_bzgbs6lx.json"
project_url_2="https://assets6.lottiefiles.com/packages/lf20_eeuhulsy.json"
report_url="https://assets9.lottiefiles.com/packages/lf20_zrqthn6o.json"
about_url="https://assets2.lottiefiles.com/packages/lf20_k86wxpgr.json"

about_1=load_lottieurl(about_url)
report_1=load_lottieurl(report_url)
project_1=load_lottieurl(project_url_1)
project_2=load_lottieurl(project_url_2)

lottie_download = load_lottieurl(lottie_url_download)


def st_display_sweetviz(report_html,width=1000,height=500):
	report_file = codecs.open(report_html,'r')
	page = report_file.read()
	components.html(page,width=width,height=height,scrolling=True)



###############################################Streamlit Main###############################################

def main():
    # set page title 
    terror_data=pd.read_csv("./globalterrorism.csv",encoding="latin-1")
    data= clean_dataframe(terror_data)       
    # 2. horizontal menu with custom style
    selected = option_menu(menu_title= None,options=["Home", "Project","Report" ,"About"], icons=["house-door", "cast","clipboard","file-person"],  menu_icon="cast", default_index=0, orientation="horizontal", styles={"container": {"padding": "0!important", "background-color": "cyan"},"icon": {"color": "#6c3483", "font-size": "25px"}, "nav-link": {"font-size": "25px","text-align": "left","margin": "0px","--hover-color": "#ff5733", },"nav-link-selected":{"background-color":"#2874a6"},},)
    
    #horizontal Home selected
    if selected == "Home":
        image= Image.open("world_img.png")
        st.image(image,use_column_width=True)
        #image= Image.open("home_img.jpg")
        #st.image(image,use_column_width=True)
        #st.title("Home") 
        terror= data
# =============================================================================
#         terror_fol=terror.copy()
#         terror_fol.dropna(subset=['latitude','longitude'],inplace=True)
#         location_fol=terror_fol[['latitude','longitude']][5000:15000]
#         country_fol=terror_fol['Country'][5000:15000]
#         city_fol=terror_fol['city'][5000:15000]
#         killed_fol=terror_fol['Killed'][5000:15000]
#         wound_fol=terror_fol['Wounded'][5000:15000]
#         def color_point(x):
#             if x>=30:
#                 color='red'
#             elif ((x>0 and x<30)):
#                 color='blue'
#             else:
#                 color='green'
#             return color 
# 
#         def point_size(x):
#             if (x>30 and x<100):
#                 size=2
#             elif (x>=100 and x<500):
#                 size=8
#             elif x>=500:
#                 size=16
#             else:
#                 size=0.5
#             return size 
# 
#         map2 = folium.Map(location=[30,0],tiles='CartoDB dark_matter',zoom_start=2) # dark_matter
#         for point in location_fol.index:
#             info='<b>Country: </b>'+str(country_fol[point])+'<br><b>City: </b>: '+str(city_fol[point])+'<br><b>Killed </b>: '+str(killed_fol[point])+'<br><b>Wounded</b> : '+str(wound_fol[point])
#             iframe = folium.IFrame(html=info, width=200, height=200)
#             folium.CircleMarker(list(location_fol.loc[point].values),popup=folium.Popup(iframe),radius=point_size(killed_fol[point]),color=color_point(killed_fol[point])).add_to(map2)
#         folium_static(map2)
# =============================================================================
                                 
            
        st.sidebar.title("Home")        
        with st.sidebar:
            lottie_url_hello2 = "https://assets10.lottiefiles.com/packages/lf20_zdm1abxk.json"
            lottie_hello2 = load_lottieurl(lottie_url_hello2)
            st_lottie(lottie_hello2, key="hello2",)
            image= Image.open("home1.jpeg")
            st.image(image,use_column_width=True)                       
                   
        def header(url):
            st.sidebar.markdown(f'<p style="background-color:#a569bd ;color:white;font-size:15px;border-radius:1%;">{url}', unsafe_allow_html=True)    
        html_45=""" A Quick Youtube Video for understanding the MOVIE RECOMMENDATION SYSTEMS for Educational Purpose ."""
        
        #st.sidebar.video("https://www.youtube.com/watch?v=n3RKsY2H-NE")
        with st.sidebar:
            #image= Image.open("Home1.png")
            st.write('Author@ Siddhartha Sarkar')
            st.write('Data Scientist ')
            st.write('References:')
            st.markdown("[ Terrorism - Our World in Data:](https://ourworldindata.org)")
            st.markdown("[  Wikipedia:](https://en.wikipedia.org/wiki/Terrorism)")
            st.markdown("[  Global Terrorism Index:](https://www.start.umd.edu/gtd/)")
        st.balloons()
        #header(html_45)
        #st.sidebar.video("https://www.youtube.com/watch?v=n3RKsY2H-NE")
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;"> 🎥 Exploratory Data Analysis(EDA) on Global Terrorism Dataset</h1>
		</div>  """
        
		
        components.html(html_temp)
        def header(url):
            st.markdown(f'<p style="background-color:#d7bde2 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        
        html_temp11 = """
		           Table of  CONTENTS:<br>
                  + Description of Terrorism <br>
                  + Brief Description of the dataset <br>
                  + Importing of libraries/modules <br>
                  + Information of Dataset <br>
                  + Data Cleaning and Prepartion of Dataset <br>
                  + Analysis of the Dataset <br>
                  + Count of attacks occuring in each Year <br>
                         Which year had the highest numbers of terrorist attacks? <br>
                  + Number of attacks happening in each country <br>
                         Which country has suffered the most due to these terrorist attacks?(Hot Zones Of Terrorism) <br>
                  + Top 10 cities having most number of terrorist attacks. <br>
                         Which city has recieved the most number of acts of Terrorism?(Hot Zones of Terrorism) <br>
                  + What are the most dangerous terrorist groups? <br>
                  + What is the most used type of weapon for these attacks? <br>
                  + Pie Chart of Types of Weapons used by Terrorist Groups <br>
                  + An interactive World map indicating the countries with highest number of attacks <br>
                  


        
		  """
        
		
        header(html_temp11)
        image= Image.open("home_main.png")
        st.image(image,use_column_width=True)
        def header(url):
            st.markdown(f'<p style="background-color: #d7bde2 ;color:black;font-size:20px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_temp12 = """
		 What is Terrorism?<br>
         Terrorism is defined in the Oxford Dictionary as “the unlawful use of violence and intimidation, 
         especially against civilians, in the pursuit of political aims.”.Definitions of terrorism are usually complex 
         and controversial, and, because of the inherent ferocity and violence of terrorism.<br>
         Terrorism, the calculated use of violence to create a general climate of fear in a population 
         and thereby to bring about a particular political objective.<br>
         
   
		  """
        		
        
        
        
        header(html_temp12)
        def plot11():
            
            import plotly.graph_objects as go

            values = [['Year', 'Month', 'Day', 'Country', 'Region','city',\
                       'latitude','longitude','AttackType','Killed','Wounded',\
                           'Target','Summary','Group',\
                               'Target_type','Targetsubtype','Nationality','Weapon_type','Motive',\
                                   'Casualities'], #1st col
            ["Year",
            "Month",
            "Day",
            "Country",
            "Region",
            "city",
            "  latitude ",
            "longitude",
            " AttackType -Various AttackTypes",
            " Killed ",
            " Wounded ",
            "Target",'Summary','Group',
            " Target_type",'Targetsubtype',
            'Nationality','Weapon_type',
            'Motive  ','Casualities',
            ]]

            fig = go.Figure(data=[go.Table(
            columnorder = [1,2],
            columnwidth = [80,400],
            header = dict(
            values = [['<b>Columns of<br>  Dataset </b>'],
                  ['<b>Attribute Information</b>']],
            line_color='red',
            fill_color='royalblue',
            align=['left','center'],
            font=dict(color='black', size=14),
            height=40
                  ),
            cells=dict(
            values=values,
            line_color='red',
            fill=dict(color=['aqua', '#4d004c']),
            font=dict(color='white', size=14),
            align=['left', 'center'],
              font_size=12,
            height=20)
              )
           ])
            return fig

        p11=plot11()
        st.plotly_chart(p11)
        
        def plot12():
            import plotly.figure_factory as ff
            df_sample = data.iloc[0:10,0:11]
            colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
            font_colors=[[0,'#ffffff'], [.5,'#000000'], [1,'#000000']]
            fig =  ff.create_table(df_sample,colorscale=colorscale,index=True,font_colors=['#ffffff', '#000000','#000000'])
            fig.show()
            return fig
        p12=plot12()
        st.write("Data Table")
        st.plotly_chart(p12)
        
        
        def header(url):
            
            st.markdown(f'<p style="background-color: #d7bde2 ;color:black;font-size:20px;border-radius:0%;">{url}', unsafe_allow_html=True)
        html_temp13 = """
		 Brief Description about the Dataset:<br>
         =================================================================<br>
         1) Consists Information of around 1,81,000 terrorist attacks all over the globe.<br>
         2) The data consists of attacks of 47 years from 1970 to 2017.<br>
         3) It also includes some of the motives behind these attacks.<br>
         4) It contains data of Domestic as well as International occurrences of terrorism.<br>
         =================================================================
      
		  """      		
        header(html_temp13)

        def header(url):
            
            st.markdown(f'<p style="background-color: #d7bde2 ;color:black;font-size:20px;border-radius:0%;">{url}', unsafe_allow_html=True)
        html_temp13 = """
		 Objective:<br>
         =================================================================<br>
         1.Perform 'Exploratory Data Analysis' on 'Global Terrorism' dataset.<br>
         2.As a part analysis,my task is to find out the hot zones of terrorism.<br>
         3.What other security issues can you find through this EDA<br>
         =================================================================
      
		  """      		
        header(html_temp13)
# =============================================================================
#         st.markdown("""
#                 #### Tasks Perform by the app:
#                 + App covers the most basic Machine Learning task of  Analysis, Correlation between variables,project report.
#                 + Machine Learning on different Machine Learning Algorithms, building different models and lastly  prediction.
#                 
#                 """)
# =============================================================================
                
    #Horizontal About selected
    if selected == "About":
        #st.title(f"You have selected {selected}")
        
        st.sidebar.title("About")
        with st.sidebar:
            image= Image.open("About-Us-PNG-Isolated-Photo.png")
            add_image=st.image(image,use_column_width=True)
        
        st_lottie(about_1,key='ab1')
        #image2= Image.open("about.jpg")
        #st.image(image2,use_column_width=True)
        #st.sidebar.write("This Youtube Video Shows and Describes Different Kind Of Mushrooms for Learning Purpose ")
        #st.sidebar.video('https://www.youtube.com/watch?v=6PNq6paMBXU')
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Exploratory Data Analysis on Global Terrorism Data </h1>
		</div>  """
        
		
        components.html(html_temp)
        def header(url):
            st.markdown(f'<p style="background-color: #d7bde2 ;color:black;font-size:30px;border-radius:2%;">{url}', unsafe_allow_html=True)
        html_99   =  """
        In this project,I tried to analyse the Global Terrorism data to gain insight about the several factors related to 
        this and the most effected region of the World"""
        header(html_99)
        
        st.sidebar.markdown("""
                    #### + Project Done By :        
                    #### @Author Mr. Siddhartha Sarkar
                    
        
                    """)
        st.snow()
        
        #st.sidebar.markdown("[ Visit To Github Repositories](.git)")
    #Horizontal Project_Report selected
    if selected == "Report":
        #report_1
        #st.title("Profile Report")
        st.sidebar.title("Project_Profile_Report")
        
        with st.sidebar:
            st_lottie(report_1, key="report1")
            #image= Image.open("report_project.png")
            #add_image=st.image(image,use_column_width=True)
        
        st.balloons()    
        html_temp = """
		<div style="background-color:royalblue;padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Profile Report Generation </h1>
		</div>  """
        
		
        components.html(html_temp)
        html_temp1 = """
			<style>
			* {box-sizing: border-box}
			body {font-family: Verdana, sans-serif; margin:0}
			.mySlides {display: none}
			img {vertical-align: middle;}
			/* Slideshow container */
			.slideshow-container {
			  max-width: 1500px;
			  position: relative;
			  margin: auto;
			}
			/* Next & previous buttons */
			.prev, .next {
			  cursor: pointer;
			  position: absolute;
			  top: 50%;
			  width: auto;
			  padding: 16px;
			  margin-top: -22px;
			  color: white;
			  font-weight: bold;
			  font-size: 18px;
			  transition: 0.6s ease;
			  border-radius: 0 3px 3px 0;
			  user-select: none;
			}
			/* Position the "next button" to the right */
			.next {
			  right: 0;
			  border-radius: 3px 0 0 3px;
			}
			/* On hover, add a black background color with a little bit see-through */
			.prev:hover, .next:hover {
			  background-color: rgba(0,0,0,0.8);
			}
			/* Caption text */
			.text {
			  color: #f2f2f2;
			  font-size: 15px;
			  padding: 8px 12px;
			  position: absolute;
			  bottom: 8px;
			  width: 100%;
			  text-align: center;
			}
			/* Number text (1/3 etc) */
			.numbertext {
			  color: #f2f2f2;
			  font-size: 12px;
			  padding: 8px 12px;
			  position: absolute;
			  top: 0;
			}
			/* The dots/bullets/indicators */
			.dot {
			  cursor: pointer;
			  height: 15px;
			  width: 15px;
			  margin: 0 2px;
			  background-color: #bbb;
			  border-radius: 50%;
			  display: inline-block;
			  transition: background-color 0.6s ease;
			}
			.active, .dot:hover {
			  background-color: #717171;
			}
			/* Fading animation */
			.fade {
			  -webkit-animation-name: fade;
			  -webkit-animation-duration: 1.5s;
			  animation-name: fade;
			  animation-duration: 1.5s;
			}
			@-webkit-keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			@keyframes fade {
			  from {opacity: .4} 
			  to {opacity: 1}
			}
			/* On smaller screens, decrease text size */
			@media only screen and (max-width: 300px) {
			  .prev, .next,.text {font-size: 11px}
			}
			</style>
			</head>
			<body>
			<div class="slideshow-container">
			<div class="mySlides fade">
			  <div class="numbertext">1 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_nature_wide.jpg" style="width:100%">
			  <div class="text">Caption Text</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">2 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_snow_wide.jpg" style="width:100%">
			  <div class="text">Caption Two</div>
			</div>
			<div class="mySlides fade">
			  <div class="numbertext">3 / 3</div>
			  <img src="https://www.w3schools.com/howto/img_mountains_wide.jpg" style="width:100%">
			  <div class="text">Caption Three</div>
			</div>
			<a class="prev" onclick="plusSlides(-1)">&#10094;</a>
			<a class="next" onclick="plusSlides(1)">&#10095;</a>
			</div>
			<br>
			<div style="text-align:center">
			  <span class="dot" onclick="currentSlide(1)"></span> 
			  <span class="dot" onclick="currentSlide(2)"></span> 
			  <span class="dot" onclick="currentSlide(3)"></span> 
			</div>
			<script>
			var slideIndex = 1;
			showSlides(slideIndex);
			function plusSlides(n) {
			  showSlides(slideIndex += n);
			}
			function currentSlide(n) {
			  showSlides(slideIndex = n);
			}
			function showSlides(n) {
			  var i;
			  var slides = document.getElementsByClassName("mySlides");
			  var dots = document.getElementsByClassName("dot");
			  if (n > slides.length) {slideIndex = 1}    
			  if (n < 1) {slideIndex = slides.length}
			  for (i = 0; i < slides.length; i++) {
			      slides[i].style.display = "none";  
			  }
			  for (i = 0; i < dots.length; i++) {
			      dots[i].className = dots[i].className.replace(" active", "");
			  }
			  slides[slideIndex-1].style.display = "block";  
			  dots[slideIndex-1].className += " active";
			}
			</script>
			"""
        components.html(html_temp1)
        st.sidebar.title("Navigation")
        menu = ['None',"Sweetviz","Pandas Profile"]
        choice = st.sidebar.radio("Menu",menu)
        if choice == 'None':
            st.markdown("""
                        #### Kindly select from left Menu.
                       # """)
        elif choice == "Pandas Profile":
            st.subheader("Automated EDA with Pandas Profile")
            st.subheader("Upload Your File to Perform Report analysis")
            data_file= st.file_uploader("Upload CSV",type=['csv'])
            if data_file is not None:
                df = pd.read_csv(data_file)
                st.table(df.head(10))
                m = st.markdown("""
                       <style>
                    div.stButton > button:first-child {
                  background-color: #0099ff;
                      color:#ffffff;
                                   }
                    div.stButton > button:hover {
                     background-color: #00ff00;
                         color:#ff0000;
                          }
                    </style>""", unsafe_allow_html=True)

                if st.button("Generate Profile Report"):
                    profile= ProfileReport(df)
                    st_profile_report(profile)
            
            
        elif choice == "Sweetviz":
            st.subheader("Automated EDA with Sweetviz")
            st.subheader("Upload Your File to Perform Report analysis")
            data_file = st.file_uploader("Upload CSV",type=['csv'])
            if data_file is not None:
                df =  pd.read_csv(data_file)
                st.dataframe(df.head(10))
                m = st.markdown("""
                            <style>
                         div.stButton > button:first-child {
                                 background-color: #0099ff;
                                      color:#ffffff;
                                        }
                            div.stButton > button:hover {
                               background-color: #00ff00;
                                       color:#ff0000;
                                             }
                        </style>""", unsafe_allow_html=True)

                if st.button("Generate Sweetviz Report"):
                    # Normal Workflow
                    report = sv.analyze(df)
                    report.show_html()
                    st_display_sweetviz("SWEETVIZ_REPORT.html") 
               			
		      
    #Horizontal Project selected
    if selected == "Project":
            
            terror_data=pd.read_csv("./globalterrorism.csv",encoding="latin-1")
            data= clean_dataframe(terror_data)
            
            
            with st.sidebar:
                st_lottie(project_1, key="pro1")                
            import time                
            st_lottie(project_2, key="pro22")
            st.title("Projects")              
            #image2= Image.open("project11.jpeg")
            #st.image(image2,use_column_width=True)
            my_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1)
            st.sidebar.title("Navigation")
            with st.sidebar:
                
                menu_Pre_Exp = option_menu("Exploratory Data Analysis", ['Global',"India", "United States","West Europe","East Europe","Middle East & North Africa","East Asia","Central Asia","Australasia & Oceania"],
                         icons=['globe'],
                         menu_icon="app-indicator", default_index=0,orientation="vertical",
                         styles={
        "container": {"padding": "5!important", "background-color": "#c39bd3"},
        "icon": {"color": "#2980b9", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#ec7063"},
        "nav-link-selected": {"background-color": "green"}, } )

#['Global',"India", "United States","West Europe","East Europe","Middle East & North Africa","East Asia","Central Asia","Australasia & Oceania"]            
            #EDA On Document File
                       
            
            if  menu_Pre_Exp == 'Global' : # and selected == "Projects"
                    st.title('Global')
                    image= Image.open("world_img.png")
                    st.image(image,use_column_width=True)
                    st.write("Dataset")
                    st.write(data.head(10))
                    st.write("Some Basic Insights Of the Data")
                    st.write("Year with the most attacks:",data['Year'].value_counts().idxmax())
                    
                    st.write("Country with the most attacks:",data['Country'].value_counts().idxmax())
                    
                    st.write("City with the most attacks:",data['city'].value_counts().index[1])
                    
                    st.write("Region with the most attacks:",data['Region'].value_counts().idxmax())
                    
                    st.write("Most Attack Types:",data['AttackType'].value_counts().idxmax())
                    
                    st.write("Most Targeted Person/Org :",data['Target'].value_counts().idxmax())
                    
                    st.write("Most Target_type :",data['Target_type'].value_counts().idxmax())
                    
                    st.write("Most Target subtype :",data['Targetsubtype'].value_counts().idxmax())
                    
                    st.write("Most used Weapon_type :",data['Weapon_type'].value_counts().idxmax())

                    st.write("-"*65)
                    st.write('Country with Highest Terrorist Attacks:\n',data['Country'].value_counts().head(10))
                    st.write('\n\nRegions with Highest Terrorist Attacks:\n',data['Region'].value_counts().head())
                    st.write('\n\nMaximum people killed in an attack are:\n',data['Killed'].max(),'that took place in',data.loc[data['Killed'].idxmax()].Country)

                    st.write("Basic Statistics Of the Data:")
                    def plot12():
                        import plotly.figure_factory as ff
                        df_sample = data.describe(include='all').round(2).T
                        colorscale = [[0, '#4d004c'],[.5, '#f2e5ff'],[1, '#ffffff']]
                        font_colors=[[0,'#ffffff'], [.5,'#000000'], [1,'#000000']]
                        fig =  ff.create_table(df_sample,colorscale=colorscale,index=True,font_colors=['#ffffff', '#000000','#000000'])
                        fig.show()
                        return fig
                    p12=plot12()
                    st.plotly_chart(p12)
                    m = st.markdown("""
                                  <style>
                                  div.stButton > button:first-child {
                                   background-color: #0099ff;
                                       color:#ffffff;
                                                         }
                                   div.stButton > button:hover {
                                   background-color: #00ff00;
                                   color:#ff0000;
                                                   }
                                  </style>""", unsafe_allow_html=True)
                    submit = st.button(label='Generate Visualizations')
                    if submit:
                        plt.subplots(figsize=(15,6))
                        plt.style.use('fivethirtyeight')
                        sns.countplot('Year',data=data)
                        plt.xticks(rotation=90)
                        plt.title('Number Of Terrorist Activities Year Wise')
                        st.pyplot()
                        
                        plt.subplots(figsize=(15,6))
                        sns.countplot('AttackType',data=data,palette='inferno',order=data['AttackType'].value_counts().index)
                        plt.xticks(rotation=90)
                        plt.title('Attacking Methods by Terrorists')
                        st.pyplot()
                        attackType_filtered =data['AttackType'] #.apply(lambda x: x if x in ['Explosives','Firearms','Unknown',
                                                           #    'Incendiary'] else 'Others')
                        attackType = attackType_filtered.value_counts().tolist()

                        # Pie chart of weapons types
                        attackType_labels = ['Assassination', 'Hostage Taking', 'Bombing/Explosion',
                                      'Facility/Infrastructure Attack', 'Armed Assault', 'Hijacking',
                        'Unknown', 'Unarmed Assault']

                        attackType_sizes = []

                        for j in attackType:
                            percent = j*100/len(data['AttackType'])
                            attackType_sizes.append(percent)
                            
                        

                        fig, ax = plt.subplots(figsize=(14,7))
                        patches, texts, autotexts = ax.pie(attackType_sizes, labels=attackType_labels, autopct='%1.1f%%',
                             startangle = -20, shadow = True,
                          explode = (0.05, 0, 0, 0, 0,0.002,0.05,0),
                        colors = sns.color_palette("cool", 10)[:4:1]+
                                   [(0.5, 0.6, 0.65)],
                        textprops={'fontsize':12,'weight':'light','color':'red','rotation':-15})

                        ax.axis('equal')
                        plt.title('Attack types', fontsize= 20, pad= 20, weight ='bold', 
                         color = 'red') 
                        ax.legend(loc='lower right',framealpha = 0.5,bbox_to_anchor=(1.2,0.5,0.1,1), prop={'size': 14})
                        st.pyplot()
                        
                        
                        plt.subplots(figsize=(15,6))
                        sns.countplot(data['Target_type'],order=data['Target_type'].value_counts().index,\
                                 palette=  sns.color_palette("viridis", 10)[:4:1]+
                                   [(0.5, 0.6, 0.65)])
                        plt.xticks(rotation=90)
                        plt.title('Targeted Sectors of the society')
                        st.pyplot()
                        
                        plt.subplots(figsize=(15,6))
                        sns.countplot('Region',data=data,palette='magma',\
                                    edgecolor=sns.color_palette('dark',7),order=data['Region'].value_counts().index)
                        plt.xticks(rotation=90)
                        plt.title('Number Of Terrorist Activities By Region')
                        st.pyplot()
                        
                        
                        weapontype_filtered =data['Weapon_type'].apply(lambda x: x if x in ['Explosives/Bombs/Dynamite','Firearms','Unknown',
                                                               'Incendiary'] else 'Others')
                        weapontype = weapontype_filtered.value_counts().tolist()

                        # Pie chart of weapons types
                        weap_labels = ['Explosives/Bombs/Dynamite','Firearms','Unknown','Incendiary','Others']

                        weap_sizes = []

                        for j in weapontype:
                            percent = j*100/len(data['Weapon_type'])
                            weap_sizes.append(percent)
                        

                        fig, ax = plt.subplots(figsize=(14,7))
                        patches, texts, autotexts = ax.pie(weap_sizes , labels=weap_labels,autopct='%1.1f%%',
                        startangle = -20, shadow = True,
                        #explode = (0.05, 0, 0, 0, 0),
                        colors = sns.color_palette("tab20", 5)[:4:1]+
                                   [(0.5, 0.6, 0.65)],
                        textprops={'fontsize':12,'weight':'light','color':'black','rotation':-15})

                        ax.axis('equal')
                        plt.title('Weapon types', fontsize= 20, pad= 20, weight ='bold', 
                        color = 'k') 
                        ax.legend(loc='lower right',framealpha = 0.5,bbox_to_anchor=(1.2,0.5,0.1,1), prop={'size': 14})
                        st.pyplot()
                        
                        #AttackType vs Region¶
                        pd.crosstab(data.Region,data.AttackType).plot.barh(stacked=True,width=1,color=sns.color_palette('viridis',9))
                        fig=plt.gcf()
                        fig.set_size_inches(12,8)
                        plt.title("Attack type In Various Regions")
                        st.pyplot()
                        
                        #AttackType vs Region¶
                        pd.crosstab(data.Region,data.Target_type).plot.barh(stacked=True,width=1,color=sns.color_palette('viridis',9))
                        fig=plt.gcf()
                        fig.set_size_inches(15,8)
                        plt.title("Targeted Part of the Society In Various Regions")
                        st.pyplot()
                        
                        #Terrorism By Country
                        plt.subplots(figsize=(18,6))
                        sns.barplot(data['Country'].value_counts()[:15].index,data['Country'].value_counts()[:15].values,palette='inferno')
                        plt.title('Top Affected Countries')
                        st.pyplot()
                        
                        #Most Notorious Groups
                        sns.barplot(data['Group'].value_counts()[1:15].values,data['Group'].value_counts()[1:15].index,palette=('rainbow'))
                        plt.xticks(rotation=90)
                        fig=plt.gcf()
                        fig.set_size_inches(15,7)
                        plt.title('Terrorist Groups with Highest Terror Attacks')
                        st.pyplot()
                        
                        #Activity of Top Terrorist Groups
                        top_groups10=data[data['Group'].isin(data['Group'].value_counts()[1:11].index)]
                        pd.crosstab(top_groups10.Year,top_groups10.Group).plot(color=sns.color_palette('Paired',10))
                        fig=plt.gcf()
                        fig.set_size_inches(18,6)
                        plt.title('Activity of Top Terrorist Groups')
                        st.pyplot()
                        
                        terror=data.copy()
                        terror_region=pd.crosstab(terror.Year,terror.Region)
                        terror_region.plot(color=sns.color_palette('Set1',12))
                        fig=plt.gcf()
                        fig.set_size_inches(18,6)
                        plt.title('Activity of Terrorist Groups in Various regions')
                        st.pyplot()
                        
                        #Attacks vs Killed
                        terror=data.copy()
                        coun_terror=terror['Country'].value_counts()[:15].to_frame()
                        coun_terror.columns=['Attacks']
                        coun_kill=terror.groupby('Country')['Killed'].sum().to_frame()
                        coun_terror.merge(coun_kill,left_index=True,right_index=True,how='left').plot.bar(width=0.9)
                        fig=plt.gcf()
                        fig.set_size_inches(18,6)
                        plt.title('Attacks vs Killed')
                        st.pyplot()
                        
                        terror=data.copy()
                        top_groups=terror[terror['Group'].isin(terror['Group'].value_counts()[:14].index)]
                        m4 = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
                        m4.drawcoastlines()
                        m4.drawcountries()
                        m4.fillcontinents(lake_color='aqua')
                        m4.drawmapboundary(fill_color='aqua')
                        fig=plt.gcf()
                        fig.set_size_inches(22,10)
                        colors=['r','g','b','y','#800000','#ff1100','#8202fa','#20fad9','#ff5733','#fa02c7',"#f99505",'#b3b6b7','#8e44ad','#1a2b3c']
                        group=list(top_groups['Group'].unique())
                        def group_point(group,color,label):
                            lat_group=list(top_groups[top_groups['Group']==group].latitude)
                            long_group=list(top_groups[top_groups['Group']==group].longitude)
                            x_group,y_group=m4(long_group,lat_group)
                            m4.plot(x_group,y_group,'go',markersize=3,color=j,label=i)
                        
                        for i,j in zip(group,colors):
                            group_point(i,j,i)
                        
                        legend=plt.legend(loc='lower left',frameon=True,prop={'size':10})
                        frame=legend.get_frame()
                        frame.set_facecolor('white')
                        plt.title('Regional Activities of Terrorist Groups with Map')
                        st.pyplot()
                        
                        #Global Terror Attacks
                        m3 = Basemap(projection='cea',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
                        lat_100=list(terror[terror['Casualties']>=75].latitude)
                        long_100=list(terror[terror['Casualties']>=75].longitude)
                        x_100,y_100=m3(long_100,lat_100)
                        m3.plot(x_100, y_100,'go',markersize=5,color = 'r')
                        lat_=list(terror[terror['Casualties']<75].latitude)
                        long_=list(terror[terror['Casualties']<75].longitude)
                        x_,y_=m3(long_,lat_)
                        m3.plot(x_, y_,'go',markersize=2,color = 'gold',alpha=0.4)
                        m3.drawcoastlines()
                        m3.drawcountries()
                        m3.fillcontinents(lake_color='aqua')
                        m3.drawmapboundary(fill_color='aqua')
                        fig=plt.gcf()
                        fig.set_size_inches(18,10)
                        plt.title('Global Terrorist Attacks with  Casualties plotting')
                        plt.legend(loc='lower left',handles=[mpatches.Patch(color='gold', label = "< 75 Casualties"),
                                   mpatches.Patch(color='red',label='> 75 Casualties')])
                        st.pyplot()
                        
                        #Global Terror Attacks
                        m3 = Basemap(projection='cea',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c',lat_0=True,lat_1=True)
                        lat_100=list(terror[terror['Casualties']>=75].latitude)
                        long_100=list(terror[terror['Casualties']>=75].longitude)
                        x_100,y_100=m3(long_100,lat_100)
                        m3.plot(x_100, y_100,'go',markersize=5,color = 'r')
                        lat_=list(terror[terror['Casualties']<75].latitude)
                        long_=list(terror[terror['Casualties']<75].longitude)
                        x_,y_=m3(long_,lat_)
                        m3.plot(x_, y_,'go',markersize=2,color = 'blue',alpha=0.4)
                        m3.drawcoastlines()
                        m3.drawcountries()
                        m3.fillcontinents(lake_color='aqua')
                        m3.drawmapboundary(fill_color='aqua')
                        fig=plt.gcf()
                        fig.set_size_inches(18,10)
                        plt.title('Global Terrorist Attacks')
                        plt.legend(loc='lower left',handles=[mpatches.Patch(color='blue', label = "< 75 Casualties"),
                                mpatches.Patch(color='red',label='> 75 Casualties')])
                        st.pyplot()
                        
                        terror_fol=terror.copy()
                        terror_fol.dropna(subset=['latitude','longitude'],inplace=True)
                        location_fol=terror_fol[['latitude','longitude']][5000:15000]
                        country_fol=terror_fol['Country'][5000:15000]
                        city_fol=terror_fol['city'][5000:15000]
                        killed_fol=terror_fol['Killed'][5000:15000]
                        wound_fol=terror_fol['Wounded'][5000:15000]
                        def color_point(x):
                            if x>=30:
                                color='red'
                            elif ((x>0 and x<30)):
                                color='blue'
                            else:
                                color='green'
                            return color 
      
                        def point_size(x):
                            if (x>30 and x<100):
                                size=2
                            elif (x>=100 and x<500):
                                size=8
                            elif x>=500:
                                size=16
                            else:
                                size=0.5
                            return size 
      
                        map2 = folium.Map(location=[30,0],tiles='CartoDB dark_matter',zoom_start=2) # dark_matter
                        for point in location_fol.index:
                            info='<b>Country: </b>'+str(country_fol[point])+'<br><b>City: </b>: '+str(city_fol[point])+'<br><b>Killed </b>: '+str(killed_fol[point])+'<br><b>Wounded</b> : '+str(wound_fol[point])
                            iframe = folium.IFrame(html=info, width=200, height=200)
                            folium.CircleMarker(list(location_fol.loc[point].values),popup=folium.Popup(iframe),radius=point_size(killed_fol[point]),color=color_point(killed_fol[point])).add_to(map2)
                        folium_static(map2)
    
            
            if  menu_Pre_Exp == 'India':
                terror=data.copy()
                st.title('India')
                image= Image.open("india.jpg")
                st.image(image,use_column_width=True)
                m = st.markdown("""
                              <style>
                              div.stButton > button:first-child {
                               background-color: #0099ff;
                                   color:#ffffff;
                                                     }
                               div.stButton > button:hover {
                               background-color: #00ff00;
                               color:#ff0000;
                                               }
                              </style>""", unsafe_allow_html=True)
                submit = st.button(label='Generate Visualizations')
                if submit:
                    
                    terror_india=terror[terror['Country']=='India']
                    terror_fol=terror_india
                    terror_fol.dropna(subset=['latitude','longitude'],inplace=True)
                    location_fol=terror_fol[['latitude','longitude']][5000:15000]
                    country_fol=terror_fol['Country'][5000:15000]
                    city_fol=terror_fol['city'][5000:15000]
                    killed_fol=terror_fol['Killed'][5000:15000]
                    wound_fol=terror_fol['Wounded'][5000:15000]
                    def color_point(x):
                        if x>=30:
                            color='red'
                        elif ((x>0 and x<30)):
                            color='blue'
                        else:
                            color='green'
                        return color 
  
                    def point_size(x):
                        if (x>30 and x<100):
                            size=2
                        elif (x>=100 and x<500):
                            size=8
                        elif x>=500:
                            size=16
                        else:
                            size=0.5
                        return size 
  
                    map2 = folium.Map(location=[30,0],tiles='CartoDB dark_matter',zoom_start=2) # dark_matter
                    for point in location_fol.index:
                        info='<b>Country: </b>'+str(country_fol[point])+'<br><b>City: </b>: '+str(city_fol[point])+'<br><b>Killed </b>: '+str(killed_fol[point])+'<br><b>Wounded</b> : '+str(wound_fol[point])
                        iframe = folium.IFrame(html=info, width=200, height=200)
                        folium.CircleMarker(list(location_fol.loc[point].values),popup=folium.Popup(iframe),radius=point_size(killed_fol[point]),color=color_point(killed_fol[point])).add_to(map2)
                    folium_static(map2)
                    plt.figure(figsize=(13,6))
                    ind_groups=terror_india['Group'].value_counts()[1:11].index
                    ind_groups=terror_india[terror_india['Group'].isin(ind_groups)]
                    sns.countplot(y='Group',data=ind_groups,order=terror_india['Group'].value_counts()[1:11].index,palette='inferno')
                    #plt.title('Top Groups')
                    st.pyplot()  
                    Target_type_filtered =terror_india['Target_type']
                    Target_type = Target_type_filtered.value_counts().tolist()

                    # Pie chart of weapons types
                    Target_type_labels = ['Airports & Aircraft', 'Government (General)', 'Police',\
                     'Telecommunication', 'Private Citizens & Property',\
                        'Religious Figures/Institutions', 'Transportation', 'NGO',\
                        'Utilities', 'Military', 'Violent Political Party',\
                           'Government (Diplomatic)', 'Maritime', 'Business',\
                       'Educational Institution', 'Journalists & Media',\
                      'Terrorists/Non-State Militia', 'Food or Water Supply', 'Tourists',\
                                  'Other']

                    Target_type_sizes = []

                    for j in Target_type:
                       percent = j*100/len(terror_india['Target_type'])
                       Target_type_sizes.append(percent)

                    fig, ax = plt.subplots(figsize=(14,7))
                    patches, texts, autotexts = ax.pie(Target_type_sizes, labels=Target_type_labels, autopct='%1.1f%%',
                    startangle = -20, shadow = True,
                    #explode = (0.05, 0, 0, 0, 0,0.002,0.05,0),
                    colors = sns.color_palette("plasma", 10)[:4:1]+
                                   [(0.5, 0.6, 0.65)],
                    textprops={'fontsize':12,'weight':'light','color':'red','rotation':-15})

                    ax.axis('equal')
                    plt.title('Targeted Areas of Socities of INDIA', fontsize= 20, pad= 20, 
                             color = 'red') 
                    ax.legend(loc='lower right',framealpha = 0.5,bbox_to_anchor=(1.2,0.5,0.1,1), prop={'size': 10})
                    st.pyplot()
                    plt.figure(figsize=(13,6))
                    sns.countplot(y='AttackType',data=terror_india,order=terror_india['AttackType'].value_counts().index,\
                                      palette='plasma')
                    plt.title('Attack Types of Terrorist groups')
                    plt.subplots_adjust(hspace=0.3,wspace=0.6)
                    plt.xticks()
                    st.pyplot()
                    #AttackType vs Region¶
                    pd.crosstab(terror_india.Target_type,terror_india.AttackType).plot.barh(stacked=True,width=1,color=sns.color_palette('viridis',9))
                    fig=plt.gcf()
                    fig.set_size_inches(15,8)
                    plt.title(" Various types of attacks on Different Part of the Society")
                    st.pyplot()
                    weapontype_filtered =terror_india['Weapon_type'].apply(lambda x: x if x in ['Explosives/Bombs/Dynamite','Firearms','Unknown',
                                                               'Incendiary'] else 'Others')
                    weapontype = weapontype_filtered.value_counts().tolist()

                    # Pie chart of weapons types
                    weap_labels = ['Explosives/Bombs/Dynamite','Firearms','Unknown','Incendiary','Others']

                    weap_sizes = []

                    for j in weapontype:
                        percent = j*100/len(terror_india['Weapon_type'])
                        weap_sizes.append(percent)
                        
                     

                    fig, ax = plt.subplots(figsize=(14,7))
                    patches, texts, autotexts = ax.pie(weap_sizes , labels=weap_labels,autopct='%1.1f%%',
                    startangle = -20, shadow = True,
                    #explode = (0.05, 0, 0, 0, 0),
                    colors = sns.color_palette("cool", 5)[:4:1]+
                                   [(0.5, 0.6, 0.65)],
                    textprops={'fontsize':12,'weight':'light','color':'black','rotation':-15})

                    ax.axis('equal')
                    plt.title('Weapon types', fontsize= 20, pad= 20, weight ='bold', 
                                color = 'k') 
                    ax.legend(loc='lower right',framealpha = 0.5,bbox_to_anchor=(1.2,0.5,0.1,1), prop={'size': 14})
                    st.pyplot()



                    st.success('Done!')       
                        
                        
            if  menu_Pre_Exp == 'United States':
                terror=data.copy()
                st.title('United States')
                m = st.markdown("""
                              <style>
                              div.stButton > button:first-child {
                               background-color: #0099ff;
                                   color:#ffffff;
                                                     }
                               div.stButton > button:hover {
                               background-color: #00ff00;
                               color:#ff0000;
                                               }
                              </style>""", unsafe_allow_html=True)
                submit = st.button(label='Generate Visualizations')
                if submit:
                    image= Image.open("usa.jpg")
                    st.image(image,use_column_width=True)
                    image= Image.open("usa5.png")
                    st.image(image,use_column_width=True)
                    image= Image.open("usa1.png")
                    st.image(image,use_column_width=True)
                    image= Image.open("usa2.png")
                    st.image(image,use_column_width=True)
                    image= Image.open("usa3.png")
                    st.image(image,use_column_width=True)
                    image= Image.open("usa4.png")
                    st.image(image,use_column_width=True)
                    
                
                
                    

    

            if  menu_Pre_Exp == 'West Europe':
                image= Image.open("west.png")
                st.image(image,use_column_width=True)
                image= Image.open("west1.png")
                st.image(image,use_column_width=True)
                image= Image.open("west2.png")
                st.image(image,use_column_width=True)
                image= Image.open("west3.png")
                st.image(image,use_column_width=True)
                image= Image.open("west4.png")
                st.image(image,use_column_width=True)
                
                   
   

            if  menu_Pre_Exp == 'East Europe':
                image= Image.open("east.png")
                st.image(image,use_column_width=True)
                image= Image.open("east1.png")
                st.image(image,use_column_width=True)
                image= Image.open("east2.png")
                st.image(image,use_column_width=True)
                image= Image.open("east3.png")
                st.image(image,use_column_width=True)
                image= Image.open("west4.png")
                st.image(image,use_column_width=True)
            if  menu_Pre_Exp == 'Middle East & North Africa':
                image= Image.open("me.png")
                st.image(image,use_column_width=True)
                image= Image.open("me1.png")
                st.image(image,use_column_width=True)
                image= Image.open("me2.png")
                st.image(image,use_column_width=True)
                image= Image.open("me3.png")
                st.image(image,use_column_width=True)
                image= Image.open("me4.png")
                st.image(image,use_column_width=True)
                image= Image.open("me5.png")
                st.image(image,use_column_width=True)
            if  menu_Pre_Exp == 'East Asia':
                image= Image.open("ea.png")
                st.image(image,use_column_width=True)
                image= Image.open("ea1.png")
                st.image(image,use_column_width=True)
                image= Image.open("ea2.png")
                st.image(image,use_column_width=True)
                image= Image.open("ea3.png")
                st.image(image,use_column_width=True)
                
            if  menu_Pre_Exp == 'Central Asia':
                image= Image.open("ca.png")
                st.image(image,use_column_width=True)
                image= Image.open("ca1.png")
                st.image(image,use_column_width=True)
                image= Image.open("ca2.png")
                st.image(image,use_column_width=True)
                image= Image.open("ca3.png")
                st.image(image,use_column_width=True)
            if  menu_Pre_Exp == 'Australasia & Oceania': 
                image= Image.open("aus.png")
                st.image(image,use_column_width=True)
                image= Image.open("aus1.png")
                st.image(image,use_column_width=True)
                image= Image.open("aus2.png")
                st.image(image,use_column_width=True)
                image= Image.open("aus2.png")
                st.image(image,use_column_width=True)
                           
                                
#"United States","West Europe","East Europe","Middle East & North Africa","East Asia","Central Asia","Australasia & Oceania
                                                      
if __name__=='__main__':
    main()            


