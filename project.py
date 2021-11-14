from IPython.display import display
from matplotlib import pyplot as plt
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import altair as alt
import timeit
import math
import streamlit as st
import s3fs
import os

# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)

data_url_2020 = "s3://streamlitabdel/full_2020.csv"
data_url_2019 = "s3://streamlitabdel/full_2019.csv"
data_url_2018 = "s3://streamlitabdel/full_2018.csv"
data_url_2017 = "s3://streamlitabdel/full_2017.csv"


@st.cache(ttl=600)
def read_file(filename):
    with fs.open(filename) as f:
        return f.read().decode("utf-8")

@st.cache(allow_output_mutation=True)
def load_data(url):
    with fs.open(url) as f:
        df = pd.read_csv(url, low_memory=False, nrows = 100000)
    return df



@st.cache(allow_output_mutation=True)
def drop_data(df):
    temp_df = df
    drop_col = []
    for col in temp_df.columns:
        missing = round((temp_df[col].isna().sum()*100)/len(temp_df[col]), 2)
        print("  {}   {}% " .format(col, missing))
        if missing > 80:
            drop_col.append(col)
    return df.drop(drop_col, axis=1, inplace=True)


@st.cache(allow_output_mutation=True)
def drop_dataNA(dfa):
    dfa = dfa.dropna()
    return dfa


@st.cache(allow_output_mutation=True)
def drop_dupicated(df):
    df = df.drop_duplicates()
    return df
@st.cache(allow_output_mutation=True)
def transform_Codep_ValF(df):
    nature = df.groupby('code_departement')['valeur_fonciere'].sum().reset_index().sort_values('valeur_fonciere', ascending=False).head(20)
    nature = nature.rename(columns={'valeur_fonciere': 'sum'})
    
    return nature


@st.cache(allow_output_mutation=True)
def transform_Codp_Surface(df):
    nature = df.groupby('code_departement')['surface_terrain'].sum().reset_index().sort_values('surface_terrain', ascending=False).head(20)
    nature = nature.rename(columns={'surface_terrain': 'sum'})
    return nature


@st.cache(allow_output_mutation=True)
def  transform_mut_valF(df):
    nature = df.groupby('nature_mutation')['valeur_fonciere'].sum(
    ).reset_index().sort_values('valeur_fonciere', ascending=False).head(6)


    nature = nature.rename(columns={'valeur_fonciere': 'sum'})
    return nature
@st.cache(allow_output_mutation=True)
def  transform_mut_surf(df):
    nature = df.groupby('nature_mutation')['surface_terrain'].sum(
    ).reset_index().sort_values('surface_terrain', ascending=False).head(6)


    nature = nature.rename(columns={'surface_terrain': 'sum'})
    return nature

@st.cache(allow_output_mutation=True)
def transform_nat_valF(df):
    nature = df.groupby('nature_culture')['valeur_fonciere'].sum(
    ).reset_index().sort_values('valeur_fonciere', ascending=False).head(6)


    nature = nature.rename(columns={'valeur_fonciere': 'sum'})
    
    return nature


@st.cache(allow_output_mutation=True)
def distance_dvf(latlong1, latlong2):
    """
    Approximatly middle lat and Lon - Place name - "Grand-Corent"
   
    5.419434 46.195202
  
    """
    lat1, long1 = latlong1
    lat2, long2 = latlong2

    degrees_to_radians = math.pi/180.0
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    cos = (math.sin(phi1)*math.sin(phi2)*math.cos(theta1 - theta2) +
           math.cos(phi1)*math.cos(phi2))

    arc = math.acos(cos)
    kilomerters = arc*6373

    return kilomerters


def Grand_Corent(df):
    Grand_Corent = (5.419434, 46.195202)


    df['distance'] = df.apply(lambda row: distance_dvf(
    Grand_Corent, (row['latitude'], row['longitude'])), axis=1)
    nature = df.valeur_fonciere
    nature.index = df.distance
    return nature
def corrmat(df):
    return df.corr()


option = st.sidebar.selectbox(
    "choix de l'année",
    ("2020", "2019","2018","2017"))
if option == "2020":
    st.header("Load_data")
    df_2020 = load_data(data_url_2020)
    st.write(df_2020.head(10))
    st.header("Nettoyage données")
    st.markdown("Calcul du pourcentage (%) de valeurs NAN existant dans le dataset.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.heatmap(df_2020.isnull(), yticklabels=False, cbar=False)
    st.pyplot()
    drop_data(df_2020)

    df_2020 = drop_dataNA(df_2020)
    df_2020 = drop_dupicated(df_2020)
    df_2020['lot1_numero'] = df_2020['lot1_numero'].astype('object')
    df_2020['code_postal'] = df_2020['code_postal'].astype('object')
    df_2020['code_type_local'] = df_2020['code_type_local'].astype('object')
    df_2020['type_local'] = df_2020['type_local'].astype('object')
    df_2020['nombre_pieces_principales'] = df_2020['nombre_pieces_principales'].astype(
    'object')
    df_2020['code_departement'] = df_2020['code_departement'].astype('object')
    df_2020['code_commune'] = df_2020['code_commune'].astype('object')
    df_2020['nombre_lots'] = df_2020['nombre_lots'].astype('float64')
    df_2020['latitude'] = pd.to_numeric(df_2020['latitude'])
    df_2020['longitude'] = pd.to_numeric(df_2020['longitude'])


    st.write("Si la colonne a plus de 80 % de valeurs NAN on supprime la colonne ")
    st.write("On supprime les doublons, et les lignes avec des valeurs null")
    st.write(df_2020.astype(object))
    st.write("On s'assure qu'on a plus de valeurs null")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.heatmap(df_2020.isnull(), yticklabels=False, cbar=False)
    st.pyplot()
    st.header("Visualisation")
    st.header("valeur_fonciere & code_departement")
    transform_Codep_ValF(df_2020).plot.bar(x='code_departement', y='sum', color=['green', 'yellow'])
    st.pyplot()

    st.header("code_departement & surface_terrain ")
    transform_Codp_Surface(df_2020).plot.bar(x='code_departement', y='sum', color=['orange', 'green'])
    st.pyplot()

    st.header("valeur_fonciere & nature_mutation")
    transform_mut_valF(df_2020).plot(kind='pie', y='sum',
                        autopct='%.2f', labels=df_2020['nature_mutation'], labelsize = 4)
    st.pyplot()

    st.header("surface_terrain' & nature_mutation")
    transform_mut_surf(df_2020).plot(kind='pie', y='sum',
                                     autopct='%.2f', labels=df_2020['nature_mutation'], labelsize=4)
    st.pyplot()

    st.header("nature_culture & valeur_fonciere")
    transform_nat_valF(df_2020).plot.bar(x="nature_culture", y="sum")
    st.pyplot()

    st.header("Prix du terrain par rapport à la distance de Grand-Corent")
    Grand_Corent(df_2020).plot(style='.')
    plt.ylim(0, 1e7)
    plt.xlabel('Distance from Grand-Corent (Kilometeres)')
    plt.ylabel('Land Price')
    plt.title('Land Price vs. Distance from Grand-Corent')
    st.pyplot()

    st.header("Matrice de corrélation")
    sns.heatmap(corrmat(df_2020), vmax=.9, square=True) 
    st.pyplot()
    st.header("MAP")
    if st.checkbox('show the map'):
        st.map(df_2020)
elif option=="2019":
    st.header("Load_data")
    df_2019 = load_data(data_url_2019)
    st.write(df_2019.head(10))
    st.header("Nettoyage données")
    st.markdown("Calcul du pourcentage (%) de valeurs NAN existant dans le dataset.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.heatmap(df_2019.isnull(), yticklabels=False, cbar=False)
    st.pyplot()
    drop_data(df_2019)

    df_2019 = drop_dataNA(df_2019)
    df_2019 = drop_dupicated(df_2019)
    df_2019['code_postal'] = df_2019['code_postal'].astype('object')
    df_2019['code_type_local'] = df_2019['code_type_local'].astype('object')
    df_2019['type_local'] = df_2019['type_local'].astype('object')
    df_2019['nombre_pieces_principales'] = df_2019['nombre_pieces_principales'].astype(
    'object')
    df_2019['code_departement'] = df_2019['code_departement'].astype('object')
    df_2019['code_commune'] = df_2019['code_commune'].astype('object')
    df_2019['nombre_lots'] = df_2019['nombre_lots'].astype('float64')
    df_2019['latitude'] = pd.to_numeric(df_2019['latitude'])
    df_2019['longitude'] = pd.to_numeric(df_2019['longitude'])


    st.write("Si la colonne a plus de 80 % de valeurs NAN on supprime la colonne ")
    st.write("On supprime les doublons, et les lignes avec des valeurs null")
    st.write(df_2019.astype(object))
    st.write("On s'assure qu'on a plus de valeurs null")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.heatmap(df_2019.isnull(), yticklabels=False, cbar=False)
    st.pyplot()
    st.header("Visualisation")
    st.header("valeur_fonciere & code_departement")
    transform_Codep_ValF(df_2019).plot.bar(x='code_departement', y='sum', color=['green', 'yellow'])
    st.pyplot()

    st.header("code_departement & surface_terrain ")
    transform_Codp_Surface(df_2019).plot.bar(x='code_departement', y='sum', color=['orange', 'green'])
    st.pyplot()

    st.header("valeur_fonciere & nature_mutation")
    transform_mut_valF(df_2019).plot(kind='pie', y='sum',
                        autopct='%.2f', labels=df_2019['nature_mutation'],labelsize = 4)
    st.pyplot()

    st.header("surface_terrain' & nature_mutation")
    transform_mut_surf(df_2019).plot(kind='pie', y='sum',
                        autopct='%.2f', labels=df_2019['nature_mutation'],labelsize = 4)
    st.pyplot()

    st.header("nature_culture & valeur_fonciere")
    transform_nat_valF(df_2019).plot.bar(x="nature_culture", y="sum")
    st.pyplot()

    st.header("Prix du terrain par rapport à la distance de Grand-Corent")
    Grand_Corent(df_2019).plot(style='.')
    plt.ylim(0, 1e7)
    plt.xlabel('Distance from Grand-Corent (Kilometeres)')
    plt.ylabel('Land Price')
    plt.title('Land Price vs. Distance from Grand-Corent')
    st.pyplot()

    st.header("Matrice de corrélation")
    sns.heatmap(corrmat(df_2019), vmax=.9, square=True) 
    st.pyplot()
    st.header("MAP")
    if st.checkbox('show the map'):
        st.map(df_2019)
elif option=="2018":
    st.header("Load_data")
    df_2018 = load_data(data_url_2018)
    st.write(df_2018.head(10))
    st.header("Nettoyage données")
    st.markdown("Calcul du pourcentage (%) de valeurs NAN existant dans le dataset.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.heatmap(df_2018.isnull(), yticklabels=False, cbar=False)
    st.pyplot()
    drop_data(df_2018)

    df_2018 = drop_dataNA(df_2018)
    df_2018 = drop_dupicated(df_2018)
    df_2018['code_postal'] = df_2018['code_postal'].astype('object')
    df_2018['code_type_local'] = df_2018['code_type_local'].astype('object')
    df_2018['type_local'] = df_2018['type_local'].astype('object')
    df_2018['nombre_pieces_principales'] = df_2018['nombre_pieces_principales'].astype(
    'object')
    df_2018['code_departement'] = df_2018['code_departement'].astype('object')
    df_2018['code_commune'] = df_2018['code_commune'].astype('object')
    df_2018['nombre_lots'] = df_2018['nombre_lots'].astype('float64')
    df_2018['latitude'] = pd.to_numeric(df_2018['latitude'])
    df_2018['longitude'] = pd.to_numeric(df_2018['longitude'])


    st.write("Si la colonne a plus de 80 % de valeurs NAN on supprime la colonne ")
    st.write("On supprime les doublons, et les lignes avec des valeurs null")
    st.write(df_2018.astype(object))
    st.write("On s'assure qu'on a plus de valeurs null")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.heatmap(df_2018.isnull(), yticklabels=False, cbar=False)
    st.pyplot()
    st.header("Visualisation")
    st.header("valeur_fonciere & code_departement")
    transform_Codep_ValF(df_2018).plot.bar(x='code_departement', y='sum', color=['green', 'yellow'])
    st.pyplot()

    st.header("code_departement & surface_terrain ")
    transform_Codp_Surface(df_2018).plot.bar(x='code_departement', y='sum', color=['orange', 'green'])
    st.pyplot()

    st.header("valeur_fonciere & nature_mutation")
    transform_mut_valF(df_2018).plot(kind='pie', y='sum',
                        autopct='%.2f', labels=df_2018['nature_mutation'], labelsize = 4)
    st.pyplot()

    st.header("surface_terrain' & nature_mutation")
    transform_mut_surf(df_2018).plot(kind='pie', y='sum',
                        autopct='%.2f', labels=df_2018['nature_mutation'], labelsize = 4)
    st.pyplot()

    st.header("nature_culture & valeur_fonciere")
    transform_nat_valF(df_2018).plot.bar(x="nature_culture", y="sum")
    st.pyplot()

    st.header("Prix du terrain par rapport à la distance de Grand-Corent")
    Grand_Corent(df_2018).plot(style='.')
    plt.ylim(0, 1e7)
    plt.xlabel('Distance from Grand-Corent (Kilometeres)')
    plt.ylabel('Land Price')
    plt.title('Land Price vs. Distance from Grand-Corent')
    st.pyplot()

    st.header("Matrice de corrélation")
    sns.heatmap(corrmat(df_2018), vmax=.9, square=True) 
    st.pyplot()
    st.header("MAP")
    if st.checkbox('show the map'):
        st.map(df_2018)
elif option == "2017":
    st.header("Load_data")
    df_2017 = load_data(data_url_2017)
    st.write(df_2017.head(10))
    st.header("Nettoyage données")
    st.markdown("Calcul du pourcentage (%) de valeurs NAN existant dans le dataset.")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.heatmap(df_2017.isnull(), yticklabels=False, cbar=False)
    st.pyplot()
    drop_data(df_2017)

    df_2017 = drop_dataNA(df_2017)
    df_2017 = drop_dupicated(df_2017)
    df_2017['code_postal'] = df_2017['code_postal'].astype('object')
    df_2017['code_type_local'] = df_2017['code_type_local'].astype('object')
    df_2017['type_local'] = df_2017['type_local'].astype('object')
    df_2017['nombre_pieces_principales'] = df_2017['nombre_pieces_principales'].astype(
    'object')
    df_2017['code_departement'] = df_2017['code_departement'].astype('object')
    df_2017['code_commune'] = df_2017['code_commune'].astype('object')
    df_2017['nombre_lots'] = df_2017['nombre_lots'].astype('float64')
    df_2017['latitude'] = pd.to_numeric(df_2017['latitude'])
    df_2017['longitude'] = pd.to_numeric(df_2017['longitude'])


    st.write("Si la colonne a plus de 80 % de valeurs NAN on supprime la colonne ")
    st.write("On supprime les doublons, et les lignes avec des valeurs null")
    st.write(df_2017.astype(object))
    st.write("On s'assure qu'on a plus de valeurs null")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    sns.heatmap(df_2017.isnull(), yticklabels=False, cbar=False)
    st.pyplot()
    st.header("Visualisation")
    st.header("valeur_fonciere & code_departement")
    transform_Codep_ValF(df_2017).plot.bar(x='code_departement', y='sum', color=['green', 'yellow'])
    st.pyplot()

    st.header("code_departement & surface_terrain ")
    transform_Codp_Surface(df_2017).plot.bar(x='code_departement', y='sum', color=['orange', 'green'])
    st.pyplot()

    st.header("valeur_fonciere & nature_mutation")
    transform_mut_valF(df_2017).plot(kind='pie', y='sum',
                        autopct='%.2f', labels=df_2017['nature_mutation'], labelsize = 4)
    st.pyplot()

    st.header("surface_terrain' & nature_mutation")
    transform_mut_surf(df_2017).plot(kind='pie', y='sum',
                                     autopct='%.2f', labels=df_2017['nature_mutation'], labelsize=4)
    st.pyplot()

    st.header("nature_culture & valeur_fonciere")
    transform_nat_valF(df_2017).plot.bar(x="nature_culture", y="sum")
    st.pyplot()

    st.header("Prix du terrain par rapport à la distance de Grand-Corent")
    Grand_Corent(df_2017).plot(style='.')
    plt.ylim(0, 1e7)
    plt.xlabel('Distance from Grand-Corent (Kilometeres)')
    plt.ylabel('Land Price')
    plt.title('Land Price vs. Distance from Grand-Corent')
    st.pyplot()

    st.header("Matrice de corrélation")
    sns.heatmap(corrmat(df_2017), vmax=.9, square=True) 
    st.pyplot()
    st.header("MAP")
    if st.checkbox('show the map'):
        st.map(df_2017)
    



