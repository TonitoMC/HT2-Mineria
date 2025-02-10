import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from sklearn.metrics import adjusted_rand_score

sns.set_theme()

# Cargar los datos al DF
data = pd.read_csv('data/iris.csv')
 
#Visualizacion inicial de los datos
plt.scatter(data['sepal_length'], data['sepal_width'])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.savefig('out/sepal_initial_visualization.png')

# Filtrado de datos para utilizar únicamente información del sépalo
sepal_data = data.filter(['sepal_length', 'sepal_width'])

# Aplicación KMeans con 2 clusters
kmeans = KMeans(n_clusters = 2, random_state = 42)
kmeans.fit(sepal_data)
identified_clusters = kmeans.fit_predict(sepal_data)

# Adición de columna de clusters al Dataframe
clustered_sepal_data = sepal_data.copy()
clustered_sepal_data['cluster'] = identified_clusters
clustered_sepal_data['cluster'] = clustered_sepal_data['cluster'].astype("category")

# Gráfica de datos interactiva con clusters identificados
fig = px.scatter(clustered_sepal_data,
                  x = "sepal_length",
                  y = "sepal_width",
                  color = "cluster",
)

fig.update_layout(
    title = "Sepal Length vs Sepal Width, 2 clusters identified via KMeans. Unstandardized data",
    xaxis_title = "Sepal Length",
    yaxis_title = "Sepal Width",
)

fig.show()

# Estandarizacion de los datos
scaler = StandardScaler()
scaled_sepal_data = scaler.fit_transform(sepal_data)

# Aplicación de KMeans con 2 clusters, utilizando la misma instancia del inciso anterior
identified_clusters = kmeans.fit_predict(scaled_sepal_data)
clustered_scaled_sepal_data = scaled_sepal_data.copy()
clustered_scaled_sepal_data = pd.DataFrame(scaled_sepal_data, columns=['sepal_length', 'sepal_width'])

# Adición de columna de clusters al DataFrame
clustered_scaled_sepal_data['cluster'] = identified_clusters
clustered_scaled_sepal_data['cluster'] = clustered_scaled_sepal_data['cluster'].astype("category")

# Gráfica de datos interactiva con clusters identificados
fig = px.scatter(clustered_scaled_sepal_data,
                  x = "sepal_length",
                  y = "sepal_width",
                  color = "cluster",
)

fig.update_layout(
    title = "Sepal Length vs Sepal Width, 2 clusters identified via KMeans. Standardized data",
    xaxis_title = "Sepal Length",   
    yaxis_title = "Sepal Width",
)

fig.show()

# Aplicación método del codo
cluster_numbers = range(1, 11)
wcss = []

for i in cluster_numbers:
    # Crear KMeans con i clusters, aplicar a los datos y calcular el wcss
    kmeans = KMeans(n_clusters = i, random_state = 42)
    kmeans.fit(scaled_sepal_data)
    wcss.append(kmeans.inertia_)

# Gráfica de codo
plt.figure(figsize=(8, 5))
plt.plot(cluster_numbers, wcss, marker='o')
plt.xlabel("Número de clusters")
plt.ylabel("WCSS")
plt.title("Gráfico de Codo")
plt.grid(True)
plt.savefig("out/sepal_elbow_plot.png")

# Gráfica de 3, 4 y 6 clusters identificados como posibles puntos de inflexión en la gráfica de codo
cluster_nums = [3, 4, 6]

for i in cluster_nums:
    kmeans = KMeans(n_clusters = i, random_state = 42)
    identified_clusters = kmeans.fit_predict(scaled_sepal_data)
    clustered_scaled_sepal_data = scaled_sepal_data.copy()
    clustered_scaled_sepal_data = pd.DataFrame(scaled_sepal_data, columns = ['sepal_length', 'sepal_width'])
    clustered_scaled_sepal_data['cluster'] = identified_clusters
    clustered_scaled_sepal_data['cluster'] = clustered_scaled_sepal_data['cluster'].astype("category")

    fig = px.scatter(clustered_scaled_sepal_data,
                     x = "sepal_length",
                     y = "sepal_width",
                     color = "cluster")
    
    fig.update_layout(
        title = f"Sepal Length vs Sepal Width, {i} clusters identified via KMeans. Standardized data",
        xaxis_title = "Sepal Length",
        yaxis_title = "Sepal Width"
    )

    fig.show()

# Kneed para encontrar numero optimo de clusters
knee_locator = KneeLocator(cluster_numbers, wcss, curve="convex", direction="decreasing")

optimal_clusters = knee_locator.knee

plt.figure(figsize=(8, 5))
plt.plot(cluster_numbers, wcss, marker='o', label="WCSS")
plt.axvline(x=optimal_clusters, color='r', linestyle='--', label=f"Optimal Clusters: {optimal_clusters}")
plt.xlabel("Número de clusters")
plt.ylabel("WCSS")
plt.title("Gráfico de Codo con KneeLocator")
plt.legend()
plt.grid(True)

plt.savefig("out/sepal_kneed_elbow.png")

sepal_data_with_species = pd.read_csv('data/iris-con-respuestas.csv')

# Gráfica de datos interactiva con especies identificadas
fig = px.scatter(sepal_data_with_species,
                  x = "sepal_length",
                  y = "sepal_width",
                  color = "species",
)

fig.update_layout(
    title = "Sepal Length vs Sepal Width, Species Identified",
    xaxis_title = "Sepal Length",   
    yaxis_title = "Sepal Width",
)

fig.show()

# Clustering con 3 clusters, comparacion a datos reales via adjusted_rand_score
true_labels = sepal_data_with_species['species']

kmeans = KMeans(n_clusters=3, random_state = 42)
predicted_clusters = kmeans.fit_predict(scaled_sepal_data)

true_labels_numeric = pd.factorize(true_labels)[0]

ari_score = adjusted_rand_score(true_labels_numeric, predicted_clusters)
print(ari_score)