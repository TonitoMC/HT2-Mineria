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

# Cargar los datos al df
data = pd.read_csv('data/iris.csv')
 
# Visualizacion inicial de los datos
plt.scatter(data['petal_length'], data['petal_width'])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.savefig('out/petal_initial_visualization.png')

# Filtrado de datos para utilizar unicamente  informacion del petalo
petal_data = data.filter(['petal_length', 'petal_width'])

# Aplicacion KMeans con 2 clusters
kmeans = KMeans(n_clusters = 2, random_state = 42)
kmeans.fit(petal_data)
identified_clusters = kmeans.fit_predict(petal_data)

# Adicion de columna de clusters al DataFrame
clustered_petal_data = petal_data.copy()
clustered_petal_data['cluster'] = identified_clusters
clustered_petal_data['cluster'] = clustered_petal_data['cluster'].astype("category")

# Grafica de datos interactiva con clusters identificados
fig = px.scatter(clustered_petal_data,
                  x = "petal_length",
                  y = "petal_width",
                  color = "cluster",
)

fig.update_layout(
    title = "Petal Length vs Petal Width, 2 clusters identified via KMeans. Unstandardized data",
    xaxis_title = "Petal Length",
    yaxis_title = "Petal Width",
)

fig.show()

# Estandarizacion de los datos
scaler = StandardScaler()
scaled_petal_data = scaler.fit_transform(petal_data)

# Aplicacion de KMeans con 2 clusters, utilizando la misma instancia del inciso anterior
identified_clusters = kmeans.fit_predict(scaled_petal_data)
clustered_scaled_petal_data = scaled_petal_data.copy()
clustered_scaled_petal_data = pd.DataFrame(scaled_petal_data, columns=['petal_length', 'petal_width'])

# Adicion de columna de clusters al DataFrame
clustered_scaled_petal_data['cluster'] = identified_clusters
clustered_scaled_petal_data['cluster'] = clustered_scaled_petal_data['cluster'].astype("category")

# Grafica de datos interactiva con clusters identificados
fig = px.scatter(clustered_scaled_petal_data,
                  x = "petal_length",
                  y = "petal_width",
                  color = "cluster",
)

fig.update_layout(
    title = " Petal Length vs Petal Width, 2 clusters identified via KMeans. Standardized data",
    xaxis_title = "Petal Length",
    yaxis_title = "Petal Width",
)

fig.show()

# Aplicacion metodo del codo
cluster_numbers = range(1, 11)
wcss = []

for i in cluster_numbers:
    # Crear KMeans con i clusters, aplicar a los datos y calcular el wcss
    kmeans = KMeans(n_clusters = i, random_state = 42, n_init = 10)
    kmeans.fit(scaled_petal_data)
    wcss.append(kmeans.inertia_)

# Grafica de codo
plt.figure(figsize=(8, 5))
plt.plot(cluster_numbers, wcss, marker='o')
plt.xlabel("Número de clusters")
plt.ylabel("WCSS")
plt.title("Gráfico de Codo")
plt.grid(True)
plt.savefig("out/petal_elbow_plot.png")

# Graficas de 3, 4 y 5 clusters identificados como posibles puntos de inflexion en la grafica de codo
for i in range(3, 6):
    kmeans = KMeans(n_clusters = i, random_state = 42)
    identified_clusters = kmeans.fit_predict(scaled_petal_data)
    clustered_scaled_petal_data = scaled_petal_data.copy()
    clustered_scaled_petal_data = pd.DataFrame(scaled_petal_data, columns = ['petal_length', 'petal_width'])
    clustered_scaled_petal_data['cluster'] = identified_clusters
    clustered_scaled_petal_data['cluster'] = clustered_scaled_petal_data['cluster'].astype("category")

    fig = px.scatter(clustered_scaled_petal_data,
                     x = "petal_length",
                     y = "petal_width",
                     color = "cluster")
    
    fig.update_layout(
        title = f"Petal Length vs Petal Width, {i} clusters identified via KMeans. Standardized data",
        xaxis_title = "Petal Length",
        yaxis_title = "Petal Width"
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

plt.savefig("out/petal_kneed_elbow.png")

petal_data_with_species = pd.read_csv('data/iris-con-respuestas.csv')

# Gráfica de datos interactiva con especies identificadas
fig = px.scatter(petal_data_with_species,
                  x = "petal_length",
                  y = "petal_width",
                  color = "species",
)

fig.update_layout(
    title = "Petal Length vs Petal Width, Species Identified",
    xaxis_title = "Petal Length",   
    yaxis_title = "Petal Width",
)

fig.show()

# Clustering con 3 clusters, comparacion a datos reales via adjusted_rand_score
true_labels = petal_data_with_species['species']

kmeans = KMeans(n_clusters=3, random_state = 42)
predicted_clusters = kmeans.fit_predict(scaled_petal_data)

true_labels_numeric = pd.factorize(true_labels)[0]

ari_score = adjusted_rand_score(true_labels_numeric, predicted_clusters)
print(ari_score)