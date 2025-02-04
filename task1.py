import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.plot import show
from sklearn.preprocessing import MinMaxScaler

# Funkcja do odczytania pliku .tif i przekształcenia go do tablicy numpy
def read_raster(file_path):
    with rasterio.open(file_path) as src:
        data = src.read(1)  # Odczytujemy pierwszą warstwę (2D)
    return data

# Ścieżki do plików .tif (wskazujemy pełną ścieżkę)
temp_file_1 = r'C:\Users\gabry\Documents\MAGISTERKA\Matlab_python\exam\team5\t5_lst2023_Jun_Aug.tif'
temp_file_2 = r'C:\Users\gabry\Documents\MAGISTERKA\Matlab_python\exam\team5\t5_lst2024May.tif'
ndvi_file_1 = r'C:\Users\gabry\Documents\MAGISTERKA\Matlab_python\exam\team5\t5_ndvi2024_Jul_Aug.tif'
ndvi_file_2 = r'C:\Users\gabry\Documents\MAGISTERKA\Matlab_python\exam\team5\t5_ndvi2024May.tif'

# Odczytanie danych
temp1 = read_raster(temp_file_1)
temp2 = read_raster(temp_file_2)
ndvi1 = read_raster(ndvi_file_1)
ndvi2 = read_raster(ndvi_file_2)

# Normalizacja danych, aby były w tym samym zakresie
scaler = MinMaxScaler()
temp1 = scaler.fit_transform(temp1.reshape(-1, 1)).reshape(temp1.shape)
temp2 = scaler.fit_transform(temp2.reshape(-1, 1)).reshape(temp2.shape)
ndvi1 = scaler.fit_transform(ndvi1.reshape(-1, 1)).reshape(ndvi1.shape)
ndvi2 = scaler.fit_transform(ndvi2.reshape(-1, 1)).reshape(ndvi2.shape)

# Wyświetlanie trzech obrazów (jednego dla temperatury i dwóch dla NDVI)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Obraz 1 - Temperatura (LST)
show(temp1, ax=axes[0], title="LST 2023 Jun-Aug", cmap='coolwarm')

# Obraz 2 - NDVI 1
show(ndvi1, ax=axes[1], title="NDVI 2024 Jul-Aug", cmap='YlGn')

# Obraz 3 - NDVI 2
show(ndvi2, ax=axes[2], title="NDVI 2024 May", cmap='YlGn')

plt.tight_layout()
plt.show()

# Tworzenie histogramów
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Histogram dla temperatury (LST)
axes[0].hist(temp1.flatten(), bins=50, color='skyblue', edgecolor='black')
axes[0].set_title("Histogram LST 2023 Jun-Aug")
axes[0].set_xlabel("Temperature (scaled)")
axes[0].set_ylabel("Frequency")

# Histogram dla NDVI 1
axes[1].hist(ndvi1.flatten(), bins=50, color='green', edgecolor='black')
axes[1].set_title("Histogram NDVI 2024 Jul-Aug")
axes[1].set_xlabel("NDVI (scaled)")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# Tworzenie wykresu rozrzutu (scatter plot) NDVI vs. Temperatura
plt.figure(figsize=(8, 6))
plt.scatter(ndvi1.flatten(), temp1.flatten(), color='purple', alpha=0.5)
plt.title("Scatter Plot: NDVI vs Temperature")
plt.xlabel("NDVI")
plt.ylabel("Temperature")
plt.show()
