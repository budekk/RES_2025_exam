import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Ścieżka do pliku .tif
file_path = r'C:\Users\gabry\Documents\MAGISTERKA\Matlab_python\exam\raster.tif'

# Wczytanie obrazu z pliku .tif
with rasterio.open(file_path) as src:
    img_data = src.read()  # Wczytanie wszystkich pasm obrazu (wielokanałowego)
    transform = src.transform  # Informacje o transformacji przestrzennej
    metadata = src.meta  # Metadane obrazu

# Konwersja obrazu do formatu 2D (rozpłaszczony macierz danych)
# Każde pasmo będzie traktowane jako oddzielny kanał
num_bands, height, width = img_data.shape
image_2d = img_data.reshape((num_bands, -1)).T  # Przekształcenie do kształtu (ilość pikseli, liczba pasm)

# Normalizacja danych (ważne dla K-Means)
scaler = StandardScaler()
image_2d_scaled = scaler.fit_transform(image_2d)

# Zastosowanie K-Means
num_clusters = 5  # Liczba klastrów (możesz to dostosować)
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(image_2d_scaled)

# Przekształcenie wyników klasyfikacji z powrotem do kształtu obrazu
classified_image = labels.reshape((height, width))

# Wyświetlenie wyników
plt.figure(figsize=(10, 8))
plt.imshow(classified_image, cmap='viridis')
plt.colorbar()
plt.title(f'K-Means Classification with {num_clusters} Clusters')
plt.show()

# Można zapisać klasyfikowany obraz do nowego pliku .tif
output_path = r'C:\Users\gabry\Documents\MAGISTERKA\Matlab_python\exam\classified_raster.tif'
with rasterio.open(output_path, 'w', **metadata) as dst:
    dst.write(classified_image.astype(rasterio.uint8), 1)
