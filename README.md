# Team Work Report

## 1. General Information
**Date:** 04.02.2025
**Team:** Team 5 
**Project:** Exam 
**Team Leader:** Bartosz Budek  

---

## 2. Work Summary
A brief description of overall project progress, key achievements, and any encountered issues.
Exam rules
You are working with teams. Team Leader: Created the repository. There is ONE document for the team report. Please organize the structure, and finally, each student should update the report. After the exam, you will exchange your reports and evaluate the reports of other teams (with individual scores for each student – the author must be listed in the report). Everyone in the team should provide an individual score, and I will calculate the average. You can vote anonymously within the team 🙂. Please organize this process.

You can use any help you want, but using your colleague's work is forbidden.

The data is in Teams in the corresponding folder.

NOTICE: The tasks are ordered by difficulty: Task 1 is the easiest, and Task 3 is the most difficult.

Task 1 Analysis of the correlation between temperature and NDVI
read temp 1 ndvi1
display 2 images with defiened palete
create 2 histograms
create scatter plot x=ndvi y=temp
repeat point 1-4 for temp2 and ndvi2

Perform analysis in Python and Matlab/Octave (both)

If you want to see the GEE code: https://code.earthengine.google.com/150e730dc6ea42320cc4091a9d6c1de0

Task 2 DEM analysis
read DEM
read point cloud
calculate differences: deltaH=point cloud H - DEM
calculate accuracy metrics
If you are able do all in Python, if not perform pre-processing in QGIS/ArcGIS/SAGA.. only accuracy anaysis make in Python OR Matlab/Octave

calculate difference between DEM 2024 - DEM 2021
check Groud Motion Service
Task 3 Classification accuracy
perform Sentinel-2 classification - K-Means any tool
download LULC map (i.e. Urban Atlas or Corine)
perform accuracy analysis using https://github.com/RemoteSys/accuracy
---

## 3. Team Members' Results

### 3.1 Member 1 - Bartosz Budek
- **Completed Tasks:**
  - [ ] Task 1
  - [ ] Task 2
  - [ ] Task 3


- **Code Implementation:**
  The following Python script calculates accuracy metrics.

  ```python
  !git clone https://github.com/RemoteSys/accuracy.git
  %cd accuracy
  !pip install -r requirements.txt
  !pip install .
  !accuracy /content/LULC_200_2021.tif
  !accuracy /content/data2cols.csv
  !accuracy /content/data2cols.csv -f "ac = (TP * TN - FP * FN) / ((TP + FP) * (TP + FN))**0.5"
  import pandas as pd
```python

import numpy as np

df = pd.read_csv("/content/data2cols.csv")

TP = df['TP'].sum()
TN = df['TN'].sum()
FP = df['FP'].sum()
FN = df['FN'].sum()

OA = (TP + TN) / (TP + TN + FP + FN)
PA = TP / (TP + FN)
UA = TP / (TP + FP)
OME = FN / (TP + FN)
CME = FP / (TP + FP)
NPV = TN / (TN + FN)
ACC = (TP + TN) / (TP + TN + FP + FN)
PPV = TP / (TP + FP)
TPR = TP / (TP + FN)
TNR = TN / (TN + FP)
FNR = FN / (FN + TP)
FPR = FP / (FP + TN)
FDR = FP / (FP + TP)
FOR = FN / (FN + TN)
TS = TP / (TP + FN + FP)
MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
BA = (TPR + TNR) / 2
F1 = 2 * (PPV * TPR) / (PPV + TPR)
FM = np.sqrt(PPV * TPR)
BM = TPR + TNR - 1
MK = PPV + NPV - 1

metrics = {
    "Overall Accuracy (OA)": OA,
    "Producer Accuracy (PA)": PA,
    "User Accuracy (UA)": UA,
    "Omission Error (OME)": OME,
    "Commission Error (CME)": CME,
    "Negative Predictive Value (NPV)": NPV,
    "Accuracy (ACC)": ACC,
    "Precision (PPV)": PPV,
    "Recall (TPR)": TPR,
    "Specificity (TNR)": TNR,
    "False Negative Rate (FNR)": FNR,
    "False Positive Rate (FPR)": FPR,
    "False Discovery Rate (FDR)": FDR,
    "False Omission Rate (FOR)": FOR,
    "Threat Score (TS)": TS,
    "Matthews Correlation Coefficient (MCC)": MCC,
    "Balanced Accuracy (BA)": BA,
    "F1 Score": F1,
    "Fowlkes-Mallows Index (FM)": FM,
    "Informedness (BM)": BM,
    "Markedness (MK)": MK,
}

for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
    
  import pandas as pd

  df = pd.read_csv("data2cols.csv")

  df["OA"] = df["TP"] / (df["TP"] + df["TN"] + df["FP"] + df["FN"])
  df["PA"] = df["TP"] / (df["TP"] + df["FN"])
  df["UA"] = df["TP"] / (df["TP"] + df["FP"])

  df.to_csv("classic_acc.csv", index=False)
  print("Results saved to classic_acc.csv")

import pandas as pd

file_paths = [
    "/content/data2cols_results/cross_full.csv",
    "/content/data2cols_results/binary_cross.csv",
    "/content/data2cols_results/classic_acc.csv",
    "/content/data2cols_results/simple_acc.csv",
    "/content/data2cols_results/complex_acc.csv",
    "/content/data2cols_results/average_acc.csv"
]

results = {}
for path in file_paths:
    try:
        df = pd.read_csv(path)
        results[path] = df.head()  # Wyświetlenie pierwszych kilku wierszy
    except Exception as e:
        results[path] = f"Error loading file: {e}"

results
```
![Screenshot1](screen1.png)
![Screenshot2](screen2.png)

---

### 3.2 Member 2 - Aleksandra Barnach 
- **Completed Tasks:**
  - [ ] Task 1
  - [ ] Task 2
  - [X] Task 3

 - **K-means Code Implementation:**
  The following Python script calculates K-means classification.

  ```python
!pip install rasterio
import rasterio
print("Rasterio imported successfully!")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Ścieżka do pliku
raster_path = "/content/raster.tif"

# Odczytanie pliku GeoTIFF
with rasterio.open(raster_path) as src:
    image = src.read()  # Załaduj wszystkie pasma (bands)
    profile = src.profile  # Zapisz metadane

# Przekształcenie danych na format 2D (piksele x kanały)
num_bands, height, width = image.shape
image_2d = image.reshape(num_bands, -1).T  # Transponowanie dla K-Means

# Usunięcie NaN i wartości zerowych
valid_pixels = np.all(image_2d > 0, axis=1)
filtered_data = image_2d[valid_pixels]

# Klasyfikacja K-Means (np. 5 klas)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(filtered_data)

# Przypisanie etykiet do pikseli
labels = np.full(image_2d.shape[0], -1)  # Domyślnie -1 dla odfiltrowanych pikseli
labels[valid_pixels] = kmeans.labels_

# Rekonstrukcja do oryginalnych wymiarów
classified_image = labels.reshape(height, width)

# Wizualizacja
plt.figure(figsize=(10, 10))
plt.imshow(classified_image, cmap='tab10')
plt.colorbar(label='Cluster Label')
plt.title("Sentinel-2 K-Means Classification")
plt.show()

# Zapis wyników
output_path = "/content/classified.tif"
with rasterio.open(
    output_path, 'w', driver='GTiff',
    height=height, width=width, count=1,
    dtype=rasterio.uint8, crs=profile['crs'],
    transform=profile['transform']
) as dst:
    dst.write(classified_image.astype(rasterio.uint8), 1)

print(f"Classified image saved to {output_path}")

```

![Screenshot3](1.png)

 - **Accuracy Analysis Code Implementation:**
  The following Python script calculates K-means classification.

  ```python
!pip install rasterio
import rasterio
print("Rasterio imported successfully!")
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

# Wczytaj raster
raster_path = "/content/raster.tif"
with rasterio.open(raster_path) as src:
    raster_data = src.read(1)  # Zakładamy, że raster ma tylko jedną warstwę
    profile = src.profile

# Sprawdź wartości unikalne (przydatne dla klasyfikacji)
unique_values = np.unique(raster_data)
print("Unikalne wartości w rastrze:", unique_values)

ground_truth_data = np.random.choice(unique_values, raster_data.shape)  # Symulacja referencji


# Spłaszczanie danych (potrzebne do analizy klasyfikacji)
flattened_raster = raster_data.flatten()
flattened_truth = ground_truth_data.flatten()

# Usunięcie wartości NoData (jeśli istnieją, np. -9999)
mask = (flattened_truth >= 0)  # Dostosuj w zależności od NoData w Twoim rastrze
flattened_raster = flattened_raster[mask]
flattened_truth = flattened_truth[mask]

# Obliczenie macierzy błędów
conf_matrix = confusion_matrix(flattened_truth, flattened_raster)
print("Macierz błędów:\n", conf_matrix)

# Obliczenie dokładności
overall_acc = accuracy_score(flattened_truth, flattened_raster)
print("Ogólna dokładność:", overall_acc)

# Szczegółowy raport klasyfikacji
report = classification_report(flattened_truth, flattened_raster, zero_division=0)
print("Raport klasyfikacji:\n", report)

# Wizualizacja macierzy błędów
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap="Blues", interpolation="nearest")
plt.colorbar(label="Liczba pikseli")
plt.xlabel("Przewidywane klasy")
plt.ylabel("Referencyjne klasy")
plt.title("Macierz błędów")
plt.show()

```
Macierz błędów:
 [[4079 2320 3226 6818 1297]
 [4227 2245 3227 6867 1255]
 [4160 2288 3262 6747 1282]
 [4308 2258 3293 6850 1292]
 [4247 2334 3215 6917 1306]]



 Raport klasyfikacji:
               precision    recall  f1-score   support

           1       0.19      0.23      0.21     17740
           3       0.20      0.13      0.15     17821
           4       0.20      0.18      0.19     17739
           6       0.20      0.38      0.26     18001
           8       0.20      0.07      0.11     18019

    accuracy                           0.20     89320
   macro avg       0.20      0.20      0.19     89320
weighted avg       0.20      0.20      0.19     89320


![Screenshot4](2.png)



- **Notes:**
- **Issues/Challenges:**
- **Plans for the Next Period:**

---

### 3.3 Member 3 - Wiktoria Gądek
- **Completed Tasks:**
  - [ ] Task 1
  - [ ] Task 2
- **Notes:**
- **Issues/Challenges:**
- **Plans for the Next Period:**

---

### 3.4 Member 4 - Full Name
- **Completed Tasks:**
  - [ ] Task 1
  - [ ] Task 2
- **Notes:**
- **Issues/Challenges:**
- **Plans for the Next Period:**

---

### 3.5 Member 5 - Full Name
- **Completed Tasks:**
  - [ ] Task 1
  - [ ] Task 2
- **Notes:**
- **Issues/Challenges:**
- **Plans for the Next Period:**

---

## 4. Conclusions and Recommendations
Summary of key findings and suggestions for improvements in the future.

## 5. Additional Notes
Space for any other information relevant to the team.

---

*Prepared by the Team YYYY-MM-DD*
