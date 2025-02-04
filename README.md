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


- **Code Implementation for Task3:**
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
```python

import rasterio
import numpy as np
import geopandas as gpd

# Ścieżki do plików
dem_file = path
point_cloud_file = path

def load_dem(file_path):
    """Wczytuje dane DEM oraz macierz transformacji."""
    with rasterio.open(file_path) as src:
        elevation_data = src.read(1)  # Pobranie pierwszego pasma
        affine_transform = src.transform
    return elevation_data, affine_transform

def load_point_cloud(file_path):
    """Wczytuje chmurę punktów z pliku wektorowego (SHP)."""
    return gpd.read_file(file_path)

def extract_dem_elevation(elevation_data, transform, x_coord, y_coord):
    """Interpoluje wysokość DEM dla podanych współrzędnych."""
    col, row = ~transform * (x_coord, y_coord)  # Transformacja współrzędnych
    row, col = round(row), round(col)
    
    if 0 <= row < elevation_data.shape[0] and 0 <= col < elevation_data.shape[1]:
        return elevation_data[row, col]
    return np.nan  # Wartość poza zakresem

def compute_height_differences(dem_data, transform, points):
    """Oblicza różnice wysokości między chmurą punktów a DEM."""
    points['DEM_Height'] = points.apply(lambda p: extract_dem_elevation(dem_data, transform, p.geometry.x, p.geometry.y), axis=1)
    points['Height_Diff'] = points['Z'] - points['DEM_Height']
    return points

def evaluate_accuracy(errors):
    """Oblicza metryki dokładnościowe na podstawie różnic wysokości."""
    height_diff = errors['Height_Diff'].dropna()
    mean_err = np.mean(height_diff)
    rmse_val = np.sqrt(np.mean(height_diff ** 2))
    std_dev_val = np.std(height_diff)
    
    return {
        'Mean Error': mean_err,
        'RMSE': rmse_val,
        'Standard Deviation': std_dev_val
    }

def display_metrics(results):
    """Wyświetla wyniki dokładności."""
    print("\n--- Accuracy Metrics ---")
    print(f"Mean Error: {results['Mean Error']:.2f} m")
    print(f"RMSE: {results['RMSE']:.2f} m")
    print(f"Standard Deviation: {results['Standard Deviation']:.2f} m")

def main(dem_file, point_cloud_file):
    """Główna funkcja przetwarzania danych."""
    dem_data, affine = load_dem(dem_file)
    point_cloud = load_point_cloud(point_cloud_file)
    processed_points = compute_height_differences(dem_data, affine, point_cloud)
    accuracy_results = evaluate_accuracy(processed_points)
    display_metrics(accuracy_results)
    return processed_points, accuracy_results

if __name__ == "__main__":
    main(dem_file, point_cloud_file)

```

### 3.2 Member 2 - Aleksandra Barnach 
- **Completed Tasks:**
  - [ ] Task 3.1
  - [ ] Task 3.3
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

### 3.4 Member 4 - Gabriela Bebak
- **Completed Tasks:**
  - [ ] Task 1
  - [ ] Task 2
  

  ```python
  import rasterio
  import numpy as np
  import geopandas as gpd

  dem_path = "C:/Users/gabry/Documents/MAGISTERKA/Matlab_python/exam/Lubin_2024_03_27.asc"
  point_cloud_path = "C:/Users/gabry/Documents/MAGISTERKA/Matlab_python/exam/Lubin_2024_03_27_pc_t1.shp"

  def read_dem(dem_path):
    with rasterio.open(dem_path) as dem:
        dem_data = dem.read(1)  # Odczyt pierwszego kanału
        transform = dem.transform
    return dem_data, transform

  def read_point_cloud(point_cloud_path):
    gdf = gpd.read_file(point_cloud_path)  # Wczytanie pliku SHP
    return gdf

  def interpolate_dem_height(dem_data, transform, x, y):
    col, row = ~transform * (x, y)  # Konwersja współrzędnych
    row, col = int(round(row)), int(round(col))
    if 0 <= row < dem_data.shape[0] and 0 <= col < dem_data.shape[1]:
        return dem_data[row, col]
    else:
        return np.nan  # Poza zakresem DEM

  def calculate_differences(dem_data, transform, point_cloud):
    point_cloud['DEM_H'] = point_cloud.apply(lambda row: interpolate_dem_height(dem_data, transform, row.geometry.x, row.geometry.y), axis=1)
    point_cloud['Delta_H'] = point_cloud['Z'] - point_cloud['DEM_H']
    return point_cloud

  def calculate_accuracy_metrics(differences):
    delta_h = differences['Delta_H'].dropna()
    mean_error = np.mean(delta_h)
    rmse = np.sqrt(np.mean(delta_h ** 2))
    std_dev = np.std(delta_h)
    return {
        'Mean Error': mean_error,
        'RMSE': rmse,
        'Standard Deviation': std_dev
    }

  def print_metrics(metrics):
    print("\n--- Accuracy Metrics ---")
    print(f"Mean Error: {metrics['Mean Error']:.2f} meters")
    print(f"RMSE: {metrics['RMSE']:.2f} meters")
    print(f"Standard Deviation: {metrics['Standard Deviation']:.2f} meters")

  def main(dem_path, point_cloud_path):
    dem_data, transform = read_dem(dem_path)
    point_cloud = read_point_cloud(point_cloud_path)
    differences = calculate_differences(dem_data, transform, point_cloud)
    metrics = calculate_accuracy_metrics(differences)
    print_metrics(metrics)
    return differences, metrics

  if __name__ == "__main__":
    dem_path = "C:/Users/gabry/Documents/MAGISTERKA/Matlab_python/exam/Lubin_2024_03_27.asc"
    point_cloud_path = "C:/Users/gabry/Documents/MAGISTERKA/Matlab_python/exam/Lubin_2024_03_27_pc_t1.shp"
    main(dem_path, point_cloud_path)
``
  
![image](https://github.com/user-attachments/assets/ad29c30e-c511-464d-a10c-5eedc9d92a4b)
This code loads elevation model (DEM) data and a 3D point cloud in shapefile format, and then calculates the elevation differences between the DEM values and the points in the cloud. From these differences, accuracy metrics such as mean error, RMSE (root mean square error) and standard deviation are calculated. The results show an average height difference of 4.05 meters, which is quite a large error. The RMSE is 9.25 meters, suggesting a significant discrepancy, and the standard deviation of 8.31 meters indicates a wide spread of errors at the analyzed points.

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
