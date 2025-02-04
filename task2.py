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
