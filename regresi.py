import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

# Membaca dataset dari file CSV
file_path = 'C:\metnum tugas 2\Student_Performance.csv'
data = pd.read_csv('Student_Performance.csv')

# Ekstraksi kolom yang diperlukan
x = data['Sample Question Papers Practiced'].values.reshape(-1, 1)
y = data['Performance Index'].values

# Model Linear (Metode 1)
linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred_linear = linear_model.predict(x)
rmse_linear = np.sqrt(mean_squared_error(y, y_pred_linear))

# Model Eksponensial (Metode 3)
def exponential_model(x, a, b):
    return a * np.exp(b * x)

popt, _ = curve_fit(exponential_model, x.flatten(), y, maxfev=10000)
y_pred_exponential = exponential_model(x.flatten(), *popt)
rmse_exponential = np.sqrt(mean_squared_error(y, y_pred_exponential))

# Plot grafik
plt.figure(figsize=(14, 6))

# Plot untuk model linear
plt.subplot(1, 2, 1)
plt.scatter(x, y, label='Data Sebenarnya')
plt.plot(x, y_pred_linear, color='red', label='Regresi Linear')
plt.title(f'Regresi Linear\nRMSE: {rmse_linear:.2f}')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()

# Plot untuk model eksponensial
plt.subplot(1, 2, 2)
plt.scatter(x, y, label='Data Sebenarnya')
plt.plot(x, y_pred_exponential, color='red', label='Regresi Eksponensial')
plt.title(f'Regresi Eksponensial\nRMSE: {rmse_exponential:.2f}')
plt.xlabel('Jumlah Latihan Soal (NL)')
plt.ylabel('Nilai Ujian (NT)')
plt.legend()

plt.tight_layout()
plt.show()

print(f'RMSE Linear: {rmse_linear}')
print(f'RMSE Eksponensial: {rmse_exponential}')
