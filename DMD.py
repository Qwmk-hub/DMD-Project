import pandas as pd
import numpy as np
from scipy.linalg import svd, eig
from sklearn.metrics import mean_squared_error
import folium
from folium.plugins import HeatMap

df = pd.read_csv('pm_data.csv', parse_dates=['date'])

df_avg = df.groupby('date')[['PM10', 'PM2.5']].mean().reset_index()
df_avg.set_index('date', inplace=True)
df_avg = df_avg.interpolate()

X = df_avg['PM10'].values.reshape(1, -1)

def apply_dmd(X, r):
    X1 = X[:, :-1]
    X2 = X[:, 1:]
    
    U, S, Vh = svd(X1, full_matrices=False)
    Ur = U[:, :r]
    Sr = np.diag(S[:r])
    Vr = Vh.conj().T[:, :r]

    A_tilde = Ur.T @ X2 @ Vr @ np.linalg.inv(Sr)
    eigvals, W = eig(A_tilde)
    Phi = X2 @ Vr @ np.linalg.inv(Sr) @ W

    return eigvals, Phi

eigvals, modes = apply_dmd(X, r=2)

def forecast_dmd(eigvals, modes, timesteps):
    omega = np.log(eigvals)
    t = np.arange(timesteps)
    b = np.linalg.lstsq(modes, X[:, 0], rcond=None)[0]
    
    forecast = np.zeros((modes.shape[0], timesteps), dtype='complex')
    for i in range(timesteps):
        forecast[:, i] = modes @ (b * np.exp(omega * t[i]))
    
    return forecast.real

X_forecast = forecast_dmd(eigvals, modes, timesteps=20)

errors = []
for days in range(1, 21):
    pred = X_forecast[0, :days]
    true = df_avg['PM10'].values[-days:]
    mse = mean_squared_error(true, pred)
    errors.append(mse)

optimal_day = np.argmin(errors) + 1
print(f"최적 예측 일수: {optimal_day}, 최소 MSE: {errors[optimal_day-1]:.4f}")

def visualize_heatmap(df_map, value_col, title):
    m = folium.Map(location=[36.5, 127.5], zoom_start=7)
    heat_data = [[row['lat'], row['lon'], row[value_col]] for _, row in df_map.iterrows()]
    HeatMap(heat_data).add_to(m)
    return m

df_map = pd.read_csv('station_coords.csv')  # lat, lon 포함

df_map['PM10'] = df_avg['PM10'].iloc[-1]
m_actual = visualize_heatmap(df_map, 'PM10', 'Actual PM10')
m_actual.save('actual_pm10.html')

df_map['PM10'] = X_forecast[0, optimal_day - 1]
m_predicted = visualize_heatmap(df_map, 'PM10', 'Predicted PM10')
m_predicted.save('predicted_pm10.html')
