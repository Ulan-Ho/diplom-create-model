# import pandas as pd
# import numpy as np
# import math
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
#
#
# def haversine_distance(lat1, lon1, lat2, lon2):
#     R = 6371e3  # радиус Земли в метрах
#     phi1, phi2 = math.radians(lat1), math.radians(lat2)
#     delta_phi = math.radians(lat2 - lat1)
#     delta_lambda = math.radians(lon2 - lon1)
#     a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     return R * c
#
#
# # Синтетические данные (пример)
# N = 10000
# data = []
# for i in range(N):
#     # Расстояние между точками (в метрах)
#     distance = np.random.uniform(1000, 100000)
#
#     # Рельеф: средний уклон в процентах (0-10%)
#     slope = np.random.uniform(0, 10)
#
#     # Время суток (0-23)
#     hour = np.random.randint(0, 24)
#
#     # День недели (0=Понедельник, ..., 6=Воскресенье)
#     weekday = np.random.randint(0, 7)
#
#     # Погодные условия: 0 - ясная, 1 - дождь, 2 - снег
#     weather = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
#
#     # Дорожные условия: индекс качества дороги от 0 до 1 (1 - отличное покрытие)
#     road_quality = np.random.uniform(0.5, 1.0)
#
#     # Ограничение скорости (в км/ч)
#     speed_limit = np.random.choice([50, 80, 100, 120])
#
#     # Тип транспорта: 0 - легковой автомобиль, 1 - грузовик, 2 - автобус
#     vehicle_type = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
#
#     # Вес груза (в кг): для легковых машин 0-200, для грузовиков 500-5000, для автобусов 0-100
#     if vehicle_type == 0:
#         weight = np.random.uniform(0, 200)
#     elif vehicle_type == 1:
#         weight = np.random.uniform(500, 5000)
#     else:
#         weight = np.random.uniform(0, 100)
#
#     # Исторические данные: среднее время в пути (в секундах) по аналогичным маршрутам
#     base_time = distance / (speed_limit * 1000 / 3600)  # теоретическое время без влияния дополнительных факторов
#
#     # Влияние уклона: увеличение времени на 1% за каждый процент уклона
#     time_after_slope = base_time * (1 + slope / 100)
#
#     # Влияние времени суток: пиковые часы (7-9, 17-19) замедляют движение на 20%
#     if 7 <= hour <= 9 or 17 <= hour <= 19:
#         time_after_time = time_after_slope * 1.2
#     else:
#         time_after_time = time_after_slope
#
#     # Влияние погоды: дождь увеличивает время на 10%, снег на 20%
#     if weather == 1:
#         time_after_weather = time_after_time * 1.1
#     elif weather == 2:
#         time_after_weather = time_after_time * 1.2
#     else:
#         time_after_weather = time_after_time
#
#     # Влияние дорожных условий: плохое качество увеличивает время на 15%
#     time_after_road = time_after_weather * (1 + (1 - road_quality) * 0.15)
#
#     # Влияние транспортного средства: грузовики и автобусы движутся медленнее
#     if vehicle_type == 1:
#         time_after_vehicle = time_after_road * 1.2
#     elif vehicle_type == 2:
#         time_after_vehicle = time_after_road * 1.1
#     else:
#         time_after_vehicle = time_after_road
#
#     travel_time = time_after_vehicle
#
#     data.append({
#         'distance': distance,  # Расстояние в метрах
#         'slope': slope,  # Уклон
#         'hour': hour,  # Время суток
#         'weekday': weekday,  # День недели
#         'weather': weather,  # Погодные условия
#         'road_quality': road_quality,  # Качество дороги
#         'speed_limit': speed_limit,  # Ограничение скорости
#         'vehicle_type': vehicle_type,  # Тип транспорта (0, 1, 2)
#         'weight': weight,  # Вес груза в кг
#         'travel_time': travel_time  # Фактическое время в пути в секундах
#     })
#
# df = pd.DataFrame(data)
# print(df.head())
#
#
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error
#
# # Выбираем признаки и целевую переменную
# features = ['distance', 'slope', 'hour', 'weekday', 'weather', 'road_quality', 'speed_limit', 'vehicle_type', 'weight']
# X = df[features]
# y = df['travel_time']
#
# # Разбиваем данные на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Инициализируем и обучаем модель градиентного бустинга
# model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Предсказание и оценка модели
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred, squared=False)
#
# print(f'MAE: {mae}')
# print(f'RMSE: {rmse}')

import pandas as pd
import numpy as np
import math
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Функция для вычисления расстояния между двумя точками по формуле Хаверсина
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371e3  # радиус Земли в метрах
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Генерация синтетических данных
def generate_synthetic_data(num_samples=10000, seed=42):
    np.random.seed(seed)
    data = []

    for i in range(num_samples):
        # Расстояние между точками в метрах (от 1 км до 100 км)
        distance = np.random.uniform(1000, 100000)

        # Рельеф: средний уклон маршрута в процентах (0-10%)
        slope = np.random.uniform(0, 10)

        # Время суток: час от 0 до 23
        hour = np.random.randint(0, 24)

        # День недели: 0 - Понедельник, ..., 6 - Воскресенье
        weekday = np.random.randint(0, 7)

        # Погодные условия: 0 - ясная погода, 1 - дождь, 2 - снег
        weather = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

        # Дорожные условия: индекс качества дороги (от 0.5 до 1.0; 1.0 – отличное покрытие)
        road_quality = np.random.uniform(0.5, 1.0)

        # Ограничение скорости (в км/ч): варианты 50, 80, 100, 120
        speed_limit = np.random.choice([50, 80, 100, 120])

        # Тип транспорта: 0 - легковой автомобиль, 1 - грузовик, 2 - автобус
        vehicle_type = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])

        # Вес груза (в кг):
        if vehicle_type == 0:
            weight = np.random.uniform(0, 200)
        elif vehicle_type == 1:
            weight = np.random.uniform(500, 5000)
        else:
            weight = np.random.uniform(0, 100)

        # Теоретическое время в пути (без учета дополнительных факторов)
        base_time = distance / (speed_limit * 1000 / 3600)  # в секундах

        # Влияние уклона: увеличение времени на 1% за каждый процент уклона
        time_after_slope = base_time * (1 + slope / 100)

        # Влияние времени суток: пиковые часы (7-9 и 17-19) увеличивают время на 20%
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            time_after_time = time_after_slope * 1.2
        else:
            time_after_time = time_after_slope

        # Влияние погоды: дождь увеличивает время на 10%, снег – на 20%
        if weather == 1:
            time_after_weather = time_after_time * 1.1
        elif weather == 2:
            time_after_weather = time_after_time * 1.2
        else:
            time_after_weather = time_after_time

        # Влияние дорожных условий: плохое покрытие увеличивает время на 15%
        time_after_road = time_after_weather * (1 + (1 - road_quality) * 0.15)

        # Влияние типа транспорта: грузовики и автобусы движутся медленнее
        if vehicle_type == 1:
            time_after_vehicle = time_after_road * 1.2
        elif vehicle_type == 2:
            time_after_vehicle = time_after_road * 1.1
        else:
            time_after_vehicle = time_after_road

        travel_time = time_after_vehicle  # итоговое время в пути в секундах

        data.append({
            'distance': distance,  # Расстояние (м)
            'slope': slope,  # Уклон (%)
            'hour': hour,  # Время суток (0-23)
            'weekday': weekday,  # День недели (0-6)
            'weather': weather,  # Погодные условия (0,1,2)
            'road_quality': road_quality,  # Качество дороги (0.5-1.0)
            'speed_limit': speed_limit,  # Ограничение скорости (км/ч)
            'vehicle_type': vehicle_type,  # Тип транспорта (0,1,2)
            'weight': weight,  # Вес груза (кг)
            'travel_time': travel_time  # Время в пути (сек)
        })

    df = pd.DataFrame(data)
    return df


# Генерируем данные
df = generate_synthetic_data(num_samples=10000)

# Сохраняем данные в CSV с отметкой времени (чтобы каждый запуск создавал уникальный файл)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"synthetic_travel_data_{timestamp}.csv"
df.to_csv(csv_filename, index=False)
print(f"Данные сохранены в файл: {csv_filename}")

# # Обучение модели на сгенерированных данных
# features = ['distance', 'slope', 'hour', 'weekday', 'weather', 'road_quality', 'speed_limit', 'vehicle_type', 'weight']
# X = df[features]
# y = df['travel_time']
#
# # Разделение данных на обучающую и тестовую выборки
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Используем ансамблевый метод: градиентный бустинг
# model = GradientBoostingRegressor(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Предсказание и оценка модели
# y_pred = model.predict(X_test)
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
#
# print(f'MAE: {mae:.2f} секунд')
# print(f'RMSE: {rmse:.2f} секунд')
#
# # Пример вывода нескольких предсказаний
# predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
# print(predictions.head())
# Признаки и целевая переменная
features = ['distance', 'slope', 'hour', 'weekday', 'weather', 'road_quality', 'speed_limit', 'vehicle_type', 'weight']
X = df[features].values  # Признаки
y = df['travel_time'].values  # Время в пути (сек)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Нормализация признаков (стандартизация)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создание модели нейронной сети
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Выходной слой для регрессии
])

# Компиляция модели
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Раннее остановка, чтобы избежать переобучения
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Обучение модели
history = model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# Оценка модели
loss, mae = model.evaluate(X_test, y_test)
print(f"Test MAE: {mae:.2f} секунд")

# Пример предсказаний
predictions = model.predict(X_test).flatten()
results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print(results.head())