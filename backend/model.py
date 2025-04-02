# import requests
# import pandas as pd
# from geopy.geocoders import Nominatim
#
# # Определяем границы Казахстана (приблизительно)
# KZ_BOUNDS = {
#     "min_lat": 40.0,
#     "max_lat": 56.0,
#     "min_lon": 46.0,
#     "max_lon": 87.0
# }
#
#
# # Фильтрация координат по Казахстану
# def is_in_kazakhstan(lat, lon):
#     return (KZ_BOUNDS["min_lat"] <= lat <= KZ_BOUNDS["max_lat"]) and \
#         (KZ_BOUNDS["min_lon"] <= lon <= KZ_BOUNDS["max_lon"])
#
#
# # Функция для получения маршрута из OSRM
# def get_route(start, end, osrm_url="http://localhost:5000/route/v1/car"):
#     url = f"{osrm_url}/{start[1]},{start[0]};{end[1]},{end[0]}?overview=false&steps=true&alternatives=true"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json()
#     return None
#
#
# # Функция для подготовки входных данных
# def prepare_training_data(routes, weather_data):
#     training_data = []
#     for route in routes:
#         date = route["date"]
#         start, end = route["start"], route["end"]
#
#         if date in weather_data.index:
#             weather = weather_data.loc[date]
#             osrm_info = get_route(start, end)
#
#             if osrm_info:
#                 distance = osrm_info["routes"][0]["distance"]
#                 duration = osrm_info["routes"][0]["duration"]
#
#                 training_data.append({
#                     "date": date,
#                     "start_lat": start[0],
#                     "start_lon": start[1],
#                     "end_lat": end[0],
#                     "end_lon": end[1],
#                     "distance": distance,
#                     "duration": duration,
#                     **weather.to_dict()
#                 })
#     return pd.DataFrame(training_data)
#
#
# weather_data = pd.read_csv("weather_data.csv", index_col="time", parse_dates=True)
#
# # Пример маршрутов
# routes = [
#     {"date": "2018-01-01", "start": (43.222, 76.851), "end": (51.169, 71.449)},  # Алматы -> Астана
#     {"date": "2018-01-02", "start": (42.317, 69.590), "end": (47.094, 51.923)}  # Шымкент -> Атырау
# ]
#
# # Генерация данных
# training_df = prepare_training_data(routes, weather_data)
# print(training_df.head())










# from datetime import datetime
# from meteostat import Point, Daily
#
# # Set time period
# start = datetime(2018, 1, 1)
# end = datetime(2018, 1, 1)
#
# # Create Point for Vancouver, BC
# vancouver = Point(49.2497, -123.1193, 70)
#
# # Get daily data for 2018
# data = Daily(vancouver, start, end)
# data = data.fetch()
#
# print(data)
#
#
#



#
# import requests
# import pandas as pd
# from geopy.geocoders import Nominatim
#
#
# class TrainingData:
#     def __init__(self, coordinates, weather_data, distance, duration, slope, speed_limit, vehicle_type, weight):
#         """
#         Инициализация класса с данными.
#
#         :param coordinates: Список координат (latitude, longitude).
#         :param weather_data: Данные о погоде (средняя температура, минимальная, максимальная, осадки, снег, ветер, и т.д.)
#         :param distance: Расстояние между точками маршрута.
#         :param duration: Время в пути.
#         :param slope: Уклон маршрута.
#         :param speed_limit: Ограничение скорости.
#         :param vehicle_type: Тип транспорта.
#         :param weight: Вес груза.
#         """
#         self.coordinates = coordinates
#         self.weather_data = weather_data
#         self.distance = distance
#         self.duration = duration
#         self.slope = slope
#         self.speed_limit = speed_limit
#         self.vehicle_type = vehicle_type
#         self.weight = weight
#
#     def prepare_data(self):
#         """
#         Подготовка данных для обучения модели.
#
#         Преобразует данные в pandas DataFrame для легкости использования в модели машинного обучения.
#         """
#         # Формируем структуру данных для обучения
#         data = {
#             "latitude_start": self.coordinates[0][0],
#             "longitude_start": self.coordinates[0][1],
#             "latitude_end": self.coordinates[1][0],
#             "longitude_end": self.coordinates[1][1],
#             "avg_temp": self.weather_data["tavg"],  # Средняя температура
#             "min_temp": self.weather_data["tmin"],  # Минимальная температура
#             "max_temp": self.weather_data["tmax"],  # Максимальная температура
#             "precipitation": self.weather_data["prcp"],  # Осадки
#             "snow": self.weather_data["snow"],  # Снег
#             "wind_speed": self.weather_data["wspd"],  # Скорость ветра
#             "wind_direction": self.weather_data["wdir"],  # Направление ветра
#             "pressure": self.weather_data["pres"],  # Давление
#             "sunshine_duration": self.weather_data["tsun"],  # Длительность солнечного света
#             "distance": self.distance,  # Расстояние
#             "duration": self.duration,  # Продолжительность
#             "slope": self.slope,  # Уклон
#             "speed_limit": self.speed_limit,  # Ограничение скорости
#             "vehicle_type": self.vehicle_type,  # Тип транспортного средства
#             "weight": self.weight  # Вес груза
#         }
#
#         # Преобразуем в DataFrame
#         df = pd.DataFrame([data])
#
#         return df
#
#
# KZ_BOUNDS = {
#     "min_lat": 40.0,
#     "max_lat": 56.0,
#     "min_lon": 46.0,
#     "max_lon": 87.0
# }
#
#
# def is_in_Kazakhstan(lat, lon):
#     return (KZ_BOUNDS["min_lat"] <= lat <= KZ_BOUNDS["max_lat"] and KZ_BOUNDS["min_lon"] <= lon <= KZ_BOUNDS["max_lon"])
#
#
# def get_route(start, end, osrm_url = "http://localhost:5000/route/v1/car"):
#     url = f"{osrm_url}/{start[0]},{start[1]};{end[0]},{end[1]}?overview=full&geometries=geojson&annotations=speed&steps=true"
#     response = requests.get(url)
#     if response.status_code == 200:
#         return response.json()
#     return None
#
# route = {"start": (76.90789752058801,43.23938399030151), "end": (77.23529376017227,44.28901504591738)}
# start, end = route["start"], route["end"]
# # print(route["start"], route["end"])
# route = get_route(start, end)
# print(route)
#


# Дистанция маршрута: Общая протяжённость пути.​
#
# Тип местности: Информация о рельефе, наличии гор, водоёмов и других природных препятствий.​
#
# Время суток: Часы пик, ночное время и т.д.​
#
# Погодные условия: Дождь, снег, туман и их влияние на движение.​
#
# Скоростные ограничения: Максимально допустимая скорость на различных участках дороги.​
#
# Исторические данные о трафике: Информация о загруженности дорог в разные периоды времени.​
#
# Дополнительные услуги: Наличие платных дорог, паромных переправ и других факторов, влияющих на время в пути.


# from datetime import datetime
# import matplotlib.pyplot as plt
# from meteostat import Point, Daily

# # Set time period
# start = datetime(2018, 1, 1)
# end = datetime(2018, 12, 31, 23, 59)

# # Create Point for Vancouver, BC
# vancouver = Point(49.2497, -123.1193, 70)

# # Get daily data for 2018
# data = Daily(vancouver, start, end)
# data = data.fetch()
# print(data)
# Plot line chart including average, minimum and maximum temperature
# data.plot(y=['tavg', 'tmin', 'tmax'])
# plt.show()
             





















# from datetime import datetime
# import pandas as pd
# from meteostat import Point, Hourly
# import pytz
# import gpxpy
# import gpxpy.gpx

# def parse_gpx(file_path):
#     # Чтение и парсинг GPX файла
#     with open(file_path, 'r') as gpx_file:
#         gpx = gpxpy.parse(gpx_file)

#         data = []

#         for track in gpx.tracks:
#             for segment in track.segments:
#                 for point in segment.points:ю
#                     data.append({
#                         'latitude': point.latitude,
#                         'longitude': point.longitude,
#                         'elevation': point.elevation,
#                         'time': point.time
#                     })

#     df = pd.DataFrame(data)
#     return df

# def time_to_datetime(time_str):
#     # Преобразование времени в datetime без временной зоны
#     dt = datetime.fromisoformat(time_str)
#     start = datetime(dt.year, dt.month, dt.day, dt.hour)
#     end = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
#     return start, end

# # Пример использования
# file_path = "11947853.gpx"  # Укажи путь к GPX-файлу
# parsed_data = parse_gpx(file_path)
# print(parsed_data.head())

# def parse_weather(data):
#     start, end = time_to_datetime(data['time'])
#     vancouver = Point(data['latitude'], data['longitude'], data['elevation'])
#     data = Hourly(vancouver, start, end)
#     data = data.fetch()
#     return data


# # Пример использования с Meteostat
# date_str = '2025-04-01 12:05:30+00:00'

# start, end = time_to_datetime(date_str)
# vancouver = Point(49.2497, -123.1193, 400)

# # Получение данных о погоде с Meteostat
# data = Hourly(vancouver, start, end)
# data = data.fetch()

# print(data)

import os
from datetime import datetime
import pandas as pd
from meteostat import Point, Hourly
import pytz
import gpxpy
import gpxpy.gpx
from geopy.distance import geodesic
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pyttsx3


# Чтение и парсинг GPX файла
def parse_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx = gpxpy.parse(gpx_file)

        data = []

        for track in gpx.tracks:
            for segment in track.segments:
                for point in segment.points:
                    data.append({
                        'latitude': point.latitude,
                        'longitude': point.longitude,
                        'elevation': point.elevation,
                        'time': point.time
                    })

    df = pd.DataFrame(data)
    return df

# Преобразование времени в datetime без временной зоны
def time_to_datetime(time_str):
    # Проверяем, является ли time_str строкой
    if isinstance(time_str, str):
        dt = datetime.fromisoformat(time_str)
    elif isinstance(time_str, datetime):
        dt = time_str
    else:
        raise ValueError("Invalid time format: must be str or datetime")

    start = datetime(dt.year, dt.month, dt.day, dt.hour)
    end = datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute)
    return start, end


# Функция для запроса данных о погоде для каждой точки
def get_weather_for_point(latitude, longitude, elevation, time_str):
    start, end = time_to_datetime(time_str)
    point = Point(latitude, longitude, elevation)

    # Получаем погодные данные за данный интервал времени
    weather_data = Hourly(point, start, end)
    weather_data = weather_data.fetch()

    # Если данные есть, возвращаем их, иначе возвращаем None
    if not weather_data.empty:
        return weather_data.iloc[0]
    else:
        return None


def add_time_and_day(df):
    if 'time' not in df.columns:
        raise KeyError("В DataFrame отсутствует колонка 'time'")

    df['time'] = pd.to_datetime(df['time'])  # Преобразуем в datetime
    df['exact_time'] = df['time'].dt.strftime('%H:%M:%S')  # Часы, минуты, секунды
    df['day_of_week'] = df['time'].dt.day_name(locale="ru_RU")  # День недели на русском
    return df



# Обработка данных о трекере и добавление данных о погоде
def process_gpx_with_weather(file_path):
    # engine = pyttsx3.init()
    # engine.say("Я начал зугрузку")
    # engine.runAndWait()
    parsed_data = parse_gpx(file_path)

    weather_data = []
    for _, row in parsed_data.iterrows():

        if pd.isnull(row['time']):
            print(f"Пропущена строка без времени: {row}")
            continue
        
        weather = get_weather_for_point(row['latitude'], row['longitude'], row['elevation'], row['time'])
        if weather is not None:
            # print(f"Хорошие данные {row['latitude'], row['longitude'], row['elevation'], row['time']}")
            row_weather = {
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'elevation': row['elevation'],
                'time': row['time'],
                'temp': weather['temp'],
                'dwpt': weather['dwpt'],
                'rhum': weather['rhum'],
                'prcp': weather['prcp'],
                # 'snow': weather['snow'],
                'wdir': weather['wdir'],
                'wspd': weather['wspd'],
                'pres': weather['pres'],
                # 'tsun': weather['tsun'],
                'coco': weather['coco']
            }
            weather_data.append(row_weather)
        else:
            print(f"Нет данных о погоде для точки {row['latitude'], row['longitude'], row['elevation'], row['time']}")

    weather_df = pd.DataFrame(weather_data)
    weather_df = add_time_and_day(weather_df)
    # engine.say("Я Закончил зугрузку")
    # engine.runAndWait()
    # print("End")

    return weather_df





def process_all_gpx_in_folder(folder_path):
    all_weather_data = []

    # Получаем список всех файлов в папке
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.gpx'):
            file_path = os.path.join(folder_path, file_name)
            print(f"Обрабатываю файл: {file_name}")
            weather_df = process_gpx_with_weather(file_path)
            all_weather_data.append(weather_df)

    # Объединяем все данные в один DataFrame
    all_weather_df = pd.concat(all_weather_data, ignore_index=True)

    # Сохраняем все данные в CSV
    output_file = os.path.join(folder_path, "weather_data.csv")
    all_weather_df.to_csv(output_file, index=False)
    print(f"Данные сохранены в {output_file}")


# folder_path = "A:/diplom-supply-optimization/backend/gpx"  # Путь к папке с GPX файлами
# df = process_all_gpx_in_folder(folder_path)


# Пример использования
file_path = "./gpx/11947793.gpx"  # Укажи путь к GPX-файлу
df = process_gpx_with_weather(file_path)

# Выводим объединенные данные
print(df.head())


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# df = pd.read_csv("./gpx/weather_data.csv")

# Преобразование времени
df['time'] = pd.to_datetime(df['time'], errors='coerce')  # Ошибки преобразования заменяются на NaT
df['time_seconds'] = (df['time'] - df['time'].min()).dt.total_seconds()

# Преобразование дня недели в числовой формат
day_of_week_map = {'Понедельник': 0, 'Вторник': 1, 'Среда': 2, 'Четверг': 3, 'Пятница': 4, 'Суббота': 5, 'Воскресенье': 6}
df['day_of_week'] = df['day_of_week'].map(day_of_week_map).fillna(-1)  # Заполняем отсутствующие значения -1

# Функция для вычисления расстояния между точками
def calculate_distance(row1, row2):
    point1 = (row1['latitude'], row1['longitude'])
    point2 = (row2['latitude'], row2['longitude'])
    return geodesic(point1, point2).km

df['distance'] = df.apply(lambda row: calculate_distance(row, df.iloc[row.name + 1]) if row.name + 1 < len(df) else None, axis=1)

# Обрабатываем деление на ноль
df['time_to_next_point'] = df.apply(lambda row: row['distance'] / row['wspd'] if row['wspd'] > 0 else None, axis=1)
print(df.isna().sum())  # Выводим количество NaN по каждому столбцу

df.dropna(inplace=True)  # Убираем ошибки


# Признаки
X = df[['latitude', 'longitude', 'elevation', 'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres', 'coco', 'day_of_week', 'time_seconds', 'distance']]
y = df['time_to_next_point']

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Обучение модели
# model = RandomForestRegressor(n_estimators=50, random_state=42)
# model.fit(X_train, y_train)
import joblib

# Допустим, у тебя обученная модель:

model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "travel_time_model.pkl")  # Сохранение в файл

print("MAE на обучающем наборе:", mean_absolute_error(y_train, model.predict(X_train)))
print("MAE на тестовом наборе:", mean_absolute_error(y_test, model.predict(X_test)))



print(set(X.columns) & {"time_to_next_point"})
print(df[['distance', 'time_to_next_point']].corr())



# Оценка модели
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae * 3600:.2f} секунд')

# Кросс-валидация
scores = cross_val_score(model, X_train, y_train, scoring="neg_mean_absolute_error", cv=5)
print(f"Средняя MAE на кросс-валидации: {-scores.mean() * 3600:.2f} секунд")

# Прогноз для новых данных
new_data = pd.DataFrame({
    'latitude': [48.359150],
    'longitude': [10.751000],
    'elevation': [506.00],
    'temp': [7.0],
    'dwpt': [2.0],
    'rhum': [70.0],
    'prcp': [0.0],
    # 'snow': [0.0],
    'wdir': [90.0],
    'wspd': [20.0],
    'pres': [1023.5],
    # 'tsun': [3.0],
    'coco': [8.0],
    'day_of_week': [1],  # Вторник
    'time_seconds': [3600],  # 1 час
    'distance': [10]  # 10 км
})

predicted_time = model.predict(new_data)
print(f'Прогнозируемое время в пути: {predicted_time[0] * 3600:.2f} секунд')




import matplotlib.pyplot as plt
import seaborn as sns

# Предсказания модели
y_pred = model.predict(X_test)

# График 1: Фактические vs. предсказанные значения
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Идеальная линия (y=x)
plt.xlabel("Фактическое время в пути (сек.)")
plt.ylabel("Предсказанное время в пути (сек.)")
plt.title("Сравнение фактических и предсказанных значений")
plt.show()

# # График 2: Ошибки предсказаний
errors = y_test - y_pred

plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=30, kde=True)
plt.xlabel("Ошибка предсказания (сек.)")
plt.ylabel("Количество наблюдений")
plt.title("Распределение ошибок предсказаний")
plt.show()

sns.boxplot(x=errors)
plt.title("Boxplot ошибок предсказаний")
plt.show()

# plt.scatter(df['distance'], errors, alpha=0.5)
# plt.xlabel("Дистанция (км)")
# plt.ylabel("Ошибка предсказания (сек.)")
# plt.title("Зависимость ошибки от дистанции")
# plt.axhline(y=0, color='r', linestyle='--')  # Линия 0 ошибки
# plt.show()


plt.hist(df['time_to_next_point'], bins=50)
plt.xlabel("Время в пути (сек.)")
plt.ylabel("Частота")
plt.title("Распределение времени в пути")
plt.show()













# from requests import get
# from pandas import json_normalize


# def get_elevation(lat = None, long = None):
#     '''
#         script for returning elevation in m from lat, long
#     '''
#     if lat is None or long is None: return None
    
#     query = ('https://api.open-elevation.com/api/v1/lookup'
#              f'?locations={lat},{long}')
    
#     # Request with a timeout for slow responses
#     r = get(query, timeout = 20)

#     # Only get the json response in case of 200 or 201
#     if r.status_code == 200 or r.status_code == 201:
#         elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
#     else:
#         elevation = None
#     return elevation

# print(get_elevation(48.359097,  10.751560))