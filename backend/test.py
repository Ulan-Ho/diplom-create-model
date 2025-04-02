# from datetime import datetime
# from meteostat import Point, Daily

# # Set time period
# start = datetime(2022, 4, 1)
# end = datetime(2025, 4, 1, 12, 42)

# # Create Point for Vancouver, BC
# vancouver = Point(39.1168569, -1.0266075, 70)

# # Get daily data for 2018
# data = Daily(vancouver, start, end)
# data = data.fetch()
# print(data)
# # Plot line chart including average, minimum and maximum temperature
# # data.plot(y=['tavg', 'tmin', 'tmax'])
# # plt.show()


import joblib
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

def calculate_distance(row1, row2):
    point1 = (row1['latitude'], row1['longitude'])
    point2 = (row2['latitude'], row2['longitude'])
    return geodesic(point1, point2).km



def calculate_distance_between_points(df):
    distances = []
    for i in range(1, len(df)):
        point1 = (df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude'])
        point2 = (df.iloc[i]['latitude'], df.iloc[i]['longitude'])
        distance = geodesic(point1, point2).meters  # расстояние в метрах
        distances.append(distance)
    
    # Добавляем начальную точку (с нулевым расстоянием)
    distances = [0] + distances
    df['distance'] = distances
    return df['distance']


# Загружаем обученную модель
model = joblib.load("travel_time_model.pkl")

# Обрабатываем данные с трекера
file_path = "./gpx/11948042.gpx"
weather_data_df = process_gpx_with_weather(file_path)
weather_data_df['time_seconds'] = (weather_data_df['time'] - weather_data_df['time'].min()).dt.total_seconds()
weather_data_df['distance'] = calculate_distance_between_points(weather_data_df)
day_of_week_map = {'Понедельник': 0, 'Вторник': 1, 'Среда': 2, 'Четверг': 3, 'Пятница': 4, 'Суббота': 5, 'Воскресенье': 6}
weather_data_df['day_of_week'] = weather_data_df['day_of_week'].map(day_of_week_map).fillna(-1)


# Список признаков, которые используются в модели
features = ['latitude', 'longitude', 'elevation', 'temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres', 'coco', 'day_of_week', 'time_seconds', 'distance']
X_new = weather_data_df[features].dropna()


weather_data_df = weather_data_df.loc[X_new.index]  # Убираем те же строки, что и в X_new
weather_data_df["predicted_time"] = model.predict(X_new)

# Выводим результаты
print(weather_data_df[["latitude", "longitude", "time", "predicted_time"]])


print(X_new.describe())
print(X_new.head())

# Если хочешь сохранить результаты
weather_data_df.to_csv("predicted_travel_times.csv", index=False)
