from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import requests
import math
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

app = FastAPI()

# Класс Point оставляем, если понадобится в дальнейшем
class Point:
    def __init__(self, lon: float, lat: float):
        self.lon = lon
        self.lat = lat


OSRM_URL = "http://localhost:5000"  # Адрес локального OSRM


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371e3  # Радиус Земли в метрах
    phi1, phi2 = map(math.radians, [lat1, lat2])
    dphi, dlambda = map(math.radians, [lat2 - lat1, lon2 - lon1])
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def build_distance_matrix(points):
    n = len(points)
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = haversine_distance(*points[i], *points[j])
    return matrix


def solve_tsp_fixed_endpoints(points_str: str):
    point_items = points_str.split(";")
    points = []
    for item in point_items:
        try:
            lon, lat = map(float, item.split(','))
            points.append((lat, lon))
        except Exception as e:
            print("Ошибка разбора точки:", item, e)
            continue

    if len(points) < 3:
        return points_str

    start, end = points[0], points[-1]
    waypoints = points[1:-1]

    if not waypoints:
        return points_str

    distance_matrix = build_distance_matrix([start] + waypoints + [end])
    n = len(distance_matrix)

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node, to_node = manager.IndexToNode(from_index), manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    routing.SetFixedCostOfAllVehicles(0)
    routing.AddDisjunction([manager.NodeToIndex(n - 1)], 0)
    routing.SetFixedCostOfAllVehicles(0)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        index = routing.Start(0)
        order = []
        while not routing.IsEnd(index):
            order.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        order.append(manager.IndexToNode(index))

        ordered_points = [start] + [waypoints[i - 1] for i in order[1:-1]] + [end]
        return ";".join(f"{lon},{lat}" for lat, lon in ordered_points)

    print("Решение TSP не найдено")
    return points_str


@app.get("/route/")
def get_route(points: str, algorithm: str = "dijkstra", alternatives: bool = False):
    optimized_points = solve_tsp_fixed_endpoints(points)
    
    coordinate_pairs = points.split(';')

# Перебираем каждую пару и разделяем по запятой
    coordinates = []
    for pair in coordinate_pairs:
        lon, lat = map(float, pair.split(','))
        elevation = get_elevation(lat, lon)
        print("Высота:", elevation)
        coordinates.append((lat, lon))  # Добавляем в список в формате (широта, долгота)

    print("Оптимизированный порядок точек:", optimized_points)

    alternatives_param = "true" if alternatives else "false"
    url = (
        f"{OSRM_URL}/route/v1/car/{optimized_points}"
        f"?alternatives=true&overview=full&geometries=geojson&steps=true&annotations=speed&annotations=datasources"
    )
    # print(url)
    response = requests.get(url)
    return response.json()


@app.get("/routes")
def get_routes(points: str, profiles: list[str] = Query(["foot"])):
    """
    Пример альтернативного эндпоинта, который запрашивает маршруты сразу для нескольких профилей.
    Параметр points аналогичный: "lon,lat;lon,lat;lon,lat"
    """
    routes = {}
    for profile in profiles:
        url = (
            f"{OSRM_URL}/table/v1/{profile}/{points}"
            f"?alternatives=true"
            "&overview=full"
            "&geometries=geojson"
            "&steps=true"
            "&annotations=speed"
        )
        print(f"OSRM запрос для профиля {profile}:", url)
        response = requests.get(url)
        if response.status_code == 200:
            routes[profile] = response.json()
        else:
            routes[profile] = {"error": f"Ошибка при запросе маршрута для профиля {profile}"}

    return routes


# Настройка CORS
origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



from requests import get
from pandas import json_normalize


def get_elevation(lat = None, long = None):
    '''
        script for returning elevation in m from lat, long
    '''
    if lat is None or long is None: return None
    
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={lat},{long}')
    
    # Request with a timeout for slow responses
    r = get(query, timeout = 20)

    # Only get the json response in case of 200 or 201
    if r.status_code == 200 or r.status_code == 201:
        elevation = json_normalize(r.json(), 'results')['elevation'].values[0]
    else:
        elevation = None
    return elevation

