Запускаем обработку карты (подготовка данных для маршрутизации)
GERMANY:
docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/germany-latest.osm.pbf
docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-contract /data/germany-latest.osrm


KAZAKHSTAN:
docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/kazakhstan-latest.osm.pbf
docker run -t -v ${PWD}:/data osrm/osrm-backend osrm-contract /data/kazakhstan-latest.osrm

OSRM API-сервер
docker run -t -i -p 5000:5000 -v ${PWD}:/data osrm/osrm-backend osrm-routed --algorithm mld /data/kazakhstan-latest.osrm


docker run -t -i -p 5000:5000 -v A:\diplom-supply-optimization\osrm-data\data:/data osrm/osrm-backend osrm-routed --algorithm mld /data/kazakhstan-latest.osrm


TEST: curl "http://localhost:5000/route/v1/driving/76.926,43.222;71.414,51.169?overview=full"






