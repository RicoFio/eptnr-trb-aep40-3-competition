from enum import Enum


class GTFSNetworkCostsPerDistanceUnit(Enum):
    WALK = 0
    TRAM = 200
    STREETCAR = 200
    LIGHT_RAIL = 200
    METRO = 500
    SUBWAY = 500
    RAIL = 400
    BUS = 50
    FERRY = 200
    CABLE_TRAM = 200
    AERIAL_LIFT = 200
    SUSPENDED_CABLE_CAR = 200
    GONDOLA_LIFT = 200
    AERIAL_TRAMWAY = 700
    FUNICULAR = 200
    TROLLEYBUS = 500
    MONORAIL = 800
