
from math import asin, cos, radians, sin, sqrt
import numpy as np
from loadmydata.load_molene_meteo import load_molene_meteo_dataset


################################################################################################
########################################## QUESTION 4 ##########################################
################################################################################################


def load_data(whiteslist_stations=None):
    data_df, stations_df, description = load_molene_meteo_dataset()
    # convert temperature from Kelvin to Celsius
    data_df["temp"] = data_df.t - 273.15  # temperature in Celsius

    if whiteslist_stations is not None:
        keep_cond = stations_df.Nom.isin(whiteslist_stations)
        stations_df = stations_df[keep_cond]

        keep_cond = data_df.station_name.isin(whiteslist_stations)
        data_df = data_df[keep_cond].reset_index().drop("index", axis="columns")

    temperature_df = data_df.pivot(index="date", values="temp", columns="station_name")
    return data_df, stations_df, temperature_df

# util functions from the last tutorial
def get_geodesic_distance(point_1, point_2):
    """
    Calculate the great circle distance (in km) between two points
    on the earth (specified in decimal degrees)

    https://stackoverflow.com/a/4913653
    """

    lon1, lat1 = point_1
    lon2, lat2 = point_2

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# util functions from the last tutorial

def get_exponential_similarity(condensed_distance_matrix, bandwidth, threshold):
    exp_similarity = np.exp(-(condensed_distance_matrix**2) / bandwidth / bandwidth)
    res_arr = np.where(exp_similarity > threshold, exp_similarity, 0.0)
    return res_arr