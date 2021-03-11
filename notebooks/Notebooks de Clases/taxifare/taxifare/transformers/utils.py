
def minkowski_distance(df, p,
                       start_lat="pickup_latitude",
                       start_lon="pickup_longitude",
                       end_lat="dropoff_latitude",
                       end_lon="dropoff_longitude"):
    x1 = df[start_lon]
    x2 = df[end_lon]
    y1 = df[start_lat]
    y2 = df[end_lat]
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)
