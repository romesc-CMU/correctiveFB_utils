def get_time_horizon(params):
    time_horizon = params.get_float("horizon/time", default=-1.0)
    if time_horizon > 0.0:
        return time_horizon
    else:
        dist_horizon = params.get_float("horizon/distance", default=2.0)
        vel = params.get_float("trajgen/desired_speed", default=1.0)
        return dist_horizon / vel
