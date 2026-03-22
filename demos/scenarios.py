from bfx import get_timestamp

SCENARIOS = {
    "google_leak": {
        "input": "data/Google_Leak.json",
        "outdir": "out",
        "anomaly_name": "Google_Leak",
        "start_time": get_timestamp(2017, 8, 25, 3, 0, 0),
        "end_time": get_timestamp(2017, 8, 25, 4, 0, 0),
        "anomaly_start_time": get_timestamp(2017, 8, 25, 3, 22, 0),
        "anomaly_end_time": get_timestamp(2017, 8, 25, 3, 36, 0),
        "period": 1,
        "features": None,  # or a list of feature names
    },
    "india_leak": {
        "input": "data/India_Leak.json",
        "outdir": "out",
        "anomaly_name": "India_Leak",
        "start_time": get_timestamp(2021, 4, 16, 13, 20, 0),
        "end_time": get_timestamp(2021, 4, 16, 14, 20, 0),
        "anomaly_start_time": get_timestamp(2021, 4, 16, 13, 48, 0),
        "anomaly_end_time": get_timestamp(2021, 4, 16, 14, 0, 0),
        "period": 1,
        "features": None,  # or a list of feature names
    },
}
