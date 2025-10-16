# Demo: FeatureChartExaminer for "Google Leak" anomaly
# Window: Aug 25, 2017, 03:00–04:00 UTC
# Anomaly shading: 03:22:00–03:36:00 UTC
# Input: data/Features_1.json
# Output: out/Google_Leak/google-leak-timeseries.png

from bfx import get_timestamp, Dataset, FeaturesChartExaminer

def main() -> None:
    y, m, d = 2017, 8, 25

    # Investigation window
    start_time = get_timestamp(y, m, d, 3, 0, 0)   # 03:00:00 UTC
    end_time   = get_timestamp(y, m, d, 4, 0, 0)   # 04:00:00 UTC

    # Anomaly window
    anomaly_start_time = get_timestamp(y, m, d, 3, 22, 0)  # 03:22:00
    anomaly_end_time   = get_timestamp(y, m, d, 3, 36, 0)  # 03:36:00

    # Dataset
    ds = Dataset()
    ds.setParams({
        "input": "data/Features_1.json",  # root-level data/
        "outdir": "out",                  # root-level out/
        "anomaly_name": "Google_Leak",
        "start_time": start_time,
        "end_time": end_time,
        "anomaly_start_time": anomaly_start_time,
        "anomaly_end_time": anomaly_end_time,
        "period": 1, 
    })

    # Chart examiner
    chart = FeaturesChartExaminer(ds)
    chart.setParams({
        "chart_type": "feature_timeseries",
        "scaling": "minmax",
        "features": [
            "nb_A",
            "nb_W",
            "nb_implicit_W",
            "nb_dup_A",
            # "nb_dup_W", 
            # "nb_A_prefix", 
            # "nb_W_prefix",
            # "max_A_AS", 

            # "avg_A_AS", 
            # "nb_orign_change", 
            # "nb_new_A", 
            # "nb_new_A_afterW", 
            
            # "nb_tolonger",
            # "nb_toshorter",
            # "editdist_7",
            # "editdist_8", 

            # "editdist_9",
            # "editdist_10", 
            # "editdist_11",
            # "editdist_12",

            # "editdist_13", 
            # "editdist_14",
            # "editdist_15",
            # "editdist_16",
            
            # "editdist_17",
            # "max_path_len",
            # "avg_path_len", 
            # "max_editdist", 
         
            # "avg_editdist",
            # "avg_interarrival",
            # "avg_A_prefix", 
            # "max_A_prefix",
        ],  # None -> all; list -> subset
        "filename": "google-leak-timeseries.png",
        "dpi": 140,
    })

    fig, ax, saved = chart.run()
    print(f"Saved chart to: {saved}" if saved else "Chart rendered (not saved).")

if __name__ == "__main__":
    main()
