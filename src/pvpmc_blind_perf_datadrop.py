# Created by A. MATHIEU at 10/11/2023
"""Script which exports performance data (S1) from the PVPMC blind comparison in order to estimate how accurate generic models can be"""

import pandas as pd

from tqdm import tqdm
from pvlib.solarposition import get_solarposition

from src.config import DATA_PATH

# Constants provided from the report: The 2021 Blind PVPMC Modeling Intercomparison, Marios Theristis, and Joshua S. Stein
LAT_ALB, LONG_ALB = 35.05, -106.54  # °
ALT_ALB = 1600  # m
TZ_ALB = "GMT-7"

if __name__ == "__main__":
    url = 'https://datahub.duramat.org/dataset/293db0cb-e838-4f7a-8e77-f62e85328c47/resource/b54bdc36-1864-48a9-abab' \
          '-daf0e3f8dcf5/download/pvpmc_2021_blind_modeling_comparison_data_s1-s6.xlsx'
    data = pd.read_excel(url, sheet_name="S1")  # Albuquerque, USA

    data["Month"] = data["Month"].astype(str).str.zfill(2)
    data["Day"] = data["Day"].astype(str).str.zfill(2)
    data["Hour"] = data["Hour"].astype(str).str.zfill(2)

    index = pd.DatetimeIndex([])
    for idx, row in tqdm(data.iterrows(), total=len(data.index)):
        if row['Hour'] == "24":
            row['Hour'] = "00"
        dt = pd.to_datetime(f"{row['Year']}-{row['Month']}-{row['Day']} {row['Hour']}:00")
        if row['Hour'] == "00":
            dt = dt + pd.DateOffset(days=1)
        dt_index = pd.DatetimeIndex([dt])
        index = index.append(dt_index)

    data.index = index
    data = data.tz_localize("Etc/GMT+7")
    data = data.sort_index()

    data = data.rename(columns={"GHI (W/m2)": "ghi", "DHI (W/m2)": "dhi", "DNI (W/m2)": "dni"})
    data = data.rename(columns={"Ambient Temp (°C) ": "temp_air", "Wind Speed (m/s)": "wind_speed",
                                "Measured front POA irradiance (W/m2)": "gpoa"})
    data = data.rename(columns={'Measured module temperature (°C)': "t_mod", "Measured DC power (W)": "pdc"})

    data_xtrain = data[["ghi", "dhi", "dni", "temp_air", "wind_speed"]].copy()
    data_ytrain = data[["gpoa", 't_mod', 'pdc']].copy()

    # Extract solar position and use th ²e average as an estimate
    min_index = pd.date_range(data.index[0], data.index[-1] + pd.Timedelta(hours=1), freq="1min").tz_convert("UTC")
    solar_pos = get_solarposition(min_index, LAT_ALB, LONG_ALB, ALT_ALB)
    solar_pos_tz = solar_pos.tz_convert("Etc/GMT+7")
    # Same convention as in the report: Hourly averages, reported at the end of the hour hourly average
    solar_pos_H = solar_pos_tz.resample("H", origin='end').mean()

    data_xtrain["sun_azimuth"] = solar_pos_H["azimuth"].copy()
    data_xtrain["sun_elevation"] = solar_pos_H["apparent_elevation"].copy()

    data_xtrain.to_csv(DATA_PATH / "xtrain.csv")
    data_ytrain.to_csv(DATA_PATH / "ytrain.csv")
