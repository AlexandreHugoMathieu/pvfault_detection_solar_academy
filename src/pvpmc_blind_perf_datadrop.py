# Created by A. MATHIEU at 10/11/2023
"""Script which exports performance data from the PVPMC blind comparison in order to estimate how accurate generic models can be"""
import numpy as np
import pandas as pd

from tqdm import tqdm
from pvlib.solarposition import get_solarposition
from pvlib.pvsystem import retrieve_sam
from pvlib.iam import martin_ruiz_diffuse, martin_ruiz
from pvlib.spectrum import spectral_factor_sapm
from pvlib.irradiance import aoi
from pvlib.atmosphere import get_absolute_airmass, get_relative_airmass

from src.config import DATA_PATH
from src.failure_ivmpp_modeling import scale_system_iv_curve_pdc, fixed_shading, clipping, bdiode_sc, default_effect_plot
from src.inverter_scenario import generate_inv_scenarios
from src.pv_models import get_Pmpp
from src.utils.helio_fmt import setup_helio_plt

setup_helio_plt()
# Constants provided from the report: The 2021 Blind PVPMC Modeling Intercomparison, Marios Theristis, and Joshua S. Stein
LAT_ALB, LONG_ALB = 35.05, -106.54  # °
ALT_ALB = 1600  # m
TZ_ALB = "Etc/GMT+7"

TILT_S2 = 35  # °
AZIMUTH_S2 = 180  # °


def pvpmc_perf_data(system: str):
    """PVPMC blind comparison data

    Reference
    ---------

     Theristis, M, Riedel-Lyngskær, N, Stein, JS, et al. Blind photovoltaic modeling intercomparison:
     A multidimensional data analysis and lessons learned.
     Prog Photovolt Res Appl. 2023; 31(11): 1144-1157. doi:10.1002/pip.3729
    """

    # Alburquerque
    url = 'https://datahub.duramat.org/dataset/293db0cb-e838-4f7a-8e77-f62e85328c47/resource/b54bdc36-1864-48a9-abab' \
          '-daf0e3f8dcf5/download/pvpmc_2021_blind_modeling_comparison_data_s1-s6.xlsx'
    data = pd.read_excel(url, sheet_name=system)

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

    data = data.sort_index()

    data = data.rename(columns={"GHI (W/m2)": "ghi", "DHI (W/m2)": "dhi", "DNI (W/m2)": "dni"})
    data = data.rename(columns={"Ambient Temp (°C) ": "temp_air", "Wind Speed (m/s)": "wind_speed",
                                "Measured front POA irradiance (W/m2)": "gpoa"})
    data = data.rename(columns={'Measured module temperature (°C)': "t_mod", "Measured DC power (W)": "pdc"})

    return data


def get_iam_smm(aoi_angle, tilt, poa_beam, gpoa, solar_elevation, sandia_params):
    iam_b = martin_ruiz(aoi_angle)
    iam_d = martin_ruiz_diffuse(tilt)
    iam_total = ((iam_b * poa_beam.clip(upper=gpoa) + (gpoa - poa_beam).clip(lower=0) * iam_d[0]) / gpoa)
    iam_total = iam_total.clip(lower=0, upper=1)
    am_rel = get_relative_airmass(90 - solar_elevation)
    am_abs = get_absolute_airmass(am_rel)
    smm = spectral_factor_sapm(am_abs, sandia_params).replace(0, 1)
    return iam_total, smm


def hourly_solarposition(index, lat, long, alt, tz):
    # Extract solar position and use the average as an estimate
    min_index = pd.date_range(index[0], index[-1] + pd.Timedelta(hours=1), freq="1min").tz_convert("UTC")
    solar_pos = get_solarposition(min_index, lat, long, alt)
    solar_pos_tz = solar_pos.tz_convert(tz)
    # Same convention as in the report: Hourly averages, reported at the end of the hour hourly average
    solar_pos_H = solar_pos_tz.resample("H", origin='end').mean()

    return solar_pos_H


if __name__ == "__main__":
    # System 1, Alburquerque
    data = pvpmc_perf_data("S1")
    data = data.tz_localize(TZ_ALB)
    solar_pos_H = hourly_solarposition(data.index, LAT_ALB, LONG_ALB, ALT_ALB, TZ_ALB)

    data_xtrain = data[["ghi", "dhi", "dni", "temp_air", "wind_speed"]].copy()
    data_ytrain = data[["gpoa", 't_mod', 'pdc']].copy()
    data_xtrain["sun_azimuth"] = solar_pos_H["azimuth"].copy()
    data_xtrain["sun_elevation"] = solar_pos_H["apparent_elevation"].copy()

    data_xtrain.to_csv(DATA_PATH / "xtrain.csv")
    data_ytrain.to_csv(DATA_PATH / "ytrain.csv")

    # # System 2, Alburquerque
    data = pvpmc_perf_data("S2")
    data = data.tz_localize(TZ_ALB)
    pdc0 = 3900
    pac = generate_inv_scenarios(data["pdc"], 3900, n_scenarios=1)[0]
    data["pac"] = pac
    solar_pos_H = hourly_solarposition(data.index, LAT_ALB, LONG_ALB, ALT_ALB, TZ_ALB)

    data_xtrain = data[["ghi", "dhi", "dni", "temp_air", "wind_speed"]].copy()
    data_ytrain = data[["gpoa", 't_mod', 'pdc', 'pac']].copy()

    data_xtrain["solar_azimuth"] = solar_pos_H["azimuth"].copy()
    data_xtrain["solar_elevation"] = solar_pos_H["apparent_elevation"].copy()

    data_xtrain.to_csv(DATA_PATH / "project_data" / "project_xtrain.csv")
    data_ytrain.to_csv(DATA_PATH / "project_data" / "project_ytrain.csv")

    ####### Inject failure defects
    data_xtrain = pd.read_csv(DATA_PATH / "project_data" / "project_xtrain.csv", index_col=0)
    data_model = pd.read_csv(DATA_PATH / "project_data" / "project_ytrain.csv", index_col=0)
    data = pd.concat([data_xtrain, data_model], axis=1)
    data.index = pd.to_datetime(data.index)

    # Get IAM and SMM
    aoi_angle = aoi(TILT_S2, AZIMUTH_S2, 90 - data["solar_elevation"], data["solar_azimuth"])
    data["poa_direct"] = (data["ghi"] * np.cos(aoi_angle * np.pi / 180)).clip(lower=0)
    data["poa_diffuse"] = (data["gpoa"] - data["poa_direct"]).clip(lower=0)
    iam_total, smm = get_iam_smm(aoi_angle, TILT_S2, data["poa_direct"], data["gpoa"], data["solar_elevation"],
                                 retrieve_sam("Sandiamod")['Canadian_Solar_CS6X_300M__2013_'])

    pv_params = retrieve_sam("CECMod")['Canadian_Solar_Inc__CS6P_275M']
    for idx, row in tqdm(data.iterrows(), total=len(data.index)):
        if row["gpoa"] > 0:
            g_poa_effective = (row["gpoa"] * smm.loc[idx] * iam_total.loc[idx]).copy()
            iv_curve = scale_system_iv_curve_pdc(pv_params, pdc=row["pdc"], g_poa_effective=g_poa_effective,
                                                 temp_cell=row["t_mod"], n_module=12, vmax=12 * 39)
            _, i_dc, v_dc = get_Pmpp(iv_curve, VI_max=True, v_col='v_system', i_col='i_system')
            pdc_model = get_Pmpp(iv_curve) * 12
            data.loc[idx, "pdc_model"] = pdc_model
            data.loc[idx, "v_dc_model"] = float(v_dc)
            data.loc[idx, "i_dc_model"] = float(i_dc)


    # Shading
    idc_s, vdc_s, pdc_s, pac_s = fixed_shading(data["gpoa"], data["poa_diffuse"], data["i_dc_model"],
                                               data["v_dc_model"], data["pac"], data["solar_azimuth"],
                                               data["solar_elevation"], data["t_mod"])

    # Clipping
    idc_c, vdc_c, pdc_c, pac_c = clipping(data["gpoa"], data["i_dc_model"], data["v_dc_model"], data["pac"],
                                          data["t_mod"], pac_max=2666, pv_params=pv_params)

    # Bypass short circuit
    sc_date = pd.to_datetime("202007171200").tz_localize(TZ_ALB)
    idc_sc, vdc_sc, pdc_sc, pac_sc = \
        bdiode_sc(data["gpoa"], data["i_dc_model"], data["v_dc_model"], data["pac"], data["t_mod"],
                  sc_date, pv_params, n_mod=12, n_diode=3)

    # Visualize
    setup_helio_plt()
    _ = default_effect_plot(data["i_dc_model"], data["v_dc_model"], data["pdc"], data["pac"], idc_s, vdc_s, pdc_s,
                            pac_s, "Shading")
    _ = default_effect_plot(data["i_dc_model"], data["v_dc_model"], data["pdc"], data["pac"], idc_c, vdc_c, pdc_c,
                            pac_c, "Clipping")
    _ = default_effect_plot(data["i_dc_model"], data["v_dc_model"], data["pdc"], data["pac"], idc_sc, vdc_sc, pdc_sc,
                            pac_sc, "Short-circuit")

    # Store
    data_shading = data[['gpoa', 'pdc', 'pac']].copy()
    data_clipping = data[['gpoa', 'pdc', 'pac']].copy()
    data_sc = data[['gpoa', 'pdc', 'pac']].copy()
    data_shading["pdc"] = pdc_s
    data_shading["pac"] = pac_s
    data_clipping["pdc"] = pdc_c
    data_clipping["pac"] = pac_c
    data_sc["pdc"] = pdc_sc
    data_sc["pac"] = pac_sc

    data_shading.to_csv(DATA_PATH / "project_data" / "project_shading.csv")
    data_clipping.to_csv(DATA_PATH / "project_data" / "project_clipping.csv")
    data_sc.to_csv(DATA_PATH / "project_data" / "project_sc.csv")
