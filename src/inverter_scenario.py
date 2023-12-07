# Created by A. MATHIEU at 06/12/2023
"""Script to generate inverter AC timeserie scenario which includes some error compared to initial design"""
import numpy as np
import pandas as pd

from tqdm import tqdm
from pvlib.inverter import pvwatts

MU_uncertainty = pd.DataFrame(index=np.arange(0, 1.3, 0.1).round(1))
MU_uncertainty["min"] = [-0.24, -0.1, -0.07, -0.06, -0.06, -0.06, -0.06, -0.06, -0.06, -0.06, -0.01, 0.05, 0.2]
MU_uncertainty["max"] = [0.19, 0.06, 0.05, 0.04, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.06, 0.14, 0.24]
STD_uncertainty = pd.Series(data=[0.072, 0.041, 0.029, 0.016, 0.009, 0.009, 0.008, 0.008, 0.006, 0.006, 0.026, 0.017,
                                  0.041], index=MU_uncertainty.index)


def generate_inv_scenarios(dc_power, pdc0, n_scenarios=1000, mus=MU_uncertainty, stds=STD_uncertainty):
    # Get efficiency
    ac_model = pvwatts(dc_power, pdc0)
    efficiency_model = ac_model / dc_power

    # Compute mean systematic error
    mu_delta = np.random.uniform(size=1000)
    mu_deltas = pd.DataFrame(columns=range(n_scenarios), index=mus.index)
    for n in range(n_scenarios):
        mu_deltas[n] = mu_delta[n] * (mus["max"] - mus["min"]) + mus["min"]

    # Get mus and stds for each timestep
    mu_scenarios = pd.DataFrame(columns=range(n_scenarios), index=dc_power.index)
    std_scenarios = pd.DataFrame(columns=range(n_scenarios), index=dc_power.index)
    for i in tqdm(range(len(mus.index)), disable=n_scenarios < 10):
        lim = mus.index[i]
        if i < (len(mus.index) - 1):
            lim_1 = mus.index[i + 1]
            index_filter = efficiency_model.loc[(efficiency_model >= lim) & (efficiency_model < lim_1)].index
        else:
            index_filter = efficiency_model.loc[(efficiency_model >= lim)]

        for idx in index_filter:
            mu_scenarios.loc[idx] = mu_deltas.loc[lim]

        stds_lim = np.random.normal(loc=0, scale=stds.loc[lim], size=(len(index_filter), n_scenarios))
        std_scenarios.loc[index_filter, :] = stds_lim

    # Combine systematic error + random error
    eff_scns = mu_scenarios.add(efficiency_model, axis=0).clip(lower=0)
    eff_scns_noised = eff_scns.add(std_scenarios, axis=0).clip(lower=0, upper=1.5)
    ac_scns = eff_scns_noised.mul(dc_power, axis=0)

    return ac_scns
