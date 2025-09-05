import argparse
import pickle

import h5py
import numpy as np
from uncertainty_tools import (
    all_mc_corrections,
    background_syst,
    eta_phi_systematic,
    get_era_vals,
    get_mc_lumis,
    luminometer_syst,
    make_ones_hist,
)

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    divideHists,
    expand_hist_by_duplicate_axes,
    expand_hist_by_duplicate_axis,
    multiplyHists,
    scaleHist,
)

parser = argparse.ArgumentParser()
args = parser.parse_args()

slope_ramses = 0.0006
slope_hfoc = 0.0007
mass_bin = 9


file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

data_output = results["dataPostVFP"]["output"]
lumi_output = results["dataPostVFP"]["lumi_outout"]
MC_Zmumu = results["ZmumuPostVFP"]["output"]


reco_dtdt_data = data_output["time_mll"].get()
reco_dtst_data = data_output["time_dtst"].get()
reco_stst_data = data_output["time_stst"].get()
time_proj_low_all = data_output["time_proj_2"].get()
time_proj_hlt_all = data_output["time_proj"].get()

time_proj_hlt_all = expand_hist_by_duplicate_axis(time_proj_hlt_all, "mll", "gen_mll")
time_proj_low_all = expand_hist_by_duplicate_axis(time_proj_low_all, "mll", "gen_mll")

time_proj_low_all = time_proj_low_all.project(
    "time", "mll", "gen_mll", "pt_lead", "eta_lead", "pt_sublead", "eta_sublead"
)
time_proj_hlt_all = time_proj_hlt_all.project(
    "time", "mll", "gen_mll", "pt_lead", "eta_lead", "pt_sublead", "eta_sublead"
)

time_proj_low = time_proj_low_all[{"mll": mass_bin, "gen_mll": mass_bin}]
time_proj_hlt = time_proj_hlt_all[{"mll": mass_bin, "gen_mll": mass_bin}]


### pass reco, pass generator

dtdt_prpg = MC_Zmumu["dtdt_prpg"].get()
dtst_prpg = MC_Zmumu["dtst_prpg"].get()
stst_prpg = MC_Zmumu["stst_prpg"].get()
weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

dtdt_prfg = MC_Zmumu["dtdt_prfg"].get()
dtst_prfg = MC_Zmumu["dtst_prfg"].get()
stst_prfg = MC_Zmumu["stst_prfg"].get()

### should loop over these instead of calling them explicitly

### MAKE THIS IMPLEMENTATION NOT STUPID
dtdt_prpg_BG, dtdt_prpg_BG_syst, dtdt_prpg_BG_stat = get_era_vals(
    MC_Zmumu, "dtdt", "BG"
)
dtst_prpg_BG, dtst_prpg_BG_syst, dtst_prpg_BG_stat = get_era_vals(
    MC_Zmumu, "dtst", "BG"
)
stst_prpg_BG, stst_prpg_BG_syst, stst_prpg_BG_stat = get_era_vals(
    MC_Zmumu, "stst", "BG"
)

dtdt_prpg_H, dtdt_prpg_H_syst, dtdt_prpg_H_stat = get_era_vals(MC_Zmumu, "dtdt", "H")
dtst_prpg_H, dtst_prpg_H_syst, dtst_prpg_H_stat = get_era_vals(MC_Zmumu, "dtst", "H")
stst_prpg_H, stst_prpg_H_syst, stst_prpg_H_stat = get_era_vals(MC_Zmumu, "stst", "H")

pass_gen = MC_Zmumu["pass_gen"].get()

### probably need to pull these back'
lumi_scaling = lumi_output["lumi_nom"].get()
lumi_scaling_h = lumi_output["lumi_pre"].get()
lumi_scaling_bg = lumi_output["lumi_post"].get()

### pulling for cross detector scaling
lumi_hfoc = lumi_output["lumi_hfoc"].get()
lumi_pcc = lumi_output["lumi_pcc"].get()
lumi_ramses = lumi_output["lumi_ramses"].get()

lumi_hfoc_nom = lumi_output["lumi_in_hfoc"].get()
lumi_pcc_nom = lumi_output["lumi_in_pcc"].get()
lumi_ramses_nom = lumi_output["lumi_in_ramses"].get()
## pulling for linearity
sbil_pcc = lumi_output["sbil_pcc"].get()
count_pcc = lumi_output["count_pcc"].get()


weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

nbins_mll = len(dtdt_prfg.axes["mll"])
nbins_time = len(reco_dtst_data.axes["time"])
nbins_pt_subleading = len(reco_dtst_data.axes["pt_sublead"])
nbins_pt_leading = len(reco_dtst_data.axes["pt_lead"])

nbins_eta_leading = len(reco_dtdt_data.axes["eta_lead"])


hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
hfoc_scaling = multiplyHists(hfoc_scaling, lumi_scaling)

pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
pcc_scaling = multiplyHists(pcc_scaling, lumi_scaling)

ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
ramses_scaling = multiplyHists(ramses_scaling, lumi_scaling)

dtdt_prpg_H = dtdt_prpg_H[{"mll": mass_bin, "gen_mll": mass_bin}]
dtst_prpg_H = dtst_prpg_H[{"mll": mass_bin, "gen_mll": mass_bin}]
stst_prpg_H = stst_prpg_H[{"mll": mass_bin, "gen_mll": mass_bin}]
dtdt_prpg_BG = dtdt_prpg_BG[{"mll": mass_bin, "gen_mll": mass_bin}]
dtst_prpg_BG = dtst_prpg_BG[{"mll": mass_bin, "gen_mll": mass_bin}]
stst_prpg_BG = stst_prpg_BG[{"mll": mass_bin, "gen_mll": mass_bin}]

prpg_all = [
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
]
prpg_syst = [
    dtdt_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
    dtst_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
    stst_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
    dtdt_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
    dtst_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
    stst_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
]


time_hists = [time_proj_hlt, time_proj_low]

lumi_hists = [lumi_scaling_h, lumi_scaling_bg]

dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc = get_mc_lumis(
    prpg_all,
    time_hists,
    hfoc_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)

dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc = get_mc_lumis(
    prpg_all,
    time_hists,
    pcc_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)

dtdt_prpg_ramses, dtst_prpg_ramses, stst_prpg_ramses = get_mc_lumis(
    prpg_all,
    time_hists,
    ramses_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)

avg_sbil_pcc = scaleHist(divideHists(sbil_pcc, count_pcc), 1e9)

sbil_hfoc_fit = scaleHist(avg_sbil_pcc, slope_hfoc)

sbil_ones = make_ones_hist(sbil_hfoc_fit)
sbil_hfoc_fit = addHists(sbil_hfoc_fit, sbil_ones)
sbil_hfoc_fit = multiplyHists(sbil_hfoc_fit, lumi_scaling)

sbil_ramses_fit = scaleHist(avg_sbil_pcc, slope_ramses)
sbil_ramses_fit = addHists(sbil_ramses_fit, sbil_ones)
sbil_ramses_fit = multiplyHists(sbil_ramses_fit, lumi_scaling)

dtdt_prpg_sbil_hfoc, dtst_prpg_sbil_hfoc, stst_prpg_sbil_hfoc = get_mc_lumis(
    prpg_all,
    time_hists,
    sbil_hfoc_fit,
    lumi_hists,
    weightsum,
    cross_sec,
)
dtdt_prpg_sbil_ramses, dtst_prpg_sbil_ramses, stst_prpg_sbil_ramses = get_mc_lumis(
    prpg_all,
    time_hists,
    sbil_ramses_fit,
    lumi_hists,
    weightsum,
    cross_sec,
)

#### normal
dtdt_prpg, dtst_prpg, stst_prpg = get_mc_lumis(
    prpg_all,
    time_hists,
    lumi_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)

(dtdt_prpg_prefiring_syst, dtst_prpg_prefiring_syst, stst_prpg_prefiring_syst) = (
    get_mc_lumis(
        prpg_syst,
        time_hists,
        lumi_scaling,
        lumi_hists,
        weightsum,
        cross_sec,
    )
)


pass_gen = all_mc_corrections(
    pass_gen[{"mll": mass_bin, "gen_mll": mass_bin}],
    time_proj_low,
    lumi_scaling,
    weightsum,
    cross_sec,
)

h2_first = dtdt_prpg.project("time", "pt_lead", "eta_lead")
h1_second = dtst_prpg.project("time", "pt_sublead", "eta_sublead")
h0_second = stst_prpg.project("time", "pt_sublead", "eta_sublead")
n_masked = pass_gen.project("time", "pt_lead", "eta_lead")
#### i think i will need to mix this too but it doesn't affect the fit

reco_dtdt_data = reco_dtdt_data[{"mll": mass_bin}].project(
    "time", "pt_lead", "eta_lead"
)
reco_dtst_data = reco_dtst_data[{"mll": mass_bin}].project(
    "time", "pt_sublead", "eta_sublead"
)
reco_stst_data = reco_stst_data[{"mll": mass_bin}].project(
    "time", "pt_sublead", "eta_sublead"
)


## create the tensor
writer = tensorwriter.TensorWriter()
##g# enerator channel
writer.add_channel(n_masked.axes, "ch_masked", masked=True)
writer.add_process(
    divideHists(n_masked, lumi_scaling), "Zmumu pass gen", "ch_masked", signal=False
)


writer.add_channel(
    reco_dtdt_data.axes, "ch_dtdt"
)  ### i have implicity selected a muon, this may be bad later
writer.add_data(reco_dtdt_data, "ch_dtdt")
writer.add_process(h2_first, "Zmumu pass gen", "ch_dtdt", signal=False)

writer.add_channel(reco_dtst_data.axes, "ch_dtst")
writer.add_data(reco_dtst_data, "ch_dtst")
writer.add_process(h1_second, "Zmumu pass gen", "ch_dtst", signal=False)

writer.add_channel(reco_stst_data.axes, "ch_stst")
writer.add_data(reco_stst_data, "ch_stst")
writer.add_process(h0_second, "Zmumu pass gen", "ch_stst", signal=False)


### adding axes as appropriate to make everything 4 dimensional
dtdt_prpg = expand_hist_by_duplicate_axis(dtdt_prpg, "time", "gen_time")
dtst_prpg = expand_hist_by_duplicate_axis(dtst_prpg, "time", "gen_time")
stst_prpg = expand_hist_by_duplicate_axis(stst_prpg, "time", "gen_time")

h2_first_var_lead = expand_hist_by_duplicate_axes(
    h2_first, ["time", "pt_lead", "eta_lead"], ["gen_time", "pt_prime", "eta_prime"]
)
h1_second_var = expand_hist_by_duplicate_axes(
    h1_second,
    ["time", "pt_sublead", "eta_sublead"],
    ["gen_time", "pt_prime", "eta_prime"],
)
h0_second_var = expand_hist_by_duplicate_axes(
    h0_second,
    ["time", "pt_sublead", "eta_sublead"],
    ["gen_time", "pt_prime", "eta_prime"],
)

pass_gen_expanded = expand_hist_by_duplicate_axes(pass_gen, ["time"], ["gen_time"])


N_eff = True
ID_eff = True
Trig_eff = True

if N_eff:
    dtdt_n = np.zeros([nbins_pt_subleading, nbins_eta_leading, nbins_time])
    dtst_n = np.zeros([nbins_pt_subleading, nbins_eta_leading, nbins_time])
    stst_n = np.zeros([nbins_pt_subleading, nbins_eta_leading, nbins_time])

if ID_eff:
    dtdt_id = np.zeros([nbins_pt_subleading, nbins_eta_leading, nbins_time])
    dtst_id = np.zeros([nbins_pt_subleading, nbins_eta_leading, nbins_time])
    stst_id = np.zeros([nbins_pt_subleading, nbins_eta_leading, nbins_time])

if Trig_eff:
    dtdt_trig = np.zeros([nbins_pt_subleading, nbins_eta_leading, nbins_time])
    dtst_trig = np.zeros([nbins_pt_subleading, nbins_eta_leading, nbins_time])
    stst_trig = np.zeros([nbins_pt_subleading, nbins_eta_leading, nbins_time])


### so at this point i have already selected the mass bin, need to iterate over pt, eta, time
for i in range(nbins_pt_subleading):  # just select two pt bins in the center
    for j in range(nbins_eta_leading):  # eta
        for k in range(nbins_time):  #  time
            var_size = 0.1

            if i > 1:  ## redo the naming convention
                eps_1 = dtdt_prpg[
                    {"gen_time": k, "pt_sublead": i - 2, "eta_sublead": j}
                ]
                eps_2 = h2_first_var_lead[
                    {"gen_time": k, "pt_prime": i - 2, "eta_prime": j}
                ]
                pass1_pass2 = addHists(
                    h2_first, addHists(eps_2 * var_size, eps_1 * var_size)
                )

                proj_2 = h2_first.project("pt_lead", "eta_lead")
                ### to allow eps to vary, i think this stays for all
                if N_eff:
                    writer.add_systematic(
                        pass1_pass2,
                        f"n_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_dtdt",
                        constrained=False,
                        groups=["nz"],
                    )
                    dtdt_n[i, j, k] = (
                        divideHists(
                            pass1_pass2.project("time"), h2_first.project("time")
                        ).values()
                    )[k]
                if Trig_eff:
                    writer.add_systematic(
                        pass1_pass2,
                        f"hlt_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_dtdt",
                        constrained=False,
                        groups=["eff_2"],
                    )
                    dtdt_trig[i, j, k] = (
                        divideHists(
                            pass1_pass2.project("time"), h2_first.project("time")
                        ).values()
                    )[k]
                if ID_eff:
                    writer.add_systematic(
                        pass1_pass2,
                        f"id_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_dtdt",
                        constrained=False,
                        groups=["eff_1"],
                    )
                    dtdt_id[i, j, k] = (
                        divideHists(
                            pass1_pass2.project("time"), h2_first.project("time")
                        ).values()
                    )[k]

            #### for double tight single trigger
            eps_2 = h1_second_var[{"gen_time": k, "pt_prime": i, "eta_prime": j}]
            proj_1 = h1_second.project("pt_sublead", "eta_sublead")

            if i > 1:

                eps_1 = dtst_prpg[{"gen_time": k, "pt_lead": i - 2, "eta_lead": j}]

                pass1_fail2 = addHists(eps_2.copy() * (-var_size), eps_1 * var_size)
                pass1_pass2 = addHists(eps_2.copy() * var_size, eps_1 * var_size)
                pass1_pass2_id = addHists(2 * pass1_pass2, h1_second)

                pass1_pass2 = addHists(pass1_pass2, h1_second)
                pass1_fail2 = addHists(2 * pass1_fail2, h1_second)

                if N_eff:
                    writer.add_systematic(
                        pass1_pass2,
                        f"n_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_dtst",
                        constrained=False,
                        groups=["nz"],
                    )
                    dtst_n[i, j, k] = (
                        divideHists(
                            pass1_pass2.project("time"), h1_second.project("time")
                        ).values()
                    )[k]

                if Trig_eff:
                    writer.add_systematic(
                        pass1_fail2,
                        f"hlt_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_dtst",
                        constrained=False,
                        groups=["eff_2"],
                    )
                    dtst_trig[i, j, k] = (
                        divideHists(
                            pass1_fail2.project("time"), h1_second.project("time")
                        ).values()
                    )[k]

            if ID_eff:
                if i > 1:
                    writer.add_systematic(
                        pass1_pass2_id,
                        f"id_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_dtst",
                        constrained=False,
                        groups=["eff_1"],
                    )
                    dtst_id[i, j, k] = (
                        divideHists(
                            pass1_pass2_id.project("time"), h1_second.project("time")
                        ).values()
                    )[k]

                else:
                    pass2 = addHists(eps_2 * var_size, h1_second)
                    writer.add_systematic(
                        pass2,
                        f"id_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_dtst",
                        constrained=False,
                        groups=["eff_1"],
                    )
                    dtst_id[i, j, k] = (
                        divideHists(
                            pass2.project("time"), h1_second.project("time")
                        ).values()
                    )[k]

            #### for single tight single trigger

            eps_2 = h0_second_var[{"gen_time": k, "pt_prime": i, "eta_prime": j}]
            proj_0 = h0_second.project("pt_sublead", "eta_sublead")

            if i > 1:
                eps_1 = stst_prpg[{"gen_time": k, "pt_lead": i - 2, "eta_lead": j}]

                pass1 = addHists(eps_1 * var_size, h0_second.copy())
                pass1_fail2 = addHists(eps_2.copy() * (-var_size), eps_1 * var_size)
                pass1_pass2 = addHists(eps_2.copy() * var_size, eps_1 * var_size)

                pass1_pass2 = addHists(pass1_pass2, h0_second)

                pass1_fail2 = addHists(2 * pass1_fail2, h0_second)
                if N_eff:
                    writer.add_systematic(
                        pass1_pass2,
                        f"n_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_stst",
                        constrained=False,
                        groups=["nz"],
                    )

                    stst_n[i, j, k] = (
                        divideHists(
                            pass1_pass2.project("time"), h0_second.project("time")
                        ).values()
                    )[k]
                if Trig_eff:
                    writer.add_systematic(
                        pass1_fail2,
                        f"hlt_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_stst",
                        constrained=False,
                        groups=["eff_2"],
                    )
                    stst_trig[i, j, k] = (
                        divideHists(
                            pass1_fail2.project("time"), h0_second.project("time")
                        ).values()
                    )[k]

            if ID_eff:
                if i > 1:

                    writer.add_systematic(
                        pass1_fail2,
                        f"id_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_stst",
                        constrained=False,
                        groups=["eff_1"],
                    )
                    stst_id[i, j, k] = (
                        divideHists(
                            pass1_fail2.project("time"), h0_second.project("time")
                        ).values()
                    )[k]
                else:
                    fail2 = addHists(eps_2 * (-var_size), h0_second)

                    writer.add_systematic(
                        fail2,
                        f"id_pt{i}_eta{j}_time{k}",
                        "Zmumu pass gen",
                        "ch_stst",
                        constrained=False,
                        groups=["eff_1"],
                    )
                    stst_id[i, j, k] = (
                        divideHists(
                            fail2.project("time"), h0_second.project("time")
                        ).values()
                    )[k]

            # # # for masked channel --- may need to change bc this has a different number of dimensions

            if i > 1:
                v_masked = pass_gen_expanded[
                    {"gen_time": k, "pt_sublead": i, "eta_sublead": j}
                ]
                var_masked = addHists(v_masked * var_size, n_masked)
                cross_section_masked = divideHists(var_masked, lumi_scaling)
                writer.add_systematic(
                    cross_section_masked,
                    f"n_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_masked",
                    constrained=False,
                    groups=["nz"],
                )


###### NONE OF THIS IS MASS DEPENDENT
num_etaphi = len(dtdt_prpg_H_stat.project("etaPhiRegion").values())
for i in range(num_etaphi):
    prpg_stat = [
        dtdt_prpg_H_stat[
            {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
        ],
        dtst_prpg_H_stat[
            {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
        ],
        stst_prpg_H_stat[
            {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
        ],
        dtdt_prpg_BG_stat[
            {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
        ],
        dtst_prpg_BG_stat[
            {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
        ],
        stst_prpg_BG_stat[
            {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
        ],
    ]

    eta_phi_systematic(
        writer,
        prpg_stat,
        time_hists,
        lumi_scaling,
        lumi_hists,
        weightsum,
        cross_sec,
        i,
    )

writer.add_systematic(
    dtdt_prpg_prefiring_syst.project("time", "pt_lead", "eta_lead"),
    f"prefiring_syst",
    "Zmumu pass gen",
    "ch_dtdt",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    dtst_prpg_prefiring_syst.project("time", "pt_sublead", "eta_sublead"),
    f"prefiring_syst",
    "Zmumu pass gen",
    "ch_dtst",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    stst_prpg_prefiring_syst.project("time", "pt_sublead", "eta_sublead"),
    f"prefiring_syst",
    "Zmumu pass gen",
    "ch_stst",
    constrained=True,
    groups=["prefiring_syst"],
)

# background_syst_names = [
#     "ZmumuPostVFP",
#     "Top",
#     "Diboson",
#     "GGToLLPostVFP",
#     "QCDmuEnrichPt15PostVFP",
#     "WplusmunuPostVFP",
#     "QGToDYQTo2LPostVFP",
#     "QGToWQToLNuPostVFP",
# ]
# background_proc = [
#     "Zmumu fail gen",
#     "Top",
#     "Diboson",
#     "GG",
#     "QCD",
#     "W",
#     "QG_2L",
#     "QG_Lnu",
# ]

background_syst_names = [
    "ZmumuPostVFP",
    "Top",
    "Diboson",
    "GGToLLPostVFP",
    # "QCDmuEnrichPt15PostVFP",
    # "WplusmunuPostVFP",
    # "QGToDYQTo2LPostVFP",
    # "QGToWQToLNuPostVFP",
]
background_proc = [
    "Zmumu fail gen",
    "Top",
    "Diboson",
    "GG",
    # "QCD",
    # "W",
    # "QG_2L",
    # "QG_Lnu",
]


for i in range(len(background_syst_names)):
    proc_name = background_proc[i]
    if proc_name == "Zmumu fail gen":
        fgen = True
    else:
        fgen = False
    print("proc_name: %s" % proc_name)
    background_syst(
        writer,
        results,
        background_syst_names[i],
        time_proj_hlt_all,
        time_proj_low_all,
        lumi_scaling,
        proc_name,
        f"bkg_{proc_name}",
        fail_gen=fgen,
    )

## PCC cross detector
luminometer_syst(
    writer, "pcc", dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc, "stability"
)
## HFOC cross detector
luminometer_syst(
    writer, "hfoc", dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc, "stability"
)

#### RAMSES cross detector
luminometer_syst(
    writer, "ramses", dtdt_prpg_ramses, dtst_prpg_ramses, stst_prpg_ramses, "stability"
)


#### HFOC linearity
luminometer_syst(
    writer,
    "hfoc",
    dtdt_prpg_sbil_hfoc,
    dtst_prpg_sbil_hfoc,
    stst_prpg_sbil_hfoc,
    "linearity",
)

#### RAMSES linearity
luminometer_syst(
    writer,
    "ramses",
    dtdt_prpg_sbil_ramses,
    dtst_prpg_sbil_ramses,
    stst_prpg_sbil_ramses,
    "linearity",
)

writer.write(outfolder="./", outfilename="liv")

dict_out = {}

if N_eff:
    dict_out["dtdt_n"] = dtdt_n
    dict_out["dtst_n"] = dtst_n
    dict_out["stst_n"] = stst_n

if ID_eff:
    dict_out["dtdt_id"] = dtdt_id
    dict_out["dtst_id"] = dtst_id
    dict_out["stst_id"] = stst_id

if Trig_eff:
    dict_out["dtdt_trig"] = dtdt_trig
    dict_out["dtst_trig"] = dtst_trig
    dict_out["stst_trig"] = stst_trig

# for name in [dtdt_n, dtdt_trig, dtdt_id,  dtst_n, dtdt_id, dtst_n, dtst_trig, dtst_id, stst_n, stst_trig]
# dict_out = {'dtdt_n': dtdt_n, 'dtdt_trig': dtdt_trig, 'dtdt_id': dtdt_id, 'dtst_n': dtst_n, 'dtst_trig': dtst_trig, 'dtst_id': dtst_id, 'stst_n': stst_n, 'stst_trig': stst_trig, 'stst_id': stst_id}

with open("efficiency_dict.pkl", "wb") as f:
    pickle.dump(dict_out, f)
