import argparse
import pickle

import h5py
from uncertainty_tools import (
    all_mc_corrections,
    get_era_vals,
    get_h0var,
    get_h0var_low,
    get_h1var,
    get_h1var_low,
    get_h2var,
    get_mc_lumis,
    make_ones_hist,
)

from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    divideHists,
    expand_hist_by_duplicate_axis,
    multiplyHists,
    scaleHist,
)

parser = argparse.ArgumentParser()
args = parser.parse_args()

slope_ramses = 0.0006
slope_hfoc = 0.0007
mass_bin = 9
var_size = 0.01


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
    "time", "mll", "gen_mll", "pt_probe", "eta_probe", "pt_tag", "eta_tag"
)
time_proj_hlt_all = time_proj_hlt_all.project(
    "time", "mll", "gen_mll", "pt_probe", "eta_probe", "pt_tag", "eta_tag"
)

time_proj_low = time_proj_low_all[{"mll": mass_bin, "gen_mll": mass_bin}]
time_proj_hlt = time_proj_hlt_all[{"mll": mass_bin, "gen_mll": mass_bin}]


time_proj_true = time_proj_low_all.project(
    "time", "pt_probe", "eta_probe", "pt_tag", "eta_tag"
)

### pass reco, pass generator

dtdt_prpg = MC_Zmumu["dtdt_prpg"].get()
dtst_prpg = MC_Zmumu["dtst_prpg"].get()
stst_prpg = MC_Zmumu["stst_prpg"].get()
weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

dtdt_prfg = MC_Zmumu["dtdt_prfg"].get()
dtst_prfg = MC_Zmumu["dtst_prfg"].get()
stst_prfg = MC_Zmumu["stst_prfg"].get()

tight = MC_Zmumu["tight"].get()
loose = MC_Zmumu["loose"].get()
trigger = MC_Zmumu["trigger"].get()


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
nbins_pt_taging = len(reco_dtst_data.axes["pt_tag"])
nbins_pt_probeing = len(reco_dtst_data.axes["pt_probe"])

nbins_eta_probeing = len(reco_dtdt_data.axes["eta_probe"])

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

n_masked = pass_gen.project("time", "pt_tag", "eta_tag")

h2 = dtdt_prpg.project("time", "pt_probe", "eta_probe")
h1 = dtst_prpg.project("time", "pt_probe", "eta_probe")
h0 = stst_prpg.project("time", "pt_probe", "eta_probe")

# h2_alt = dtdt_prpg.project("time", "pt_tag", "eta_tag")
# h1_alt = dtst_prpg.project("time", "pt_probe", "eta_probe")
# h0_alt = stst_prpg.project("time", "pt_probe", "eta_probe")

# eps_hlt_true_sublead = divideHists(h2_alt*2, addHists(h1, 2*h2_alt))
# eps_hlt_true_lead = divideHists(h2*2, addHists(h1_alt, h2*2))

# eps_id_true_lead = divideHists(addHists(h1_alt, h2*2), addHists(h1_alt, addHists(h0_alt, h2)))
# eps_id_true_sublead = divideHists(addHists(h1, h2), addHists(h1, addHists(h0, h2_alt)))

# eps_id_avg = scaleHist(addHists(eps_id_true_lead, eps_id_true_sublead), 1/2)
# # eps_hlt_avg = scaleHist(addHists(eps_hlt_true_lead, eps_hlt_true_sublead), 1/2)
trigger = trigger[{"mll": mass_bin, "gen_mll": mass_bin}]
tight = tight[{"mll": mass_bin, "gen_mll": mass_bin}]
loose = loose[{"mll": mass_bin, "gen_mll": mass_bin}]

tight = all_mc_corrections(tight, time_proj_true, lumi_scaling, weightsum, cross_sec)
loose = all_mc_corrections(loose, time_proj_true, lumi_scaling, weightsum, cross_sec)
trigger = all_mc_corrections(
    trigger, time_proj_true, lumi_scaling, weightsum, cross_sec
)

trigger_proj = trigger.project("time", "pt_probe", "eta_probe")
tight_proj = tight.project("time", "pt_probe", "eta_probe")
loose_proj = loose.project("time", "pt_probe", "eta_probe")

eps_hlt_true = divideHists(trigger_proj, tight_proj)
eps_id_true = divideHists(tight_proj, loose_proj)


def extract_diagonal(hist_in):
    ## assumes hist_in comes as time, pt1, eta1, pt2, eta2
    values = hist_in.values()  ## assumes this comeas as
    hist_out = hist_in.copy().project("time", "pt_probe", "eta_probe")
    hist_out_values = hist_out.values()
    for k in range(hist_in.shape[0]):  # should be time
        for i in range(hist_in.shape[1]):  ## pt
            for j in range(hist_in.shape[2]):  # eta
                hist_out_values[k, i, j] = values[k, i, j, i, j]
    return hist_out


h2 = trigger.project("time", "pt_probe", "eta_probe")

h1 = addHists(
    tight.project("time", "pt_probe", "eta_probe"),
    scaleHist(trigger.project("time", "pt_probe", "eta_probe"), -1),
)
h0 = addHists(
    loose.project("time", "pt_probe", "eta_probe"),
    scaleHist(tight.project("time", "pt_probe", "eta_probe"), -1),
)


# h2 = extract_diagonal(trigger)
# h1 = extract_diagonal(tight)
# h0 = extract_diagonal(loose)


efficiency_ones = make_ones_hist(h1)
# generate histogram of ones

eps_id_prime = 1.01
eps_hlt_prime = 1.01

### greater than 25 GeV
## e2 = 2*h2/(h1 + 2*h1)

eps_hlt = addHists(h1, scaleHist(h2, 2))
eps_hlt_high = scaleHist(divideHists(h2, eps_hlt), 2)

##e1 = h1/(h0*(1-e2) + h1)
eps_id = addHists(efficiency_ones, scaleHist(eps_hlt_high, -1))
eps_id = multiplyHists(h0, eps_id)
eps_id = addHists(eps_id, h1)
eps_id_high = divideHists(h1, eps_id)

heff_high = divideHists(h2, multiplyHists(eps_hlt_high, eps_hlt_high))
heff_high = divideHists(heff_high, multiplyHists(eps_id_high, eps_id_high))

eps_id_var_high = scaleHist(eps_id_high, eps_id_prime)
eps_hlt_var_high = scaleHist(eps_hlt_high, eps_hlt_prime)

h0var_id_high = get_h0var(eps_id_var_high, eps_hlt_high, heff_high, efficiency_ones)
h0var_hlt_high = get_h0var(eps_id_high, eps_hlt_var_high, heff_high, efficiency_ones)
h1var_id_high = get_h1var(eps_id_var_high, eps_hlt_high, heff_high, efficiency_ones)
h1var_hlt_high = get_h1var(eps_id_high, eps_hlt_var_high, heff_high, efficiency_ones)


h2var_id_high = get_h2var(eps_id_var_high, eps_hlt_high, heff_high)
h2var_hlt_high = get_h2var(eps_id_high, eps_hlt_var_high, heff_high)


##### below 25 GeV
eps_hlt_low = eps_hlt_high

##e1 = h1/(h0 + h1)
eps_id = addHists(h0, scaleHist(h1, 1))
eps_id = divideHists(h1, eps_id)
eps_id_low = scaleHist(eps_id, 1)

heff_low = divideHists(h0, addHists(efficiency_ones, scaleHist(eps_id_low, -1)))
heff_low = divideHists(heff_low, multiplyHists(eps_id_low, eps_hlt_high))

eps_id_var_low = scaleHist(eps_id_low.copy(), eps_id_prime)
eps_hlt_var_low = scaleHist(eps_hlt_low.copy(), eps_hlt_prime)

h0var_id_low = get_h0var_low(eps_id_var_low, eps_hlt_low, heff_low, efficiency_ones)
h1var_id_low = get_h1var_low(eps_id_var_low, eps_hlt_low, heff_low, efficiency_ones)
h2var_id_low = get_h2var(eps_id_var_low, eps_hlt_low, heff_low)


# h2var_id_low = remove_low_bins(h2var_id_low) ## this might already be projected on the wrong direction which isnt great

h2_data = reco_dtdt_data.project("time", "pt_probe", "eta_probe")
h1_data = reco_dtst_data.project("time", "pt_tag", "eta_tag")
h0_data = reco_stst_data.project("time", "pt_tag", "eta_tag")

eps_hlt_data = addHists(h1_data, scaleHist(h2_data, 2))
eps_hlt_data = divideHists(h2_data, eps_hlt_data)
eps_hlt_high_data = scaleHist(eps_hlt_data, 2)

eps_id_data = addHists(efficiency_ones, scaleHist(eps_hlt_high_data, -1))
eps_id_data = multiplyHists(h0_data, eps_id_data)
eps_id_data = addHists(eps_id_data, h1_data)
eps_id_high_data = divideHists(h1_data, eps_id_data)


eps_hlt_low = scaleHist(h2_data, 0)
eps_hlt_low_data = eps_hlt_high_data

##e1 = h1/(h0 + h1)
eps_id_data = addHists(h0_data, scaleHist(h1_data, 1))
eps_id_data = divideHists(h1_data, eps_id_data)
eps_id_low_data = scaleHist(eps_id_data, 1)

efficiencies = {
    # "h2_id_high": divideHists(h2var_id_high, heff_high).values(),
    # "h2_id_low": divideHists(h2var_id_low, heff_low).values(),
    # "h1_id_high": divideHists(h1var_id_high, heff_high).values(),
    # "h1_id_low": divideHists(h1var_id_low, heff_low).values(),
    # "h0_id_high": divideHists(h0var_id_high, heff_high).values(),
    # "h0_id_low": divideHists(h0var_id_low, heff_low).values(),
    # "h2_hlt_high": divideHists(h2var_hlt_high, heff_high).values(),
    # "h1_hlt_high": divideHists(h1var_hlt_high, heff_high).values(),
    # "h0_hlt_high": divideHists(h0var_hlt_high, heff_high).values(),
    "epsilon_hlt_high": eps_hlt_high.values(),
    "epsilon_id_high": eps_id_high.values(),
    # "epsilon_hlt_low": eps_hlt_low.values(),
    # "epsilon_id_low": eps_id_low.values(),
    "eps_hlt_true_neg": eps_hlt_true.values(),
    "eps_id_true_neg": eps_id_true.values(),
    # "epsilon_hlt_high_data": eps_hlt_high_data.values(),
    # "epsilon_id_high_data": eps_id_high_data.values(),
    # "epsilon_hlt_low_data": eps_hlt_low_data.values(),
    # "epsilon_id_low_data": eps_id_low_data.values(),
}

with open("efficiency_values.pkl", "wb") as f:
    pickle.dump(efficiencies, f)


"""
#### i think i will need to mix this too but it doesn't affect the fit

reco_dtdt_data = reco_dtdt_data[{"mll": mass_bin}].project(
    "time", "pt_probe", "eta_probe"
)
reco_dtst_data = reco_dtst_data[{"mll": mass_bin}].project(
    "time", "pt_tag", "eta_tag"
)
reco_stst_data = reco_stst_data[{"mll": mass_bin}].project(
    "time", "pt_tag", "eta_tag"
)


## create the tensor
writer = tensorwriter.TensorWriter()
##g# enerator channel
writer.add_channel(n_masked.axes, "ch_masked", masked=True)
writer.add_process(
    divideHists(n_masked, lumi_scaling), "Zmumu pass gen", "ch_masked", signal=False
)

reco_dtdt_data = remove_low_bins(reco_dtdt_data)
h2 = remove_low_bins(h2)
dtdt_prpg = remove_low_bins(dtdt_prpg)

h2var_id_low = remove_low_bins(h2var_id_low)


writer.add_channel(
    reco_dtdt_data.axes, "ch_dtdt"
)  ### i have implicity selected a muon, this may be bad later
writer.add_data(reco_dtdt_data, "ch_dtdt")
writer.add_process(h2, "Zmumu pass gen", "ch_dtdt", signal=False)

writer.add_channel(reco_dtst_data.axes, "ch_dtst")
writer.add_data(reco_dtst_data, "ch_dtst")
writer.add_process(h1, "Zmumu pass gen", "ch_dtst", signal=False)

writer.add_channel(reco_stst_data.axes, "ch_stst")
writer.add_data(reco_stst_data, "ch_stst")
writer.add_process(h0, "Zmumu pass gen", "ch_stst", signal=False)
# pdb.set_trace()

### adding axes as appropriate to make everything 4 dimensional
dtdt_prpg = expand_hist_by_duplicate_axis(dtdt_prpg, "time", "gen_time")
dtst_prpg = expand_hist_by_duplicate_axis(dtst_prpg, "time", "gen_time")
stst_prpg = expand_hist_by_duplicate_axis(stst_prpg, "time", "gen_time")


pass_gen_expanded = expand_hist_by_duplicate_axes(pass_gen, ["time"], ["gen_time"])
# for e in range(nbins_eta_probeing):
#     for t in range(nbins_time):
#         print(h0[{"eta_tag": e, "time": t}])


### so at this point i have already selected the mass bin, need to iterate over pt, eta, time
for i in range(2, nbins_pt_taging - 1):  # just select two pt bins in the center
    for j in range(nbins_eta_probeing):  # eta
        for k in range(nbins_time):  #  time

            if i > 0:  ## we only have 1 bin beneath 25 GeV

                ### be more consistent about ordering of time and mll
                ### fitting for the number of events
                v2 = dtdt_prpg[
                    {"pt_tag": i - 1, "eta_tag": j, "gen_time": k}
                ]  ## equivalent to n2
                var2 = addHists(v2 * var_size, h2)

                writer.add_systematic(
                    var2,
                    f"n_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtdt",
                    constrained=False,
                    groups=["nz"],
                )

            v1 = dtst_prpg[
                {"pt_probe": i, "eta_probe": j, "gen_time": k}
            ]  ## equivalent to n1
            var1 = addHists(v1 * var_size, h1)
            writer.add_systematic(
                var1,
                f"n_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtst",
                constrained=False,
                groups=["nz"],
            )
            v0 = stst_prpg[
                {"pt_probe": i, "eta_probe": j, "gen_time": k}
            ]  ## equivalent to n1
            var0 = addHists(v0 * var_size, h0)
            writer.add_systematic(
                var0,
                f"n_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_stst",
                constrained=False,
                groups=["nz"],
            )
            # for masked channel
            v_masked = pass_gen_expanded[{"pt_probe": i, "eta_probe": j, "gen_time": k}]
            var_masked = addHists(v_masked * var_size, n_masked)
            cross_section_masked = divideHists(var_masked, lumi_scaling)

            # pdb.set_trace()
            writer.add_systematic(
                cross_section_masked,
                f"n_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_masked",
                constrained=False,
                groups=["nz"],
            )

            if i > 0:
                # ## efficiency
                h1var_id_primed = get_eff_hist(
                    h1var_id_high, h1, i, j, k, "pt_tag", "eta_tag"
                )
                h0var_id_primed = get_eff_hist(
                    h0var_id_high, h0, i, j, k, "pt_tag", "eta_tag"
                )

                h1var_hlt_primed = get_eff_hist(
                    h1var_hlt_high, h1, i, j, k, "pt_tag", "eta_tag"
                )
                h0var_hlt_primed = get_eff_hist(
                    h0var_hlt_high, h0, i, j, k, "pt_tag", "eta_tag"
                )

                # #     ### ID EFFICIENCY
                h2var_id_primed = get_eff_hist(
                    h2var_id_high, h2, i - 1, j, k, "pt_probe", "eta_probe"
                )
                h2var_hlt_primed = get_eff_hist(
                    h2var_hlt_high, h2, i - 1, j, k, "pt_probe", "eta_probe"
                )
                #     pdb.set_trace()

                writer.add_systematic(
                    h2var_id_primed,
                    f"id_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtdt",
                    constrained=False,
                    groups=["eff_id"],
                )
                writer.add_systematic(
                    h2var_hlt_primed,
                    f"hlt_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtdt",
                    constrained=False,
                    groups=["eff_trig"],
                )

                writer.add_systematic(
                    h1var_hlt_primed,
                    f"hlt_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtst",
                    constrained=False,
                    groups=["eff_trig"],
                )
                writer.add_systematic(
                    h0var_hlt_primed,
                    f"hlt_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_stst",
                    constrained=False,
                    groups=["eff_trig"],
                )

            else:
                h1var_id_primed = get_eff_hist(
                    h1var_id_low, h1, i, j, k, "pt_tag", "eta_tag"
                )
                h0var_id_primed = get_eff_hist(
                    h0var_id_low, h0, i, j, k, "pt_tag", "eta_tag"
                )
                #     # ### ID EFFICIENCY, these two used to be i-2
                h2var_id_primed = get_eff_hist(
                    remove_low_bins(h2var_id_low.copy()),
                    h2,
                    i,
                    j,
                    k,
                    "pt_probe",
                    "eta_probe",
                )

            # ### order of these is time, pt, eta

            writer.add_systematic(
                h1var_id_primed,
                f"id_prime_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtst",
                constrained=False,
                groups=["eff_id"],
            )

            writer.add_systematic(
                h0var_id_primed,
                f"id_prime_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_stst",
                constrained=False,
                groups=["eff_id"],
            )


# ##### NONE OF THIS IS MASS DEPENDENT ## may be linked to statistical uncertainty becuase the eta region? do i still need this or am i double counting
# lowers stability uncetainty increase linearity uncertainty.
# num_etaphi = len(dtdt_prpg_H_stat.project("etaPhiRegion").values())
# for i in range(num_etaphi):
#     prpg_stat = [
#         dtdt_prpg_H_stat[
#             {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
#         ],
#         dtst_prpg_H_stat[
#             {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
#         ],
#         stst_prpg_H_stat[
#             {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
#         ],
#         dtdt_prpg_BG_stat[
#             {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
#         ],
#         dtst_prpg_BG_stat[
#             {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
#         ],
#         stst_prpg_BG_stat[
#             {"etaPhiRegion": i, "downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}
#         ],
#     ]

#     eta_phi_systematic(
#         writer,
#         prpg_stat,
#         time_hists,
#         lumi_scaling,
#         lumi_hists,
#         weightsum,
#         cross_sec,
#         i,
#     )


### these slightly increase the statistical uncertainty but dont contribute to the stability/linearity
# dtdt_prpg_prefiring_syst = dtdt_prpg_prefiring_syst.project("time", "pt_probe", "eta_probe")
# writer.add_systematic(
#     remove_low_bins(dtdt_prpg_prefiring_syst),
#     f"prefiring_syst",
#     "Zmumu pass gen",
#     "ch_dtdt",
#     constrained=True,
#     groups=["prefiring_syst"],
# )
# writer.add_systematic(
#     dtst_prpg_prefiring_syst.project("time", "pt_tag", "eta_tag"),
#     f"prefiring_syst",
#     "Zmumu pass gen",
#     "ch_dtst",
#     constrained=True,
#     groups=["prefiring_syst"],
# )
# writer.add_systematic(
#     stst_prpg_prefiring_syst.project("time", "pt_tag", "eta_tag"),
#     f"prefiring_syst",
#     "Zmumu pass gen",
#     "ch_stst",
#     constrained=True,
#     groups=["prefiring_syst"],
# )

# # background_syst_names = [
# #     "ZmumuPostVFP",
# #     "Top",
# #     "Diboson",
# #     "GGToLLPostVFP",
# #     "QCDmuEnrichPt15PostVFP",
# #     "WplusmunuPostVFP",
# #     "QGToDYQTo2LPostVFP",
# #     "QGToWQToLNuPostVFP",
# # ]
# # background_proc = [
# #     "Zmumu fail gen",
# #     "Top",
# #     "Diboson",
# #     "GG",
# #     "QCD",
# #     "W",
# #     "QG_2L",
# #     "QG_Lnu",
# # ]

# background_syst_names = [
#     "ZmumuPostVFP",
#     "Top",
#     "Diboson",
#     "GGToLLPostVFP",
#     # "QCDmuEnrichPt15PostVFP",
#     # "WplusmunuPostVFP",
#     # "QGToDYQTo2LPostVFP",
#     # "QGToWQToLNuPostVFP",
# ]
# background_proc = [
#     "Zmumu fail gen",
#     "Top",
#     "Diboson",
#     "GG",
#     # "QCD",
#     # "W",
#     # "QG_2L",
#     # "QG_Lnu",
# ]


# for i in range(len(background_syst_names)):
#     proc_name = background_proc[i]
#     if proc_name == "Zmumu fail gen":
#         fgen = True
#     else:
#         fgen = False
#     print("proc_name: %s" % proc_name)
#     background_syst(
#         writer,
#         results,
#         background_syst_names[i],
#         time_proj_hlt_all,
#         time_proj_low_all,
#         lumi_scaling,
#         proc_name,
#         f"bkg_{proc_name}",
#         fail_gen=fgen,
#     )


## PCC cross detector
luminometer_syst(
    writer, "pcc", dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc, "stability"
)
# ## HFOC cross detector
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
"""
