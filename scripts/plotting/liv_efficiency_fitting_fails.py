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
    remove_low_bins,
)

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    divideHists,
    expand_hist_by_duplicate_axes,
    multiplyHists,
    scaleHist,
)

parser = argparse.ArgumentParser()
args = parser.parse_args()

slope_ramses = 0.0006
slope_hfoc = 0.0007
# mass_bin = 9
mass_bin = 2
var_size = 0.01

######################################################################33
# DATA IMPORTS #

file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

data_output = results["dataPostVFP"]["output"]
lumi_output = results["dataPostVFP"]["lumi_outout"]
MC_Zmumu = results["ZmumuPostVFP"]["output"]

dtdt_data = data_output["time_mll"].get()
dtst_data = data_output["time_dtst"].get()
stst_data = data_output["time_stst"].get()
time_proj_low_all = data_output[
    "time_proj"
].get()  #### HONESTLY NOT SURE WHY WE HAVE THIS STILL, MAY NEED TO BE DELETED
time_proj_hlt_all = data_output["time_proj"].get()


### PASS GENERATOR CUTOFFS
### THESE ARE NOT MUTUALLY EXCLUSIVE. MAKE THEM EXCLUSIVE LATER IN THIS CODE. EVENTUALLY WILL SWITCH TO THEM BEING MUTUALLY EXCLUSIVE IN THE HISTMAKER

pass_gen = MC_Zmumu["pass_gen"].get()

weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

### FAIL GENERATOR CUTOFFS
dtdt_prfg = MC_Zmumu["dtdt_prfg"].get()  ## DON'T CURRENTLY USE THESE
dtst_prfg = MC_Zmumu["dtst_prfg"].get()
stst_prfg = MC_Zmumu["stst_prfg"].get()

### should loop over these instead of calling them explicitly
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


### DON'T QUITE REMEMBER WHAT I S
lumi_scaling = lumi_output["lumi_nom"].get()
lumi_scaling_h = lumi_output["lumi_pre"].get()
lumi_scaling_bg = lumi_output["lumi_post"].get()

### STABILITY
lumi_hfoc = lumi_output["lumi_hfoc"].get()
lumi_pcc = lumi_output["lumi_pcc"].get()
lumi_ramses = lumi_output["lumi_ramses"].get()

# NOT QUITE SURE WHAT THIS DIFFERENCE IS TBH
lumi_hfoc_nom = lumi_output["lumi_in_hfoc"].get()
lumi_pcc_nom = lumi_output["lumi_in_pcc"].get()
lumi_ramses_nom = lumi_output["lumi_in_ramses"].get()

## LINEARITY
sbil_pcc = lumi_output["sbil_pcc"].get()
count_pcc = lumi_output["count_pcc"].get()

#### A COUPLE FIXED QUANTITIES
nbins_mll = len(dtdt_prfg.axes["mll"])  ## don't currenyl use this
nbins_time = len(dtst_data.axes["time"])
nbins_pt = len(dtst_data.axes["pt_probe"])
nbins_eta = len(dtdt_data.axes["eta_probe"])

#############################################################33

# time_proj_hlt_all = expand_hist_by_duplicate_axis(time_proj_hlt_all, "mll", "gen_mll")
# time_proj_low_all = expand_hist_by_duplicate_axis(time_proj_low_all, "mll", "gen_mll")
# time_proj_hlt_all = time_proj_hlt_all.project("time", "mll", "gen_mll", "pt_probe", "eta_probe", "pt_tag", "eta_tag")
# time_proj_low_all = time_proj_low_all.project("time", "mll", "gen_mll", "pt_probe", "eta_probe", "pt_tag", "eta_tag")


# time_proj_low = time_proj_low_all[{"mll": mass_bin, "gen_mll": mass_bin}]
# time_proj_hlt = time_proj_hlt_all[{"mll": mass_bin, "gen_mll": mass_bin}]


# ### needs to come after removing the mass bins
# time_proj_true = time_proj_low_all.project(
#     "time", "pt_probe", "eta_probe", "pt_tag", "eta_tag"
# )

hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
hfoc_scaling = multiplyHists(hfoc_scaling, lumi_scaling)

pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
pcc_scaling = multiplyHists(pcc_scaling, lumi_scaling)

ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
ramses_scaling = multiplyHists(ramses_scaling, lumi_scaling)


### okay i don't want to do the mass selection here

# dtdt_prpg_H = dtdt_prpg_H[{"mll": mass_bin, "gen_mll": mass_bin}] ### instead of specifying mass bins here could i use : to use them all?
# dtst_prpg_H = dtst_prpg_H[{"mll": mass_bin, "gen_mll": mass_bin}]
# stst_prpg_H = stst_prpg_H[{"mll": mass_bin, "gen_mll": mass_bin}]
# dtdt_prpg_BG = dtdt_prpg_BG[{"mll": mass_bin, "gen_mll": mass_bin}]
# dtst_prpg_BG = dtst_prpg_BG[{"mll": mass_bin, "gen_mll": mass_bin}]
# stst_prpg_BG = stst_prpg_BG[{"mll": mass_bin, "gen_mll": mass_bin}]


### this just puts them in a list
prpg_all = [
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
]
# prpg_syst = [
#     dtdt_prpg_H_syst[{"downUpVar": 0}], #"mll": mass_bin, "gen_mll": mass_bin}],
#     dtst_prpg_H_syst[{"downUpVar": 0}], #"mll": mass_bin, "gen_mll": mass_bin}],
#     stst_prpg_H_syst[{"downUpVar": 0}], #"mll": mass_bin, "gen_mll": mass_bin}],
#     dtdt_prpg_BG_syst[{"downUpVar": 0}], #"mll": mass_bin, "gen_mll": mass_bin}],
#     dtst_prpg_BG_syst[{"downUpVar": 0}], #"mll": mass_bin, "gen_mll": mass_bin}],
#     stst_prpg_BG_syst[{"downUpVar": 0}], #"mll": mass_bin, "gen_mll": mass_bin}],
# ]

time_hists = [
    time_proj_hlt_all,
    time_proj_low_all,
]  ## do i currently differentiate between these two?
lumi_hists = [lumi_scaling_h, lumi_scaling_bg]

# dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc = get_mc_lumis(
#     prpg_all,
#     time_hists,
#     hfoc_scaling,
#     lumi_hists,
#     weightsum,
#     cross_sec,
# )

# dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc = get_mc_lumis(
#     prpg_all,
#     time_hists,
#     pcc_scaling,
#     lumi_hists,
#     weightsum,
#     cross_sec,
# )

# dtdt_prpg_ramses, dtst_prpg_ramses, stst_prpg_ramses = get_mc_lumis(
#     prpg_all,
#     time_hists,
#     ramses_scaling,
#     lumi_hists,
#     weightsum,
#     cross_sec,
# )

# avg_sbil_pcc = scaleHist(divideHists(sbil_pcc, count_pcc), 1e9)

# sbil_hfoc_fit = scaleHist(avg_sbil_pcc, slope_hfoc)

# sbil_ones = make_ones_hist(sbil_hfoc_fit)
# sbil_hfoc_fit = addHists(sbil_hfoc_fit, sbil_ones)
# sbil_hfoc_fit = multiplyHists(sbil_hfoc_fit, lumi_scaling)

# sbil_ramses_fit = scaleHist(avg_sbil_pcc, slope_ramses)
# sbil_ramses_fit = addHists(sbil_ramses_fit, sbil_ones)
# sbil_ramses_fit = multiplyHists(sbil_ramses_fit, lumi_scaling)

# dtdt_prpg_sbil_hfoc, dtst_prpg_sbil_hfoc, stst_prpg_sbil_hfoc = get_mc_lumis(
#     prpg_all,
#     time_hists,
#     sbil_hfoc_fit,
#     lumi_hists,
#     weightsum,
#     cross_sec,
# )

# dtdt_prpg_sbil_ramses, dtst_prpg_sbil_ramses, stst_prpg_sbil_ramses = get_mc_lumis(
#     prpg_all,
#     time_hists,
#     sbil_ramses_fit,
#     lumi_hists,
#     weightsum,
#     cross_sec,
# )

#### normal
dtdt_prpg, dtst_prpg, stst_prpg = get_mc_lumis(
    prpg_all,
    time_hists,
    lumi_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)

# (dtdt_prpg_prefiring_syst, dtst_prpg_prefiring_syst, stst_prpg_prefiring_syst) = (
#     get_mc_lumis(
#         prpg_syst,
#         time_hists,
#         lumi_scaling,
#         lumi_hists,
#         weightsum,
#         cross_sec,
#     )
# )

#### oh right this is why i didn't want to do this it takes FUCKING FOREVER to run the code using this binning

dtdt_m_center = dtdt_prpg[{"mll": mass_bin}]  # , "gen_mll": mass_bin}]
dtst_m_center = dtst_prpg[{"mll": mass_bin}]  # , "gen_mll": mass_bin}]
stst_m_center = stst_prpg[{"mll": mass_bin}]  # , "gen_mll": mass_bin}]

dtdt_3d = dtdt_m_center.project(
    "time", "pt_probe", "eta_probe"
)  ### this is good because it doesn't permanently modify the original
dtst_3d = dtst_m_center.project("time", "pt_probe", "eta_probe")
stst_3d = stst_m_center.project("time", "pt_probe", "eta_probe")


def make_mutually_exclusive(dtdt, dtst, stst):
    dtdt_ex = dtdt
    dtst_ex = addHists(dtst, scaleHist(dtdt, -1))
    stst_ex = addHists(stst, scaleHist(dtst, -1))
    return dtdt_ex, dtst_ex, stst_ex


dtdt_m_center, dtst_m_center, stst_m_center = make_mutually_exclusive(
    dtdt_m_center, dtst_m_center, stst_m_center
)


h2, h1, h0 = make_mutually_exclusive(dtdt_3d, dtst_3d, stst_3d)

eps_hlt_true = divideHists(dtdt_3d, dtst_3d)
eps_id_true = divideHists(dtst_3d, stst_3d)


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
eps_id = addHists(eps_id, h1)  ### weird little blip because of h1
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
eps_hlt_low = scaleHist(
    h2, 0
)  ### HERE I CHANGE IT, NOT SURE IF I SHOULD BE SETTING THIS TO 0

##e1 = h1/(h0 + h1)
eps_id = addHists(h0, scaleHist(h1, 1))
eps_id_low = divideHists(h1, eps_id)

heff_low = divideHists(h0, addHists(efficiency_ones, scaleHist(eps_id_low, -1)))
heff_low = divideHists(heff_low, eps_id_low)

eps_id_var_low = scaleHist(eps_id_low.copy(), eps_id_prime)
eps_hlt_var_low = scaleHist(eps_hlt_low.copy(), eps_hlt_prime)


h0var_id_low = get_h0var_low(eps_id_var_low, eps_hlt_low, heff_low, efficiency_ones)
h1var_id_low = get_h1var_low(eps_id_var_low, eps_hlt_low, heff_low, efficiency_ones)
h2var_id_low = get_h2var(eps_id_var_low, eps_hlt_low, heff_low)


#### this is a nice check but we don't technically neeeeed it

dtdt_data_3d = dtdt_data[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe")
dtst_data_3d = dtst_data[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe")
stst_data_3d = stst_data[{"mll": mass_bin}].project("time", "pt_probe", "eta_probe")

dtdt_data_3d_exc, dtst_data_3d_exc, stst_data_3d_exc = make_mutually_exclusive(
    dtdt_data_3d, dtst_data_3d, stst_data_3d
)


# h2_data = dtdt_data

# h1_data = addHists(
#     dtst_data,
#     scaleHist(dtdt_data, -1),
# )
# h0_data = addHists(
#     stst_data,
#     scaleHist(dtst_data, -1),
# )

# eps_hlt_data = addHists(h1_data, scaleHist(h2_data, 2))
# eps_hlt_data = divideHists(h2_data, eps_hlt_data)
# eps_hlt_high_data = scaleHist(eps_hlt_data, 2)

# eps_id_data = addHists(efficiency_ones, scaleHist(eps_hlt_high_data, -1))
# eps_id_data = multiplyHists(h0_data, eps_id_data)
# eps_id_data = addHists(eps_id_data, h1_data)
# eps_id_high_data = divideHists(h1_data, eps_id_data)

# # eps_hlt_low_data = eps_hlt_high_data
# eps_hlt_low_data = eps_hlt_low  #### SHOULD I BY DEFAULT BE SETTING THIS TO 0

# ##e1 = h1/(h0 + h1)
# eps_id_data = addHists(h0_data, scaleHist(h1_data, 1))
# eps_id_data = divideHists(h1_data, eps_id_data)
# eps_id_low_data = scaleHist(eps_id_data, 1)

# combined_eps_hlt_data = eps_hlt_high_data.values()
# combined_eps_hlt_data[:, :1, :] = eps_hlt_low_data.values()[:, :1, :]
# combined_eps_id_data = eps_id_high_data.values()
# combined_eps_id_data[:, :1, :] = eps_id_low_data.values()[:, :1, :]

combined_epsilon_hlt = eps_hlt_high.values()
combined_epsilon_hlt[:, :1, :] = eps_hlt_low.values()[:, :1, :]
combined_epsilon_id = eps_id_high.values()
combined_epsilon_id[:, :1, :] = eps_id_low.values()[:, :1, :]


efficiencies = {
    "epsilon_hlt_high": eps_hlt_high.values(),
    "epsilon_id_high": eps_id_high.values(),
    "epsilon_hlt_low": eps_hlt_low.values(),
    "epsilon_id_low": eps_id_low.values(),
    "eps_hlt_true": eps_hlt_true.values(),
    "eps_id_true": eps_id_true.values(),
    # "epsilon_hlt_high_data": eps_hlt_high_data.values(),
    # "epsilon_id_high_data": eps_id_high_data.values(),
    # "epsilon_hlt_low_data": eps_hlt_low_data.values(),
    # "epsilon_id_low_data": eps_id_low_data.values(),
    # "COMBINED_eps_hlt_data": combined_eps_hlt_data,
    # "COMBINED_eps_id_data": combined_eps_id_data,
    "COMBINED_epsilon_hlt": combined_epsilon_hlt,
    "COMBINED_epsilon_id": combined_epsilon_id,
}

with open("efficiency_values.pkl", "wb") as f:
    pickle.dump(efficiencies, f)


###################################################################333
#####  KNOW THIS IS CORRECT #####

# should rewrite the section above to not have redundant code. same with some sections in the histmaker

"""
combined epsilons: time, pt_probe, eta_probe; pt all the way down to 15 GeV
need to check on all of the variations but they should all be time, pt_probe, eta_probe
^will these two need to be expanded along another axis


n_masked = time, pt_tag, eta_tag --> IS THAT CORRECT? I THINK IT SHOULD PROBABLY BE IN TERMS OF PROBE? OR IS THE POINT THAT IT IS MASKED SO IT IS TRACKING THE OTHER CHANNEL

i am currently ignoring the low bins for hlt. i dont think i should do that. i think i should just not fit to those. 


*t*t_prpg are all still 5d in (time, pt_probe, eta_probe, pt_tag, eta_tag)
"""

#### i think i will need to mix this too but it doesn't affect the fit

pass_gen = all_mc_corrections(
    pass_gen, time_proj_hlt_all, lumi_scaling, weightsum, cross_sec
)

n_masked = pass_gen.project(
    "time", "pt_probe", "eta_probe"
)  ### THIS ONLY WORKS BECAUSE THEY ARE NOT MUTALLY EXCLUSIVE. IF I SWITCH THE FRAMEWORK TO MAKE THEM SO, THEN THIS WILL NEED TO CAHNGE


## create the tensor
writer = tensorwriter.TensorWriter()
##generator channel
writer.add_channel(n_masked.axes, "ch_masked", masked=True)  ## is this still correct?
writer.add_process(
    divideHists(n_masked, lumi_scaling), "Zmumu pass gen", "ch_masked", signal=False
)

dtdt_data_3d_exc = remove_low_bins(dtdt_data_3d_exc)
h2 = remove_low_bins(h2)
dtdt_m_center = remove_low_bins(dtdt_m_center)


### okay these are all 3d (time, pt_probe, eta_probe which is how i want it. )
writer.add_channel(
    dtdt_data_3d_exc.axes, "ch_dtdt_3d"
)  # at this point it is only looking at information about the single muon. i don't think that is right, i think we want to keep the other soooo either i could expand the axes or i could attempt to solve the root issue and not project it down


writer.add_data(dtdt_data_3d_exc, "ch_dtdt_3d")
writer.add_process(h2, "Zmumu pass gen", "ch_dtdt_3d", signal=False)


# writer.add_channel(dtst_data_3d_exc.axes, "ch_dtst_3d")
# writer.add_data(dtst_data_3d_exc, "ch_dtst_3d")
# writer.add_process(h1, "Zmumu pass gen", "ch_dtst_3d", signal=False)

# writer.add_channel(stst_data_3d_exc.axes, "ch_stst_3d")
# writer.add_data(stst_data_3d_exc, "ch_stst_3d")
# writer.add_process(h0, "Zmumu pass gen", "ch_stst_3d", signal=False)

#### this is for the fit later in terms of mass and time
# dtdt_data = dtdt_data.project("time", "mll")
# dtst_data = dtst_data.project("time", "mll")
# stst_data = stst_data.project("time", "mll")

# dtdt_2d = dtdt_prpg.project("time", "mll") ### this is good because it doesn't permanently modify the original
# dtst_2d = dtst_prpg.project("time", "mll")
# stst_2d = stst_prpg.project("time", "mll")

# h2_2d = dtdt_2d

# h1_2d = addHists(
#     dtst_2d,
#     scaleHist(dtdt_2d, -1),
# )
# h0_2d = addHists(
#     stst_2d,
#     scaleHist(dtst_2d, -1),
# )


# n_masked_2 = stst_prpg.project("time", "mll")
# ### these are the 5d ones
# writer.add_channel(n_masked.axes, "ch_masked_2d", masked=True)  ## i think the problem has something to do with the fact that i don
# writer.add_process(
#     divideHists(n_masked, lumi_scaling), "Zmumu pass gen", "ch_masked_2d", signal=False
# )
# writer.add_channel(
#     dtdt_data.axes, "ch_dtdt_2d"
# )
# writer.add_data(dtdt_data, "ch_dtdt_2d")
# writer.add_process(h2_2d, "Zmumu pass gen", "ch_dtdt_2d", signal=False)

# writer.add_channel(dtst_data.axes, "ch_dtst_2d")
# writer.add_data(dtst_data, "ch_dtst_2d")
# writer.add_process(h1_2d, "Zmumu pass gen", "ch_dtst_2d", signal=False)

# writer.add_channel(stst_data.axes, "ch_stst_2d")
# writer.add_data(stst_data, "ch_stst_2d")
# writer.add_process(h0_2d, "Zmumu pass gen", "ch_stst_2d", signal=False)

### adding axes as appropriate to make everything 6 dimensional


### so at this point i have already selected the mass bin, need to iterate over pt, eta, time


### does this remove necessary informtaion??? i dont think so

dtdt_m_center = dtdt_m_center.project("time", "pt_probe", "eta_probe")
dtst_m_center = dtst_m_center.project("time", "pt_probe", "eta_probe")
stst_m_center = stst_m_center.project("time", "pt_probe", "eta_probe")
pass_gen = pass_gen[{"mll": mass_bin}]
pass_gen = pass_gen.project("time", "pt_probe", "eta_probe")

# dtdt_m_center = expand_hist_by_duplicate_axes(dtdt_m_center, ["time", "pt_probe", "eta_probe"], ["time_copy", "pt_copy", "eta_copy"])
dtdt_m_center = expand_hist_by_duplicate_axes(
    h2.copy(), ["time", "pt_probe", "eta_probe"], ["time_copy", "pt_copy", "eta_copy"]
)
dtst_m_center = expand_hist_by_duplicate_axes(
    dtst_m_center,
    ["time", "pt_probe", "eta_probe"],
    ["time_copy", "pt_copy", "eta_copy"],
)
stst_m_center = expand_hist_by_duplicate_axes(
    stst_m_center,
    ["time", "pt_probe", "eta_probe"],
    ["time_copy", "pt_copy", "eta_copy"],
)
pass_gen = expand_hist_by_duplicate_axes(
    pass_gen, ["time", "pt_probe", "eta_probe"], ["time_copy", "pt_copy", "eta_copy"]
)

for i in range(nbins_pt):  # just select two pt bins in the center
    for j in range(nbins_eta):  # eta
        for k in range(nbins_time):  #  time

            if i > 0:  ## we only have 1 bin beneath 25 GeV

                ### be more consistent about ordering of time and mll
                ### fitting for the number of events
                v2 = dtdt_m_center[
                    {"pt_copy": i - 1, "eta_copy": j, "time_copy": k}
                ]  ## equivalent to n2, was #i - 1
                var2 = addHists(scaleHist(v2, var_size), h2)
                writer.add_systematic(
                    var2,
                    f"n_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtdt_3d",
                    constrained=False,
                    groups=["nz"],
                )

            # pdb.set_trace()

            # v1 = dtst_m_center[
            #     {"pt_copy": i, "eta_copy": j, "time_copy": k}
            # ]  ## equivalent to n1
            # var1 = addHists(scaleHist(v1, var_size), h1)
            # writer.add_systematic(
            #     var1,
            #     f"n_pt{i}_eta{j}_time{k}",
            #     "Zmumu pass gen",
            #     "ch_dtst_3d",
            #     constrained=False,
            #     groups=["nz"],
            # )

            #### well okay this looks really wrong
            # v0 = stst_m_center[
            #     {"pt_copy": i, "eta_copy": j, "time_copy": k}
            # ]  ## equivalent to n1
            # var0 = addHists(scaleHist(v0, var_size), h0)
            # writer.add_systematic(
            #     var0,
            #     f"n_pt{i}_eta{j}_time{k}",
            #     "Zmumu pass gen",
            #     "ch_stst_3d",
            #     constrained=False,
            #     groups=["nz"],
            # )

            # # # for masked channel
            # v_masked = pass_gen[{"pt_copy": i, "eta_copy": j, "time_copy":k}]
            # var_masked = addHists(scaleHist(v_masked, var_size), n_masked)
            # cross_section_masked = divideHists(var_masked, lumi_scaling)

            # writer.add_systematic(
            #     cross_section_masked,
            #     f"n_pt{i}_eta{j}_time{k}",
            #     "Zmumu pass gen",
            #     "ch_masked",
            #     constrained=False,
            #     groups=["nz"],
            # )

            # pdb.set_trace()
            # if i > 0:
            #     # ## efficiency
            #     h1var_id_primed = get_eff_hist(
            #         h1var_id_high, h1, i, j, k, "pt_probe", "eta_probe"
            #     )
            #     h0var_id_primed = get_eff_hist(
            #         h0var_id_high, h0, i, j, k, "pt_probe", "eta_probe"
            #     )

            #     h1var_hlt_primed = get_eff_hist(
            #         h1var_hlt_high, h1, i, j, k, "pt_probe", "eta_probe"
            #     )
            #     h0var_hlt_primed = get_eff_hist(
            #         h0var_hlt_high, h0, i, j, k, "pt_probe", "eta_probe"
            #     )

            #     # #     ### ID EFFICIENCY
            #     h2var_id_primed = get_eff_hist(
            #         h2var_id_high, h2, i - 1, j, k, "pt_probe", "eta_probe"
            #     )
            #     h2var_hlt_primed = get_eff_hist(
            #         h2var_hlt_high, h2, i - 1, j, k, "pt_probe", "eta_probe"
            #     )

            # pdb.set_trace()

            # writer.add_systematic(
            #     h2var_hlt_primed,
            #     f"hlt_prime_pt{i}_eta{j}_time{k}",
            #     "Zmumu pass gen",
            #     "ch_dtdt_3d",
            #     constrained=False,
            #     groups=["eff_trig"],
            # )

            # writer.add_systematic(
            #     h1var_hlt_primed,
            #     f"hlt_prime_pt{i}_eta{j}_time{k}",
            #     "Zmumu pass gen",
            #     "ch_dtst_3d",
            #     constrained=False,
            #     groups=["eff_trig"],
            # )
            # writer.add_systematic(
            #     h0var_hlt_primed,
            #     f"hlt_prime_pt{i}_eta{j}_time{k}",
            #     "Zmumu pass gen",
            #     "ch_stst_3d",
            #     constrained=False,
            #     groups=["eff_trig"],
            # )
            # writer.add_systematic(
            #     h2var_id_primed,
            #     f"id_prime_pt{i}_eta{j}_time{k}",
            #     "Zmumu pass gen",
            #     "ch_dtdt_3d",
            #     constrained=False,
            #     groups=["eff_id"],
            # )
            # else:
            #     h1var_id_primed = get_eff_hist(
            #         h1var_id_low, h1, i, j, k, "pt_probe", "eta_probe"
            #     )
            #     h0var_id_primed = get_eff_hist(
            #         h0var_id_low, h0, i, j, k, "pt_probe", "eta_probe"
            #     )
            #     #     # ### ID EFFICIENCY, these two used to be i-2
            #     # hd = get_eff_hist(
            #     #     h2var_id_low,
            #     #     h2,
            #     #     i,
            #     #     j,
            #     #     k,
            #     #     "pt_probe",
            #     #     "eta_probe",
            #     # )2var_id_prime

            # ### order of these is time, pt, eta
            # writer.add_systematic(
            #     h1var_id_primed,
            #     f"id_prime_pt{i}_eta{j}_time{k}",
            #     "Zmumu pass gen",
            #     "ch_dtst_3d",
            #     constrained=False,
            #     groups=["eff_id"],
            # )

            # writer.add_systematic(
            #     h0var_id_primed,
            #     f"id_prime_pt{i}_eta{j}_time{k}",
            #     "Zmumu pass gen",
            #     "ch_stst_3d",
            #     constrained=False,
            #     groups=["eff_id"],
            # )

            # for l in range(nbins_mll):
            #     if i > 0:
            #         dtdt_prpg.values()[:, l, l, i, j, :, :] = dtdt_prpg.values()[:, l, l, i, j, :, :] * (h2var_id_high.values() * eps_id_prime) *(h2var_hlt_high.values() * eps_hlt_prime)
            #         dtdt_prpg.values()[:, l, l, :, :, i, j] = dtdt_prpg.values()[:, l, l, :, :, i, j] * (h2var_id_high.values() * eps_id_prime) *(h2var_hlt_high.values() * eps_hlt_prime)

            #         dtst_prpg.values()[:, l, l, i, j, :, :] = dtst_prpg.values()[:, l, l, i, j, :, :] * (h1var_id_high.values() * eps_id_prime) *(h1var_hlt_high.values() * eps_hlt_prime)
            #         dtst_prpg.values()[:, l, l, :, :, i, j] = dtst_prpg.values()[:, l, l, :, :, i, j] * (h1var_id_high.values() * eps_id_prime) *(h1var_hlt_high.values() * eps_hlt_prime)

            #         stst_prpg.values()[:, l, l, i, j, :, :] = stst_prpg.values()[:, l, l, i, j, :, :] * (h0var_id_high.values() * eps_id_prime) *(h0var_hlt_high.values() * eps_hlt_prime)
            #         stst_prpg.values()[:, l, l, :, :, i, j] = stst_prpg.values()[:, l, l, :, :, i, j] * (h0var_id_high.values() * eps_id_prime) *(h0var_hlt_high.values() * eps_hlt_prime)

            #     else:
            #         dtdt_prpg.values()[:, l, l, i, j, :, :] = dtdt_prpg.values()[:, l, l, i, j, :, :] * (h2var_id_low.values() * eps_id_prime)
            #         dtdt_prpg.values()[:, l, l, :, :, i, j] = dtdt_prpg.values()[:, l, l, :, :, i, j] * (h2var_id_low.values() * eps_id_prime)

            #         dtst_prpg.values()[:, l, l, i, j, :, :] = dtst_prpg.values()[:, l, l, i, j, :, :] * (h1var_id_low.values() * eps_id_prime)
            #         dtst_prpg.values()[:, l, l, :, :, i, j] = dtst_prpg.values()[:, l, l, :, :, i, j] * (h1var_id_low.values() * eps_id_prime)

            #         stst_prpg.values()[:, l, l, i, j, :, :] = stst_prpg.values()[:, l, l, i, j, :, :] * (h0var_id_low.values() * eps_id_prime)
            #         stst_prpg.values()[:, l, l, :, :, i, j] = stst_prpg.values()[:, l, l, :, :, i, j] * (h0var_id_low.values() * eps_id_prime)


### SO THESE SHOULD BE DONE ACROSS ALL MASS BINS
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
#     "ch_dtdt_3d",
#     constrained=True,
#     groups=["prefiring_syst"],
# )
# writer.add_systematic(
#     dtst_prpg_prefiring_syst.project("time", "pt_tag", "eta_tag"),
#     f"prefiring_syst",
#     "Zmumu pass gen",
#     "ch_dtst_3d_3d",
#     constrained=True,
#     groups=["prefiring_syst"],
# )
# writer.add_systematic(
#     stst_prpg_prefiring_syst.project("time", "pt_tag", "eta_tag"),
#     f"prefiring_syst",
#     "Zmumu pass gen",
#     "ch_stst_3d",
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


### statistical uncertainty and the stability and linearity still slightly linked (~0.003%)

### stability cross detector seems to generate hte majority of that
# ## PCC cross detector
# luminometer_syst(
#     writer, "pcc", dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc, "stability"
# )
# # ## HFOC cross detector
# luminometer_syst(
#     writer, "hfoc", dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc, "stability"
# )


# # #### RAMSES cross detector


# luminometer_syst(
#     writer, "ramses", dtdt_prpg_ramses, dtst_prpg_ramses, stst_prpg_ramses, "stability",
# )


# ## seems to generate about the same amount of statistical uncertainty and together the uncertainties on each are higher so they are somehow linked which is a problem

# ### YEAH THESE ARE 100% COUPLED. CRAP.
# #### HFOC linearity
# # luminometer_syst(
# #     writer,
# #     "hfoc",
# #     dtdt_prpg_sbil_hfoc,
# #     dtst_prpg_sbil_hfoc,
# #     stst_prpg_sbil_hfoc,
# #     "linearity",
# # )

# # #### RAMSES linearity
# luminometer_syst(
#     writer,
#     "ramses",
#     dtdt_prpg_sbil_ramses,
#     dtst_prpg_sbil_ramses,
#     stst_prpg_sbil_ramses,
#     "linearity",

# )

writer.write(outfolder="./", outfilename="liv")
