import argparse
import pdb

import h5py
import hist
import numpy as np

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    broadcastSystHist,
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


def get_h2var(eps_id, eps_hlt, heff):
    #  h2var = heff * (eps_prime * eps)
    h2var = multiplyHists(heff, multiplyHists(eps_id, eps_id))
    h2var = multiplyHists(h2var, multiplyHists(eps_hlt, eps_hlt))
    return h2var


def get_h1var(eps_id, eps_hlt, heff, hist_ones):
    # h = 2*heff*(eps_hlt * hlt_prime)*(1-(eps_hlt * hlt_prime))
    h1var = addHists(hist_ones, scaleHist(eps_hlt, -1))
    h1var = multiplyHists(h1var, eps_hlt)
    h1var = multiplyHists(h1var, heff)
    h1var = scaleHist(h1var, 2)
    h1var = multiplyHists(h1var, multiplyHists(eps_id, eps_id))
    return h1var


def get_h0var(eps_id, eps_hlt, heff, hist_ones):
    # h = 2*heff*(eps_id * eps_id_prime)*(1-(eps_id * eps_id_prime))
    h0var = addHists(hist_ones, scaleHist(eps_id, -1))
    h0var = multiplyHists(h0var, eps_id)
    h0var = multiplyHists(h0var, eps_hlt)
    h0var = multiplyHists(h0var, heff)
    h0var = scaleHist(h0var, 2)
    return h0var


def get_eff_hist(eps_hist, ref_hist, i, j):
    hist_copy = ref_hist.copy()
    hist_values = hist_copy.values()
    try:
        hist_values[j, i] = eps_hist[{"time": j, "mll": i}].value
    except:
        hist_values[j, i] = eps_hist[{"time": j, "mll": i}]

    hist_copy.values()[...] = hist_values
    return hist_copy


def mc_scaling(mc_results, weightsum, cross_sec):
    temp = mc_results.copy()
    temp /= weightsum
    temp *= cross_sec
    temp *= 1000
    return temp


def all_mc_corrections(hist_in, hist_proj, lumi_scaling, weightsum, cross_sec):
    hist_in_new = hist_in.copy()
    hist_in_new = mc_scaling(hist_in_new, weightsum, cross_sec)
    hist_in_2d = broadcastSystHist(hist_in_new, hist_proj)
    hist_in_2d = multiplyHists(hist_in_2d, lumi_scaling)
    return hist_in_2d


def mc_corrections_all_cases(
    dtdt_mc, dtst_mc, stst_mc, hist_proj, lumi_scaling, weightsum, cross_sec
):

    dtdt = all_mc_corrections(
        dtdt_mc.copy(), hist_proj, lumi_scaling, weightsum, cross_sec
    )
    dtst = all_mc_corrections(
        dtst_mc.copy(), hist_proj, lumi_scaling, weightsum, cross_sec
    )
    stst = all_mc_corrections(
        stst_mc.copy(), hist_proj, lumi_scaling, weightsum, cross_sec
    )
    return dtdt, dtst, stst


def make_ones_hist(hist_ref):
    ones = np.ones_like(hist_ref.values())
    h_ones = hist_ref.copy()
    h_ones.values()[...] = ones
    return h_ones


def averaged_prefiring_hist(prefiring_hist, ref_hist):
    axis_len = len(prefiring_hist.axes[0])
    avg_arr = np.zeros([axis_len])

    for i in range(axis_len):
        this_set = prefiring_hist[{"mll": i}].values()
        avg_arr[i] = np.average(this_set)
    ### this axis needs to be an mll axis
    copy_hist = hist.Hist(ref_hist.axes[0])
    copy_hist.values()[...] = avg_arr
    copy_hist = expand_hist_by_duplicate_axis(copy_hist, "mll", "gen_mll")
    return copy_hist


def get_mc_lumis(
    dtdt_h,
    dtst_h,
    stst_h,
    dtdt_bg,
    dtst_bg,
    stst_bg,
    time_proj,
    scaling,
    lumi_h,
    lumi_bg,
    lumi_nom,
    weightsum,
    cross_sec,
):
    sum_lumis = addHists(lumi_bg, lumi_h)
    lumi_scaling_h = divideHists(lumi_h, sum_lumis)
    lumi_scaling_bg = divideHists(lumi_bg, sum_lumis)
    dtdt_h, dtst_h, stst_h = mc_corrections_all_cases(
        dtdt_h,
        dtst_h,
        stst_h,
        time_proj,
        multiplyHists(lumi_scaling_h, lumi_nom),
        weightsum,
        cross_sec,
    )

    dtdt_bg, dtst_bg, stst_bg = mc_corrections_all_cases(
        dtdt_bg,
        dtst_bg,
        stst_bg,
        time_proj,
        multiplyHists(lumi_scaling_bg, lumi_nom),
        weightsum,
        cross_sec,
    )

    dtdt = addHists(dtdt_bg, dtdt_h)
    dtst = addHists(dtst_bg, dtst_h)
    stst = addHists(stst_bg, stst_h)

    return dtdt, dtst, stst


### i need to get good at coding so i dont need to pass in all these variables
def eta_phi_systematic(
    writer,
    dtdt_H,
    dtst_H,
    stst_H,
    dtdt_BG,
    dtst_BG,
    stst_BG,
    time_proj,
    lumi_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    weightsum,
    cross_sec,
    etaphi_num,
):
    dtdt_stat, dtst_stat, stst_stat = get_mc_lumis(
        dtdt_H[{"downUpVar": 0}],
        dtst_H[{"downUpVar": 0}],
        stst_H[{"downUpVar": 0}],
        dtdt_BG[{"downUpVar": 0}],
        dtst_BG[{"downUpVar": 0}],
        stst_BG[{"downUpVar": 0}],
        time_proj,
        lumi_scaling,
        lumi_scaling_h,
        lumi_scaling_bg,
        lumi_scaling,
        weightsum,
        cross_sec,
    )
    writer.add_systematic(
        dtdt_stat.project("time", "mll"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "prpg",
        "ch_dtdt",
        constrained=True,
        groups=["prefiring_stat"],
    )
    writer.add_systematic(
        dtst_stat.project("time", "mll"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "prpg",
        "ch_dtst",
        constrained=True,
        groups=["prefiring_stat"],
    )
    writer.add_systematic(
        stst_stat.project("time", "mll"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "prpg",
        "ch_stst",
        constrained=True,
        groups=["prefiring_stat"],
    )


file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)
MC_Zmumu = results["ZmumuPostVFP"]["output"]
data_output = results["dataPostVFP"]["output"]
lumi_output = results["dataPostVFP"]["lumi_outout"]

reco_dtdt_data = data_output["time_mll"].get()
reco_dtst_data = data_output["time_mll_dtst"].get()
reco_stst_data = data_output["time_mll_stst"].get()


### pass reco, pass generator

dtdt_prpg_mc_true = MC_Zmumu["mll_dtdt_prpg"].get()
dtst_prpg_mc_true = MC_Zmumu["mll_dtst_prpg"].get()
stst_prpg_mc_true = MC_Zmumu["mll_stst_prpg"].get()

### should loop over these instead of calling them explicitly
dtdt_prpg_mc_BG = MC_Zmumu["dtdt_prpg_BG"].get()
dtst_prpg_mc_BG = MC_Zmumu["dtst_prpg_BG"].get()
stst_prpg_mc_BG = MC_Zmumu["stst_prpg_BG"].get()

dtdt_prpg_mc_H = MC_Zmumu["dtdt_prpg_H"].get()
dtst_prpg_mc_H = MC_Zmumu["dtst_prpg_H"].get()
stst_prpg_mc_H = MC_Zmumu["stst_prpg_H"].get()


dtdt_prpg_mc_BG_syst = MC_Zmumu["dtdt_prpg_BG_muonL1PrefireSyst"].get()
dtst_prpg_mc_BG_syst = MC_Zmumu["dtst_prpg_BG_muonL1PrefireSyst"].get()
stst_prpg_mc_BG_syst = MC_Zmumu["stst_prpg_BG_muonL1PrefireSyst"].get()

dtdt_prpg_mc_H_syst = MC_Zmumu["dtdt_prpg_H_muonL1PrefireSyst"].get()
dtst_prpg_mc_H_syst = MC_Zmumu["dtst_prpg_H_muonL1PrefireSyst"].get()
stst_prpg_mc_H_syst = MC_Zmumu["stst_prpg_H_muonL1PrefireSyst"].get()

dtdt_prpg_mc_BG_stat = MC_Zmumu["dtdt_prpg_BG_muonL1PrefireStat"].get()
dtst_prpg_mc_BG_stat = MC_Zmumu["dtst_prpg_BG_muonL1PrefireStat"].get()
stst_prpg_mc_BG_stat = MC_Zmumu["stst_prpg_BG_muonL1PrefireStat"].get()

dtdt_prpg_mc_H_stat = MC_Zmumu["dtdt_prpg_H_muonL1PrefireStat"].get()
dtst_prpg_mc_H_stat = MC_Zmumu["dtst_prpg_H_muonL1PrefireStat"].get()
stst_prpg_mc_H_stat = MC_Zmumu["stst_prpg_H_muonL1PrefireStat"].get()


time_proj = data_output["time_proj"].get()
time_proj_gen_mll = data_output["time_proj"].get().project("time", "gen_mll")
time_proj_mll = data_output["time_proj"].get().project("time", "mll")

pass_gen = MC_Zmumu["pass_gen"].get()

# background_processes = ### NOT SURE WHAT GOES HERE YET

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
nbins_mll = len(dtdt_prpg_mc_true.axes["mll"])
nbins_time = len(reco_dtst_data.axes["time"])

### cross-detector uncertainties

hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
hfoc_scaling = multiplyHists(hfoc_scaling, lumi_scaling)

pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
pcc_scaling = multiplyHists(pcc_scaling, lumi_scaling)

ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
ramses_scaling = multiplyHists(ramses_scaling, lumi_scaling)


############################# CURRENTLY WORKING ON ##########################################
### i should prabably do this for each of the 3 cases. but for now will just implement one
dtdt_prpg_mc_hfoc, dtst_prpg_mc_hfoc, stst_prpg_mc_hfoc = get_mc_lumis(
    dtdt_prpg_mc_H,
    dtst_prpg_mc_H,
    stst_prpg_mc_H,
    dtdt_prpg_mc_BG,
    dtst_prpg_mc_BG,
    stst_prpg_mc_BG,
    time_proj,
    hfoc_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
    weightsum,
    cross_sec,
)
dtdt_prpg_mc_pcc, dtst_prpg_mc_pcc, stst_prpg_mc_pcc = get_mc_lumis(
    dtdt_prpg_mc_H,
    dtst_prpg_mc_H,
    stst_prpg_mc_H,
    dtdt_prpg_mc_BG,
    dtst_prpg_mc_BG,
    stst_prpg_mc_BG,
    time_proj,
    pcc_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
    weightsum,
    cross_sec,
)

dtdt_prpg_mc_ramses, dtst_prpg_mc_ramses, stst_prpg_mc_ramses = get_mc_lumis(
    dtdt_prpg_mc_H,
    dtst_prpg_mc_H,
    stst_prpg_mc_H,
    dtdt_prpg_mc_BG,
    dtst_prpg_mc_BG,
    stst_prpg_mc_BG,
    time_proj,
    ramses_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
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


dtdt_prpg_mc_sbil_hfoc, dtst_prpg_mc_sbil_hfoc, stst_prpg_mc_sbil_hfoc = get_mc_lumis(
    dtdt_prpg_mc_H,
    dtst_prpg_mc_H,
    stst_prpg_mc_H,
    dtdt_prpg_mc_BG,
    dtst_prpg_mc_BG,
    stst_prpg_mc_BG,
    time_proj,
    sbil_hfoc_fit,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
    weightsum,
    cross_sec,
)

dtdt_prpg_mc_sbil_ramses, dtst_prpg_mc_sbil_ramses, stst_prpg_mc_sbil_ramses = (
    get_mc_lumis(
        dtdt_prpg_mc_H,
        dtst_prpg_mc_H,
        stst_prpg_mc_H,
        dtdt_prpg_mc_BG,
        dtst_prpg_mc_BG,
        stst_prpg_mc_BG,
        time_proj,
        sbil_ramses_fit,
        lumi_scaling_h,
        lumi_scaling_bg,
        lumi_scaling,
        weightsum,
        cross_sec,
    )
)

dtdt_prpg_mc, dtst_prpg_mc, stst_prpg_mc = get_mc_lumis(
    dtdt_prpg_mc_H,
    dtst_prpg_mc_H,
    stst_prpg_mc_H,
    dtdt_prpg_mc_BG,
    dtst_prpg_mc_BG,
    stst_prpg_mc_BG,
    time_proj,
    lumi_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
    weightsum,
    cross_sec,
)
(
    dtdt_prpg_mc_prefiring_syst,
    dtst_prpg_mc_prefiring_syst,
    stst_prpg_mc_prefiring_syst,
) = get_mc_lumis(
    dtdt_prpg_mc_H_syst[{"downUpVar": 0}],
    dtst_prpg_mc_H_syst[{"downUpVar": 0}],
    stst_prpg_mc_H_syst[{"downUpVar": 0}],
    dtdt_prpg_mc_BG_syst[{"downUpVar": 0}],
    dtst_prpg_mc_BG_syst[{"downUpVar": 0}],
    stst_prpg_mc_BG_syst[{"downUpVar": 0}],
    time_proj,
    lumi_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
    weightsum,
    cross_sec,
)


pass_gen = all_mc_corrections(
    pass_gen, time_proj_gen_mll, lumi_scaling, weightsum, cross_sec
)

### efficiencies
h2 = dtdt_prpg_mc.project("time", "mll")
h1 = dtst_prpg_mc.project("time", "mll")
h0 = stst_prpg_mc.project("time", "mll")

efficiency_ones = make_ones_hist(h1)

## e2 = 2*h2/(h1 + 2*h1)
eps_hlt = addHists(h1, scaleHist(h2, 2))
eps_hlt = divideHists(h2, eps_hlt)
eps_hlt = scaleHist(eps_hlt, 2)
##e1 = h1/(h0*(1-e2) + h1)
eps_id = addHists(efficiency_ones, scaleHist(eps_hlt, -1))
eps_id = multiplyHists(h0, eps_id)
eps_id = addHists(eps_id, h1)
eps_id = divideHists(h1, eps_id)

heff = divideHists(h2, multiplyHists(eps_hlt, eps_hlt))
heff = divideHists(heff, multiplyHists(eps_id, eps_id))

# generate histogram of ones

eps_id_prime = 1.01
eps_hlt_prime = 1.01


eps_id_var = scaleHist(eps_id.copy(), eps_id_prime)
eps_hlt_var = scaleHist(eps_hlt.copy(), eps_hlt_prime)

h0var_id = get_h0var(eps_id_var, eps_hlt, heff, efficiency_ones)
h0var_hlt = get_h0var(eps_id, eps_hlt_var, heff, efficiency_ones)
h1var_id = get_h1var(eps_id_var, eps_hlt, heff, efficiency_ones)
h1var_hlt = get_h1var(eps_id, eps_hlt_var, heff, efficiency_ones)
h2var_id = get_h2var(eps_id_var, eps_hlt, heff)
h2var_hlt = get_h2var(eps_id, eps_hlt_var, heff)

n_masked = pass_gen.project("time", "gen_mll")

## create the tensor
writer = tensorwriter.TensorWriter()

##g# enerator channel
writer.add_channel(pass_gen.axes, "ch_masked", masked=True)
writer.add_process(
    divideHists(pass_gen, lumi_scaling), "prpg", "ch_masked", signal=False
)
### efficiency channels
writer.add_channel(reco_dtdt_data.axes, "ch_dtdt")
writer.add_data(reco_dtdt_data, "ch_dtdt")
writer.add_process(h2, "prpg", "ch_dtdt", signal=False)

writer.add_channel(reco_dtst_data.axes, "ch_dtst")
writer.add_data(reco_dtst_data, "ch_dtst")
writer.add_process(h1, "prpg", "ch_dtst", signal=False)

writer.add_channel(reco_stst_data.axes, "ch_stst")
writer.add_data(reco_stst_data, "ch_stst")
writer.add_process(h0, "prpg", "ch_stst", signal=False)
### adding axes as appropriate to make everything 4 dimensional

dtdt_prpg_mc = expand_hist_by_duplicate_axis(dtdt_prpg_mc, "time", "gen_time")
dtst_prpg_mc = expand_hist_by_duplicate_axis(dtst_prpg_mc, "time", "gen_time")
stst_prpg_mc = expand_hist_by_duplicate_axis(stst_prpg_mc, "time", "gen_time")

pass_gen_expanded = expand_hist_by_duplicate_axes(
    pass_gen, ["time", "gen_mll"], ["gen_time", "gen_mll_0"]
)

h2_var_id_ALL = []
h1_var_id_ALL = []
h0_var_id_ALL = []
h2_var_hlt_ALL = []
h1_var_hlt_ALL = []
h0_var_hlt_ALL = []
# pdb.set_trace()

for i in range(3, 6):  # just select two mass bins in the center
    for j in range(nbins_time):
        ### be more consistent about ordering of time and mll
        ### fitting for the number of events
        v2 = dtdt_prpg_mc[{"gen_mll": i, "gen_time": j}]  ## equivalent to n2
        var2 = addHists(v2 * 0.1, h2)
        writer.add_systematic(
            var2,
            f"n_mll{i}_time{j}",
            "prpg",
            "ch_dtdt",
            constrained=False,
            groups=["nz"],
        )

        v1 = dtst_prpg_mc[{"gen_mll": i, "gen_time": j}]  ## equivalent to n1
        var1 = addHists(v1 * 0.1, h1)
        writer.add_systematic(
            var1,
            f"n_mll{i}_time{j}",
            "prpg",
            "ch_dtst",
            constrained=False,
            groups=["nz"],
        )
        v0 = stst_prpg_mc[{"gen_mll": i, "gen_time": j}]  ## equivalent to n1
        var0 = addHists(v0 * 0.1, h0)
        writer.add_systematic(
            var0,
            f"n_mll{i}_time{j}",
            "prpg",
            "ch_stst",
            constrained=False,
            groups=["nz"],
        )
        # for masked channel
        v_masked = pass_gen_expanded[{"gen_mll_0": i, "gen_time": j}]
        var_masked = addHists(v_masked * 0.1, n_masked)
        cross_section_masked = divideHists(var_masked, lumi_scaling)
        writer.add_systematic(
            cross_section_masked,
            f"n_mll{i}_time{j}",
            "prpg",
            "ch_masked",
            constrained=False,
            groups=["nz"],
        )

        ## efficiency
        h2var_id_primed = get_eff_hist(h2var_id, h2, i, j)
        h1var_id_primed = get_eff_hist(h1var_id, h1, i, j)
        h0var_id_primed = get_eff_hist(h0var_id, h0, i, j)

        h2var_hlt_primed = get_eff_hist(h2var_hlt, h2, i, j)
        h1var_hlt_primed = get_eff_hist(h1var_hlt, h1, i, j)
        h0var_hlt_primed = get_eff_hist(h0var_hlt, h0, i, j)

        h2_var_id_ALL.append(h2var_id_primed)
        h1_var_id_ALL.append(h1var_id_primed)
        h0_var_id_ALL.append(h0var_id_primed)
        h2_var_hlt_ALL.append(h2var_hlt_primed)
        h1_var_hlt_ALL.append(h1var_hlt_primed)
        h0_var_hlt_ALL.append(h0var_hlt_primed)

        #### ID EFFICIENCY

        writer.add_systematic(
            h2var_id_primed,
            f"id_prime_mll{i}_time{j}",
            "prpg",
            "ch_dtdt",
            constrained=False,
            groups=["eff_1"],
        )
        writer.add_systematic(
            h1var_id_primed,
            f"id_prime_mll{i}_time{j}",
            "prpg",
            "ch_dtst",
            constrained=False,
            groups=["eff_1"],
        )

        writer.add_systematic(
            h0var_id_primed,
            f"id_prime_mll{i}_time{j}",
            "prpg",
            "ch_stst",
            constrained=False,
            groups=["eff_1"],
        )

        ### HLT EFFICIENCY

        writer.add_systematic(
            h2var_hlt_primed,
            f"hlt_prime_mll{i}_time{j}",
            "prpg",
            "ch_dtdt",
            constrained=False,
            groups=["eff_2"],
        )
        writer.add_systematic(
            h1var_hlt_primed,
            f"hlt_prime_mll{i}_time{j}",
            "prpg",
            "ch_dtst",
            constrained=False,
            groups=["eff_2"],
        )
        writer.add_systematic(
            h0var_hlt_primed,
            f"hlt_prime_mll{i}_time{j}",
            "prpg",
            "ch_stst",
            constrained=False,
            groups=["eff_2"],
        )


writer.add_systematic(
    dtdt_prpg_mc_prefiring_syst.project("time", "mll"),
    f"prefiring_syst",
    "prpg",
    "ch_dtdt",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    dtst_prpg_mc_prefiring_syst.project("time", "mll"),
    f"prefiring_syst",
    "prpg",
    "ch_dtst",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    stst_prpg_mc_prefiring_syst.project("time", "mll"),
    f"prefiring_syst",
    "prpg",
    "ch_stst",
    constrained=True,
    groups=["prefiring_syst"],
)
pdb.set_trace()
num_etaphi = len(dtdt_prpg_mc_H_stat.project("etaPhiRegion").values())
for i in range(num_etaphi):
    eta_phi_systematic(
        writer,
        dtdt_prpg_mc_H_stat[{"etaPhiRegion": i}],
        dtst_prpg_mc_H_stat[{"etaPhiRegion": i}],
        stst_prpg_mc_H_stat[{"etaPhiRegion": i}],
        dtdt_prpg_mc_BG_stat[{"etaPhiRegion": i}],
        dtst_prpg_mc_BG_stat[{"etaPhiRegion": i}],
        stst_prpg_mc_BG_stat[{"etaPhiRegion": i}],
        time_proj,
        lumi_scaling,
        lumi_scaling_h,
        lumi_scaling_bg,
        weightsum,
        cross_sec,
        i,
    )

##### MAKE LESS STUPID
#### PCC cross detector
writer.add_systematic(
    dtdt_prpg_mc_pcc.project("time", "mll"),
    "pcc_stability",
    "prpg",
    "ch_dtdt",
    constrained=True,
    groups=["cross_detector_stability"],
)
writer.add_systematic(
    dtst_prpg_mc_pcc.project("time", "mll"),
    "pcc_stability",
    "prpg",
    "ch_dtst",
    constrained=True,
    groups=["cross_detector_stability"],
)
writer.add_systematic(
    stst_prpg_mc_pcc.project("time", "mll"),
    "pcc_stability",
    "prpg",
    "ch_stst",
    constrained=True,
    groups=["cross_detector_stability"],
)
## HFOC cross detector
writer.add_systematic(
    dtdt_prpg_mc_hfoc.project("time", "mll"),
    "hfoc_stability",
    "prpg",
    "ch_dtdt",
    constrained=True,
    groups=["cross_detector_stability"],
)
writer.add_systematic(
    dtst_prpg_mc_hfoc.project("time", "mll"),
    "hfoc_stability",
    "prpg",
    "ch_dtst",
    constrained=True,
    groups=["cross_detector_stability"],
)
writer.add_systematic(
    stst_prpg_mc_hfoc.project("time", "mll"),
    "hfoc_stability",
    "prpg",
    "ch_stst",
    constrained=True,
    groups=["cross_detector_stability"],
)

#### RAMSES cross detector
writer.add_systematic(
    dtdt_prpg_mc_ramses.project("time", "mll"),
    "ramses_stability",
    "prpg",
    "ch_dtdt",
    constrained=True,
    groups=["cross_detector_stability"],
)
writer.add_systematic(
    dtst_prpg_mc_ramses.project("time", "mll"),
    "ramses_stability",
    "prpg",
    "ch_dtst",
    constrained=True,
    groups=["cross_detector_stability"],
)
writer.add_systematic(
    stst_prpg_mc_ramses.project("time", "mll"),
    "ramses_stability",
    "prpg",
    "ch_stst",
    constrained=True,
    groups=["cross_detector_stability"],
)


#### HFOC linearity
writer.add_systematic(
    dtdt_prpg_mc_sbil_hfoc.project("time", "mll"),
    "hfoc_linearity",
    "prpg",
    "ch_dtdt",
    constrained=True,
    groups=["linearity"],
)
writer.add_systematic(
    dtst_prpg_mc_sbil_hfoc.project("time", "mll"),
    "hfoc_linearity",
    "prpg",
    "ch_dtst",
    constrained=True,
    groups=["linearity"],
)
writer.add_systematic(
    stst_prpg_mc_sbil_hfoc.project("time", "mll"),
    "hfoc_linearity",
    "prpg",
    "ch_stst",
    constrained=True,
    groups=["linearity"],
)


#### RAMSES linearity
writer.add_systematic(
    dtdt_prpg_mc_sbil_ramses.project("time", "mll"),
    "ramses_linearity",
    "prpg",
    "ch_dtdt",
    constrained=True,
    groups=["linearity"],
)
writer.add_systematic(
    dtst_prpg_mc_sbil_ramses.project("time", "mll"),
    "ramses_linearity",
    "prpg",
    "ch_dtst",
    constrained=True,
    groups=["linearity"],
)
writer.add_systematic(
    stst_prpg_mc_sbil_ramses.project("time", "mll"),
    "ramses_linearity",
    "prpg",
    "ch_stst",
    constrained=True,
    groups=["linearity"],
)

writer.write(outfolder="./", outfilename="liv")
