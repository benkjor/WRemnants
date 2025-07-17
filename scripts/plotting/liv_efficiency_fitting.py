import argparse

import h5py
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
    hist_values[j, i] = eps_hist[{"time": j, "mll": i}].value
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


file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

reco_dtdt_data = results["dataPostVFP"]["output"]["time_mll"].get()
reco_dtst_data = results["dataPostVFP"]["output"]["time_mll_dtst"].get()
reco_stst_data = results["dataPostVFP"]["output"]["time_mll_stst"].get()

time_proj = results["dataPostVFP"]["output"]["time_proj"].get()
time_proj_gen_mll = (
    results["dataPostVFP"]["output"]["time_proj"].get().project("time", "gen_mll")
)

### pass reco, pass generator
dtdt_prpg_mc = results["ZmumuPostVFP"]["output"]["mll_dtdt_prpg"].get()
dtst_prpg_mc = results["ZmumuPostVFP"]["output"]["mll_dtst_prpg"].get()
stst_prpg_mc = results["ZmumuPostVFP"]["output"]["mll_stst_prpg"].get()

pass_gen = results["ZmumuPostVFP"]["output"]["pass_gen"].get()

# background_processes = ### NOT SURE WHAT GOES HERE YET


### probably need to pull these back'
lumi_scaling = results["dataPostVFP"]["lumi_outout"]["lumi_nom"].get()
### pulling for cross detector scaling
lumi_hfoc = results["dataPostVFP"]["lumi_outout"]["lumi_hfoc"].get()
lumi_pcc = results["dataPostVFP"]["lumi_outout"]["lumi_pcc"].get()
lumi_ramses = results["dataPostVFP"]["lumi_outout"]["lumi_ramses"].get()

lumi_hfoc_nom = results["dataPostVFP"]["lumi_outout"]["lumi_in_hfoc"].get()
lumi_pcc_nom = results["dataPostVFP"]["lumi_outout"]["lumi_in_pcc"].get()
lumi_ramses_nom = results["dataPostVFP"]["lumi_outout"]["lumi_in_ramses"].get()
## pulling for linearity
sbil_pcc = results["dataPostVFP"]["lumi_outout"][
    "sbil_pcc"
].get()  #### turns out this is nominal so going to not create a separate nominal one
count_pcc = results["dataPostVFP"]["lumi_outout"]["count_pcc"].get()


weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

nbins_mll = len(dtdt_prpg_mc.axes["mll"])
nbins_time = len(reco_dtst_data.axes["time"])

##### caluclating the effieincies and the variation matricies
### units are initially 1/fb, want in 1/ub

### cross-detector uncertainties

hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
hfoc_scaling = multiplyHists(hfoc_scaling, lumi_scaling)

pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
pcc_scaling = multiplyHists(pcc_scaling, lumi_scaling)

ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
ramses_scaling = multiplyHists(ramses_scaling, lumi_scaling)

dtdt_prpg_mc_hfoc, dtst_prpg_mc_hfoc, stst_prpg_mc_hfoc = mc_corrections_all_cases(
    dtdt_prpg_mc,
    dtst_prpg_mc,
    stst_prpg_mc,
    time_proj,
    hfoc_scaling,
    weightsum,
    cross_sec,
)
dtdt_prpg_mc_pcc, dtst_prpg_mc_pcc, stst_prpg_mc_pcc = mc_corrections_all_cases(
    dtdt_prpg_mc,
    dtst_prpg_mc,
    stst_prpg_mc,
    time_proj,
    hfoc_scaling,
    weightsum,
    cross_sec,
)
dtdt_prpg_mc_ramses, dtst_prpg_mc_ramses, stst_prpg_mc_ramses = (
    mc_corrections_all_cases(
        dtdt_prpg_mc,
        dtst_prpg_mc,
        stst_prpg_mc,
        time_proj,
        hfoc_scaling,
        weightsum,
        cross_sec,
    )
)

avg_sbil_pcc = scaleHist(divideHists(sbil_pcc, count_pcc), 1e9)

sbil_hfoc_fit = scaleHist(avg_sbil_pcc, slope_hfoc)

sbil_ones = make_ones_hist(sbil_hfoc_fit)
sbil_hfoc_fit = addHists(sbil_hfoc_fit, sbil_ones)
sbil_hfoc_fit = multiplyHists(sbil_hfoc_fit, lumi_scaling)

sbil_ramses_fit = scaleHist(avg_sbil_pcc, slope_ramses)
sbil_ramses_fit = addHists(sbil_ramses_fit, sbil_ones)
sbil_ramses_fit = multiplyHists(sbil_ramses_fit, lumi_scaling)

dtdt_prpg_mc_sbil_hfoc, dtst_prpg_mc_sbil_hfoc, stst_prpg_mc_sbil_hfoc = (
    mc_corrections_all_cases(
        dtdt_prpg_mc,
        dtst_prpg_mc,
        stst_prpg_mc,
        time_proj,
        sbil_hfoc_fit,
        weightsum,
        cross_sec,
    )
)
dtdt_prpg_mc_sbil_ramses, dtst_prpg_mc_sbil_ramses, stst_prpg_mc_sbil_ramses = (
    mc_corrections_all_cases(
        dtdt_prpg_mc,
        dtst_prpg_mc,
        stst_prpg_mc,
        time_proj,
        sbil_ramses_fit,
        weightsum,
        cross_sec,
    )
)

#### normal

dtdt_prpg_mc, dtst_prpg_mc, stst_prpg_mc = mc_corrections_all_cases(
    dtdt_prpg_mc,
    dtst_prpg_mc,
    stst_prpg_mc,
    time_proj,
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

# pdb.set_trace()
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


for i in range(10, 13):  # just select two mass bins in the center
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
### HFOC
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
