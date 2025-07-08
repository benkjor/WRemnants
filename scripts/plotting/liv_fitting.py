# import pdb
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
### if i need additonal arguments add them here

args = parser.parse_args()


def get_h1var(eps_prime_val, ref_hist, heff, eps_hist):
    # h = 2*h_eff*eps_prime/eps*(1-eps*eps_prime) for eps_prime a number, all other vars hists
    ones = np.ones_like(ref_hist.values())
    h_ones = ref_hist.copy()
    h_ones.values()[...] = ones
    neg_eps_prime = -1 * eps_prime_val
    h1var = scaleHist(eps_hist, neg_eps_prime)
    h1var = addHists(h_ones, h1var)
    h1var = scaleHist(multiplyHists(heff, h1var), 2)
    h1var = multiplyHists(h1var, scaleHist(eps_hist, eps_prime_val))
    return h1var


def mc_scaling(mc_results, weightsum, cross_sec):
    mc_results /= weightsum
    mc_results *= cross_sec
    mc_results *= 1000
    return mc_results


def all_mc_corrections(hist_in, hist_proj, lumi_scaling, weightsum, cross_sec):
    hist_in = mc_scaling(hist_in, weightsum, cross_sec)
    hist_in_2d = broadcastSystHist(hist_in, hist_proj)
    hist_in_2d = multiplyHists(hist_in_2d, lumi_scaling)
    return hist_in_2d


file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

reco_dtdt_data = results["dataPostVFP"]["output"]["time_mll"].get()
reco_dtst_data = results["dataPostVFP"]["output"]["time_mll_dtight_strig"].get()

time_proj = results["dataPostVFP"]["output"]["time_proj"].get()
time_proj_gen_mll = (
    results["dataPostVFP"]["output"]["time_proj"].get().project("time", "gen_mll")
)
### pass reco, fail generator
dtdt_prfg_mc = results["ZmumuPostVFP"]["output"]["mll_dtdt_prfg"].get()
dtst_prfg_mc = results["ZmumuPostVFP"]["output"]["mll_dtst_prfg"].get()
stst_prfg_mc = results["ZmumuPostVFP"]["output"]["mll_stst_prfg"].get()
### pass reco, pass generator
dtdt_prpg_mc = results["ZmumuPostVFP"]["output"]["mll_dtdt_prpg"].get()
dtst_prpg_mc = results["ZmumuPostVFP"]["output"]["mll_dtst_prpg"].get()
stst_prpg_mc = results["ZmumuPostVFP"]["output"]["mll_stst_prpg"].get()

pass_gen = results["ZmumuPostVFP"]["output"]["pass_gen"].get()

# background_processes = ### NOT SURE WHAT GOES HERE YET

lumi_scaling = results["dataPostVFP"]["lumi_outout"]["time"].get()
weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

nbins_mll = len(dtdt_prpg_mc.axes["mll"])
nbins_time = len(reco_dtst_data.axes["time"])


dtdt_prfg_mc = all_mc_corrections(
    dtdt_prfg_mc, time_proj, lumi_scaling, weightsum, cross_sec
)
dtst_prfg_mc = all_mc_corrections(
    dtst_prfg_mc, time_proj, lumi_scaling, weightsum, cross_sec
)
stst_prfg_mc = all_mc_corrections(
    stst_prfg_mc, time_proj, lumi_scaling, weightsum, cross_sec
)
dtdt_prpg_mc = all_mc_corrections(
    dtdt_prpg_mc, time_proj, lumi_scaling, weightsum, cross_sec
)
dtst_prpg_mc = all_mc_corrections(
    dtst_prpg_mc, time_proj, lumi_scaling, weightsum, cross_sec
)
stst_prpg_mc = all_mc_corrections(
    stst_prpg_mc, time_proj, lumi_scaling, weightsum, cross_sec
)
pass_gen = all_mc_corrections(
    pass_gen, time_proj_gen_mll, lumi_scaling, weightsum, cross_sec
)

h1 = dtst_prpg_mc.project("time", "mll")
h2 = dtdt_prpg_mc.project("time", "mll")


eps = 2 * divideHists(h2, addHists(h1, 2 * h2))
heff = divideHists(h2, multiplyHists(eps, eps))

eps_prime = 1.01
h1varUp = get_h1var(eps_prime, h1, heff, eps)
h1varDown = get_h1var(1 / eps_prime, h1, heff, eps)

# h2var = heff * (eps_prime * eps)**2
h2var = scaleHist(h2, eps_prime**2)


dtst_prpg_mc = dtst_prpg_mc.project("time", "gen_mll")
n_masked = pass_gen.project("time", "gen_mll")

writer = tensorwriter.TensorWriter()

writer.add_channel(pass_gen.axes, "ch_masked", masked=True)
writer.add_process(
    divideHists(pass_gen, lumi_scaling), "prpg", "ch_masked", signal=False
)
### doesn't really matter which one we select here
writer.add_channel(reco_dtdt_data.axes, "ch_dtdt")
writer.add_data(reco_dtdt_data, "ch_dtdt")
writer.add_process(h2, "prpg", "ch_dtdt", signal=False)

writer.add_channel(reco_dtst_data.axes, "ch_dtst")
writer.add_data(reco_dtst_data, "ch_dtst")
writer.add_process(h1, "prpg", "ch_dtst", signal=False)

### not sure if i want this in the same process
# writer.add_process(
#     divideHists(dtst_prpg_mc, lumi_scaling), "eps_prime", "ch_dtst", signal=False
# )
dtdt_prpg_mc = expand_hist_by_duplicate_axis(dtdt_prpg_mc, "time", "gen_time")
dtst_prpg_mc = expand_hist_by_duplicate_axis(dtst_prpg_mc, "time", "gen_time")

pass_gen_expanded = expand_hist_by_duplicate_axes(
    pass_gen, ["time", "gen_mll"], ["gen_time", "gen_mll_0"]
)
reco_dtdt_data = expand_hist_by_duplicate_axes(
    reco_dtdt_data, ["time", "mll"], ["time_0", "mll_0"]
)
reco_dtst_data = expand_hist_by_duplicate_axes(
    reco_dtst_data, ["time", "mll"], ["time_0", "mll_0"]
)


for i in range(nbins_mll):
    for j in range(nbins_time):

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

        h1var = h1.copy()
        n1var = h1var.values()
        n1var[j, i] = h1varDown[{"time": j, "mll": i}].value
        h1var.values()[...] = n1var

        h2var = h2.copy()
        n2var = h2var.values()
        n2var[j, i] = h2var[{"time": j, "mll": i}].value
        h2var.values()[...] = n2var

        writer.add_systematic(
            h1var,
            f"eps_prime_mll{i}_time{j}",
            "prpg",
            "ch_dtst",
            constrained=False,
            groups=["eff"],
        )

        writer.add_systematic(
            h2var,
            f"eps_prime_mll{i}_time{j}",
            "prpg",
            "ch_dtdt",
            constrained=False,
            groups=["eff"],
        )


writer.write(outfolder="./", outfilename="liv")
