import argparse

import h5py

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    broadcastSystHist,
    divideHists,
    expand_hist_by_duplicate_axes,
    expand_hist_by_duplicate_axis,
    multiplyHists,
)

parser = argparse.ArgumentParser()
### if i need additonal arguments add them here

args = parser.parse_args()


file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr_maxFiles_20.hdf5"
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

reco_dtdt_data = results["dataPostVFP"]["output"]["time_mll"].get()
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


def mc_scaling(mc_results):
    mc_results /= weightsum
    mc_results *= cross_sec
    mc_results *= 1000
    return mc_results


def all_mc_corrections(hist_in, hist_proj):
    hist_in = mc_scaling(hist_in)
    hist_in_2d = broadcastSystHist(hist_in, hist_proj)
    hist_in_2d = multiplyHists(hist_in_2d, lumi_scaling)
    return hist_in_2d


dtdt_prfg_mc = all_mc_corrections(dtdt_prfg_mc, time_proj)
dtst_prfg_mc = all_mc_corrections(dtst_prfg_mc, time_proj)
stst_prfg_mc = all_mc_corrections(stst_prfg_mc, time_proj)
dtdt_prpg_mc = all_mc_corrections(dtdt_prpg_mc, time_proj)
dtst_prpg_mc = all_mc_corrections(dtst_prpg_mc, time_proj)
stst_prpg_mc = all_mc_corrections(stst_prpg_mc, time_proj)
pass_gen = all_mc_corrections(pass_gen, time_proj_gen_mll)

writer = tensorwriter.TensorWriter()

### doesn't really matter which one we select here
writer.add_channel(reco_dtdt_data.axes, "ch_dtdt")
writer.add_data(reco_dtdt_data, "ch_dtdt")
n = dtdt_prpg_mc.project("time", "mll")
dtdt_prpg_mc = expand_hist_by_duplicate_axis(dtdt_prpg_mc, "time", "gen_time")
writer.add_process(n, "prpg", "ch_dtdt", signal=False)


writer.add_channel(pass_gen.axes, "ch_masked", masked=True)
writer.add_process(
    divideHists(pass_gen, lumi_scaling), "prpg", "ch_masked", signal=False
)
n_masked = pass_gen.project("time", "gen_mll")
pass_gen_expanded = expand_hist_by_duplicate_axes(
    pass_gen, ["time", "gen_mll"], ["gen_time", "gen_mll_0"]
)


writer.add_channel(dtst_prpg_mc.axes, "ch_dtst", masked=True)
### not sure if i want this in the same process
writer.add_process(
    divideHists(dtst_prpg_mc, lumi_scaling), "eps_prime", "ch_dtst", signal=False
)
n_dtst = dtst_prpg_mc.project("time", "mll")


dtst_prpg_mc = expand_hist_by_duplicate_axis(dtst_prpg_mc, "time", "gen_time")
# dtst_prpg_mc = expand_hist_by_duplicate_axis(dtst_prpg_mc, "time", "gen_time")
# writer.add_process(n_dtst, "prpg_dtst", "ch_dtst", signal=False)


for i in range(len(dtdt_prpg_mc.axes["gen_mll"])):
    for j in range(len(dtdt_prpg_mc.axes["gen_time"])):
        v = dtdt_prpg_mc[{"gen_mll": i, "gen_time": j}]  ## equivalent to n2
        var = addHists(v * 0.1, n)
        writer.add_systematic(
            var,
            f"n_mll{i}_time{j}",
            "prpg",
            "ch_dtdt",
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

        n2 = v
        n1 = dtst_prpg_mc[{"gen_mll": i, "gen_time": j}]
        eps = 2 * divideHists(n2, addHists(n1, 2 * n2))

        neff = divideHists(n2, multiplyHists(eps, eps))

        eps_prime = 0.1 * eps
        eps_prime = eps_prime.project("time", "mll")
        # import pdb
        # pdb.set_trace()

        writer.add_systematic(
            eps_prime,
            f"eps_prime_mll{i}_time{j}",
            "eps_prime",
            "ch_dtst",
            constrained=False,
            groups=["nz"],
        )

        n2_prime = multiplyHists(neff, multiplyHists(eps_prime, eps_prime))
        n1_prime = 2 * multiplyHists(neff, eps_prime)
        n1_prime = divideHists(n1_prime, eps)
        n1_prime = multiplyHists(n1_prime, 1 - multiplyHists(eps, eps_prime))

# n2 = number that pass both high level triggers
# n1 = number that pass only one high level trigger
# n2 = dtdt_prpg_mc
# n1 = stst_prpg_mc

# eps = 2*n2/(n1 + 2*n2)
# neff = n2/eps**2
# n2_prime = neff*eps_prime**2
# n1_prime = 2*neff*eps_prime/eps*(1-eps*eps_prime)


writer.write(outfolder="./", outfilename="liv")
