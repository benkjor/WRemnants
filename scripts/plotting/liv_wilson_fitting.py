import argparse
import pickle

import h5py
import hist
import numpy as np
from uncertainty_tools import (
    all_mc_corrections,
    get_era_vals,
    get_mc_lumis,
)

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    expand_hist_by_duplicate_axis,
)

slope_ramses = 0.0006
slope_hfoc = 0.0007


parser = argparse.ArgumentParser()
args = parser.parse_args()

# indir_data = "/work/submit/jbenke/WRemnants/scripts/plotting/"
# infile_data = indir_data + "fitresults.hdf5"
# h5file = h5py.File(infile_data, "r")
# fit_results = input_tools.load_results_h5py(h5file)

# ### prefit is MC, postfit is data
# data = fit_results["parms_prefit"]

file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"  # _maxFiles_20
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

data_output = results["dataPostVFP"]["output"]
lumi_output = results["dataPostVFP"]["lumi_outout"]
MC_Zmumu = results["ZmumuPostVFP"]["output"]

dtdt_data = data_output["time_mll"].get()
dtst_data = data_output["time_dtst"].get()
stst_data = data_output["time_stst"].get()
iso_data = data_output["time_iso"].get()

time_proj_low_all = data_output["time_proj"].get()
time_proj_hlt_all = data_output["time_proj"].get()

time_proj_low = time_proj_low_all
time_proj_hlt = time_proj_hlt_all

dtdt_prpg_BG, dtdt_prpg_BG_syst, dtdt_prpg_BG_stat = get_era_vals(
    MC_Zmumu, "dtdt", "BG"
)
dtst_prpg_BG, dtst_prpg_BG_syst, dtst_prpg_BG_stat = get_era_vals(
    MC_Zmumu, "dtst", "BG"
)
stst_prpg_BG, stst_prpg_BG_syst, stst_prpg_BG_stat = get_era_vals(
    MC_Zmumu, "stst", "BG"
)
iso_BG, iso_BG_syst, iso_BG_stat = get_era_vals(MC_Zmumu, "pass_iso", "BG", iso=True)


dtdt_prpg_H, dtdt_prpg_H_syst, dtdt_prpg_H_stat = get_era_vals(MC_Zmumu, "dtdt", "H")
dtst_prpg_H, dtst_prpg_H_syst, dtst_prpg_H_stat = get_era_vals(MC_Zmumu, "dtst", "H")
stst_prpg_H, stst_prpg_H_syst, stst_prpg_H_stat = get_era_vals(MC_Zmumu, "stst", "H")
iso_H, iso_H_syst, iso_H_stat = get_era_vals(MC_Zmumu, "pass_iso", "H", iso=True)


pass_gen = MC_Zmumu["pass_gen"].get()


lumi_scaling = lumi_output["lumi_nom"].get()
lumi_scaling_h = lumi_output["lumi_pre"].get()
lumi_scaling_bg = lumi_output["lumi_post"].get()

weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

#### A COUPLE FIXED QUANTITIES
nbins_mll = len(dtdt_data.axes["mll"])
nbins_time = len(dtst_data.axes["time"])
nbins_pt = len(dtst_data.axes["pt_probe"])
nbins_eta = len(dtdt_data.axes["eta_probe"])


iso_data = iso_data.project("time", "mll")
prpg_all = [
    iso_H.project("mll"),
    dtdt_prpg_H.project("mll"),
    dtst_prpg_H.project("mll"),
    stst_prpg_H.project("mll"),
    iso_BG.project("mll"),
    dtdt_prpg_BG.project("mll"),
    dtst_prpg_BG.project("mll"),
    stst_prpg_BG.project("mll"),
]
time_hists = [
    time_proj_hlt.project("time", "mll"),
    time_proj_low.project("time", "mll"),
]
lumi_hists = [lumi_scaling_h, lumi_scaling_bg]

#### normal


with open(
    "/home/submit/jbenke/WRemnants/rabbit/rabbit/poi_models/precomputed_sigma/SM_15_to_120_GeV_13_bins.pkl",
    "rb",
) as f:
    precomputed_sm = pickle.load(f)

sm_values = np.array(precomputed_sm["values"])  ## should be in pb, as should 2001
test_ax = time_proj_hlt.copy().project("mll")

sm_values_hist = hist.Hist(*test_ax.axes, data=sm_values)

iso, dtdt_prpg, dtst_prpg, stst_prpg = get_mc_lumis(
    prpg_all,
    time_hists,
    lumi_scaling,
    lumi_hists,
    weightsum,
    sm_values_hist,
)


pass_gen = all_mc_corrections(
    pass_gen.project("mll"),
    time_proj_low,
    lumi_scaling,
    weightsum,
    cross_sec,
)
n_masked = pass_gen.project("time", "mll")


###################################################################

## create the tensor


### TEMPORARY WHILE MY OTHER CODE RUNS


# iso_unrolled = unrolledHist(iso)
# iso_data_unrolled = unrolledHist(iso_data)

iso_unrolled = iso[{"time": 0}]
iso_data_unrolled = iso_data[{"time": 0}]
nbins_time = 1

writer = tensorwriter.TensorWriter()
# writer.add_channel(n_masked.axes, "ch_masked", masked=True)  ## is this still correct?
# writer.add_process(divideHists(n_masked, lumi_scaling), "Zmumu pass gen", "ch_masked")

writer.add_channel(iso_unrolled.axes, "ch_iso")
writer.add_data(iso_data_unrolled, "ch_iso")
writer.add_process(
    iso_unrolled, "Zmumu pass gen", "ch_iso", signal=True
)  ### not quite sure where Zmumu pass gen came from


## unrolled is mass then time so indexing will go [i*nbins_mll + j] where i is the time bin and j is the mass bin
iso_unrolled_exp = expand_hist_by_duplicate_axis(
    iso_unrolled, "mll", "unrolled_ax"
)  ## for unrolled it is ""
var_size = 0.1

for i in range(nbins_time):
    for j in range(nbins_mll):
        v = iso_unrolled_exp[{"unrolled_ax": i * nbins_mll + j}]

        var = addHists(var_size * v, iso_unrolled)
        writer.add_systematic(
            var,
            f"c",
            "Zmumu pass gen",
            "ch_iso",
            constrained=False,
            groups=["cxx"],
        )


writer.write(outfolder="./", outfilename="wilson")
