import argparse

import h5py
import hist
import numpy as np
from uncertainty_tools import (
    all_mc_corrections,
    get_era_vals,
    get_mc_lumis,
    make_mutually_exclusive,
)

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    multiplyHists,
)

fit_type = 4  ### number of dimensions
coeff_name = ["c^xx", "c^xy", "c^xz", "c^yz"]
if fit_type == 4:
    start_range = 0
    end_range = 4
    channel_name = "4d"

else:
    start_range = 3
    end_range = start_range + 1
    channel_name = f"1d_coeff_{start_range}"

parser = argparse.ArgumentParser()
args = parser.parse_args()

file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = (
    file_in + "mz_dilepton_liv_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr.hdf5"
)  # _maxFiles_20
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)
data_output = results["SingleMuon_2016PostVFP"]["output"]
lumi_output = results["SingleMuon_2016PostVFP"]["lumi_outout"]
MC_Zmumu = results["Zmumu_2016PostVFP"]["output"]

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

weightsum = results["Zmumu_2016PostVFP"]["weight_sum"]
cross_sec = results["Zmumu_2016PostVFP"]["dataset"]["xsec"]
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
time_hists = time_proj_low.project("time", "mll")
lumi_hists = [lumi_scaling_h, lumi_scaling_bg]

#### normal

iso, dtdt_prpg, dtst_prpg, stst_prpg = get_mc_lumis(
    prpg_all,
    time_hists,
    lumi_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)

iso_data, dtdt_data, dtst_data, stst_data = make_mutually_exclusive(
    iso_data, dtdt_data, dtst_data, stst_data
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
iso_unrolled = iso[{"mll": 9}]
iso_data_unrolled = iso_data[{"mll": 9}]
nbins_mll = 1
n_masked_unrolled = n_masked[{"mll": 9}]


data_int = np.sum(iso_unrolled.values())
flat_line = hist.Hist(
    hist.axis.Regular(24, 0, 24, metadata="time", overflow=False, underflow=False),
    data=np.ones(24) * data_int / 24,  ## this finds the average
)


indir_liv_model = "/home/submit/jbenke/LIV/coupling_models/"
infile_liv_model = indir_liv_model + "coupling_before_cxx.npy"
vals = np.load(infile_liv_model)  # (SM+LV)/SM = 1 + LV/SM
var = hist.Hist(
    hist.axis.Regular(24, 0, 24, metadata="time", underflow=False, overflow=False),
    data=vals[:] - 1,
)

iso_injection = addHists(iso_data_unrolled, multiplyHists(var, iso_data_unrolled))


##generator channel
writer = tensorwriter.TensorWriter()
writer.add_channel(iso_data_unrolled.axes, f"ch{channel_name}")
writer.add_data(iso_data_unrolled, f"ch{channel_name}")
writer.add_process(iso_unrolled, "liv_fit", f"ch{channel_name}", signal=True)

for i in range(1):
    infile_liv_model = indir_liv_model + "coupling_before_cxx.npy"
    vals = np.load(infile_liv_model)  # (SM+LV)/SM = 1 + LV/SM
    var = hist.Hist(
        hist.axis.Regular(24, 0, 24, metadata="time", underflow=False, overflow=False),
        data=vals[:] - 1,
    )

    var = multiplyHists(var, iso_unrolled)  # LV

    writer.add_systematic(
        addHists(iso_unrolled, var),
        f"{coeff_name[i]}",
        "liv_fit",
        f"ch{channel_name}",
        constrained=False,
        noi=True,
    )

writer.write(outfolder="./", outfilename="old_wilson_coeff")
