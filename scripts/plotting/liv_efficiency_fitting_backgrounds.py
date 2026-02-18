import argparse

import h5py
from uncertainty_tools import (
    all_mc_corrections,
    get_era_vals,
    get_mc_lumis,
    luminometer_syst,
    make_mutually_exclusive,
    make_ones_hist,
    remove_low_bins,
)

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    divideHists,
    multiplyHists,
    scaleHist,
)

parser = argparse.ArgumentParser()
args = parser.parse_args()

slope_ramses = 0.0006
slope_hfoc = 0.0007
mass_bin = 9
var_size = 0.01

background_syst_names = [
    # "ZmumuPostVFP",
    "Top",
    "Diboson",
    "GGToLLPostVFP",
    "QCDmuEnrichPt15PostVFP",
    "WplusmunuPostVFP",
    "QGToDYQTo2LPostVFP",
    "QGToWQToLNuPostVFP",
]
background_proc = [
    # "Zmumu fail gen",
    "Top",
    "Diboson",
    "GG",
    "QCD",
    "W",
    "QG_2L",
    "QG_Lnu",
]

######################################################################
# DATA IMPORTS #

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

time_proj_low = data_output["time_proj"].get()

### should loop over these instead of calling them explicitly

### MAKE THIS IMPLEMENTATION NOT STUPID

#### i wonder if it has something to do with the fact that i do the exclusion after the rest of hte corrections
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


### STABILITY
lumi_hfoc = lumi_output["lumi_hfoc"].get()
lumi_pcc = lumi_output["lumi_pcc"].get()
lumi_ramses = lumi_output["lumi_ramses"].get()

## LINEARITY
sbil_pcc = lumi_output["sbil_pcc"].get()
count_pcc = lumi_output["count_pcc"].get()


lumi_hfoc_nom = lumi_output["lumi_in_hfoc"].get()
lumi_pcc_nom = lumi_output["lumi_in_pcc"].get()
lumi_ramses_nom = lumi_output["lumi_in_ramses"].get()

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

hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
hfoc_scaling = multiplyHists(hfoc_scaling, lumi_scaling)

pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
pcc_scaling = multiplyHists(pcc_scaling, lumi_scaling)

ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
ramses_scaling = multiplyHists(ramses_scaling, lumi_scaling)


time_proj_low = time_proj_low[{"mll": mass_bin}]
iso_H = iso_H[{"mll": mass_bin}]
dtdt_prpg_H = dtdt_prpg_H[{"mll": mass_bin}]
dtst_prpg_H = dtst_prpg_H[{"mll": mass_bin}]
stst_prpg_H = stst_prpg_H[{"mll": mass_bin}]

iso_BG = iso_BG[{"mll": mass_bin}]
dtdt_prpg_BG = dtdt_prpg_BG[{"mll": mass_bin}]
dtst_prpg_BG = dtst_prpg_BG[{"mll": mass_bin}]
stst_prpg_BG = stst_prpg_BG[{"mll": mass_bin}]


dtdt_data = dtdt_data[{"mll": mass_bin}]
dtst_data = dtst_data[{"mll": mass_bin}]
stst_data = stst_data[{"mll": mass_bin}]
iso_data = iso_data[{"mll": mass_bin}]

pass_gen = pass_gen[{"mll": mass_bin}]
prpg_all = [
    iso_H,
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    iso_BG,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
]


pass_gen = all_mc_corrections(
    pass_gen,
    time_proj_low,
    lumi_scaling,
    weightsum,
    cross_sec,
)


# prpg_syst = [
#     dtdt_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     dtst_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     stst_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     dtdt_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     dtst_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     stst_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
# ]


lumi_hists = [lumi_scaling_h, lumi_scaling_bg]
iso_hfoc, dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc = get_mc_lumis(
    prpg_all,
    time_proj_low,
    hfoc_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)


iso_pcc, dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc = get_mc_lumis(
    prpg_all,
    time_proj_low,
    pcc_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)

iso_ramses, dtdt_prpg_ramses, dtst_prpg_ramses, stst_prpg_ramses = get_mc_lumis(
    prpg_all,
    time_proj_low,
    ramses_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)


avg_sbil_pcc = scaleHist(divideHists(sbil_pcc, count_pcc), 1e9)
sbil_ones = make_ones_hist(avg_sbil_pcc)

sbil_hfoc_fit = scaleHist(avg_sbil_pcc, slope_hfoc)
sbil_hfoc_fit = addHists(sbil_hfoc_fit, sbil_ones)
sbil_hfoc_fit = multiplyHists(sbil_hfoc_fit, lumi_scaling)

sbil_ramses_fit = scaleHist(avg_sbil_pcc, slope_ramses)
sbil_ramses_fit = addHists(sbil_ramses_fit, sbil_ones)
sbil_ramses_fit = multiplyHists(sbil_ramses_fit, lumi_scaling)
iso_sbil_hfoc, dtdt_prpg_sbil_hfoc, dtst_prpg_sbil_hfoc, stst_prpg_sbil_hfoc = (
    get_mc_lumis(
        prpg_all,
        time_proj_low,
        sbil_hfoc_fit,
        lumi_hists,
        weightsum,
        cross_sec,
    )
)

iso_sbil_ramses, dtdt_prpg_sbil_ramses, dtst_prpg_sbil_ramses, stst_prpg_sbil_ramses = (
    get_mc_lumis(
        prpg_all,
        time_proj_low,
        sbil_ramses_fit,
        lumi_hists,
        weightsum,
        cross_sec,
    )
)


#### normal
iso, dtdt_prpg, dtst_prpg, stst_prpg = get_mc_lumis(
    prpg_all,
    time_proj_low,
    lumi_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
)

# (dtdt_prpg_prefiring_syst, dtst_prpg_prefiring_syst, stst_prpg_prefiring_syst) = (
#     get_mc_lumis(
#         prpg_syst,
#         time_proj_low,
#         lumi_scaling,
#         lumi_hists,
#         weightsum,
#         cross_sec,
#     )
# )

iso_data, dtdt_data, dtst_data, stst_data = make_mutually_exclusive(
    iso_data, dtdt_data, dtst_data, stst_data
)


test_nmasked = addHists(pass_gen, scaleHist(dtdt_data, -1))

dtdt_data = remove_low_bins(dtdt_data)
dtdt_prpg = remove_low_bins(dtdt_prpg)
dtdt_prpg_sbil_ramses = remove_low_bins(dtdt_prpg_sbil_ramses)
dtdt_prpg_sbil_hfoc = remove_low_bins(dtdt_prpg_sbil_hfoc)
dtdt_prpg_ramses = remove_low_bins(dtdt_prpg_ramses)
dtdt_prpg_pcc = remove_low_bins(dtdt_prpg_pcc)
dtdt_prpg_hfoc = remove_low_bins(dtdt_prpg_hfoc)


# can't do this later because i redefine iso_data
# iso_data_time_mll = iso_data.project("time", "mll")
# dtdt_data_time_mll = dtdt_data.project("time", "mll")
# dtst_data_time_mll = dtst_data.project("time", "mll")
# stst_data_time_mll = stst_data.project("time", "mll")

# iso_mc_time_mll = iso.project("time", "mll")
# dtdt_mc_time_mll = dtdt_prpg.project("time", "mll")
# dtst_mc_time_mll = dtst_prpg.project("time", "mll")
# stst_mc_time_mll = stst_prpg.project("time", "mll")

# n_masked = test_nmasked.project("time", "mll")

iso_data_time_mll = iso_data.project("time")
dtdt_data_time_mll = dtdt_data.project("time")
dtst_data_time_mll = dtst_data.project("time")
stst_data_time_mll = stst_data.project("time")

iso_mc_time_mll = iso.project("time")
dtdt_mc_time_mll = dtdt_prpg.project("time")
dtst_mc_time_mll = dtst_prpg.project("time")
stst_mc_time_mll = stst_prpg.project("time")

n_masked = test_nmasked.project("time")


###################################################################

## create the tensor
writer = tensorwriter.TensorWriter()
writer.add_channel(n_masked.axes, "ch_masked", masked=True)
writer.add_process(divideHists(n_masked, lumi_scaling), "Zmumu pass gen", "ch_masked")

#### for mass ones
writer.add_channel(iso_data_time_mll.axes, "ch_iso")
writer.add_data(iso_data_time_mll, "ch_iso")
writer.add_process(iso_mc_time_mll, "Zmumu pass gen", "ch_iso", signal=False)

writer.add_channel(dtdt_data_time_mll.axes, "ch_dtdt")
writer.add_data(dtdt_data_time_mll, "ch_dtdt")
writer.add_process(dtdt_mc_time_mll, "Zmumu pass gen", "ch_dtdt", signal=False)

writer.add_channel(dtst_data_time_mll.axes, "ch_dtst")
writer.add_data(dtst_data_time_mll, "ch_dtst")
writer.add_process(dtst_mc_time_mll, "Zmumu pass gen", "ch_dtst", signal=False)

writer.add_channel(stst_data_time_mll.axes, "ch_stst")
writer.add_data(stst_data_time_mll, "ch_stst")
writer.add_process(stst_mc_time_mll, "Zmumu pass gen", "ch_stst", signal=False)


#########################################################
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
#         time_proj_hlt,
#         time_proj_low,
#         lumi_scaling,
#         [lumi_scaling_h, lumi_scaling_bg],
#         proc_name,
#         f"bkg_{proc_name}",
#         fail_gen=fgen,
#     )


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
#         time_proj_low,
#         lumi_scaling,
#         lumi_hists,
#         weightsum,
#         cross_sec,
#         i,
#     )


### these slightly increase the statistical uncertainty but dont contribute to the stability/linearity
# dtdt_prpg_prefiring_syst = dtdt_prpg_prefiring_syst.project("time", "mll")
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

### statistical uncertainty and the stability and linearity still slightly linked (~0.003%)


### stability cross detector seems to generate hte majority of that
## PCC cross detector
# pdb.set_trace()
luminometer_syst(
    writer, "pcc", iso_pcc, dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc, "stability"
)
# pdb.set_trace()

## HFOC cross detector
# luminometer_syst(
#     writer,
#     "hfoc",
#     iso_sbil_hfoc,
#     dtdt_prpg_sbil_hfoc,
#     dtst_prpg_sbil_hfoc,
#     stst_prpg_sbil_hfoc,
#     "linearity",
# )


# luminometer_syst(
#     writer,
#     "hfoc",
#     iso_hfoc,
#     dtdt_prpg_hfoc,
#     dtst_prpg_hfoc,
#     stst_prpg_hfoc,
#     "stability",
# )

# # #### RAMSES cross detector
# luminometer_syst(
#     writer,
#     "ramses",
#     iso_ramses,
#     dtdt_prpg_ramses,
#     dtst_prpg_ramses,
#     stst_prpg_ramses,
#     "stability",
# )


### YEAH THESE ARE 100% COUPLED. CRAP.
# #### HFOC linearity

### RAMSES linearity
luminometer_syst(
    writer,
    "ramses",
    iso_sbil_ramses,
    dtdt_prpg_sbil_ramses,
    dtst_prpg_sbil_ramses,
    stst_prpg_sbil_ramses,
    "linearity",
)

writer.write(outfolder="./", outfilename="background")
# writer.write(outfolder="./")
