import argparse

import h5py
from uncertainty_tools import (
    all_mc_corrections,
    background_syst,
    create_variation,
    get_era_vals,
    get_mc_lumis,
    make_mutually_exclusive,
    remove_low_bins,
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
var_size = 0.01

background_syst_names = [
    "ZmumuPostVFP",
    "Top",
    "Diboson",
    "GGToLLPostVFP",
    "QCDmuEnrichPt15PostVFP",
    "WplusmunuPostVFP",
    "QGToDYQTo2LPostVFP",
    "QGToWQToLNuPostVFP",
]
background_proc = [
    "Zmumu fail gen",
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
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"  # _maxFiles_20
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

data_output = results["dataPostVFP"]["output"]
lumi_output = results["dataPostVFP"]["lumi_outout"]
MC_Zmumu = results["ZmumuPostVFP"]["output"]


dtdt_data = data_output["time_mll"].get()
dtst_data = data_output["time_dtst"].get()
stst_data = data_output["time_stst"].get()
time_proj_low_all = data_output["time_proj"].get()
time_proj_hlt_all = data_output["time_proj"].get()

time_proj_low = time_proj_low_all[{"mll": mass_bin}]
time_proj_hlt = time_proj_hlt_all[{"mll": mass_bin}]


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

dtdt_prpg_H, dtdt_prpg_H_syst, dtdt_prpg_H_stat = get_era_vals(MC_Zmumu, "dtdt", "H")
dtst_prpg_H, dtst_prpg_H_syst, dtst_prpg_H_stat = get_era_vals(MC_Zmumu, "dtst", "H")
stst_prpg_H, stst_prpg_H_syst, stst_prpg_H_stat = get_era_vals(MC_Zmumu, "stst", "H")

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


weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

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

dtdt_prpg_H = dtdt_prpg_H[{"mll": mass_bin}]
dtst_prpg_H = dtst_prpg_H[{"mll": mass_bin}]
stst_prpg_H = stst_prpg_H[{"mll": mass_bin}]
dtdt_prpg_BG = dtdt_prpg_BG[{"mll": mass_bin}]
dtst_prpg_BG = dtst_prpg_BG[{"mll": mass_bin}]
stst_prpg_BG = stst_prpg_BG[{"mll": mass_bin}]

prpg_all = [
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
]
# prpg_syst = [
#     dtdt_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     dtst_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     stst_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     dtdt_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     dtst_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
#     stst_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
# ]


time_hists = [time_proj_hlt, time_proj_low]
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

pass_gen = all_mc_corrections(
    pass_gen[{"mll": mass_bin}],
    time_proj_low,
    lumi_scaling,
    weightsum,
    cross_sec,
)

n_masked = pass_gen.project("time", "pt_probe", "eta_probe")


hlt_var_nom = divideHists(
    dtdt_prpg.project("time", "pt_probe", "eta_probe"),
    dtst_prpg.project("time", "pt_probe", "eta_probe"),
)
id_var_nom = divideHists(
    dtst_prpg.project("time", "pt_probe", "eta_probe"),
    stst_prpg.project("time", "pt_probe", "eta_probe"),
)
hlt_var_nom_h2 = remove_low_bins(hlt_var_nom)


dtdt_data, dtst_data, stst_data = make_mutually_exclusive(
    dtdt_data, dtst_data, stst_data
)
dtdt_prpg_ex, dtst_prpg_ex, stst_prpg_ex = make_mutually_exclusive(
    dtdt_prpg, dtst_prpg, stst_prpg
)

dtdt_prpg_ex = remove_low_bins(dtdt_prpg_ex)
# pdb.set_trace()
h2 = dtdt_prpg_ex.project("time", "pt_probe", "eta_probe")
h1 = dtst_prpg_ex.project("time", "pt_probe", "eta_probe")
h0 = stst_prpg_ex.project("time", "pt_probe", "eta_probe")

dtdt_data = remove_low_bins(dtdt_data)
h2_data = dtdt_data.project("time", "pt_probe", "eta_probe")
h1_data = dtst_data.project("time", "pt_probe", "eta_probe")
h0_data = stst_data.project("time", "pt_probe", "eta_probe")


# print("DTDT")
# print(divideHists(h2.project("pt_probe"), h2_data.project("pt_probe")))
# print("DTST")
# print(divideHists(h1.project("pt_probe"), h1_data.project("pt_probe")))
# print("STST")
# print(divideHists(h0.project("pt_probe"), h0_data.project("pt_probe")))


dtdt_prpg_proj = expand_hist_by_duplicate_axes(
    h2, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)
dtst_prpg_proj = expand_hist_by_duplicate_axes(
    h1, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)
stst_prpg_proj = expand_hist_by_duplicate_axes(
    h0, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)

id_var_h2 = remove_low_bins(id_var_nom.copy())

hlt_var_nom_h2 = expand_hist_by_duplicate_axes(
    hlt_var_nom_h2, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)
hlt_var_nom = expand_hist_by_duplicate_axes(
    hlt_var_nom, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)
id_var_nom = expand_hist_by_duplicate_axes(
    id_var_nom, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)
id_var_h2 = expand_hist_by_duplicate_axes(
    id_var_h2, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)


###################################################################333

## create the tensor
writer = tensorwriter.TensorWriter()
##generator channel --> MAY BE WRONG BECAUSE THIS ISN'T MUTUTALLY EXCLUSIVE
writer.add_channel(n_masked.axes, "ch_masked", masked=True)  ## is this still correct?
writer.add_process(divideHists(n_masked, lumi_scaling), "Zmumu pass gen", "ch_masked")

writer.add_channel(h2_data.axes, "ch_dtdt")
writer.add_data(h2_data, "ch_dtdt")
writer.add_process(h2, "Zmumu pass gen", "ch_dtdt")

writer.add_channel(h1_data.axes, "ch_dtst")
writer.add_data(h1_data, "ch_dtst")
writer.add_process(h1, "Zmumu pass gen", "ch_dtst")

writer.add_channel(h0_data.axes, "ch_stst")
writer.add_data(h0_data, "ch_stst")
writer.add_process(h0, "Zmumu pass gen", "ch_stst")


pass_gen = expand_hist_by_duplicate_axis(pass_gen, "time", "gen_time")
nbins_h2 = (nbins_pt - 1) + nbins_eta + nbins_time
nbins_h1 = nbins_pt + nbins_eta + nbins_time
### so at this point i have already selected the mass bin, need to iterate over pt, eta, time
for i in range(nbins_pt):  # just select two pt bins in the center
    print(f"pt bin: {i}")
    for j in range(nbins_eta):  # eta
        for k in range(nbins_time):  #  time

            if i > 0:  ## we only have 1 bin beneath 25 GeV
                v2 = dtdt_prpg_proj[{"gen_time": k, "pt_tag": i - 1, "eta_tag": j}]
                var2 = addHists(v2 * var_size, h2)

                writer.add_systematic(
                    var2,
                    f"n_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtdt",
                    constrained=False,
                    groups=["nz"],
                )

            v1 = dtst_prpg_proj[{"gen_time": k, "pt_tag": i, "eta_tag": j}]
            var1 = addHists(v1 * var_size, h1)
            writer.add_systematic(
                var1,
                f"n_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtst",
                constrained=False,
                groups=["nz"],
            )

            v0 = stst_prpg_proj[{"gen_time": k, "pt_tag": i, "eta_tag": j}]
            var0 = addHists(v0 * var_size, h0)
            writer.add_systematic(
                var0,
                f"n_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_stst",
                constrained=False,
                groups=["nz"],
            )
            # for masked channel --> IS NOT MUTUALLY EXCLUSIVE
            v_masked = pass_gen[{"gen_time": k, "pt_tag": i, "eta_tag": j}]
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

            if i > 0:  ### MAKE THIS IMPLEMENTATION LESS STUPID AND REPETITIVE

                hlt_var_tag_h2 = create_variation(
                    hlt_var_nom_h2, dtdt_prpg_ex, i, j, k, nbins_h2, h2=True
                )
                hlt_probe_h2 = create_variation(
                    hlt_var_nom_h2, dtdt_prpg_ex, i, j, k, 0, "probe", h2=True
                )
                hlt_probe_h2 = multiplyHists(scaleHist(hlt_probe_h2, var_size), h2)
                hlt_var_total_h2 = addHists(hlt_var_tag_h2, hlt_probe_h2)

                writer.add_systematic(
                    addHists(hlt_var_total_h2, h2),
                    f"hlt_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtdt",
                    constrained=False,
                    groups=["eff_trig"],
                )

                hlt_var_tag_h1 = create_variation(
                    hlt_var_nom, dtst_prpg_ex, i, j, k, nbins_h1
                )
                hlt_probe_h1 = create_variation(
                    hlt_var_nom, dtst_prpg_ex, i, j, k, 0, "probe"
                )
                hlt_probe_h1 = multiplyHists(scaleHist(hlt_probe_h1, var_size), h1)
                hlt_var_total_h1 = scaleHist(
                    addHists(hlt_var_tag_h1, -1 * hlt_probe_h1), 2
                )

                writer.add_systematic(
                    addHists(hlt_var_total_h1, h1),
                    f"hlt_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtst",
                    constrained=False,
                    groups=["eff_trig"],
                )

                hlt_var_tag_h0 = create_variation(
                    hlt_var_nom, stst_prpg_ex, i, j, k, nbins_h2
                )
                hlt_var_total_h0 = 2 * hlt_var_tag_h0

                writer.add_systematic(
                    addHists(hlt_var_total_h0, h0),
                    f"hlt_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_stst",
                    constrained=False,
                    groups=["eff_trig"],
                )

                id_var_tag_h2 = create_variation(
                    id_var_h2, dtdt_prpg_ex, i, j, k, nbins_h2, h2=True
                )
                id_probe_h2 = create_variation(
                    id_var_h2, dtdt_prpg_ex, i, j, k, 0, "probe", h2=True
                )
                id_probe_h2 = multiplyHists(scaleHist(id_probe_h2, var_size), h2)
                id_var_total_h2 = scaleHist(addHists(id_var_tag_h2, id_probe_h2), 1 / 2)

                writer.add_systematic(
                    addHists(id_var_total_h2, h2),
                    f"id_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtdt",
                    constrained=False,
                    groups=["eff_id"],
                )

            id_var_tag_h1 = create_variation(
                id_var_nom, dtst_prpg_ex, i, j, k, nbins_h1
            )
            id_probe_h1 = create_variation(
                id_var_nom, dtst_prpg_ex, i, j, k, 0, "probe"
            )
            id_probe_h1 = multiplyHists(scaleHist(id_probe_h1, var_size), h1)
            id_var_total_h1 = addHists(id_var_tag_h1, id_probe_h1)

            writer.add_systematic(
                addHists(id_var_total_h1, h1),
                f"id_prime_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtst",
                constrained=False,
                groups=["eff_id"],
            )

            id_var_tag_h0 = create_variation(
                id_var_nom, stst_prpg_ex, i, j, k, nbins_h1
            )
            id_probe_h0 = create_variation(
                id_var_nom, stst_prpg_ex, i, j, k, 0, "probe"
            )
            id_probe_h0 = multiplyHists(scaleHist(id_probe_h0, var_size), h0)
            id_var_total_h0 = addHists(id_var_tag_h0, -1 * id_probe_h0)

            writer.add_systematic(
                addHists(id_var_total_h0, h0),
                f"id_prime_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_stst",
                constrained=False,
                groups=["eff_id"],
            )

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
        time_proj_hlt,
        time_proj_low,
        lumi_scaling,
        [lumi_scaling_h, lumi_scaling_bg],
        proc_name,
        f"bkg_{proc_name}",
        fail_gen=fgen,
    )


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
# ## PCC cross detector
# luminometer_syst(
#     writer, "pcc", dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc, "stability"
# )
# # ## HFOC cross detector
# luminometer_syst(
#     writer, "hfoc", dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc, "stability"
# )

# #### RAMSES cross detector
# luminometer_syst(
#     writer, "ramses", dtdt_prpg_ramses, dtst_prpg_ramses, stst_prpg_ramses, "stability"
# )


## seems to generate about the same amount of statistical uncertainty and together the uncertainties on each are higher so they are somehow linked which is a problem

### YEAH THESE ARE 100% COUPLED. CRAP.
#### HFOC linearity
# luminometer_syst(
#     writer,
#     "hfoc",
#     dtdt_prpg_sbil_hfoc,
#     dtst_prpg_sbil_hfoc,
#     stst_prpg_sbil_hfoc,
#     "linearity",
# )

# #### RAMSES linearity
# luminometer_syst(
#     writer,
#     "ramses",
#     dtdt_prpg_sbil_ramses,
#     dtst_prpg_sbil_ramses,
#     stst_prpg_sbil_ramses,
#     "linearity",
# )

writer.write(outfolder="./", outfilename="liv_ALL")
