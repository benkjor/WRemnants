import argparse
import pickle

import h5py
from uncertainty_tools import (
    all_mc_corrections,
    background_syst,
    get_era_vals,
    get_mc_lumis,
    make_mutually_exclusive,
    make_ones_hist,
    remove_low_bins,
)

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
mass_bin = 9
var_size = 0.01

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


# mc_before = all_mc_corrections(MC_Zmumu["mc_before"].get(), weightsum, cross_sec)
# print("MC BEFORE")
# print()
# print("DATA BEFORE")
# print(results["dataPostVFP"]["output"]["data_before"].get())

# print("MC AFTER")
# print(mc_scaling(MC_Zmumu["mc_after"].get(), weightsum, cross_sec))
# print("DATA AFTER")
# print(results["dataPostVFP"]["output"]["data_after"].get())


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

# pdb.set_trace()
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
    pass_gen[{"mll": mass_bin}],  # , "gen_mll": mass_bin}],
    time_proj_low,
    lumi_scaling,
    weightsum,
    cross_sec,
)

n_masked = pass_gen.project(
    "time", "pt_probe", "eta_probe"
)  ### choosing tag versus probe for this did not matter


dtdt_3d = dtdt_prpg.project("time", "pt_probe", "eta_probe")
dtst_3d = dtst_prpg.project("time", "pt_probe", "eta_probe")
stst_3d = stst_prpg.project("time", "pt_probe", "eta_probe")

dtdt_data = dtdt_data.project("time", "pt_probe", "eta_probe")
dtst_data = dtst_data.project("time", "pt_probe", "eta_probe")
stst_data = stst_data.project("time", "pt_probe", "eta_probe")

h2, h1, h0 = make_mutually_exclusive(dtdt_3d, dtst_3d, stst_3d)
h2_data, h1_data, h0_data = make_mutually_exclusive(dtdt_data, dtst_data, stst_data)

#### tag and probe efficiencies
efficiency_ones = make_ones_hist(h1)
# generate histogram of ones

hlt_var_nom = divideHists(dtdt_3d, dtst_3d)
id_var_nom = divideHists(dtst_3d, stst_3d)


eps_id_prime = 1.01
eps_hlt_prime = 1.01

##### for plotting efficiencies, not for the fit
efficiencies = {
    "eps_hlt_true": hlt_var_nom.values(),
    "eps_id_true": id_var_nom.values(),
}

with open("efficiency_values.pkl", "wb") as f:
    pickle.dump(efficiencies, f)


###################################################################333
#####  KNOW THIS IS CORRECT #####


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


h2_data = remove_low_bins(h2_data)  ### data
h2 = remove_low_bins(h2)  ### mc projected to 3d
dtdt_prpg = remove_low_bins(dtdt_prpg)  ### nominal mc in 5d
hlt_var_nom_h2 = remove_low_bins(hlt_var_nom)

## create the tensor
writer = tensorwriter.TensorWriter()
##generator channel
writer.add_channel(n_masked.axes, "ch_masked", masked=True)  ## is this still correct?
writer.add_process(
    divideHists(n_masked, lumi_scaling), "Zmumu pass gen", "ch_masked", signal=False
)

### okay these are all 3d (time, pt_probe, eta_probe which is how i want it. )


writer.add_channel(h2_data.axes, "ch_dtdt")
writer.add_data(h2_data, "ch_dtdt")
writer.add_process(h2, "Zmumu pass gen", "ch_dtdt", signal=False)

writer.add_channel(h1_data.axes, "ch_dtst")
writer.add_data(h1_data, "ch_dtst")
writer.add_process(h1, "Zmumu pass gen", "ch_dtst", signal=False)

writer.add_channel(h0_data.axes, "ch_stst")
writer.add_data(h0_data, "ch_stst")
writer.add_process(h0, "Zmumu pass gen", "ch_stst", signal=False)


stst_prpg_proj = stst_prpg.project(
    "time", "pt_probe", "eta_probe"
)  ### for some reason this projection is essential for the normalization
dtst_prpg_proj = dtst_prpg.project("time", "pt_probe", "eta_probe")
dtdt_prpg_proj = dtdt_prpg.project("time", "pt_probe", "eta_probe")

dtdt_prpg_proj = expand_hist_by_duplicate_axes(
    dtdt_prpg_proj, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)
dtst_prpg_proj = expand_hist_by_duplicate_axes(
    dtst_prpg_proj, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)
stst_prpg_proj = expand_hist_by_duplicate_axes(
    stst_prpg_proj, ["time", "pt_probe", "eta_probe"], ["gen_time", "pt_tag", "eta_tag"]
)


dtdt_prpg = expand_hist_by_duplicate_axis(dtdt_prpg, "time", "gen_time")
dtst_prpg = expand_hist_by_duplicate_axis(dtst_prpg, "time", "gen_time")
stst_prpg = expand_hist_by_duplicate_axis(stst_prpg, "time", "gen_time")

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


pass_gen = expand_hist_by_duplicate_axis(pass_gen, "time", "gen_time")


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


### so at this point i have already selected the mass bin, need to iterate over pt, eta, time
for i in range(1, 2):  # just select two pt bins in the center
    print(f"pt bin: {i}")
    for j in range(nbins_eta):  # eta
        for k in range(0, 1):  #  time

            if i > 0:  ## we only have 1 bin beneath 25 GeV
                ### be more consistent about ordering of time and mll
                ### fitting for the number of events

                ### so i may need a separate one for this, unclear
                v2 = dtdt_prpg_proj[
                    {"gen_time": k, "pt_tag": i - 1, "eta_tag": j}
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

            v1 = dtst_prpg_proj[
                {"gen_time": k, "pt_tag": i, "eta_tag": j}
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
            v0 = stst_prpg_proj[
                {"gen_time": k, "pt_tag": i, "eta_tag": j}
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
                hlt_var_tag_h2 = hlt_var_nom_h2[
                    {"gen_time": k, "pt_probe": i - 1, "eta_probe": j}
                ]  ## equivalent to n2
                hlt_var_tag_h2 = hlt_var_tag_h2.project(
                    "time", "pt_tag", "eta_tag"
                )  ### smarter way would be to project it at the beginning so things come out in the right order

                hlt_var_tag_h2 = broadcastSystHist(
                    hlt_var_tag_h2, dtdt_prpg
                )  ### so i think this puts the variation in all bins where pt_tag and eta_tag are what they should be

                hlt_var_tag_h1 = hlt_var_nom[
                    {"gen_time": k, "pt_probe": i, "eta_probe": j}
                ]  ## equivalent to n2
                hlt_var_tag_h1 = hlt_var_tag_h1.project(
                    "time", "pt_tag", "eta_tag"
                )  ### smarter way would be to project it at the beginning so things come out in the right order

                hlt_var_tag_h0 = broadcastSystHist(
                    hlt_var_tag_h1.copy(), stst_prpg
                )  ### so i think this puts the variation in all bins where pt_tag and eta_tag are what they should be

                hlt_var_tag_h1 = broadcastSystHist(
                    hlt_var_tag_h1, dtst_prpg
                )  ### so i think this puts the variation in all bins where pt_tag and eta_tag are what they should be
                hlt_var_tag_h2 = multiplyHists(
                    scaleHist(
                        hlt_var_tag_h2,
                        var_size / ((nbins_pt - 1) + nbins_eta + nbins_time),
                    ),
                    dtdt_prpg,
                )
                hlt_var_tag_h1 = multiplyHists(
                    scaleHist(
                        hlt_var_tag_h1, var_size / (nbins_pt + nbins_eta + nbins_time)
                    ),
                    dtst_prpg,
                )

                hlt_var_tag_h0 = multiplyHists(
                    scaleHist(
                        hlt_var_tag_h0, var_size / (nbins_pt + nbins_eta + nbins_time)
                    ),
                    stst_prpg,
                )

                hlt_var_tag_h2 = hlt_var_tag_h2.project("time", "pt_probe", "eta_probe")
                hlt_var_tag_h1 = hlt_var_tag_h1.project("time", "pt_probe", "eta_probe")
                hlt_var_tag_h0 = hlt_var_tag_h0.project("time", "pt_probe", "eta_probe")

                hlt_var_probe_h2 = hlt_var_nom_h2[
                    {"gen_time": k, "pt_tag": i - 1, "eta_tag": j}
                ]  ## equivalent to n2
                hlt_var_probe_h1 = hlt_var_nom[
                    {"gen_time": k, "pt_tag": i, "eta_tag": j}
                ]  ## equivalent to n2

                hlt_var_probe_h0 = multiplyHists(
                    scaleHist(hlt_var_probe_h1.copy(), var_size), h0.copy()
                )

                hlt_var_probe_h1 = multiplyHists(
                    scaleHist(hlt_var_probe_h1, var_size), h1
                )

                hlt_var_probe_h2 = multiplyHists(
                    scaleHist(hlt_var_probe_h2, var_size), h2
                )

                hlt_var_total_h2 = addHists(hlt_var_probe_h2, hlt_var_tag_h2)
                hlt_var_total_h1 = addHists(hlt_var_probe_h1, -1 * hlt_var_tag_h1)
                hlt_var_total_h0 = addHists(hlt_var_probe_h0, -1 * hlt_var_tag_h0)

                ### david was saying add then project.... this is project then add. we shall seeeeee which is right
                writer.add_systematic(
                    addHists(hlt_var_total_h2, h2),
                    f"hlt_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtdt",
                    constrained=False,
                    groups=["eff_trig"],
                )

                writer.add_systematic(
                    addHists(scaleHist(hlt_var_total_h1, -2), h1),
                    f"hlt_prime_pt{i}_eta{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtst",
                    constrained=False,
                    groups=["eff_trig"],
                )

                # writer.add_systematic(
                #     addHists(scaleHist(hlt_var_total_h0, -2), h0),
                #     f"hlt_prime_pt{i}_eta{j}_time{k}",
                #     "Zmumu pass gen",
                #     "ch_stst",
                #     constrained=False,
                #     groups=["eff_trig"],
                # )

                id_var_tag_h2 = id_var_h2[
                    {"gen_time": k, "pt_probe": i - 1, "eta_probe": j}
                ]  ## equivalent to n2

                id_var_tag_h2 = id_var_tag_h2.project("time", "pt_tag", "eta_tag")
                id_var_tag_h2 = broadcastSystHist(id_var_tag_h2, dtdt_prpg)
                id_var_tag_h2 = multiplyHists(
                    scaleHist(
                        id_var_tag_h2,
                        var_size / ((nbins_pt - 1) + nbins_eta + nbins_time),
                    ),
                    dtdt_prpg,
                )
                id_var_tag_h2 = id_var_tag_h2.project("time", "pt_probe", "eta_probe")

                id_var_probe_h2 = id_var_h2[
                    {"pt_tag": i - 1, "eta_tag": j, "gen_time": k}
                ]  ## equivalent to n2
                id_var_probe_h2 = multiplyHists(
                    scaleHist(id_var_probe_h2, var_size), h2
                )
                id_var_total_h2 = addHists(id_var_probe_h2, id_var_tag_h2)

                ### this constrains it a lot
                # writer.add_systematic(
                #     addHists(id_var_total_h2, 1/2*h2),
                #     f"id_prime_pt{i}_eta{j}_time{k}",
                #     "Zmumu pass gen",
                #     "ch_dtdt",
                #     constrained=False,
                #     groups=["eff_id"],
                # )

            id_var_tag = id_var_nom[
                {"gen_time": k, "pt_probe": i, "eta_probe": j}
            ]  ## equivalent to n2

            id_var_tag = id_var_tag.project("time", "pt_tag", "eta_tag")

            id_var_tag = broadcastSystHist(
                id_var_tag, dtst_prpg
            )  ### so i think this puts the variation in all bins where pt_tag and eta_tag are what they should be
            # /(nbins_time*nbins_eta*nbins_time)

            id_var_tag_h1 = multiplyHists(
                scaleHist(id_var_tag, var_size / (nbins_pt + nbins_eta + nbins_time)),
                dtst_prpg,
            )
            id_var_tag_h0 = multiplyHists(
                scaleHist(id_var_tag, var_size / (nbins_pt + nbins_eta + nbins_time)),
                stst_prpg,
            )

            id_var_tag_h1 = id_var_tag_h1.project("time", "pt_probe", "eta_probe")
            id_var_tag_h0 = id_var_tag_h0.project("time", "pt_probe", "eta_probe")

            id_var_probe = id_var_nom[
                {"pt_tag": i, "eta_tag": j, "gen_time": k}
            ]  ## equivalent to n2
            id_var_probe_h1 = multiplyHists(scaleHist(id_var_probe, var_size), h1)
            id_var_probe_h0 = multiplyHists(scaleHist(id_var_probe, var_size), h0)

            id_var_total_h1 = addHists(id_var_probe_h1, id_var_tag_h1)
            id_var_total_h0 = addHists(id_var_probe_h0, -1 * id_var_tag_h0)
            ### order of these is time, pt, eta

            writer.add_systematic(
                addHists(id_var_total_h1, h1),
                f"id_prime_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtst",
                constrained=False,
                groups=["eff_id"],
            )

            writer.add_systematic(
                addHists(scaleHist(id_var_total_h0, -1), h0),
                f"id_prime_pt{i}_eta{j}_time{k}",
                "Zmumu pass gen",
                "ch_stst",
                constrained=False,
                groups=["eff_id"],
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

writer.write(outfolder="./", outfilename="liv")
