import argparse

import h5py
from uncertainty_tools import (
    background_syst,
    create_variation,
    eta_phi_systematic,
    get_era_vals,
    get_mc_lumis,
    luminometer_syst,
    make_mutually_exclusive,
    make_ones_hist,
    mc_scaling,
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
    unrolledHist,
)

parser = argparse.ArgumentParser()
args = parser.parse_args()

slope_ramses = 0.0006
slope_hfoc = 0.0007
mass_bin = -1
var_size = 0.01

background_syst_names = [
    # "ZmumuPostVFP",
    "Top",
    "Diboson",
    "GGToLL_2016PostVFP",
    "QCDmuEnrichPt15_2016PostVFP",
    # "Wplusmunu_2016PostVFP",
    "QGToDYQTo2L_2016PostVFP",
    "QGToWQToLNu_2016PostVFP",
]
background_proc = [
    # "Zmumu fail gen",
    "Top",
    "Diboson",
    "GG",
    "QCD",
    # "W",
    "QG_2L",
    "QG_Lnu",
]

######################################################################
# DATA IMPORTS #

file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr.hdf5"
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)
data_output = results["SingleMuon_2016PostVFP"]["output"]
lumi_output = results["SingleMuon_2016PostVFP"]["lumi_outout"]
MC_Zmumu = results["Zmumu_2016PostVFP"]["output"]

dtdt_data = data_output["time_mll"].get()
dtst_data = data_output["time_dtst"].get()
stst_data = data_output["time_stst"].get()
iso_data = data_output["time_iso"].get()

iso_data, dtdt_data, dtst_data, stst_data = make_mutually_exclusive(
    iso_data, dtdt_data, dtst_data, stst_data
)

time_proj_low = data_output["time_proj"].get()

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

lumi_hists = [lumi_scaling_h, lumi_scaling_bg]

avg_sbil_pcc = scaleHist(divideHists(sbil_pcc, count_pcc), 1e9)
sbil_hfoc_fit = scaleHist(avg_sbil_pcc, slope_hfoc)

sbil_ones = make_ones_hist(sbil_hfoc_fit)
sbil_hfoc_fit = addHists(sbil_hfoc_fit, sbil_ones)
hfoc_sbil = multiplyHists(sbil_hfoc_fit, lumi_scaling)

sbil_ramses_fit = scaleHist(avg_sbil_pcc, slope_ramses)
sbil_ramses_fit = addHists(sbil_ramses_fit, sbil_ones)
ramses_sbil = multiplyHists(sbil_ramses_fit, lumi_scaling)

# time_proj_low = time_proj_low[{"mll": mass_bin}]


def get_corrected_mc(
    results,
    process,
    time_proj_low,
    lumi_hists,
    pcc_scaling,
    hfoc_scaling,
    ramses_scaling,
    hfoc_sbil,
    ramses_sbil,
    luminometers=False,
):
    MC = results[process]["output"]

    iso_BG, iso_BG_syst, iso_BG_stat = get_era_vals(MC, "pass_iso", "BG", iso=True)
    dtdt_BG, dtdt_BG_syst, dtdt_BG_stat = get_era_vals(MC, "dtdt", "BG")
    dtst_BG, dtst_BG_syst, dtst_BG_stat = get_era_vals(MC, "dtst", "BG")
    stst_BG, stst_BG_syst, stst_BG_stat = get_era_vals(MC, "stst", "BG")

    iso_H, iso_H_syst, iso_H_stat = get_era_vals(MC, "pass_iso", "H", iso=True)
    dtdt_H, dtdt_H_syst, dtdt_H_stat = get_era_vals(MC, "dtdt", "H")
    dtst_H, dtst_H_syst, dtst_H_stat = get_era_vals(MC, "dtst", "H")
    stst_H, stst_H_syst, stst_H_stat = get_era_vals(MC, "stst", "H")
    syst = [
        iso_H_syst[{"downUpVar": 0}],
        dtdt_H_syst[{"downUpVar": 0}],
        dtst_H_syst[{"downUpVar": 0}],
        stst_H_syst[{"downUpVar": 0}],
        iso_BG_syst[{"downUpVar": 0}],
        dtdt_BG_syst[{"downUpVar": 0}],
        dtst_BG_syst[{"downUpVar": 0}],
        stst_BG_syst[{"downUpVar": 0}],
    ]
    stat = [
        iso_H_stat[{"downUpVar": 0}],
        dtdt_H_stat[{"downUpVar": 0}],
        dtst_H_stat[{"downUpVar": 0}],
        stst_H_stat[{"downUpVar": 0}],
        iso_BG_stat[{"downUpVar": 0}],
        dtdt_BG_stat[{"downUpVar": 0}],
        dtst_BG_stat[{"downUpVar": 0}],
        stst_BG_stat[{"downUpVar": 0}],
    ]

    prpg_all = [
        iso_H,
        dtdt_H,
        dtst_H,
        stst_H,
        iso_BG,
        dtdt_BG,
        dtst_BG,
        stst_BG,
    ]

    weightsum = results[process]["weight_sum"]
    xsec = results[process]["dataset"]["xsec"]

    pass_gen = MC["pass_gen"].get()

    iso, dtdt, dtst, stst = get_mc_lumis(
        prpg_all,
        time_proj_low,
        lumi_scaling,
        lumi_hists,
        weightsum,
        xsec,
    )
    print("through nominal corrections")
    if luminometers:
        ### i think the way these should work is i do it in a single mass bin then project it across all the rest?

        hfoc_rescale = divideHists(hfoc_scaling, lumi_scaling)
        iso_hfoc = multiplyHists(hfoc_rescale, iso)
        dtdt_hfoc = multiplyHists(hfoc_rescale, dtdt)
        dtst_hfoc = multiplyHists(hfoc_rescale, dtst)
        stst_hfoc = multiplyHists(hfoc_rescale, stst)

        print("hfoc stability")
        pcc_rescale = divideHists(pcc_scaling, lumi_scaling)
        iso_pcc = multiplyHists(pcc_rescale, iso)
        dtdt_pcc = multiplyHists(pcc_rescale, dtdt)
        dtst_pcc = multiplyHists(pcc_rescale, dtst)
        stst_pcc = multiplyHists(pcc_rescale, stst)
        print("pcc stability")

        ramses_rescale = divideHists(ramses_scaling, lumi_scaling)
        iso_ramses = multiplyHists(ramses_rescale, iso)
        dtdt_ramses = multiplyHists(ramses_rescale, dtdt)
        dtst_ramses = multiplyHists(ramses_rescale, dtst)
        stst_ramses = multiplyHists(ramses_rescale, stst)
        print("ramses stability")

        hfoc_sbil_rescale = divideHists(hfoc_sbil, lumi_scaling)
        iso_sbil_hfoc = multiplyHists(hfoc_sbil_rescale, iso)
        dtdt_sbil_hfoc = multiplyHists(hfoc_sbil_rescale, dtdt)
        dtst_sbil_hfoc = multiplyHists(hfoc_sbil_rescale, dtst)
        stst_sbil_hfoc = multiplyHists(hfoc_sbil_rescale, stst)
        print("hfoc linearity")

        ramses_sbil_rescale = divideHists(ramses_sbil, lumi_scaling)
        iso_sbil_ramses = multiplyHists(ramses_sbil_rescale, iso)
        dtdt_sbil_ramses = multiplyHists(ramses_sbil_rescale, dtdt)
        dtst_sbil_ramses = multiplyHists(ramses_sbil_rescale, dtst)
        stst_sbil_ramses = multiplyHists(ramses_sbil_rescale, stst)
        print("ramses linearity")

        iso_pcc, dtdt_pcc, dtst_pcc, stst_pcc = make_mutually_exclusive(
            iso_pcc, dtdt_pcc, dtst_pcc, stst_pcc
        )
        iso_hfoc, dtdt_hfoc, dtst_hfoc, stst_hfoc = make_mutually_exclusive(
            iso_hfoc, dtdt_hfoc, dtst_hfoc, stst_hfoc
        )
        iso_sbil_hfoc, dtdt_sbil_hfoc, dtst_sbil_hfoc, stst_sbil_hfoc = (
            make_mutually_exclusive(
                iso_sbil_hfoc, dtdt_sbil_hfoc, dtst_sbil_hfoc, stst_sbil_hfoc
            )
        )
        iso_sbil_ramses, dtdt_sbil_ramses, dtst_sbil_ramses, stst_sbil_ramses = (
            make_mutually_exclusive(
                iso_sbil_ramses, dtdt_sbil_ramses, dtst_sbil_ramses, stst_sbil_ramses
            )
        )

    ### dtdt was negative from before this was all passed into a single function. need to investigate further
    iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire = get_mc_lumis(
        syst,
        time_proj_low,
        lumi_scaling,
        lumi_hists,
        weightsum,
        xsec,
    )
    print("prefiring")

    iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire = make_mutually_exclusive(
        iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire
    )

    pass_gen = mc_scaling(
        pass_gen[{"mll": mass_bin}],
        time_proj_low,
        lumi_scaling,
        weightsum,
        xsec,
    )
    iso, dtdt, dtst, stst = make_mutually_exclusive(iso, dtdt, dtst, stst)
    corrected_mc = [iso, dtdt, dtst, stst]
    corrected_prefiring = [iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire]

    if luminometers:
        pcc_stability = [iso_pcc, dtdt_pcc, dtst_pcc, stst_pcc]
        hfoc_stability = [iso_hfoc, dtdt_hfoc, dtst_hfoc, stst_hfoc]
        ramses_stability = [iso_ramses, dtdt_ramses, dtst_ramses, stst_ramses]
        hfoc_linearity = [iso_sbil_hfoc, dtdt_sbil_hfoc, dtst_sbil_hfoc, stst_sbil_hfoc]
        ramses_linearity = [
            iso_sbil_ramses,
            dtdt_sbil_ramses,
            dtst_sbil_ramses,
            stst_sbil_ramses,
        ]

        return (
            corrected_mc,
            pass_gen,
            corrected_prefiring,
            stat,
            weightsum,
            xsec,
            pcc_stability,
            hfoc_stability,
            ramses_stability,
            hfoc_linearity,
            ramses_linearity,
        )

    else:
        return corrected_mc, pass_gen, corrected_prefiring, stat


(
    Zmumu_mc,
    Zmumu_pass_gen,
    Zmumu_prefire,
    Zmumu_stat,
    weightsum,
    cross_sec,
    pcc_stability,
    hfoc_stability,
    ramses_stability,
    hfoc_linearity,
    ramses_linearity,
) = get_corrected_mc(
    results,
    "Zmumu_2016PostVFP",
    time_proj_low,
    lumi_hists,
    pcc_scaling,
    hfoc_scaling,
    ramses_scaling,
    hfoc_sbil,
    ramses_sbil,
    luminometers=True,
)
iso_prefire, dtdt_prefire, dtst_prefire, stst_prefire = Zmumu_prefire


iso_mc, dtdt_mc, dtst_mc, stst_mc = Zmumu_mc
pass_gen = Zmumu_pass_gen

iso_sbil_ramses, dtdt_sbil_ramses, dtst_sbil_ramses, stst_sbil_ramses = ramses_linearity
iso_pcc, dtdt_pcc, dtst_pcc, stst_pcc = pcc_stability

iso_hfoc, dtdt_hfoc, dtst_hfoc, stst_hfoc = hfoc_stability
iso_ramses, dtdt_ramses, dtst_ramses, stst_ramses = ramses_stability

iso_sbil_hfoc, dtdt_sbil_hfoc, dtst_sbil_hfoc, stst_sbil_hfoc = hfoc_linearity


n_masked = pass_gen.project("time", "mll", "pt_probe", "eta_probe")

iso_var_nom = divideHists(
    iso_mc.project("time", "mll", "pt_probe", "eta_probe"),
    iso_mc.project("time", "mll", "pt_probe", "eta_probe"),
)  ## just want this to be one

dtdt_all = addHists(iso_mc, dtdt_mc)
dtst_all = addHists(dtdt_all, dtst_mc)
stst_all = addHists(dtst_all, stst_mc)

hlt_var_nom = divideHists(
    dtdt_all.project("time", "mll", "pt_probe", "eta_probe"),
    dtst_all.project("time", "mll", "pt_probe", "eta_probe"),
)

id_var_nom = divideHists(
    dtst_all.project("time", "mll", "pt_probe", "eta_probe"),
    stst_all.project("time", "mll", "pt_probe", "eta_probe"),
)

dtdt_mc = remove_low_bins(dtdt_mc)
dtdt_data = remove_low_bins(dtdt_data)

h3 = iso_mc.project("time", "mll", "pt_probe", "eta_probe")
h2 = dtdt_mc.project("time", "mll", "pt_probe", "eta_probe")
h1 = dtst_mc.project("time", "mll", "pt_probe", "eta_probe")
h0 = stst_mc.project("time", "mll", "pt_probe", "eta_probe")


h3_data = iso_data.project("time", "mll", "pt_probe", "eta_probe")
h2_data = dtdt_data.project("time", "mll", "pt_probe", "eta_probe")
h1_data = dtst_data.project("time", "mll", "pt_probe", "eta_probe")
h0_data = stst_data.project("time", "mll", "pt_probe", "eta_probe")

print("begin expanding")

iso_proj = expand_hist_by_duplicate_axes(
    h3,
    ["time", "mll", "pt_probe", "eta_probe"],
    ["gen_time", "gen_mll", "pt_tag", "eta_tag"],
)

dtdt_prpg_proj = expand_hist_by_duplicate_axes(
    h2,
    ["time", "mll", "pt_probe", "eta_probe"],
    ["gen_time", "gen_mll", "pt_tag", "eta_tag"],
)
dtst_prpg_proj = expand_hist_by_duplicate_axes(
    h1,
    ["time", "mll", "pt_probe", "eta_probe"],
    ["gen_time", "gen_mll", "pt_tag", "eta_tag"],
)
stst_prpg_proj = expand_hist_by_duplicate_axes(
    h0,
    ["time", "mll", "pt_probe", "eta_probe"],
    ["gen_time", "gen_mll", "pt_tag", "eta_tag"],
)

hlt_var_nom = expand_hist_by_duplicate_axes(
    hlt_var_nom,
    ["time", "mll", "pt_probe", "eta_probe"],
    ["gen_time", "gen_mll", "pt_tag", "eta_tag"],
)
print("halfway through expanding")

id_var_nom = expand_hist_by_duplicate_axes(
    id_var_nom,
    ["time", "mll", "pt_probe", "eta_probe"],
    ["gen_time", "gen_mll", "pt_tag", "eta_tag"],
)
iso_var_nom = expand_hist_by_duplicate_axes(
    iso_var_nom,
    ["time", "mll", "pt_probe", "eta_probe"],
    ["gen_time", "gen_mll", "pt_tag", "eta_tag"],
)


id_var_h2 = remove_low_bins(id_var_nom.copy())

iso_var_nom_h2 = remove_low_bins(iso_var_nom.copy())

hlt_var_nom_h2 = remove_low_bins(hlt_var_nom.copy())


###################################################################

## create the tensor
writer = tensorwriter.TensorWriter()
##generator channel --> MAY BE WRONG BECAUSE THIS ISN'T MUTUTALLY EXCLUSIVE
### unrolling
# n_masked = unrolledHist(n_masked)
# h3_data = unrolledHist(h3_data)
# h2_data = unrolledHist(h2_data)
# h1_data = unrolledHist(h1_data)
# h0_data = unrolledHist(h0_data)

h3_unrolled = unrolledHist(h3)
h2_unrolled = unrolledHist(h2)
h1_unrolled = unrolledHist(h1)
h0_unrolled = unrolledHist(h0)

writer.add_channel(n_masked.axes, "ch_masked", masked=True)  ## is this still correct?
writer.add_process(divideHists(n_masked, lumi_scaling), "Zmumu", "ch_masked")

writer.add_channel(h3_data.axes, "ch_iso")
writer.add_data(h3_data, "ch_iso")
writer.add_process(h3, "Zmumu", "ch_iso")

writer.add_channel(h2_data.axes, "ch_dtdt")
writer.add_data(h2_data, "ch_dtdt")
writer.add_process(h2, "Zmumu", "ch_dtdt")

writer.add_channel(h1_data.axes, "ch_dtst")
writer.add_data(h1_data, "ch_dtst")
writer.add_process(h1, "Zmumu", "ch_dtst")

writer.add_channel(h0_data.axes, "ch_stst")
writer.add_data(h0_data, "ch_stst")
writer.add_process(h0, "Zmumu", "ch_stst")

pass_gen = expand_hist_by_duplicate_axis(pass_gen, "time", "gen_time")
pass_gen = expand_hist_by_duplicate_axis(pass_gen, "mll", "gen_mll")
nbins_h2 = (nbins_pt - 1) + nbins_eta + nbins_time + nbins_mll
nbins_h1 = nbins_pt + nbins_eta + nbins_time + nbins_mll
### so at this point i have already selected the mass bin, need to iterate over pt, eta, time

for i in range(nbins_pt):  # pt
    # for i in range(0, 3): #pt
    print(f"pt bin: {i}")
    # for j in range(4, 6):  # eta
    for j in range(nbins_eta):  # eta

        print(f"eta_bin: {j}")
        for k in range(nbins_time):  #  time
            # for k in range(22, 24):  #  time

            if i > 0:  ## we only have 1 bin beneath 25 GeV
                #### NORMALIZATION #####
                v2 = dtdt_prpg_proj[
                    {"gen_mll": mass_bin, "gen_time": k, "pt_tag": i - 1, "eta_tag": j}
                ]
                var2 = addHists(v2 * var_size, h2)

                writer.add_systematic(
                    var2,
                    f"n_pt{i}_eta{j}_time{k}",
                    "Zmumu",
                    "ch_dtdt",
                    constrained=False,
                    groups=["nz"],
                )

                ##### HLT #####
                hlt_var_tag_h3 = create_variation(
                    hlt_var_nom, iso_mc, i, j, k, nbins_h1
                )
                hlt_probe_h3 = create_variation(
                    hlt_var_nom, iso_mc, i, j, k, 0, "probe"
                )
                hlt_probe_h3 = multiplyHists(scaleHist(hlt_probe_h3, var_size), h3)
                hlt_var_total_h3 = addHists(hlt_var_tag_h3, hlt_probe_h3)

                writer.add_systematic(
                    addHists(hlt_var_total_h3, h3),
                    f"hlt_pt{i}_eta{j}_time{k}",
                    "Zmumu",
                    "ch_iso",
                    constrained=False,
                    groups=["eff_trig"],
                )

                hlt_var_tag_h2 = create_variation(
                    hlt_var_nom_h2, dtdt_mc, i, j, k, nbins_h2, h2=True
                )
                hlt_probe_h2 = create_variation(
                    hlt_var_nom_h2, dtdt_mc, i, j, k, 0, "probe", h2=True
                )
                hlt_probe_h2 = multiplyHists(scaleHist(hlt_probe_h2, var_size), h2)
                hlt_var_total_h2 = scaleHist(addHists(hlt_var_tag_h2, hlt_probe_h2), 2)

                writer.add_systematic(
                    addHists(hlt_var_total_h2, h2),
                    f"hlt_pt{i}_eta{j}_time{k}",
                    "Zmumu",
                    "ch_dtdt",
                    constrained=False,
                    groups=["eff_trig"],
                )

                hlt_var_tag_h1 = create_variation(
                    hlt_var_nom, dtst_mc, i, j, k, nbins_h1
                )
                hlt_probe_h1 = create_variation(
                    hlt_var_nom, dtst_mc, i, j, k, 0, "probe"
                )
                hlt_probe_h1 = multiplyHists(scaleHist(hlt_probe_h1, var_size), h1)
                hlt_var_total_h1 = scaleHist(
                    addHists(hlt_var_tag_h1, -1 * hlt_probe_h1), 2
                )

                writer.add_systematic(
                    addHists(hlt_var_total_h1, h1),
                    f"hlt_pt{i}_eta{j}_time{k}",
                    "Zmumu",
                    "ch_dtst",
                    constrained=False,
                    groups=["eff_trig"],
                )

                hlt_var_tag_h0 = create_variation(
                    hlt_var_nom, stst_mc, i, j, k, nbins_h1
                )
                hlt_var_total_h0 = 2 * hlt_var_tag_h0

                writer.add_systematic(
                    addHists(hlt_var_total_h0, h0),
                    f"hlt_pt{i}_eta{j}_time{k}",
                    "Zmumu",
                    "ch_stst",
                    constrained=False,
                    groups=["eff_trig"],
                )

                #### ID ######

                id_var_tag_h2 = create_variation(
                    id_var_h2, dtdt_mc, i, j, k, nbins_h2, h2=True
                )
                id_probe_h2 = create_variation(
                    id_var_h2, dtdt_mc, i, j, k, 0, "probe", h2=True
                )
                id_probe_h2 = multiplyHists(scaleHist(id_probe_h2, var_size), h2)
                id_var_total_h2 = addHists(id_var_tag_h2, id_probe_h2)

                writer.add_systematic(
                    addHists(id_var_total_h2, h2),
                    f"id_pt{i}_eta{j}_time{k}",
                    "Zmumu",
                    "ch_dtdt",
                    constrained=False,
                    groups=["eff_id"],
                )

                ### ISOLATION ### --> can only be valid when hlt is valid

                iso_var_tag_h2 = create_variation(
                    iso_var_nom_h2, dtdt_mc, i, j, k, nbins_h2, h2=True
                )
                iso_probe_h2 = create_variation(
                    iso_var_nom_h2, dtdt_mc, i, j, k, 0, "probe", h2=True
                )
                iso_probe_h2 = multiplyHists(scaleHist(iso_probe_h2, var_size), h2)
                iso_var_total_h2 = scaleHist(
                    addHists(iso_var_tag_h2, -1 * hlt_probe_h2), 2
                )

                writer.add_systematic(
                    addHists(iso_var_total_h2, h2),
                    f"iso_pt{i}_eta{j}_time{k}",
                    "Zmumu",
                    "ch_dtdt",
                    constrained=False,
                    groups=["eff_trig"],
                )

            iso_var_tag_h3 = create_variation(iso_var_nom, iso_mc, i, j, k, nbins_h1)
            iso_probe_h3 = create_variation(iso_var_nom, iso_mc, i, j, k, 0, "probe")

            iso_probe_h3 = multiplyHists(scaleHist(iso_probe_h3, var_size), h3)
            iso_var_total_h3 = addHists(iso_var_tag_h3, iso_probe_h3)

            writer.add_systematic(
                addHists(iso_var_total_h3, h3),
                f"iso_pt{i}_eta{j}_time{k}",
                "Zmumu",
                "ch_iso",
                constrained=False,
                groups=["eff_trig"],
            )

            iso_var_tag_h1 = create_variation(iso_var_nom, dtst_mc, i, j, k, nbins_h1)
            iso_probe_h1 = create_variation(iso_var_nom, dtst_mc, i, j, k, 0, "probe")
            iso_probe_h1 = multiplyHists(scaleHist(iso_probe_h1, var_size), h1)
            iso_var_total_h1 = scaleHist(addHists(iso_var_tag_h1, -1 * iso_probe_h1), 2)

            writer.add_systematic(
                addHists(iso_var_total_h1, h1),
                f"iso_pt{i}_eta{j}_time{k}",
                "Zmumu",
                "ch_dtst",
                constrained=False,
                groups=["eff_trig"],
            )

            iso_var_tag_h0 = create_variation(iso_var_nom, stst_mc, i, j, k, nbins_h1)
            iso_var_total_h0 = 2 * iso_var_tag_h0

            writer.add_systematic(
                addHists(iso_var_total_h0, h0),
                f"iso_pt{i}_eta{j}_time{k}",
                "Zmumu",
                "ch_stst",
                constrained=False,
                groups=["eff_trig"],
            )

            #### FOR THE LOWEST PT BIN ####

            ##### NORMALIZATION #####

            v3 = iso_proj[
                {"gen_mll": mass_bin, "gen_time": k, "pt_tag": i, "eta_tag": j}
            ]
            var3 = addHists(v3 * var_size, h3)

            writer.add_systematic(
                var3,
                f"n_pt{i}_eta{j}_time{k}",
                "Zmumu",
                "ch_iso",
                constrained=False,
                groups=["nz"],
            )

            v1 = dtst_prpg_proj[
                {"gen_mll": mass_bin, "gen_time": k, "pt_tag": i, "eta_tag": j}
            ]
            var1 = addHists(v1 * var_size, h1)
            writer.add_systematic(
                var1,
                f"n_pt{i}_eta{j}_time{k}",
                "Zmumu",
                "ch_dtst",
                constrained=False,
                groups=["nz"],
            )

            v0 = stst_prpg_proj[
                {"gen_mll": mass_bin, "gen_time": k, "pt_tag": i, "eta_tag": j}
            ]
            var0 = addHists(v0 * var_size, h0)
            writer.add_systematic(
                var0,
                f"n_pt{i}_eta{j}_time{k}",
                "Zmumu",
                "ch_stst",
                constrained=False,
                groups=["nz"],
            )
            # for masked channel --> IS NOT MUTUALLY EXCLUSIVE
            v_masked = pass_gen[
                {"gen_mll": mass_bin, "gen_time": k, "pt_tag": i, "eta_tag": j}
            ]
            var_masked = addHists(v_masked * var_size, n_masked)
            cross_section_masked = divideHists(var_masked, lumi_scaling)

            writer.add_systematic(
                cross_section_masked,
                f"n_pt{i}_eta{j}_time{k}",
                "Zmumu",
                "ch_masked",
                constrained=False,
                groups=["nz"],
            )

            ###### ID #####
            id_var_tag_h3 = create_variation(id_var_nom, iso_mc, i, j, k, nbins_h1)
            id_probe_h3 = create_variation(id_var_nom, iso_mc, i, j, k, 0, "probe")
            id_probe_h3 = multiplyHists(scaleHist(id_probe_h3, var_size), h3)
            id_var_total_h3 = scaleHist(addHists(id_var_tag_h3, id_probe_h3), 1 / 2)

            writer.add_systematic(
                addHists(id_var_total_h3, h3),
                f"id_pt{i}_eta{j}_time{k}",
                "Zmumu",
                "ch_iso",
                constrained=False,
                groups=["eff_id"],
            )

            id_var_tag_h1 = create_variation(id_var_nom, dtst_mc, i, j, k, nbins_h1)
            id_probe_h1 = create_variation(id_var_nom, dtst_mc, i, j, k, 0, "probe")
            id_probe_h1 = multiplyHists(scaleHist(id_probe_h1, var_size), h1)
            id_var_total_h1 = addHists(id_var_tag_h1, id_probe_h1)

            writer.add_systematic(
                addHists(id_var_total_h1, h1),
                f"id_pt{i}_eta{j}_time{k}",
                "Zmumu",
                "ch_dtst",
                constrained=False,
                groups=["eff_id"],
            )

            id_var_tag_h0 = create_variation(id_var_nom, stst_mc, i, j, k, nbins_h1)
            id_probe_h0 = create_variation(id_var_nom, stst_mc, i, j, k, 0, "probe")
            id_probe_h0 = multiplyHists(scaleHist(id_probe_h0, var_size), h0)
            id_var_total_h0 = addHists(id_var_tag_h0, -1 * id_probe_h0)

            writer.add_systematic(
                addHists(id_var_total_h0, h0),
                f"id_pt{i}_eta{j}_time{k}",
                "Zmumu",
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
        time_proj_low,
        lumi_scaling,
        [lumi_scaling_h, lumi_scaling_bg],
        proc_name,
        f"bkg_{proc_name}",
        fail_gen=fgen,
    )

### SO THESE SHOULD BE DONE ACROSS ALL MASS BINS
##### NONE OF THIS IS MASS DEPENDENT ## may be linked to statistical uncertainty becuase the eta region? do i still need this or am i double counting
# lowers stability uncetainty increase linearity uncertainty.
num_etaphi = len(Zmumu_stat[0].project("etaPhiRegion").values())
for i in range(num_etaphi):
    eta_phi_systematic(
        writer,
        Zmumu_stat,
        time_proj_low,
        lumi_scaling,
        lumi_hists,
        weightsum,
        cross_sec,
        i,
        mass_bin,
    )

### these slightly increase the statistical uncertainty but dont contribute to the stability/linearity

writer.add_systematic(
    iso_prefire.project("time", "mll", "pt_probe", "eta_probe"),
    f"prefiring_syst",
    "Zmumu",
    "ch_iso",
    constrained=True,
    groups=["prefiring_syst"],
)
dtdt_prefire = remove_low_bins(dtdt_prefire)
writer.add_systematic(
    dtdt_prefire.project(
        "time", "mll", "pt_probe", "eta_probe"
    ),  # used to be mll as well
    f"prefiring_syst",
    "Zmumu",
    "ch_dtdt",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    dtst_prefire.project(
        "time", "mll", "pt_probe", "eta_probe"
    ),  # used to be tag pt and eta
    f"prefiring_syst",
    "Zmumu",
    "ch_dtst",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    stst_prefire.project(
        "time", "mll", "pt_probe", "eta_probe"
    ),  # used to be tag pt and eta
    f"prefiring_syst",
    "Zmumu",
    "ch_stst",
    constrained=True,
    groups=["prefiring_syst"],
)


### statistical uncertainty and the stability and linearity still slightly linked (~0.003%)
luminometer_syst(writer, "pcc", iso_pcc, dtdt_pcc, dtst_pcc, stst_pcc, "stability")

## HFOC cross detector
luminometer_syst(
    writer,
    "hfoc",
    iso_sbil_hfoc,
    dtdt_sbil_hfoc,
    dtst_sbil_hfoc,
    stst_sbil_hfoc,
    "linearity",
)


luminometer_syst(
    writer,
    "hfoc",
    iso_hfoc,
    dtdt_hfoc,
    dtst_hfoc,
    stst_hfoc,
    "stability",
)

# #### RAMSES cross detector
luminometer_syst(
    writer,
    "ramses",
    iso_ramses,
    dtdt_ramses,
    dtst_ramses,
    stst_ramses,
    "stability",
)


### YEAH THESE ARE 100% COUPLED. CRAP.
# #### HFOC linearity

### RAMSES linearity
luminometer_syst(
    writer,
    "ramses",
    iso_sbil_ramses,
    dtdt_sbil_ramses,
    dtst_sbil_ramses,
    stst_sbil_ramses,
    "linearity",
)

writer.write(outfolder="./", outfilename="liv")
# writer.write(outfolder="./")
