import argparse
import pdb

import h5py
from uncertainty_tools import (
    all_mc_corrections,
    eta_phi_systematic,
    get_era_vals,
    get_mc_lumis,
    luminometer_syst,
    make_ones_hist,
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


file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"
h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

data_output = results["dataPostVFP"]["output"]
lumi_output = results["dataPostVFP"]["lumi_outout"]
MC_Zmumu = results["ZmumuPostVFP"]["output"]


reco_dtdt_data = data_output["time_mll"].get()
reco_dtst_data = data_output["time_mll_dtst"].get()
reco_stst_data = data_output["time_mll_stst"].get()
time_proj_low = data_output["time_proj_2"].get()[{"mll": mass_bin, "gen_mll": mass_bin}]
time_proj_hlt = data_output["time_proj"].get()[{"mll": mass_bin, "gen_mll": mass_bin}]
time_proj_low_all = data_output["time_proj_2"].get()
time_proj_hlt_all = data_output["time_proj"].get()
### pass reco, pass generator

dtdt_prpg = MC_Zmumu["mll_dtdt_prpg"].get()
dtst_prpg = MC_Zmumu["mll_dtst_prpg"].get()
stst_prpg = MC_Zmumu["mll_stst_prpg"].get()
weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

dtdt_prfg = MC_Zmumu["mll_dtdt_prfg"].get()
dtst_prfg = MC_Zmumu["mll_dtst_prfg"].get()
stst_prfg = MC_Zmumu["mll_stst_prfg"].get()

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

nbins_mll = len(dtdt_prfg.axes["mll"])
nbins_time = len(reco_dtst_data.axes["time"])
nbins_pt_leading = len(reco_dtst_data.axes["pt_lead"])
nbins_eta_leading = len(reco_dtdt_data.axes["eta_lead"])


hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
hfoc_scaling = multiplyHists(hfoc_scaling, lumi_scaling)

pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
pcc_scaling = multiplyHists(pcc_scaling, lumi_scaling)

ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
ramses_scaling = multiplyHists(ramses_scaling, lumi_scaling)

dtdt_prpg_H = dtdt_prpg_H[{"mll": mass_bin, "gen_mll": mass_bin}]
dtst_prpg_H = dtst_prpg_H[{"mll": mass_bin, "gen_mll": mass_bin}]
stst_prpg_H = stst_prpg_H[{"mll": mass_bin, "gen_mll": mass_bin}]
dtdt_prpg_BG = dtdt_prpg_BG[{"mll": mass_bin, "gen_mll": mass_bin}]
dtst_prpg_BG = dtst_prpg_BG[{"mll": mass_bin, "gen_mll": mass_bin}]
stst_prpg_BG = stst_prpg_BG[{"mll": mass_bin, "gen_mll": mass_bin}]


dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc = get_mc_lumis(
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
    time_proj_hlt,
    time_proj_low,
    hfoc_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    weightsum,
    cross_sec,
)

dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc = get_mc_lumis(
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
    time_proj_hlt,
    time_proj_low,
    pcc_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    weightsum,
    cross_sec,
)


dtdt_prpg_ramses, dtst_prpg_ramses, stst_prpg_ramses = get_mc_lumis(
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
    time_proj_hlt,
    time_proj_low,
    ramses_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
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

dtdt_prpg_sbil_hfoc, dtst_prpg_sbil_hfoc, stst_prpg_sbil_hfoc = get_mc_lumis(
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
    time_proj_hlt,
    time_proj_low,
    sbil_hfoc_fit,
    lumi_scaling_h,
    lumi_scaling_bg,
    weightsum,
    cross_sec,
)


dtdt_prpg_sbil_ramses, dtst_prpg_sbil_ramses, stst_prpg_sbil_ramses = get_mc_lumis(
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
    time_proj_hlt,
    time_proj_low,
    sbil_ramses_fit,
    lumi_scaling_h,
    lumi_scaling_bg,
    weightsum,
    cross_sec,
)


#### normal
dtdt_prpg, dtst_prpg, stst_prpg = get_mc_lumis(
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
    time_proj_hlt,
    time_proj_low,
    lumi_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    weightsum,
    cross_sec,
)

(dtdt_prpg_prefiring_syst, dtst_prpg_prefiring_syst, stst_prpg_prefiring_syst) = (
    get_mc_lumis(
        dtdt_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
        dtst_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
        stst_prpg_H_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
        dtdt_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
        dtst_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
        stst_prpg_BG_syst[{"downUpVar": 0, "mll": mass_bin, "gen_mll": mass_bin}],
        time_proj_hlt,
        time_proj_low,
        lumi_scaling,
        lumi_scaling_h,
        lumi_scaling_bg,
        weightsum,
        cross_sec,
    )
)


pass_gen = all_mc_corrections(
    pass_gen[{"mll": mass_bin, "gen_mll": mass_bin}],
    time_proj_hlt,
    lumi_scaling,
    weightsum,
    cross_sec,
)

h2_first = dtdt_prpg.project("time", "pt_lead", "eta_lead")
h1_second = dtst_prpg.project("time", "pt_sublead", "eta_sublead")
h0_second = stst_prpg.project("time", "pt_sublead", "eta_sublead")
n_masked = pass_gen.project("time", "pt_lead", "eta_lead")
#### i think i will need to mix this too but it doesn't affect the fit

reco_dtdt_data = reco_dtdt_data[{"mll": mass_bin}].project(
    "time", "pt_lead", "eta_lead"
)
reco_dtst_data = reco_dtst_data[{"mll": mass_bin}].project(
    "time", "pt_sublead", "eta_sublead"
)
reco_stst_data = reco_stst_data[{"mll": mass_bin}].project(
    "time", "pt_sublead", "eta_sublead"
)


## create the tensor
writer = tensorwriter.TensorWriter()
##g# enerator channel
writer.add_channel(n_masked.axes, "ch_masked", masked=True)
writer.add_process(
    divideHists(n_masked, lumi_scaling), "Zmumu pass gen", "ch_masked", signal=False
)
### efficiency channels


writer.add_channel(
    reco_dtdt_data.axes, "ch_dtdt"
)  ### i have implicity selected a muon, this may be bad later
writer.add_data(reco_dtdt_data, "ch_dtdt")
writer.add_process(h2_first, "Zmumu pass gen", "ch_dtdt", signal=False)

writer.add_channel(reco_dtst_data.axes, "ch_dtst")
writer.add_data(reco_dtst_data, "ch_dtst")
writer.add_process(h1_second, "Zmumu pass gen", "ch_dtst", signal=False)

writer.add_channel(reco_stst_data.axes, "ch_stst")
writer.add_data(reco_stst_data, "ch_stst")
writer.add_process(h0_second, "Zmumu pass gen", "ch_stst", signal=False)

### adding axes as appropriate to make everything 4 dimensional
dtdt_prpg = expand_hist_by_duplicate_axis(dtdt_prpg, "time", "gen_time")
dtst_prpg = expand_hist_by_duplicate_axis(dtst_prpg, "time", "gen_time")
stst_prpg = expand_hist_by_duplicate_axis(stst_prpg, "time", "gen_time")

h2_first_var_lead = expand_hist_by_duplicate_axes(
    h2_first, ["time", "pt_lead", "eta_lead"], ["gen_time", "pt_prime", "eta_prime"]
)
h1_second_var = expand_hist_by_duplicate_axes(
    h1_second,
    ["time", "pt_sublead", "eta_sublead"],
    ["gen_time", "pt_prime", "eta_prime"],
)
h0_second_var = expand_hist_by_duplicate_axes(
    h0_second,
    ["time", "pt_sublead", "eta_sublead"],
    ["gen_time", "pt_prime", "eta_prime"],
)


pdb.set_trace()
pass_gen_expanded = expand_hist_by_duplicate_axes(pass_gen, ["time"], ["gen_time"])
### so at this point i have already selected the mass bin, need to iterate over pt, eta, time
for i in range(0, nbins_pt_leading):  # just select two mass bins in the center
    for j in range(nbins_eta_leading):  # eta
        for k in range(nbins_time):  #  time
            var_size = 0.01

            # ### be more consistent about ordering of time and mll
            ## redo the naming convention
            v21 = dtdt_prpg[{"pt_sublead": i, "eta_sublead": j, "gen_time": k}]
            var2 = addHists(v21 * var_size, h2_first)
            v22 = h2_first_var_lead[{"pt_prime": i, "eta_prime": j, "gen_time": k}]
            # below same as
            # var2[{"pt_lead":i, "eta_lead":j, "time":k}] = var2[{"pt_lead":i, "eta_lead":j, "time":k}] + var_size + h2_first[{"pt_lead":i, "eta_lead":j, "time":k}]
            var2 = addHists(v22 * var_size, var2)
            writer.add_systematic(
                var2,
                f"n_pt{i}_eta_{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtdt",
                constrained=False,
                groups=["nz"],
            )

            writer.add_systematic(
                var2,
                f"hlt_prime_pt{i}_eta_{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtdt",
                constrained=False,
                groups=["eff_2"],
            )

            writer.add_systematic(
                var2,
                f"id_prime_pt{i}_eta_{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtdt",
                constrained=False,
                groups=["eff_1"],
            )

            v11 = dtst_prpg[{"pt_lead": i, "eta_lead": j, "gen_time": k}]
            var1 = addHists(v11 * var_size, h1_second)
            v12 = h1_second_var[{"pt_prime": i, "eta_prime": j, "gen_time": k}]
            var1_hlt = addHists(v12 * (-var_size), var1.copy())
            var1_id = addHists(v12 * var_size, var1)

            writer.add_systematic(
                var1_id,
                f"n_pt{i}_eta_{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtst",
                constrained=False,
                groups=["nz"],
            )
            if (
                i > 1
            ):  ### need to check pt bins, if i choose a pt bin above 1, this doesn't matter. it poses a problem
                writer.add_systematic(
                    var1_hlt * 2,
                    f"hlt_prime_pt{i}_eta_{j}_time{k}",
                    "Zmumu pass gen",
                    "ch_dtst",
                    constrained=False,
                    groups=["eff_2"],
                )

            writer.add_systematic(
                var1_id * 2,
                f"id_prime_pt{i}_eta_{j}_time{k}",
                "Zmumu pass gen",
                "ch_dtst",
                constrained=False,
                groups=["eff_1"],
            )

            v01 = stst_prpg[{"pt_lead": i, "eta_lead": j, "gen_time": k}]
            var0_hlt = addHists(v01 * var_size, h0_second)
            v02 = h0_second_var[{"pt_prime": i, "eta_prime": j, "gen_time": k}]
            var0_id = addHists(v02 * (-var_size), var0_hlt.copy())
            var0 = addHists(v02 * var_size, var0_hlt)
            # pdb.set_trace()
            writer.add_systematic(
                var0,
                f"n_pt{i}_eta_{j}_time{k}",
                "Zmumu pass gen",
                "ch_stst",
                constrained=False,
                groups=["nz"],
            )

            writer.add_systematic(
                var0_hlt * 2,
                f"hlt_prime_pt{i}_eta_{j}_time{k}",
                "Zmumu pass gen",
                "ch_stst",
                constrained=False,
                groups=["eff_2"],
            )

            writer.add_systematic(
                var0_id * 2,
                f"id_prime_pt{i}_eta_{j}_time{k}",
                "Zmumu pass gen",
                "ch_stst",
                constrained=False,
                groups=["eff_1"],
            )

            # # for masked channel --- may need to change bc this has a different number of dimensions

            v_masked = pass_gen_expanded[
                {"pt_sublead": i, "eta_sublead": j, "gen_time": k}
            ]
            var_masked = addHists(v_masked * var_size, n_masked)
            cross_section_masked = divideHists(var_masked, lumi_scaling)
            writer.add_systematic(
                cross_section_masked,
                f"n_pt{i}_eta_{j}_time{k}",
                "Zmumu pass gen",
                "ch_masked",
                constrained=False,
                groups=["nz"],
            )


###### NONE OF THIS IS MASS DEPENDENT
num_etaphi = len(dtdt_prpg_H_stat.project("etaPhiRegion").values())
for i in range(num_etaphi):
    eta_phi_systematic(
        writer,
        dtdt_prpg_H_stat[{"etaPhiRegion": i, "mll": mass_bin, "gen_mll": mass_bin}],
        dtst_prpg_H_stat[{"etaPhiRegion": i, "mll": mass_bin, "gen_mll": mass_bin}],
        stst_prpg_H_stat[{"etaPhiRegion": i, "mll": mass_bin, "gen_mll": mass_bin}],
        dtdt_prpg_BG_stat[{"etaPhiRegion": i, "mll": mass_bin, "gen_mll": mass_bin}],
        dtst_prpg_BG_stat[{"etaPhiRegion": i, "mll": mass_bin, "gen_mll": mass_bin}],
        stst_prpg_BG_stat[{"etaPhiRegion": i, "mll": mass_bin, "gen_mll": mass_bin}],
        time_proj_hlt,
        time_proj_low,
        lumi_scaling,
        lumi_scaling_h,
        lumi_scaling_bg,
        weightsum,
        cross_sec,
        i,
    )

writer.add_systematic(
    dtdt_prpg_prefiring_syst.project("time", "pt_lead", "eta_lead"),
    f"prefiring_syst",
    "Zmumu pass gen",
    "ch_dtdt",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    dtst_prpg_prefiring_syst.project("time", "pt_sublead", "eta_sublead"),
    f"prefiring_syst",
    "Zmumu pass gen",
    "ch_dtst",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    stst_prpg_prefiring_syst.project("time", "pt_sublead", "eta_sublead"),
    f"prefiring_syst",
    "Zmumu pass gen",
    "ch_stst",
    constrained=True,
    groups=["prefiring_syst"],
)

# background_syst_names = [
#     "ZmumuPostVFP",
#     "Top",
#     "Diboson",
#     "GGToLLPostVFP",
#     "QCDmuEnrichPt15PostVFP",
#     "WplusmunuPostVFP",
#     "QGToDYQTo2LPostVFP",
#     "QGToWQToLNuPostVFP",
# ]
# background_proc = [
#     "Zmumu fail gen",
#     "Top",
#     "Diboson",
#     "GG",
#     "QCD",
#     "W",
#     "QG_2L",
#     "QG_Lnu",
# ]
# for i in range(len(background_syst_names)):
#     proc_name = background_proc[i]
#     if proc_name == "Zmumu fail gen":
#         fgen = True
#     else:
#         fgen = False
#     background_syst(
#         writer,
#         results,
#         background_syst_names[i],
#         time_proj_hlt_all,
#         time_proj_low_all,
#         lumi_scaling,
#         proc_name,
#         f"bkg_{proc_name}",
#         fail_gen=fgen,
#     )


### PCC cross detector
luminometer_syst(
    writer, "pcc", dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc, "stability"
)
## HFOC cross detector
luminometer_syst(
    writer, "hfoc", dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc, "stability"
)

#### RAMSES cross detector
luminometer_syst(
    writer, "ramses", dtdt_prpg_ramses, dtst_prpg_ramses, stst_prpg_ramses, "stability"
)


#### HFOC linearity
luminometer_syst(
    writer,
    "hfoc",
    dtdt_prpg_sbil_hfoc,
    dtst_prpg_sbil_hfoc,
    stst_prpg_sbil_hfoc,
    "linearity",
)

#### RAMSES linearity
luminometer_syst(
    writer,
    "ramses",
    dtdt_prpg_sbil_ramses,
    dtst_prpg_sbil_ramses,
    stst_prpg_sbil_ramses,
    "linearity",
)

writer.write(outfolder="./", outfilename="liv")
