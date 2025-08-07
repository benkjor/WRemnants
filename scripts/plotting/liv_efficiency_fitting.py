import argparse

import h5py
from uncertainty_tools import (
    all_mc_corrections,
    background_syst,
    eta_phi_systematic,
    get_eff_hist,
    get_era_vals,
    get_h0var,
    get_h1var,
    get_h2var,
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
time_proj = data_output["time_proj"].get()
time_proj_gen_mll = data_output["time_proj"].get().project("time", "gen_mll")
time_proj_mll = data_output["time_proj"].get().project("time", "mll")

### pass reco, pass generator

dtdt_prpg_true = MC_Zmumu["mll_dtdt_prpg"].get()
dtst_prpg_true = MC_Zmumu["mll_dtst_prpg"].get()
stst_prpg_true = MC_Zmumu["mll_stst_prpg"].get()
weightsum = results["ZmumuPostVFP"]["weight_sum"]
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]


dtdt_prfg = MC_Zmumu["mll_dtdt_prfg"].get()
dtst_prfg = MC_Zmumu["mll_dtst_prfg"].get()
stst_prfg = MC_Zmumu["mll_stst_prfg"].get()

### should loop over these instead of calling them explicitly

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


nbins_mll = len(dtdt_prpg_true.axes["mll"])
nbins_time = len(reco_dtst_data.axes["time"])

### cross-detector uncertainties

hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
hfoc_scaling = multiplyHists(hfoc_scaling, lumi_scaling)

pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
pcc_scaling = multiplyHists(pcc_scaling, lumi_scaling)

ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
ramses_scaling = multiplyHists(ramses_scaling, lumi_scaling)


############################# CURRENTLY WORKING ON ##########################################
### i should prabably do this for each of the 3 cases. but for now will just implement one
dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc = get_mc_lumis(
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
    time_proj,
    hfoc_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
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
    time_proj,
    pcc_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
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
    time_proj,
    ramses_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
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
    time_proj,
    sbil_hfoc_fit,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
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
    time_proj,
    sbil_ramses_fit,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
    weightsum,
    cross_sec,
)

dtdt_prpg, dtst_prpg, stst_prpg = get_mc_lumis(
    dtdt_prpg_H,
    dtst_prpg_H,
    stst_prpg_H,
    dtdt_prpg_BG,
    dtst_prpg_BG,
    stst_prpg_BG,
    time_proj,
    lumi_scaling,
    lumi_scaling_h,
    lumi_scaling_bg,
    lumi_scaling,
    weightsum,
    cross_sec,
)
(dtdt_prpg_prefiring_syst, dtst_prpg_prefiring_syst, stst_prpg_prefiring_syst) = (
    get_mc_lumis(
        dtdt_prpg_H_syst[{"downUpVar": 0}],
        dtst_prpg_H_syst[{"downUpVar": 0}],
        stst_prpg_H_syst[{"downUpVar": 0}],
        dtdt_prpg_BG_syst[{"downUpVar": 0}],
        dtst_prpg_BG_syst[{"downUpVar": 0}],
        stst_prpg_BG_syst[{"downUpVar": 0}],
        time_proj,
        lumi_scaling,
        lumi_scaling_h,
        lumi_scaling_bg,
        lumi_scaling,
        weightsum,
        cross_sec,
    )
)


pass_gen = all_mc_corrections(
    pass_gen, time_proj_gen_mll, lumi_scaling, weightsum, cross_sec
)

### efficiencies0
h2 = dtdt_prpg.project("time", "mll")
h1 = dtst_prpg.project("time", "mll")
h0 = stst_prpg.project("time", "mll")

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

dtdt_prpg = expand_hist_by_duplicate_axis(dtdt_prpg, "time", "gen_time")
dtst_prpg = expand_hist_by_duplicate_axis(dtst_prpg, "time", "gen_time")
stst_prpg = expand_hist_by_duplicate_axis(stst_prpg, "time", "gen_time")

### all
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
    "zmumu_fail_gen",
    "top",
    "diboson",
    "gg",
    "qcd",
    "w",
    "qcd_2l",
    "qcd_lnu",
]

# ### large contributions
# background_syst_names = ["ZmumuPostVFP", "Top",  'Diboson', "GGToLLPostVFP"]
# background_proc = ["zmumu_fail_gen","top", 'diboson', 'gg']

for i in range(len(background_syst_names)):
    proc_name = background_proc[i]
    if proc_name == "zmumu_fail_gen":
        fgen = True
    else:
        fgen = False
    background_syst(
        writer,
        results,
        background_syst_names[i],
        time_proj,
        lumi_scaling,
        proc_name,
        f"bkg_{proc_name}",
        fail_gen=fgen,
    )


pass_gen_expanded = expand_hist_by_duplicate_axes(
    pass_gen, ["time", "gen_mll"], ["gen_time", "gen_mll_0"]
)

h2_var_id_ALL = []
h1_var_id_ALL = []
h0_var_id_ALL = []
h2_var_hlt_ALL = []
h1_var_hlt_ALL = []
h0_var_hlt_ALL = []

# for i in range(3, 6):  # just select two mass bins in the center
for i in range(10):
    for j in range(nbins_time):
        ### be more consistent about ordering of time and mll
        ### fitting for the number of events

        ### MAKE NAMING LESS STUPID
        v2 = dtdt_prpg[{"gen_mll": i, "gen_time": j}]  ## equivalent to n2
        var2 = addHists(v2 * 0.1, h2)
        writer.add_systematic(
            var2,
            f"n_mll{i}_time{j}",
            "prpg",
            "ch_dtdt",
            constrained=False,
            groups=["nz"],
        )

        v1 = dtst_prpg[{"gen_mll": i, "gen_time": j}]  ## equivalent to n1
        var1 = addHists(v1 * 0.1, h1)
        writer.add_systematic(
            var1,
            f"n_mll{i}_time{j}",
            "prpg",
            "ch_dtst",
            constrained=False,
            groups=["nz"],
        )
        v0 = stst_prpg[{"gen_mll": i, "gen_time": j}]  ## equivalent to n1
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

        h2_var_id_ALL.append(h2var_id_primed)
        h1_var_id_ALL.append(h1var_id_primed)
        h0_var_id_ALL.append(h0var_id_primed)
        h2_var_hlt_ALL.append(h2var_hlt_primed)
        h1_var_hlt_ALL.append(h1var_hlt_primed)
        h0_var_hlt_ALL.append(h0var_hlt_primed)

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


writer.add_systematic(
    dtdt_prpg_prefiring_syst.project("time", "mll"),
    f"prefiring_syst",
    "prpg",
    "ch_dtdt",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    dtst_prpg_prefiring_syst.project("time", "mll"),
    f"prefiring_syst",
    "prpg",
    "ch_dtst",
    constrained=True,
    groups=["prefiring_syst"],
)
writer.add_systematic(
    stst_prpg_prefiring_syst.project("time", "mll"),
    f"prefiring_syst",
    "prpg",
    "ch_stst",
    constrained=True,
    groups=["prefiring_syst"],
)
num_etaphi = len(dtdt_prpg_H_stat.project("etaPhiRegion").values())
for i in range(num_etaphi):
    eta_phi_systematic(
        writer,
        dtdt_prpg_H_stat[{"etaPhiRegion": i}],
        dtst_prpg_H_stat[{"etaPhiRegion": i}],
        stst_prpg_H_stat[{"etaPhiRegion": i}],
        dtdt_prpg_BG_stat[{"etaPhiRegion": i}],
        dtst_prpg_BG_stat[{"etaPhiRegion": i}],
        stst_prpg_BG_stat[{"etaPhiRegion": i}],
        time_proj,
        lumi_scaling,
        lumi_scaling_h,
        lumi_scaling_bg,
        weightsum,
        cross_sec,
        i,
    )

##### MAKE LESS STUPID
#### PCC cross detector


luminometer_syst(
    writer, "pcc", dtdt_prpg_pcc, dtst_prpg_pcc, stst_prpg_pcc, "stability"
)

luminometer_syst(
    writer, "ramses", dtdt_prpg_ramses, dtst_prpg_ramses, stst_prpg_ramses, "stability"
)
luminometer_syst(
    writer,
    "ramses",
    dtdt_prpg_sbil_ramses,
    dtst_prpg_sbil_ramses,
    stst_prpg_sbil_ramses,
    "linearity",
)

luminometer_syst(
    writer, "hfoc", dtdt_prpg_hfoc, dtst_prpg_hfoc, stst_prpg_hfoc, "stability"
)
luminometer_syst(
    writer,
    "hfoc",
    dtdt_prpg_sbil_hfoc,
    dtst_prpg_sbil_hfoc,
    stst_prpg_sbil_hfoc,
    "linearity",
)


writer.write(outfolder="./", outfilename="liv")
