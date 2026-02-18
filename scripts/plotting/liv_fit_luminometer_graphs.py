import h5py
import matplotlib
import matplotlib.pyplot as plt

from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    divideHists,
    multiplyHists,
    scaleHist,
)

mass_bin = 9
var_size = 0.01


matplotlib.rcParams.update({"font.size": 12})

file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_out = "/home/submit/jbenke/public_html/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturbo_CT18Z_N3p0LL_N2LO_Corr.hdf5"

ramses_slope = 0.0007
hfoc_slope = 0.0006


def make_plot(
    data_all,
    plotname,
    legend_all=["MC", "Data"],
    ylim=[],
    error_bars=True,
    ylabel="",
    colors=[],
    linestyles=[],
    file_out_modifier="",
):
    plt.clf()
    plt.tight_layout()
    plt.subplots_adjust(left=0.17)
    for i in range(len(legend_all)):
        if len(linestyles) != 0:
            if len(colors) != 0:
                data_all[i].plot1d(
                    yerr=error_bars, color=colors[i], linestyle=linestyles[i]
                )
        elif len(colors) != 0:
            data_all[i].plot1d(yerr=error_bars, color=colors[i])
        else:
            data_all[i].plot1d(yerr=error_bars)
    plt.xlim([0.01, 24])
    if len(ylim) != 0:
        plt.ylim(ylim)
    if len(ylabel) != 0:
        plt.ylabel(ylabel, fontsize=16)

    plt.xlabel("Sidereal Time [hr]", fontsize=16)
    # plt.title(plotname)
    plt.legend(legend_all)
    plt.savefig(file_out + file_out_modifier + plotname + ".png")


h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

data_output = results["SingleMuon_2016PostVFP"]["output"]
lumi_output = results["SingleMuon_2016PostVFP"]["lumi_outout"]
MC_Zmumu = results["Zmumu_2016PostVFP"]["output"]


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


hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
hfoc_scaling = multiplyHists(hfoc_scaling, lumi_scaling)

pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
pcc_scaling = multiplyHists(pcc_scaling, lumi_scaling)

ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
ramses_scaling = multiplyHists(ramses_scaling, lumi_scaling)

hfoc_stability = 0.99117
ramses_stability = 0.81021


hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
ones_hist = divideHists(hfoc_scaling, hfoc_scaling)

hfoc_fitted = addHists(
    scaleHist(addHists(hfoc_scaling, scaleHist(ones_hist, -1)), hfoc_stability),
    ones_hist,
)
ramses_fitted = addHists(
    scaleHist(addHists(ramses_scaling, scaleHist(ones_hist, -1)), ramses_stability),
    ones_hist,
)
# hfoc_fitted = divideHists(scaleHist(hfoc_scaling, hfoc_stability), pcc_scaling)
# ramses_fitted = divideHists(scaleHist(ramses_scaling, ramses_stability), pcc_scaling)
make_plot(
    [pcc_scaling, hfoc_fitted, ramses_fitted],
    "lumi_ratios fitted",
    ["PCC", "HFOC", "RAMSES"],
    [0.999, 1.001],
    False,
    "ratio of inst. lumi to PHYSICS",
    colors=["black", "red", "blue"],
    file_out_modifier="liv_uncert/lumi/2026-02-17/",
)


### this doesn't plot the fitted version, just the theoretical one
# make_plot([hfoc_scaling], "lumi_ratios", ["HFOC/nominal"], [0.9, 1.1],file_out_modifier="liv_uncert/lumi/2026-02-16/")
# make_plot([pcc_scaling], "lumi_ratios", ["PCC/nominal"], [0.999, 1.001], file_out_modifier="liv_uncert/lumi/2026-02-16/")
# make_plot([ramses_scaling], "lumi_ratios", ["RAMSES/nominal"], [0.9, 1.1], file_out_modifier="liv_uncert/lumi/2026-02-16/")
