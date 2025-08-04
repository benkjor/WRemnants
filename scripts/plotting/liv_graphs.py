import h5py
import matplotlib.pyplot as plt
import numpy as np

from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    broadcastSystHist,
    divideHists,
    expand_hist_by_duplicate_axis,
    multiplyHists,
)

file_in = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
file_out = "/home/submit/jbenke/public_html/"
file_in_name = file_in + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"

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
        plt.ylabel(ylabel, fontsize=12)

    plt.xlabel("Sidereal Time [hrs]", fontsize=12)
    # plt.title(plotname)
    plt.legend(legend_all)
    plt.savefig(file_out + file_out_modifier + plotname + ".png")


def mc_scaling(mc_results, weightsum, cross_sec):
    temp = mc_results.copy()
    temp /= weightsum
    temp *= cross_sec
    temp *= 1000
    return temp


def all_mc_corrections(hist_in, hist_proj, lumi_scaling, weightsum, cross_sec):
    hist_in_new = hist_in.copy()
    hist_in_new = mc_scaling(hist_in_new, weightsum, cross_sec)
    hist_in_2d = broadcastSystHist(hist_in_new, hist_proj)
    hist_in_2d = multiplyHists(hist_in_2d, lumi_scaling)
    return hist_in_2d


def mc_corrections_all_cases(
    dtdt_mc, dtst_mc, stst_mc, hist_proj, lumi_scaling, weightsum, cross_sec
):

    dtdt = all_mc_corrections(
        dtdt_mc.copy(), hist_proj, lumi_scaling, weightsum, cross_sec
    )
    dtst = all_mc_corrections(
        dtst_mc.copy(), hist_proj, lumi_scaling, weightsum, cross_sec
    )
    stst = all_mc_corrections(
        stst_mc.copy(), hist_proj, lumi_scaling, weightsum, cross_sec
    )
    return dtdt, dtst, stst


def get_mc_lumis(
    dtdt_pre,
    dtst_pre,
    stst_pre,
    dtdt_post,
    dtst_post,
    stst_post,
    time_proj,
    scaling,
    lumi_pre,
    lumi_post,
    lumi_nom,
    weightsum,
    cross_sec,
):
    dtdt_pre, dtst_pre, stst_pre = mc_corrections_all_cases(
        dtdt_pre,
        dtst_pre,
        stst_pre,
        time_proj,
        multiplyHists(scaling, divideHists(lumi_pre, lumi_nom)),
        weightsum,
        cross_sec,
    )
    dtdt_post, dtst_post, stst_post = mc_corrections_all_cases(
        dtdt_post,
        dtst_post,
        stst_post,
        time_proj,
        multiplyHists(scaling, divideHists(lumi_post, lumi_nom)),
        weightsum,
        cross_sec,
    )

    dtdt = addHists(dtdt_pre, dtdt_post)
    dtst = addHists(dtst_pre, dtst_post)
    stst = addHists(stst_pre, stst_post)

    return dtdt, dtst, stst


h5file = h5py.File(file_in_name, "r")
results = input_tools.load_results_h5py(h5file)

### accessing the necessary data
dtdt_data = results["dataPostVFP"]["output"]["time_mll"].get()
stst_data = results["dataPostVFP"]["output"]["time_mll_stst"].get()
dtst_data = results["dataPostVFP"]["output"]["time_mll_dtst"].get()

dtdt_mc = results["ZmumuPostVFP"]["output"]["mll_dtdt_prpg"].get().project("mll")
stst_mc = results["ZmumuPostVFP"]["output"]["mll_stst_prpg"].get().project("mll")
dtst_mc = results["ZmumuPostVFP"]["output"]["mll_dtst_prpg"].get().project("mll")


generator = results["ZmumuPostVFP"]["output"]["pass_gen"].get()

wsum = results["ZmumuPostVFP"]["weight_sum"]
xsec = results["ZmumuPostVFP"]["dataset"]["xsec"]

lumi_scaling = results["dataPostVFP"]["lumi_outout"]["lumi_nom"].get()
lumi_hfoc = results["dataPostVFP"]["lumi_outout"]["lumi_hfoc"].get()
lumi_pcc = results["dataPostVFP"]["lumi_outout"]["lumi_pcc"].get()
lumi_ramses = results["dataPostVFP"]["lumi_outout"]["lumi_ramses"].get()
lumi_hfoc_nom = results["dataPostVFP"]["lumi_outout"]["lumi_in_hfoc"].get()
lumi_pcc_nom = results["dataPostVFP"]["lumi_outout"]["lumi_in_pcc"].get()
lumi_ramses_nom = results["dataPostVFP"]["lumi_outout"]["lumi_in_ramses"].get()
sbil_pcc = results["dataPostVFP"]["lumi_outout"]["sbil_pcc"].get()

mll_data = dtdt_data.project("mll")
time_data = dtdt_data.project("time")
dtdt_data = expand_hist_by_duplicate_axis(dtdt_data, "mll", "gen_mll")
MC_Zmumu = results["ZmumuPostVFP"]["output"]

dtdt_prpg_mc_prevfp = MC_Zmumu["dtdt_prpg_prevfp"].get()
dtst_prpg_mc_prevfp = MC_Zmumu["dtst_prpg_prevfp"].get()
stst_prpg_mc_prevfp = MC_Zmumu["stst_prpg_prevfp"].get()

dtdt_prpg_mc_postvfp = MC_Zmumu["dtst_prpg_postvfp"].get()
dtst_prpg_mc_postvfp = MC_Zmumu["dtst_prpg_postvfp"].get()
stst_prpg_mc_postvfp = MC_Zmumu["stst_prpg_postvfp"].get()
lumi_output = results["dataPostVFP"]["lumi_outout"]

lumi_scaling_pre = lumi_output["lumi_pre"].get()
lumi_scaling_post = lumi_output["lumi_post"].get()
data_output = results["dataPostVFP"]["output"]

time_proj = data_output["time_proj"].get()

dtdt_prpg_mc, dtst_prpg_mc, stst_prpg_mc = get_mc_lumis(
    dtdt_prpg_mc_prevfp,
    dtst_prpg_mc_prevfp,
    stst_prpg_mc_prevfp,
    dtdt_prpg_mc_postvfp,
    dtst_prpg_mc_postvfp,
    stst_prpg_mc_postvfp,
    time_proj,
    lumi_scaling,
    lumi_scaling_pre,
    lumi_scaling_post,
    lumi_scaling,
    wsum,
    xsec,
)


### lumi
hfoc_scaling = divideHists(lumi_hfoc, lumi_hfoc_nom)
pcc_scaling = divideHists(lumi_pcc, lumi_pcc_nom)
ramses_scaling = divideHists(lumi_ramses, lumi_ramses_nom)
## add the time axis to mc data
dtdt_mc_2d = all_mc_corrections(
    dtdt_mc, dtdt_data.project("mll", "time"), lumi_scaling, wsum, xsec
)
stst_mc_2d = all_mc_corrections(
    stst_mc, dtdt_data.project("mll", "time"), lumi_scaling, wsum, xsec
)
dtst_mc_2d = all_mc_corrections(
    dtst_mc, dtdt_data.project("mll", "time"), lumi_scaling, wsum, xsec
)
generator_2d = all_mc_corrections(
    generator, dtdt_data.project("gen_mll", "time"), lumi_scaling, wsum, xsec
)

#### should look into whether i need these
dtdt_data = dtdt_data.project("mll", "time")


# pdb.set_trace()
def empty_hist_copy(hist):
    # creates am empty hist of the same shape
    copy_array = np.ones_like(hist.values())
    copy_hist = hist.copy()
    copy_hist.values()[...] = copy_array
    return copy_hist


sum_mc = empty_hist_copy(dtdt_mc_2d[{"mll": 0}])
sum_data = empty_hist_copy(dtdt_data[{"mll": 0}])
sum_stst_mc = empty_hist_copy(generator_2d[{"gen_mll": 0}])
sum_dtst_mc = empty_hist_copy(stst_mc_2d[{"mll": 0}])
sum_generator = empty_hist_copy(dtst_mc_2d[{"mll": 0}])

sum_stst_data = empty_hist_copy(stst_data[{"mll": 0}])
sum_dtst_data = empty_hist_copy(dtst_data[{"mll": 0}])


for i in range(len(mll_data.values())):
    # for i in range(2):
    ### UGH THIS DOESN'T WORK ANYMORE
    time_mc_proj = dtdt_mc_2d[{"mll": i}]
    time_data_proj = dtdt_data[{"mll": i}]
    generator_proj = generator_2d[{"gen_mll": i}]

    stst_mc_proj = stst_mc_2d[{"mll": i}]
    dtst_mc_proj = dtst_mc_2d[{"mll": i}]

    stst_data_proj = stst_data[{"mll": i}]
    dtst_data_proj = dtst_data[{"mll": i}]

    sum_mc = addHists(sum_mc, time_mc_proj)
    sum_data = addHists(sum_data, time_data_proj)
    sum_stst_mc = addHists(sum_stst_mc, stst_mc_proj)
    sum_dtst_mc = addHists(sum_dtst_mc, dtst_mc_proj)
    sum_stst_data = addHists(sum_stst_data, stst_data_proj)
    sum_dtst_data = addHists(sum_dtst_data, dtst_data_proj)
    sum_generator = addHists(sum_generator, generator_proj)

    make_plot(
        [time_mc_proj, time_data_proj],
        "mass_bin_" + str(i),
        file_out_modifier="liv_uncert/efficiency/mass_binning/",
    )

make_plot(
    [
        sum_data,
        sum_mc,
        sum_dtst_data,
        sum_dtst_mc,
        sum_stst_data,
        sum_stst_mc,
        sum_generator,
    ],
    "sum_all",
    legend_all=[
        "Data: 2 tightID, 2 trig",
        "MC: 2 tightID, 2 trig",
        "Data: 2 tightID, 1 trig",
        "MC: 2 tightID, 1 trig",
        "Data: 1 tightID, 1 trig",
        "MC: 1 tightID, 1 trig",
        "Generator",
    ],
    ylabel="Events",
    colors=["red", "red", "blue", "blue", "orange", "orange", "black"],
    linestyles=["-", "--", "-", "--", "-", "--", "-"],
    file_out_modifier="liv_uncert/efficiency/mass_binning/",
)


make_plot(
    [pcc_scaling, hfoc_scaling, ramses_scaling],
    "lumi_ratios",
    ["PCC", "HFOC", "RAMSES"],
    [0.995, 1.005],
    False,
    "ratio of inst. lumi to PHYSICS",
    colors=["black", "red", "blue"],
    file_out_modifier="liv_uncert/lumi/",
)

# make_plot([hfoc_scaling], "lumi_ratios", ["HFOC/nominal"], [0.9, 1.1])
# make_plot([pcc_scaling], "lumi_ratios", ["PCC/nominal"], [0.999, 1.001])
# make_plot([ramses_scaling], "lumi_ratios", ["RAMSES/nominal"], [0.9, 1.1])
# make_plot([sum_generator], "generator", legend_all=["Generator"])

make_plot(
    [sum_mc, sum_data],
    "sum_new",
    file_out_modifier="liv_uncert/efficiency/mass_binning/",
)
ratio_sum = divideHists(sum_mc, sum_data)
make_plot(
    [ratio_sum],
    "sum_ratio",
    ["MC/Data"],
    [1, 1.5],
    file_out_modifier="liv_uncert/efficiency/mass_binning/",
)
