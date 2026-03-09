import hist
import numpy as np

from wums.boostHistHelpers import (
    addHists,
    broadcastSystHist,
    divideHists,
    multiplyHists,
    scaleHist,
    unrolledHist,
)


def mc_scaling(hist_in, hist_proj, lumi_scaling, weightsum, cross_sec):
    hist_in /= weightsum
    hist_in *= cross_sec
    hist_in *= 1000
    hist_in_2d = broadcastSystHist(hist_in, hist_proj)
    hist_in_2d = multiplyHists(hist_in_2d, lumi_scaling)

    return hist_in_2d


def correct_all_channels(
    iso_mc,
    dtdt_mc,
    dtst_mc,
    stst_mc,
    hist_proj_low,
    lumi_scaling,
    weightsum,
    cross_sec,
):
    iso = mc_scaling(iso_mc.copy(), hist_proj_low, lumi_scaling, weightsum, cross_sec)
    dtdt = mc_scaling(dtdt_mc.copy(), hist_proj_low, lumi_scaling, weightsum, cross_sec)
    dtst = mc_scaling(
        dtst_mc.copy(), hist_proj_low.copy(), lumi_scaling, weightsum, cross_sec
    )
    stst = mc_scaling(
        stst_mc.copy(), hist_proj_low.copy(), lumi_scaling, weightsum, cross_sec
    )
    return iso, dtdt, dtst, stst


def make_ones_hist(hist_ref):
    ones = np.ones_like(hist_ref.values())
    h_ones = hist_ref.copy()
    h_ones.values()[...] = ones
    return h_ones


def get_mc_lumis(
    input_data,
    time_proj_low,
    scaling,
    lumi_hists,
    weightsum,
    cross_sec,
):
    iso_h, dtdt_h, dtst_h, stst_h, iso_bg, dtdt_bg, dtst_bg, stst_bg = input_data

    lumi_h, lumi_bg = lumi_hists
    sum_lumis = addHists(lumi_bg, lumi_h)
    lumi_scaling_h = divideHists(lumi_h, sum_lumis)
    lumi_scaling_bg = divideHists(lumi_bg, sum_lumis)
    iso_h, dtdt_h, dtst_h, stst_h = correct_all_channels(
        iso_h,
        dtdt_h,
        dtst_h,
        stst_h,
        time_proj_low,
        multiplyHists(lumi_scaling_h, scaling),
        weightsum,
        cross_sec,
    )

    iso_bg, dtdt_bg, dtst_bg, stst_bg = correct_all_channels(
        iso_bg,
        dtdt_bg,
        dtst_bg,
        stst_bg,
        time_proj_low,
        multiplyHists(lumi_scaling_bg, scaling),
        weightsum,
        cross_sec,
    )

    iso = addHists(iso_bg, iso_h)
    dtdt = addHists(dtdt_bg, dtdt_h)
    dtst = addHists(dtst_bg, dtst_h)
    stst = addHists(stst_bg, stst_h)

    return iso, dtdt, dtst, stst


### i need to get good at coding so i dont need to pass in all these variables
def eta_phi_systematic(
    writer,
    input_data,
    time_hists,
    lumi_scaling,
    lumi_hists,
    weightsum,
    cross_sec,
    etaphi_num,
    mass_bin,
):
    (
        iso_H_stat,
        dtdt_H_stat,
        dtst_H_stat,
        stst_H_stat,
        iso_BG_stat,
        dtdt_BG_stat,
        dtst_BG_stat,
        stst_BG_stat,
    ) = input_data

    input_data = [
        iso_H_stat[{"etaPhiRegion": etaphi_num}],
        dtdt_H_stat[{"etaPhiRegion": etaphi_num}],
        dtst_H_stat[{"etaPhiRegion": etaphi_num}],
        stst_H_stat[{"etaPhiRegion": etaphi_num}],
        iso_BG_stat[{"etaPhiRegion": etaphi_num}],
        dtdt_BG_stat[{"etaPhiRegion": etaphi_num}],
        dtst_BG_stat[{"etaPhiRegion": etaphi_num}],
        stst_BG_stat[{"etaPhiRegion": etaphi_num}],
    ]

    iso_stat, dtdt_stat, dtst_stat, stst_stat = get_mc_lumis(
        input_data,
        time_hists,
        lumi_scaling,
        lumi_hists,
        weightsum,
        cross_sec,
    )
    iso_stat, dtdt_stat, dtst_stat, stst_stat = make_mutually_exclusive(
        iso_stat, dtdt_stat, dtst_stat, stst_stat
    )
    writer.add_systematic(
        unrolledHist(iso_stat.project("time", "mll", "pt_probe", "eta_probe")),  # ),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_iso",
        constrained=True,
        groups=["prefiring_stat"],
    )
    dtdt_stat = remove_low_bins(dtdt_stat)
    writer.add_systematic(
        unrolledHist(
            dtdt_stat.project("time", "mll", "pt_probe", "eta_probe")
        ),  # , "mll"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_dtdt",
        constrained=True,
        groups=["prefiring_stat"],
    )
    writer.add_systematic(
        unrolledHist(
            dtst_stat.project("time", "mll", "pt_probe", "eta_probe")
        ),  # , "mll"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_dtst",
        constrained=True,
        groups=["prefiring_stat"],
    )
    writer.add_systematic(
        unrolledHist(
            stst_stat.project("time", "mll", "pt_probe", "eta_probe")
        ),  # , "mll"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu",
        "ch_stst",
        constrained=True,
        groups=["prefiring_stat"],
    )


def get_era_vals(mc, trigger_cut, era, type_gen="pass", iso=False):
    if type_gen == "pass":
        if not iso:
            return (
                mc[f"{trigger_cut}_prpg_{era}"].get(),
                mc[f"{trigger_cut}_prpg_{era}_muonL1PrefireSyst"].get(),
                mc[f"{trigger_cut}_prpg_{era}_muonL1PrefireStat"].get(),
            )
        else:
            return (
                mc[f"{trigger_cut}_{era}"].get(),
                mc[f"{trigger_cut}_{era}_muonL1PrefireSyst"].get(),
                mc[f"{trigger_cut}_{era}_muonL1PrefireStat"].get(),
            )

    else:
        return (
            mc[f"{trigger_cut}_prfg_{era}"].get(),
            mc[f"{trigger_cut}_prfg_{era}_muonL1PrefireSyst"].get(),
            mc[f"{trigger_cut}_prfg_{era}_muonL1PrefireStat"].get(),
        )


def luminometer_syst(writer, luminometer, iso, dtdt, dtst, stst, syst):
    dtdt = remove_low_bins(dtdt)
    writer.add_systematic(
        unrolledHist(iso.project("time", "mll", "pt_probe", "eta_probe")),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_iso",
        constrained=True,
        groups=[f"{syst}"],
    )

    writer.add_systematic(
        unrolledHist(dtdt.project("time", "mll", "pt_probe", "eta_probe")),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_dtdt",
        constrained=True,
        groups=[f"{syst}"],
    )
    writer.add_systematic(
        unrolledHist(dtst.project("time", "mll", "pt_probe", "eta_probe")),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_dtst",
        constrained=True,
        groups=[f"{syst}"],
    )
    writer.add_systematic(
        unrolledHist(stst.project("time", "mll", "pt_probe", "eta_probe")),
        f"{luminometer}_{syst}",
        "Zmumu",
        "ch_stst",
        constrained=True,
        groups=[f"{syst}"],
    )


def background_syst(
    writer,
    results,
    res_str,
    time_proj_low,
    lumi_scaling,
    lumi_hists,
    proc_name,
    bkg_name,
    fail_gen=False,
    mass_bin=-1,  # 9,
):

    MC = results[res_str]["output"]
    try:
        weightsum = results[res_str]["weight_sum"]
        cross_sec = results[res_str]["dataset"]["xsec"]
    except:
        weightsum = results["SingleMuon_2016PostVFP"]["weight_sum"]
        cross_sec = results["SingleMuon_2016PostVFP"]["dataset"]["xsec"]
    print(res_str)

    ### MAKE THIS IMPLEMENTATION NOT STUPID
    if fail_gen:
        dtdt_prpg_BG, _, _ = get_era_vals(MC, "dtdt", "BG", "fail")
        dtst_prpg_BG, _, _ = get_era_vals(MC, "dtst", "BG", "fail")
        stst_prpg_BG, _, _ = get_era_vals(MC, "stst", "BG", "fail")

        dtdt_prpg_H, _, _ = get_era_vals(MC, "dtdt", "H", "fail")
        dtst_prpg_H, _, _ = get_era_vals(MC, "dtst", "H", "fail")
        stst_prpg_H, _, _ = get_era_vals(MC, "stst", "H", "fail")
    else:
        iso_BG, _, _ = get_era_vals(MC, "pass_iso", "BG", iso=True)
        dtdt_prpg_BG, _, _ = get_era_vals(MC, "dtdt", "BG")
        dtst_prpg_BG, _, _ = get_era_vals(MC, "dtst", "BG")
        stst_prpg_BG, _, _ = get_era_vals(MC, "stst", "BG")

        iso_H, _, _ = get_era_vals(MC, "pass_iso", "H", iso=True)
        dtdt_prpg_H, _, _ = get_era_vals(MC, "dtdt", "H")
        dtst_prpg_H, _, _ = get_era_vals(MC, "dtst", "H")
        stst_prpg_H, _, _ = get_era_vals(MC, "stst", "H")

    iso_H = iso_H
    dtdt_prpg_H = dtdt_prpg_H
    dtst_prpg_H = dtst_prpg_H
    stst_prpg_H = stst_prpg_H

    iso_BG = iso_BG
    dtdt_prpg_BG = dtdt_prpg_BG
    dtst_prpg_BG = dtst_prpg_BG
    stst_prpg_BG = stst_prpg_BG

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

    time_hists = time_proj_low

    iso, dtdt, dtst, stst = get_mc_lumis(
        prpg_all,
        time_hists,
        lumi_scaling,
        lumi_hists,
        weightsum,
        cross_sec,
    )
    iso, dtdt, dtst, stst = make_mutually_exclusive(iso, dtdt, dtst, stst)
    dtdt = remove_low_bins(dtdt)

    iso_proc = unrolledHist(
        iso.project("time", "mll", "pt_probe", "eta_probe")
    )  # , "mll")
    dtdt_proc = unrolledHist(
        dtdt.project("time", "mll", "pt_probe", "eta_probe")
    )  # , "mll")
    dtst_proc = unrolledHist(
        dtst.project("time", "mll", "pt_probe", "eta_probe")
    )  # , "mll")
    stst_proc = unrolledHist(
        stst.project("time", "mll", "pt_probe", "eta_probe")
    )  # , "mll")

    # iso_proc = remove_low_bins(iso_proc)

    writer.add_process(iso_proc, f"{proc_name}", "ch_iso", signal=False)
    writer.add_process(dtdt_proc, f"{proc_name}", "ch_dtdt", signal=False)
    writer.add_process(dtst_proc, f"{proc_name}", "ch_dtst", signal=False)
    writer.add_process(stst_proc, f"{proc_name}", "ch_stst", signal=False)

    var = 1.01
    if proc_name == "W":
        var = 1.001
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_iso", var, groups=["bkg"]
    )
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_dtdt", var, groups=["bkg"]
    )
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_dtst", var, groups=["bkg"]
    )
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_stst", var, groups=["bkg"]
    )


def remove_low_bins(old_hist, ax_name="pt_probe", nbins=1):

    if ax_name == "pt_probe":
        nbins = old_hist.axes[ax_name].index(25)

    new_edges = old_hist.axes[ax_name].edges[nbins:]

    new_axis = hist.axis.Variable(new_edges, name=ax_name)
    ax_name_ind = old_hist.axes.name.index(ax_name)

    axes = list(old_hist.axes)
    axes[ax_name_ind] = new_axis

    slices = [slice(None)] * old_hist.ndim
    slices[ax_name_ind] = slice(nbins, None)

    if "probe" in ax_name:
        try:
            new_tag_axis = hist.axis.Variable(new_edges, name=f"{ax_name[:-5]}tag")
            tag_ind = old_hist.axes.name.index(f"{ax_name[:-5]}tag")
            axes[tag_ind] = new_tag_axis
            slices[tag_ind] = slice(nbins, None)
        except:  ## if there is no tag access for some reason
            pass

    new_hist = hist.Hist(*axes)
    new_hist.values()[...] = old_hist.values()[tuple(slices)]

    return new_hist


def make_mutually_exclusive(iso, dtdt, dtst, stst):
    iso_ex = iso
    dtdt_ex = addHists(dtdt, scaleHist(iso, -1))
    dtst_ex = addHists(dtst, scaleHist(dtdt, -1))
    stst_ex = addHists(stst, scaleHist(dtst, -1))
    return iso_ex, dtdt_ex, dtst_ex, stst_ex


def create_variation(
    variation_hist,
    refererence_hist,
    i,
    j,
    k,
    nbins_total,
    muon="tag",
    h2=False,
    var_size=0.01,
):
    if h2:
        i -= 1
    not_muon = "probe"
    if muon == "probe":
        not_muon = "tag"

    var = variation_hist[{"gen_time": k, f"pt_{not_muon}": i, f"eta_{not_muon}": j}]

    if muon == "tag":
        var = var.project("time", f"pt_{muon}", f"eta_{muon}")
        var = broadcastSystHist(var, refererence_hist)
        var = multiplyHists(scaleHist(var, var_size / (nbins_total)), refererence_hist)
        var = var.project("time", "pt_probe", "eta_probe")
    return var
