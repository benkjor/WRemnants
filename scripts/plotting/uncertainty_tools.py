import hist
import numpy as np

from wums.boostHistHelpers import (
    addHists,
    broadcastSystHist,
    divideHists,
    multiplyHists,
    scaleHist,
)


def get_h2var(eps_id, eps_hlt, heff):
    #  h2var = heff * (eps_prime * eps)
    h2var = multiplyHists(heff, multiplyHists(eps_id, eps_id))
    h2var = multiplyHists(h2var, multiplyHists(eps_hlt, eps_hlt))
    return h2var


def get_h1var(eps_id, eps_hlt, heff, hist_ones):
    # h = 2*heff*(eps_hlt * hlt_prime)*(1-(eps_hlt * hlt_prime))
    h1var = addHists(hist_ones, scaleHist(eps_hlt, -1))
    h1var = multiplyHists(h1var, eps_hlt)
    h1var = scaleHist(h1var, 2)
    h1var = multiplyHists(h1var, multiplyHists(eps_id, eps_id))
    h1var = multiplyHists(h1var, heff)

    return h1var


def get_h1var_low(eps_id, eps_hlt, heff, hist_ones):
    # h = heff*(eps_hlt * hlt_prime)*(1-(eps_hlt * hlt_prime))
    # h1var = addHists(hist_ones, scaleHist(eps_hlt, -1))
    # h1var = multiplyHists(h1var, )

    h1var = multiplyHists(eps_id, eps_id)
    # h1var = multiplyHists(eps_hlt, multiplyHists(eps_id, eps_id))
    h1var = multiplyHists(h1var, heff)
    return h1var


def get_h0var(eps_id, eps_hlt, heff, hist_ones):
    # h = 2*heff*(eps_id * eps_id_prime)*(1-(eps_id * eps_id_prime))
    h0var = addHists(hist_ones, scaleHist(eps_id, -1))
    h0var = multiplyHists(h0var, eps_id)
    h0var = multiplyHists(h0var, eps_hlt)
    h0var = scaleHist(h0var, 2)
    h0var = multiplyHists(h0var, heff)
    return h0var


def get_h0var_low(eps_id, eps_hlt, heff, hist_ones):
    # h = heff*(eps_id * eps_id_prime)*(1-(eps_id * eps_id_prime))
    h0var = addHists(hist_ones, scaleHist(eps_id, -1))
    h0var = multiplyHists(h0var, eps_id)
    # h0var = multiplyHists(h0var, eps_hlt)
    h0var = multiplyHists(h0var, heff)
    return h0var


def get_eff_hist(eps_hist, ref_hist, i, j, k, first_ind, second_ind):
    hist_copy = ref_hist.copy()
    hist_values = hist_copy.values()
    try:
        # hist_values[j, i] = eps_hist[{"time": j, f"{first_ind}": i}, f"{second_ind}: k"].value

        hist_values[k, i, j] = eps_hist[
            {"time": k, f"{first_ind}": i, f"{second_ind}": j}
        ].value
    except:
        # hist_values[j, i] = eps_hist[{"time": j, "mll": i}]
        hist_values[k, i, j] = eps_hist[
            {"time": k, f"{first_ind}": i, f"{second_ind}": j}
        ]

    hist_copy.values()[...] = hist_values
    return hist_copy


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
    iso_mc,
    dtdt_mc,
    dtst_mc,
    stst_mc,
    hist_proj_hlt,
    hist_proj_low,
    lumi_scaling,
    weightsum,
    cross_sec,
):
    iso = all_mc_corrections(
        iso_mc.copy(), hist_proj_low, lumi_scaling, weightsum, cross_sec
    )
    dtdt = all_mc_corrections(
        dtdt_mc.copy(), hist_proj_low, lumi_scaling, weightsum, cross_sec
    )
    dtst = all_mc_corrections(
        dtst_mc.copy(), hist_proj_low.copy(), lumi_scaling, weightsum, cross_sec
    )

    stst = all_mc_corrections(
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
    time_hists,
    scaling,
    lumi_hists,
    weightsum,
    cross_sec,
):

    iso_h, dtdt_h, dtst_h, stst_h, iso_bg, dtdt_bg, dtst_bg, stst_bg = input_data
    time_proj_hlt, time_proj_low = time_hists

    lumi_h, lumi_bg = lumi_hists
    sum_lumis = addHists(lumi_bg, lumi_h)
    lumi_scaling_h = divideHists(lumi_h, sum_lumis)
    lumi_scaling_bg = divideHists(lumi_bg, sum_lumis)
    iso_h, dtdt_h, dtst_h, stst_h = mc_corrections_all_cases(
        iso_h,
        dtdt_h,
        dtst_h,
        stst_h,
        time_proj_hlt,
        time_proj_low,
        multiplyHists(lumi_scaling_h, scaling),
        weightsum,
        cross_sec,
    )

    iso_bg, dtdt_bg, dtst_bg, stst_bg = mc_corrections_all_cases(
        iso_bg,
        dtdt_bg,
        dtst_bg,
        stst_bg,
        time_proj_hlt,
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
):

    dtdt_stat, dtst_stat, stst_stat = get_mc_lumis(
        input_data,
        time_hists,
        lumi_scaling,
        lumi_hists,
        weightsum,
        cross_sec,
    )

    # dtdt_stat = remove_low_bins(dtdt_stat.copy())

    writer.add_systematic(
        dtdt_stat.project("time", "pt_probe", "eta_probe"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu pass gen",
        "ch_dtdt_5d",
        constrained=True,
        groups=["prefiring_stat"],
    )
    writer.add_systematic(
        dtst_stat.project("time", "pt_probe", "eta_probe"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu pass gen",
        "ch_dtst_5d",
        constrained=True,
        groups=["prefiring_stat"],
    )
    writer.add_systematic(
        stst_stat.project("time", "pt_probe", "eta_probe"),
        f"prefiring_stat_etaphi_{etaphi_num}",
        "Zmumu pass gen",
        "ch_stst_5d",
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


def luminometer_syst(writer, luminometer, dtdt, dtst, stst, syst):
    # dtdt = remove_low_bins(dtdt.copy())
    writer.add_systematic(
        dtdt.project("time", "mll"),
        f"{luminometer}_{syst}",
        "Zmumu pass gen",
        "ch_dtdt_2d",
        constrained=True,
        groups=[f"{syst}"],
    )
    writer.add_systematic(
        dtst.project("time", "mll"),
        f"{luminometer}_{syst}",
        "Zmumu pass gen",
        "ch_dtst_2d",
        constrained=True,
        groups=[f"{syst}"],
    )
    writer.add_systematic(
        stst.project("time", "mll"),
        f"{luminometer}_{syst}",
        "Zmumu pass gen",
        "ch_stst_2d",
        constrained=True,
        groups=[f"{syst}"],
    )


def background_syst(
    writer,
    results,
    res_str,
    time_proj_hlt,
    time_proj_low,
    lumi_scaling,
    lumi_hists,
    proc_name,
    bkg_name,
    fail_gen=False,
    mass_bin=9,
):

    MC = results[res_str]["output"]

    try:
        weightsum = results[proc_name]["weight_sum"]
        cross_sec = results[proc_name]["dataset"]["xsec"]
    except:
        weightsum = results["ZmumuPostVFP"]["weight_sum"]
        cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

    ### MAKE THIS IMPLEMENTATION NOT STUPID
    if fail_gen:

        dtdt_prpg_BG, _, _ = get_era_vals(MC, "dtdt", "BG", "fail")
        dtst_prpg_BG, _, _ = get_era_vals(MC, "dtst", "BG", "fail")
        stst_prpg_BG, _, _ = get_era_vals(MC, "stst", "BG", "fail")

        dtdt_prpg_H, _, _ = get_era_vals(MC, "dtdt", "H", "fail")
        dtst_prpg_H, _, _ = get_era_vals(MC, "dtst", "H", "fail")
        stst_prpg_H, _, _ = get_era_vals(MC, "stst", "H", "fail")
    else:
        dtdt_prpg_BG, _, _ = get_era_vals(MC, "dtdt", "BG")
        dtst_prpg_BG, _, _ = get_era_vals(MC, "dtst", "BG")
        stst_prpg_BG, _, _ = get_era_vals(MC, "stst", "BG")

        dtdt_prpg_H, _, _ = get_era_vals(MC, "dtdt", "H")
        dtst_prpg_H, _, _ = get_era_vals(MC, "dtst", "H")
        stst_prpg_H, _, _ = get_era_vals(MC, "stst", "H")

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

    time_hists = [time_proj_hlt, time_proj_low]

    dtdt, dtst, stst = get_mc_lumis(
        prpg_all,
        time_hists,
        lumi_scaling,
        lumi_hists,
        weightsum,
        cross_sec,
    )

    dtdt_proc = dtdt.project("time", "pt_probe", "eta_probe")
    dtst_proc = dtst.project("time", "pt_probe", "eta_probe")
    stst_proc = stst.project("time", "pt_probe", "eta_probe")

    dtdt_proc, dtst_proc, stst_proc = make_mutually_exclusive(
        dtdt_proc, dtst_proc, stst_proc
    )

    dtdt_proc = remove_low_bins(dtdt_proc)

    writer.add_process(dtdt_proc, f"{proc_name}", "ch_dtdt", signal=False)
    writer.add_process(dtst_proc, f"{proc_name}", "ch_dtst", signal=False)
    writer.add_process(stst_proc, f"{proc_name}", "ch_stst", signal=False)

    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_dtdt", 1.01, groups=["bkg"]
    )
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_dtst", 1.01, groups=["bkg"]
    )
    writer.add_norm_systematic(
        f"{bkg_name}", f"{proc_name}", "ch_stst", 1.01, groups=["bkg"]
    )
    # return dtdt, dtst, stst


def get_eff_variations(h1_leading, h2_leading, h0_leading, eps_id_prime, eps_hlt_prime):
    ### no longer used
    efficiency_ones = make_ones_hist(h1_leading)

    #### okay need to make versions of this for
    ## e2 = 2*h2/(h1 + 2*h1)
    eps_hlt = addHists(h1_leading, scaleHist(h2_leading, 2))
    eps_hlt = divideHists(h2_leading, eps_hlt)
    eps_hlt = scaleHist(eps_hlt, 2)
    ##e1 = h1/(h0*(1-e2) + h1)
    eps_id = addHists(efficiency_ones, scaleHist(eps_hlt, -1))
    eps_id = multiplyHists(h0_leading, eps_id)
    eps_id = addHists(eps_id, h1_leading)
    eps_id = divideHists(h1_leading, eps_id)

    heff = divideHists(h2_leading, multiplyHists(eps_hlt, eps_hlt))
    heff = divideHists(heff, multiplyHists(eps_id, eps_id))

    # generate histogram of ones
    eps_id_var = scaleHist(eps_id.copy(), eps_id_prime)
    eps_hlt_var = scaleHist(eps_hlt.copy(), eps_hlt_prime)

    h0var_id = get_h0var(eps_id_var, eps_hlt, heff, efficiency_ones)
    h0var_hlt = get_h0var(eps_id, eps_hlt_var, heff, efficiency_ones)
    h1var_id = get_h1var(eps_id_var, eps_hlt, heff, efficiency_ones)
    h1var_hlt = get_h1var(eps_id, eps_hlt_var, heff, efficiency_ones)
    h2var_id = get_h2var(eps_id_var, eps_hlt, heff)
    h2var_hlt = get_h2var(eps_id, eps_hlt_var, heff)
    return h0var_id, h0var_hlt, h1var_id, h1var_hlt, h2var_id, h2var_hlt


def remove_low_bins(old_hist, ax_name="pt_probe", nbins=1):
    if len(old_hist.axes) != 6:
        org_pt_axis = old_hist.axes[1]
    else:
        org_pt_axis = old_hist.axes[2]
    edges = org_pt_axis.edges
    new_pt_edges = edges[nbins:]
    # eta_edges =
    new_pt_axis = hist.axis.Variable(new_pt_edges, name=ax_name)
    new_pt_axis_2 = hist.axis.Variable(new_pt_edges, name="pt_tag")
    new_eta_axis = hist.axis.Variable(
        old_hist.axes[2].edges, name="eta_probe"
    )  ## actually a regular axis but whatever
    # pdb.set_trace()

    if len(old_hist.axes) == 3:
        new_hist = hist.Hist(old_hist.axes[0], new_pt_axis, new_eta_axis)
        new_hist.values()[...] = old_hist.values()[:, nbins:, :]

    elif len(old_hist.axes) == 5:
        new_hist = hist.Hist(
            old_hist.axes[0],
            new_pt_axis,
            old_hist.axes[2],
            new_pt_axis_2,
            old_hist.axes[4],
        )
        new_hist.values()[...] = old_hist.values()[:, nbins:, :, nbins:, :]

    elif len(old_hist.axes) == 6:
        new_hist = hist.Hist(
            old_hist.axes[0],
            old_hist.axes[1],
            new_pt_axis,
            old_hist.axes[3],
            old_hist.axes[4],
            old_hist.axes[5],
        )
        new_hist.values()[...] = old_hist.values()[:, :, nbins:, :, :, :]

    return new_hist


def probe_to_tag(old_hist):
    new_pt_axis = hist.axis.Variable(old_hist.axes[1].edges, name="pt_tag")
    new_eta_axis = hist.axis.Variable(old_hist.axes[2].edges, name="eta_tag")
    new_hist = hist.Hist(old_hist.axes[0], new_pt_axis, new_eta_axis)
    new_hist.values()[...] = old_hist.values()[...]
    return new_hist


def make_mutually_exclusive(iso, dtdt, dtst, stst):
    iso_ex = iso
    dtdt_ex = addHists(dtdt, scaleHist(iso, -1))
    dtst_ex = addHists(dtst, scaleHist(dtdt, -1))
    stst_ex = addHists(stst, scaleHist(dtst, -1))
    return iso_ex, dtdt_ex, dtst_ex, stst_ex


# def make_mutually_exclusive(dtdt, dtst, stst):
#     dtdt_ex = dtdt
#     dtst_ex = addHists(dtst, scaleHist(dtdt, -1))
#     stst_ex = addHists(stst, scaleHist(dtst, -1))
#     return dtdt_ex, dtst_ex, stst_ex


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
