import os
from datetime import datetime

import hist
import numpy as np

import narf
from narf.lumitools import (
    make_brilcalc_filter_helper,
    make_brilcalc_helper,
    make_lumihelper,
)
from utilities import common, parsing
from wremnants import muon_prefiring, syst_tools, theory_tools
from wremnants.datasets.datagroups import Datagroups
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import (
    aggregate_groups,
    scale_to_data,
    write_analysis_output,
)
from wums import logging

analysis_label = Datagroups.analysisLabel(os.path.basename(__file__))
parser, initargs = parsing.common_parser(analysis_label)


parser.add_argument(
    "--muonIsolation",
    type=int,
    nargs=2,
    default=[1, 1],
    choices=[-1, 0, 1],
    help="Apply isolation cut to triggering and not-triggering muon (in this order): -1/1 for failing/passing isolation, 0 for skipping it. If using --useDileptonTriggerSelection, then the sorting is based on the muon charge as -/+",
)
parser.add_argument("--axes", type=str, nargs="*", default=["mll", "ptll"], help="")

parser.add_argument(
    "--useDileptonTriggerSelection",
    action="store_true",
    help="Use dilepton trigger selection (default uses the Wlike one, with one triggering muon and odd/even event selection to define its charge, staying agnostic to the other)",
)
parser.add_argument(
    "--flipEventNumberSplitting",
    action="store_true",
    help="Flip even with odd event numbers to consider the positive or negative muon as the W-like muon",
)

parser.add_argument(
    "--selectNonPromptFromSV",
    action="store_true",
    help="Test: define a non-prompt muon enriched control region",
)
parser.add_argument(
    "--selectNonPromptFromLightMesonDecay",
    action="store_true",
    help="Test: define a non-prompt muon enriched control region with muons from light meson decays",
)

parser.add_argument(
    "--useGlobalOrTrackerVeto",
    action="store_true",
    help="Use global-or-tracker veto definition and scale factors instead of global only",
)
parser.add_argument(
    "--vetoGenPartPt",
    type=float,
    default=15.0,
    help="Minimum pT for the postFSR gen muon when defining the variation of the veto efficiency",
)


def make_timehelper(filename):
    def to_time(x):
        timestamp = datetime.strptime(x, "%m/%d/%y %H:%M:%S")
        ## calculate julian date since 01.01.2000
        jd = (
            367 * timestamp.year
            - np.floor(7 * (timestamp.year + np.floor((timestamp.month + 9) / 12)) / 4)
            + np.floor(275 * timestamp.month / 9)
            + timestamp.day
            - 730531.5
            + (timestamp.hour + timestamp.minute / 60 + timestamp.second / 3600) / 24
        )
        # calculate greenwich mst
        gmst = (
            67310.54841
            + (876600 * 3600 + 8640184.812866) * jd
            + 0.093104 * jd**2
            - 6.2e-6 * jd**3
        ) % 86400
        gmst /= 3600

        lst = (
            gmst + 46.309879 / 15
        ) % 24  ## include longitudinal correction, based on pt 5 at cern, could try to get a more accurate (and precise) number
        return lst

    return make_brilcalc_helper(filename, idx=2, action=to_time)


def mass_extraction(dataframe, name, root_dataype, filter_name):

    condition = (
        lambda x, idx, f=filter_name: f"Sum({f}) > {idx} ? ROOT::Math::PtEtaPhiMVector({x}_pt[{f}][{idx}], {x}_eta[{f}][{idx}], {x}_phi[{f}][{idx}], wrem::muon_mass) : ROOT::Math::PtEtaPhiMVector(0,0,0,0)"
    )

    new_df = dataframe.Define(f"{name}mu_mom4", condition(f"{root_dataype}", 0))
    new_df = new_df.Define(f"{name}smu_mom4", condition(f"{root_dataype}", 1))

    new_df = new_df.Define(
        f"{name}ll_mom4",
        f"ROOT::Math::PxPyPzEVector({name}mu_mom4)+ROOT::Math::PxPyPzEVector({name}smu_mom4)",
    )
    new_df = new_df.Define(f"{name}mll", f"{name}ll_mom4.mass()")
    new_df = new_df.Define(
        f"{name}pass", f"{name}mll >= 15 && {name}mll <= 120 && Sum({filter_name})==2"
    )

    return new_df


def trigger_tightID_sep(dataframe):
    dataframe = dataframe.Define(
        "leading_muon_passTrigger",
        "wrem::hasTriggerMatch(mu_mom4.eta(),mu_mom4.phi(),TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])",
    )
    dataframe = dataframe.Define(
        "subleading_muon_passTrigger",
        "wrem::hasTriggerMatch(smu_mom4.eta(),smu_mom4.phi(),TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])",
    )

    ### detects two muons but only one has the right momentum
    dtight = dataframe.Filter("Sum(Muon_tightId) == 2")

    dtdt = dtight.Filter("subleading_muon_passTrigger && leading_muon_passTrigger")

    dtst = dtight.Filter("subleading_muon_passTrigger != leading_muon_passTrigger")

    stst = dataframe.Filter(
        "(subleading_muon_passTrigger && Muon_tightId[1]) != (leading_muon_passTrigger && Muon_tightId[0])"
    )

    return dtdt, dtst, stst


def luminometer_filter(df, lumi_name, filter_helper, helper):
    df_filtered = df.Define(
        f"in_{lumi_name}", filter_helper, ["run", "luminosityBlock"]
    )
    df_filtered = df_filtered.Filter(f"in_{lumi_name}")
    df_filtered = df_filtered.Define(
        f"lumival_{lumi_name}", helper, ["run", "luminosityBlock"]
    )
    df_filtered = df_filtered.Define(
        f"sbilval_{lumi_name}", f"lumival_{lumi_name}/fill_count*1/24"
    )

    df_filtered = df_filtered.Define(
        f"sbilval_nom_{lumi_name}", "lumival/fill_count*1/24"
    )
    ### histogram
    df_count_hist = df_filtered.HistoBoost(f"count_{lumi_name}", [axis_date], ["time"])

    df_filtered_hist = df_filtered.HistoBoost(
        f"lumi_{lumi_name}", [axis_date], ["time", f"lumival_{lumi_name}"]
    )

    df_filtered_hist_nominal = df_filtered.HistoBoost(
        f"lumi_in_{lumi_name}", [axis_date], ["time", "lumival"]
    )

    df_filtered_hist_sbil = df_filtered.HistoBoost(
        f"sbil_{lumi_name}", [axis_date], ["time", f"sbilval_{lumi_name}"]
    )

    if lumi_name == "pcc":
        return (
            df_filtered_hist,
            df_filtered_hist_nominal,
            df_count_hist,
            df_filtered_hist_sbil,
        )
    else:
        return df_filtered_hist, df_filtered_hist_nominal, df_count_hist


def make_prefire_hists(df, results, name, axes=2):
    df_H = df.Define(
        "weight_newMuonPrefiringSF_H",
        muon_prefiring_helper_H,
        ["Muon_eta", "Muon_pt", "Muon_phi", "Muon_charge", "Muon_tightId"],
    )
    df_BG = df.Define(
        "weight_newMuonPrefiringSF_BG",
        muon_prefiring_helper_BG,
        ["Muon_eta", "Muon_pt", "Muon_phi", "Muon_charge", "Muon_tightId"],
    )

    if axes == 1:
        a1 = axis_pt_high
        a2 = axis_pt_high_copy
        pt_1 = "pt_first"
        pt_2 = "pt_second"
        eta_1 = "eta_first"
        eta_2 = "eta_second"
    elif axes == 2:
        a1 = axis_pt_high
        a2 = axis_pt_low
        pt_1 = "pt_leading"
        pt_2 = "pt_subleading"
        eta_1 = "eta_leading"
        eta_2 = "eta_subleading"

    h_weights = df_H.HistoBoost(
        f"{name}_H",
        [axis_mll, axis_mll_copy, a1, axis_eta, a2, axis_eta_copy],
        [
            "mll",
            "gen_mll",
            pt_1,
            eta_1,
            pt_2,
            eta_2,
            "weight_newMuonPrefiringSF_H",
        ],
    )
    bg_weights = df_BG.HistoBoost(
        f"{name}_BG",
        [axis_mll, axis_mll_copy, a1, axis_eta, a2, axis_eta_copy],
        [
            "mll",
            "gen_mll",
            pt_1,
            eta_1,
            pt_2,
            eta_2,
            "weight_newMuonPrefiringSF_BG",
        ],
    )
    results.append(h_weights)
    results.append(bg_weights)

    syst_tools.add_L1Prefire_unc_hists(
        results,
        df_BG,
        [axis_mll, axis_mll_copy, a1, axis_eta, a2, axis_eta_copy],
        [
            "mll",
            "gen_mll",
            pt_1,
            eta_1,
            pt_2,
            eta_2,
        ],
        helper_stat=muon_prefiring_helper_stat_BG,
        helper_syst=muon_prefiring_helper_syst_BG,
        storage_type=hist.storage.Double(),
        base_name=name + "_BG",
        weight="weight_newMuonPrefiringSF_BG",
    )

    syst_tools.add_L1Prefire_unc_hists(
        results,
        df_H,
        [axis_mll, axis_mll_copy, a1, axis_eta, a2, axis_eta_copy],
        [
            "mll",
            "gen_mll",
            pt_1,
            eta_1,
            pt_2,
            eta_2,
        ],
        helper_stat=muon_prefiring_helper_stat_H,
        helper_syst=muon_prefiring_helper_syst_H,
        storage_type=hist.storage.Double(),
        base_name=name + "_H",
        weight="weight_newMuonPrefiringSF_H",
    )

    return bg_weights, h_weights


redo_cdf = True
args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
era = args.era
calib_filepaths = common.calib_filepaths
lumi_files_path = "/work/submit/jbenke/WRemnants/wremnants/datasets"  #### THIS IS A REALLY DUMB WAY TO DO THIS
# hoping this can go up top
lumicsv = f"{lumi_files_path}/bylsoutput_nBunches.csv"
hfoc_csv = f"{lumi_files_path}/bylsoutput_nBunches_HFOC.csv"
pcc_csv = f"{lumi_files_path}/bylsoutput_nBunches_PCC.csv"
ramses_csv = f"{lumi_files_path}/bylsoutput_nBunches_RAMSES.csv"

brilcalc_helper = make_timehelper(lumicsv)
lumi_no_time = make_lumihelper(lumicsv)  # post_vfp
lumi_bunch_helper = make_brilcalc_helper(lumicsv, idx=9, action=float)

hfoc_helper = make_lumihelper(hfoc_csv)
pcc_helper = make_lumihelper(pcc_csv)
ramses_helper = make_lumihelper(ramses_csv)

hfoc_bunch_helper = make_brilcalc_helper(hfoc_csv, idx=9, action=float)
pcc_bunch_helper = make_brilcalc_helper(pcc_csv, idx=9, action=float)
ramses_bunch_helper = make_brilcalc_helper(ramses_csv, idx=9, action=float)

hfoc_filter_helper = make_brilcalc_filter_helper(hfoc_csv)
pcc_filter_helper = make_brilcalc_filter_helper(pcc_csv)
ramses_filter_helper = make_brilcalc_filter_helper(ramses_csv)

### or are these the right ones?
(
    muon_prefiring_helper_BG,
    muon_prefiring_helper_stat_BG,
    muon_prefiring_helper_syst_BG,
) = muon_prefiring.make_muon_prefiring_helpers(era="2016BG")
muon_prefiring_helper_H, muon_prefiring_helper_stat_H, muon_prefiring_helper_syst_H = (
    muon_prefiring.make_muon_prefiring_helpers(era="2016H")
)

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    nanoVersion="v9",
    base_path=args.dataPath,
    extended="msht20an3lo" not in args.pdfs,
    era=era,
)

axis_date = hist.axis.Regular(24, 0, 24, name="time", overflow=False, underflow=False)

axis_sbil = hist.axis.Regular(24, 9e-7, 3e-8, name="sbil")
axis_num_muons = hist.axis.Regular(3, -0.5, 2.5, name="num_muons")

axis_eta = hist.axis.Regular(12, -2.4, 2.4, name="eta_lead")
axis_eta_copy = hist.axis.Regular(12, -2.4, 2.4, name="eta_sublead")

# axis_pt_high = hist.axis.Regular(12, 25, 80, name="pt_lead")
# # axis_pt_high_copy = hist.axis.Regular(12, 25, 80, name="pt_sublead")
# axis_pt_high_copy = hist.axis.Regular(14, 15, 80, name="pt_sublead")

# axis_pt_low = hist.axis.Regular(14, 15, 80, name="pt_sublead")
# axis_pt_low_copy = hist.axis.Regular(14, 15, 80, name="pt_lead")


axis_pt_high = hist.axis.Variable(
    [
        24.9125,
        34.9961,
        38.9813,
        41.8262,
        43.9753,
        45.6349,
        47.1672,
        49.1107,
        52.1681,
        58.071,
        80,
    ],
    name="pt_lead",
)

axis_pt_high_copy = hist.axis.Variable(
    [
        24.9125,
        34.9961,
        38.9813,
        41.8262,
        43.9753,
        45.6349,
        47.1672,
        49.1107,
        52.1681,
        58.071,
        80,
    ],
    name="pt_sublead",
)

axis_pt_low = hist.axis.Variable(
    [
        15,
        21,
        24.9125,
        34.9961,
        38.9813,
        41.8262,
        43.9753,
        45.6349,
        47.1672,
        49.1107,
        52.1681,
        58.071,
        80,
    ],
    name="pt_sublead",
)

axis_pt_low_copy = hist.axis.Variable(
    [
        15,
        21,
        24.9125,
        34.9961,
        38.9813,
        41.8262,
        43.9753,
        45.6349,
        47.1672,
        49.1107,
        52.1681,
        58.071,
        80,
    ],
    name="pt_lead",
)

axis_mll = hist.axis.Variable(
    [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120], name="mll"
)
axis_mll_copy = hist.axis.Variable(
    [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120], name="gen_mll"
)


########################################################
def build_graph_lumi(df, dataset):
    df = df.Define("time", brilcalc_helper, ["run", "luminosityBlock"])
    hist_lumi_nom = df.HistoBoost("lumi_nom", [axis_date], ["time", "lumival"])
    df = df.Define("fill_count", lumi_bunch_helper, ["run", "luminosityBlock"])

    df_H = df.Filter("run >= 281613")
    hist_lumi_post = df_H.HistoBoost("lumi_post", [axis_date], ["time", "lumival"])
    df_B = df.Filter("run < 281613")
    hist_lumi_pre = df_B.HistoBoost("lumi_pre", [axis_date], ["time", "lumival"])

    hist_hfoc_filtered, hist_nominal_in_hfoc_filtered, hist_hfoc_count = (
        luminometer_filter(df, "hfoc", hfoc_filter_helper, hfoc_helper)
    )
    hist_pcc_filtered, hist_nominal_in_pcc_filtered, hist_pcc_count, hist_pcc_sbil = (
        luminometer_filter(df, "pcc", pcc_filter_helper, pcc_helper)
    )
    hist_ramses_filtered, hist_nominal_in_ramses_filtered, hist_ramses_count = (
        luminometer_filter(df, "ramses", ramses_filter_helper, ramses_helper)
    )

    results = [
        hist_lumi_nom,
        hist_hfoc_filtered,
        hist_pcc_filtered,
        hist_ramses_filtered,
        hist_nominal_in_hfoc_filtered,
        hist_nominal_in_pcc_filtered,
        hist_nominal_in_ramses_filtered,
        hist_pcc_sbil,
        hist_hfoc_count,
        hist_pcc_count,
        hist_ramses_count,
        hist_lumi_pre,
        hist_lumi_post,
    ]
    return results


def build_graph(df, dataset):

    logger.info(f"fomrbuild graph for dataset: {dataset.name}")
    era = args.era
    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
        df = df.Define("time", brilcalc_helper, ["run", "luminosityBlock"])
        hist_time = df.HistoBoost("time", [axis_date], ["time"])
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")
    weightsum = df.SumAndCount("weight")

    ###### NEED TO GENERATE THE RIGHT VARIABLES ####
    ## need the original pt and eta and whatnot, this
    df = df.Define(
        "vetoMuonsPre",
        "Muon_looseId && abs(Muon_dxybs) < 0.05 && Muon_charge != -99",
    )
    df = df.Define(
        "Muon_isGoodGlobal",
        "Muon_isGlobal && Muon_highPurity",
    )

    df = df.Define(
        "goodTrigObjs",
        f"wrem::goodMuonTriggerCandidate<wrem::Era::Era_2016PostVFP>(TrigObj_id,TrigObj_filterBits)",
    )

    df = df.Define(
        "veto_muon",
        "vetoMuonsPre && Muon_isGoodGlobal && Muon_pt>=25 && abs(Muon_eta) < 2.4",
    )

    df = df.Define(
        "pt_loose_muon",
        f"vetoMuonsPre && Muon_isGoodGlobal && Muon_pt>=15 && abs(Muon_eta) < 2.4",
    )

    df = df.Define("sum_veto_muons", "Sum(veto_muon)")
    df = df.Define("sum_pt_loose_muons", "Sum(pt_loose_muon)")

    df = mass_extraction(df, "", "Muon", "pt_loose_muon")

    df = df.Define("pt_leading", "mu_mom4.pt()")
    df = df.Define("pt_subleading", "smu_mom4.pt()")
    df = df.Define("eta_leading", "mu_mom4.eta()")
    df = df.Define("eta_subleading", "smu_mom4.eta()")

    ## may need to change this later, technically if our sample is biased then we won't get 50/50 from subleading and leading muons
    df = df.Define("n2_hist_def", "rand() % 2")
    df = df.Define("pt_first", "n2_hist_def == 1 ? pt_leading : pt_subleading")
    df = df.Define("eta_first", "n2_hist_def == 1 ? eta_leading : eta_subleading")
    df = df.Define("pt_second", "n2_hist_def == 1 ? pt_subleading : pt_leading")
    df = df.Define("eta_second", "n2_hist_def ==1 ? eta_subleading : eta_leading")

    if not dataset.is_data:
        df = theory_tools.define_postfsr_vars(df)
        df = df.Define(
            "postfsrMuons_inAcc",
            f"postfsrMuons && abs(GenPart_eta) < 2.4 && GenPart_pt > 25",
        )
        df = df.Define("sum_gen_muons", "Sum(postfsrMuons_inAcc)")
        df = mass_extraction(df, "gen_", "GenPart", "postfsrMuons_inAcc")

        ### these are all for the case that there are two veto muons

        df_1 = df.Filter("gen_pass")
        df_2 = df.Filter("!gen_pass")
        # pass generator
        df_pg = df_1.Filter("sum_veto_muons == 2")
        ### fail generator
        df_fg = df_2.Filter("sum_veto_muons == 2")

        df_pt_loose_pg = df_1.Filter(
            "(sum_veto_muons == 1 && sum_pt_loose_muons == 2) || (sum_veto_muons == 2)"
        )
        df_pt_loose_fg = df_2.Filter(
            "(sum_veto_muons == 1 && sum_pt_loose_muons == 2) || (sum_veto_muons == 2)"
        )

        ## double tight double trigger
        df_dtdt_fg, _, _ = trigger_tightID_sep(df_fg)
        df_dtdt_pg, _, _ = trigger_tightID_sep(df_pg)

        # # #### same extraction but using the more liberal dataset for muons
        _, df_dtst_pg, df_stst_pg = trigger_tightID_sep(
            df_pt_loose_pg
        )  ## calculating single tight single trigger
        _, df_dtst_fg, df_stst_fg = trigger_tightID_sep(df_pt_loose_fg)

        hist_pass_gen = df_1.HistoBoost(
            "pass_gen",
            [
                axis_mll,
                axis_mll_copy,
                axis_pt_high,
                axis_eta,
                axis_pt_low,
                axis_eta_copy,
            ],
            [
                "mll",
                "gen_mll",
                "pt_first",
                "eta_first",
                "pt_second",
                "eta_second",
                "weight",
            ],
        )

        ## was pt1copy
        hist_prfg = df_dtdt_fg.HistoBoost(
            "dtdt_prfg",
            [
                axis_mll,
                axis_mll_copy,
                axis_pt_high,
                axis_eta,
                axis_pt_high_copy,
                axis_eta_copy,
            ],
            [
                "mll",
                "gen_mll",
                "pt_first",
                "eta_first",
                "pt_second",
                "eta_second",
                "weight",
            ],
        )

        ### this one needs to account for only one muon passing the momentum
        hist_dtst_prfg = df_dtst_fg.HistoBoost(
            "dtst_prfg",
            [
                axis_mll,
                axis_mll_copy,
                axis_pt_high,
                axis_eta,
                axis_pt_low,
                axis_eta_copy,
            ],
            [
                "mll",
                "gen_mll",
                "pt_leading",
                "eta_leading",
                "pt_subleading",
                "eta_subleading",
                "weight",
            ],  # double tight single trigger
        )

        hist_stst_prfg = df_stst_fg.HistoBoost(
            "stst_prfg",
            [
                axis_mll,
                axis_mll_copy,
                axis_pt_high,
                axis_eta,
                axis_pt_low,
                axis_eta_copy,
            ],
            [
                "mll",
                "gen_mll",
                "pt_leading",
                "eta_leading",
                "pt_subleading",
                "eta_subleading",
                "weight",
            ],
        )

        ### was pt1 copy
        hist_prpg = df_dtdt_pg.HistoBoost(
            "dtdt_prpg",
            [
                axis_mll,
                axis_mll_copy,
                axis_pt_high,
                axis_eta,
                axis_pt_high_copy,
                axis_eta_copy,
            ],
            [
                "mll",
                "gen_mll",
                "pt_first",
                "eta_first",
                "pt_second",
                "eta_second",
                "weight",
            ],
        )

        ### back to old
        hist_dtst_prpg = df_dtst_pg.HistoBoost(
            "dtst_prpg",
            [
                axis_mll,
                axis_mll_copy,
                axis_pt_high,
                axis_eta,
                axis_pt_low,
                axis_eta_copy,
            ],
            [
                "mll",
                "gen_mll",
                "pt_leading",
                "eta_leading",
                "pt_subleading",
                "eta_subleading",
                "weight",
            ],  # double tight single trigger
        )
        hist_stst_prpg = df_stst_pg.HistoBoost(
            "stst_prpg",
            [
                axis_mll,
                axis_mll_copy,
                axis_pt_high,
                axis_eta,
                axis_pt_low,
                axis_eta_copy,
            ],
            [
                "mll",
                "gen_mll",
                "pt_leading",
                "eta_leading",
                "pt_subleading",
                "eta_subleading",
                "weight",
            ],
        )

        if redo_cdf:
            # fine_bin_axis = hist.axis.Regular(400, 15, 120, name="mll_fine_bin")
            fine_bin_axis = hist.axis.Regular(400, 15, 80, name="pt_fine_bin")
            # fine_bin_axis = hist.axis.Regular(400, -2.4, 2.4, name="eta_fine_bin")

            fine_bin_mll = df_dtdt_pg.HistoBoost(
                "fine_bin_axis_gen", [fine_bin_axis], ["pt_leading", "weight"]
            )
            results.append(fine_bin_mll)

        results.append(hist_pass_gen)
        results.append(hist_prpg)
        results.append(hist_dtst_prpg)
        results.append(hist_stst_prpg)
        results.append(hist_prfg)
        results.append(hist_dtst_prfg)
        results.append(hist_stst_prfg)

        make_prefire_hists(df_dtdt_pg, results, "dtdt_prpg", axes=1)
        make_prefire_hists(df_dtst_pg, results, "dtst_prpg")
        make_prefire_hists(df_stst_pg, results, "stst_prpg")

    else:  ### this is for real data

        df_veto = df.Filter("sum_veto_muons == 2")

        dtdt, _, _ = trigger_tightID_sep(df_veto)

        df_pt_loose = df.Filter(
            "(sum_veto_muons == 1 && sum_pt_loose_muons == 2) || (sum_veto_muons == 2)"
        )

        _, dtst, stst = trigger_tightID_sep(df_pt_loose)

        hist_time_proj = df.HistoBoost(
            "time_proj",
            [
                axis_date,
                axis_mll,
                axis_pt_high,
                axis_eta,
                axis_pt_high_copy,
                axis_eta_copy,
            ],
            [
                "time",
                "mll",
                "pt_leading",
                "eta_leading",
                "pt_subleading",
                "eta_subleading",
            ],
        )

        hist_time_proj_2 = df.HistoBoost(
            "time_proj_2",
            [
                axis_date,
                axis_mll,
                axis_pt_high,
                axis_eta,
                axis_pt_low,
                axis_eta_copy,
            ],
            [
                "time",
                "mll",
                "pt_leading",
                "eta_leading",
                "pt_subleading",
                "eta_subleading",
            ],
        )

        hist_time_mll = dtdt.HistoBoost(
            "time_mll",
            [
                axis_date,
                axis_mll,
                axis_pt_high,
                axis_eta,
                axis_pt_high_copy,
                axis_eta_copy,
            ],
            ["time", "mll", "pt_first", "eta_first", "pt_second", "eta_second"],
        )
        hist_time_dtst = dtst.HistoBoost(
            "time_dtst",
            [axis_date, axis_mll, axis_pt_high, axis_eta, axis_pt_low, axis_eta_copy],
            [
                "time",
                "mll",
                "pt_leading",
                "eta_leading",
                "pt_subleading",
                "eta_subleading",
            ],
        )
        hist_time_stst = stst.HistoBoost(
            "time_stst",
            [axis_date, axis_mll, axis_pt_high, axis_eta, axis_pt_low, axis_eta_copy],
            [
                "time",
                "mll",
                "pt_leading",
                "eta_leading",
                "pt_subleading",
                "eta_subleading",
            ],
        )

        if redo_cdf:
            # fine_bin_axis = hist.axis.Regular(400, 15, 120, name="mll_fine_bin")
            fine_bin_axis = hist.axis.Regular(400, 15, 80, name="pt_fine_bin")
            # fine_bin_axis = hist.axis.Regular(400, -2.4, 2.4, name="eta_fine_bin")

            fine_bin_mll = df.HistoBoost(
                "fine_bin_axis_gen", [fine_bin_axis], ["pt_leading", "weight"]
            )
            results.append(fine_bin_mll)

        results.append(hist_time_proj)
        results.append(hist_time_proj_2)
        results.append(hist_time)
        results.append(hist_time_mll)
        results.append(hist_time_dtst)
        results.append(hist_time_stst)
    return results, weightsum


logger.debug(f"Datasets are {[d.name for d in datasets]}")
resultdict = narf.build_and_run(datasets[::-1], build_graph, build_graph_lumi)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)
