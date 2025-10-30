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
from wremnants import muon_prefiring, muon_selections, syst_tools, theory_tools
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

    a1 = axis_pt_low
    a2 = axis_pt_high
    pt_1 = "pt_probe"
    pt_2 = "pt_tag"
    eta_1 = "eta_probe"
    eta_2 = "eta_tag"

    h_weights = df_H.HistoBoost(
        f"{name}_H",
        [
            axis_mll,
            a1,
            axis_eta,
            a2,
            axis_eta_copy,
        ],
        [
            "goodLoose_mll",
            pt_1,
            eta_1,
            pt_2,
            eta_2,
            "weight_newMuonPrefiringSF_H",
        ],
    )
    bg_weights = df_BG.HistoBoost(
        f"{name}_BG",
        [
            axis_mll,
            a1,
            axis_eta,
            a2,
            axis_eta_copy,
        ],
        [
            "goodLoose_mll",
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
        [
            axis_mll,
            a1,
            axis_eta,
            a2,
            axis_eta_copy,
        ],
        [
            "goodLoose_mll",
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
        [
            axis_mll,
            a1,
            axis_eta,
            a2,
            axis_eta_copy,
        ],
        [
            "goodLoose_mll",
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


## make the two sets of prefiring helpers for each port of the data
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

axis_sbil = hist.axis.Regular(
    24, 9e-7, 3e-8, name="sbil", overflow=False, underflow=False
)

### need copies of each axis so that i can have one be the tag and probe. s
# hould probably have a function to take in a probe axis and spit out a tag axis
axis_eta = hist.axis.Variable(
    [-2.4, -1.40655, -0.68156, -0.00848, 0.66796, 1.4006, 2.4], name="eta_probe"
)
axis_eta_copy = hist.axis.Variable(
    [-2.4, -1.40655, -0.68156, -0.00848, 0.66796, 1.4006, 2.4], name="eta_tag"
)


axis_pt_high = hist.axis.Variable(
    [
        15,
        # 21,
        25,
        32.35393,
        35.70991,
        38.30856,
        40.43642,
        42.22635,
        43.92092,
        45.87573,
        48.56281,
        53.1789,
        80,
    ],
    name="pt_tag",
)

axis_pt_high_copy = hist.axis.Variable(
    [
        15,
        # 21,
        25,
        32.35393,
        35.70991,
        38.30856,
        40.43642,
        42.22635,
        43.92092,
        45.87573,
        48.56281,
        53.1789,
        80,
    ],
    name="pt_tag",
)

axis_pt_low = hist.axis.Variable(
    [
        15,
        # 21,
        25,
        32.35393,
        35.70991,
        38.30856,
        40.43642,
        42.22635,
        43.92092,
        45.87573,
        48.56281,
        53.1789,
        80,
    ],
    name="pt_probe",
)

axis_pt_low_copy = hist.axis.Variable(
    [
        15,
        # 21,
        25,
        32.35393,
        35.70991,
        38.30856,
        40.43642,
        42.22635,
        43.92092,
        45.87573,
        48.56281,
        53.1789,
        80,
    ],
    name="pt_probe",
)

axis_mll = hist.axis.Variable(
    [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120], name="mll"
)
axis_mll_copy = hist.axis.Variable(
    [15, 30, 40, 45, 50, 55, 60, 65, 70, 76, 106, 110, 115, 120], name="goodLoose_mll"
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

    isoThreshold = args.isolationThreshold

    isoBranch = muon_selections.getIsoBranch(args.isolationDefinition)

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

    df = df.Define(
        "Muon_isGoodGlobal",
        f" Muon_isGlobal && Muon_highPurity && Muon_standaloneNumberOfValidHits > 0 && Muon_standalonePt > 15 &&  wrem::vectDeltaR2(Muon_standaloneEta, Muon_standalonePhi, Muon_eta, Muon_phi) < 0.09 && Muon_pt >= 15 && abs(Muon_eta) <= 2.4 && Muon_charge != -99",
    )

    # && {isoBranch} < {isoThreshold} this selection seems strange. becuase isoBranch is a name but the threshold is a number
    #

    #### THIS IS WHAT CAUSES THE DIFFERENCE
    df = df.Filter("Sum(Muon_isGoodGlobal) == 2")

    ###     okay so this is already built in. im confused why the isolation is less than but seems to already be in place

    df = df.Filter(
        "Muon_charge[Muon_isGoodGlobal][0] != Muon_charge[Muon_isGoodGlobal][1]"
    )
    df = df.Define(
        "Muon_isGoodMedium",
        f"Muon_isGoodGlobal && Muon_mediumId && abs(Muon_dxybs) < 0.05 && {isoBranch} < {isoThreshold}",
    )

    df = df.Define(
        "goodTrigObjs",
        f"wrem::goodMuonTriggerCandidate<wrem::Era::Era_2016PostVFP>(TrigObj_id,TrigObj_filterBits)",
    )

    df = df.Define(
        "Muon_isGoodTrigger",
        "Muon_pt>=25 && Muon_isGoodMedium && wrem::hasTriggerMatch(Muon_eta,Muon_phi,TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])",
    )
    #### filter to ensure that at least one muon passes HLT
    df = df.Filter(
        "(Muon_isGoodTrigger[Muon_isGoodGlobal][0] == 1) || (Muon_isGoodTrigger[Muon_isGoodGlobal][1] == 1)"
    )
    df = mass_extraction(df, "goodLoose_", "Muon", "Muon_isGoodGlobal")
    df = df.Filter("goodLoose_pass")

    ## define the muon that does not pass as the probe

    # if muon0 passes trigger & tight id (if muon1 passes trigger and tight id: then randomly select 0 vs 1, if muon1 fails, make it the probe), if muon0 fails make it 0
    df = df.Define(
        "mu_probe",
        "Muon_isGoodTrigger[Muon_isGoodGlobal][0] == 1 && Muon_isGoodMedium[Muon_isGoodGlobal][0] == 1 ? (Muon_isGoodTrigger[Muon_isGoodGlobal][1] == 1 && Muon_isGoodMedium[Muon_isGoodGlobal][1] == 1 ? abs(rand()%2): 1) : 0",
    )  ## looked through nanoaod for event number, couldnt find it. genEventcount did not work. i checked and this does evenly split it.
    df = df.Define("mu_tag", "abs(mu_probe - 1)")

    ### isEvenEvent is the thing that i could use
    df = df.Define(
        "pt_probe", "mu_probe == 0 ? goodLoose_mu_mom4.pt() : goodLoose_smu_mom4.pt()"
    )
    df = df.Define(
        "eta_probe",
        "mu_probe == 0 ? goodLoose_mu_mom4.eta() : goodLoose_smu_mom4.eta()",
    )

    df = df.Define(
        "pt_tag", "mu_probe == 0 ? goodLoose_smu_mom4.pt() : goodLoose_mu_mom4.pt()"
    )
    df = df.Define(
        "eta_tag", "mu_probe == 0 ? goodLoose_smu_mom4.eta() :goodLoose_mu_mom4.eta()"
    )

    if not dataset.is_data:

        df = theory_tools.define_postfsr_vars(df)
        df = df.Define(
            "postfsrMuons_loose",
            f"postfsrMuons && abs(GenPart_eta) < 2.4 && GenPart_pt > 15",
        )  ### used to be > 25, not sure if that is changes anything

        #### THIS IS HTE PROBLEM, THIS CRITERION DOES NOT MATCH ALL THE SELECTION CASES I DONT THINKG

        df = mass_extraction(df, "gen_", "GenPart", "postfsrMuons_loose")
        ### these are all for the case that there are two veto muons
        df_pg = df.Filter("gen_pass")

        hist_pass_gen = df.HistoBoost(
            "pass_gen",
            [
                axis_mll,
                axis_pt_low,
                axis_eta,
                axis_pt_high,
                axis_eta_copy,
            ],
            [
                "goodLoose_mll",
                "pt_probe",
                "eta_probe",
                "pt_tag",
                "eta_tag",
                "weight",
            ],
        )

        df_loose = df_pg  ### had it as df_pg
        df_tight = df_loose.Filter(
            "Muon_isGoodMedium[Muon_isGoodGlobal][0] == 1 && Muon_isGoodMedium[Muon_isGoodGlobal][1] == 1"
        )  ## have already defined that this passes the looseId
        df_trig = df_tight.Filter(
            "Muon_isGoodTrigger[Muon_isGoodGlobal][0] == 1 && Muon_isGoodTrigger[Muon_isGoodGlobal][1] == 1"
        )  ## filter for it.

        # ### fail generator
        df_loose_fg = df.Filter("!gen_pass")
        df_tight_fg = df_loose_fg.Filter(
            "Muon_isGoodMedium[Muon_isGoodGlobal][0] == 1 && Muon_isGoodMedium[Muon_isGoodGlobal][1] == 1"
        )  ## have already defined that this passes the looseId
        df_trig_fg = df_tight_fg.Filter(
            "Muon_isGoodTrigger[Muon_isGoodGlobal][0] == 1 && Muon_isGoodTrigger[Muon_isGoodGlobal][1] == 1"
        )  ## filter for it.

        hist_tight_muons_fg = df_tight_fg.HistoBoost(
            "dtst_prfg",
            [
                axis_mll,
                axis_pt_low,
                axis_eta,
                axis_pt_high,
                axis_eta_copy,
            ],
            [
                "goodLoose_mll",
                "pt_probe",
                "eta_probe",
                "pt_tag",
                "eta_tag",
                "weight",
            ],
        )

        hist_loose_muons_fg = df_loose_fg.HistoBoost(
            "stst_prfg",
            [
                axis_mll,
                axis_pt_low,
                axis_eta,
                axis_pt_high,
                axis_eta_copy,
            ],
            [
                "goodLoose_mll",
                "pt_probe",
                "eta_probe",
                "pt_tag",
                "eta_tag",
                "weight",
            ],
        )

        hist_trigger_muons_fg = df_trig_fg.HistoBoost(
            "dtdt_prfg",
            [
                axis_mll,
                axis_pt_low,
                axis_eta,
                axis_pt_high,
                axis_eta_copy,
            ],
            [
                "goodLoose_mll",
                "pt_probe",
                "eta_probe",
                "pt_tag",
                "eta_tag",
                "weight",
            ],
        )

        if redo_cdf:
            # fine_bin_axis = hist.axis.Regular(400, 15, 120, name="mll_fine_bin")
            # fine_bin_axis = hist.axis.Regular(400, 25, 80, name="pt_fine_bin")
            fine_bin_axis = hist.axis.Regular(600, -2.4, 2.4, name="eta_fine_bin")

            fine_bin_mll = df_trig.HistoBoost(
                "fine_bin_axis_gen", [fine_bin_axis], ["eta_probe", "weight"]
            )
            results.append(fine_bin_mll)

        results.append(hist_tight_muons_fg)
        results.append(hist_loose_muons_fg)
        results.append(hist_trigger_muons_fg)
        # results.append(hist_tight_muons_pg)
        # results.append(hist_loose_muons_pg)
        # results.append(hist_trigger_muons_pg)

        results.append(hist_pass_gen)
        make_prefire_hists(df_trig, results, "dtdt_prpg")
        make_prefire_hists(df_tight, results, "dtst_prpg")
        make_prefire_hists(df_loose, results, "stst_prpg")

        make_prefire_hists(df_trig_fg, results, "dtdt_prfg")
        make_prefire_hists(df_tight_fg, results, "dtst_prfg")
        make_prefire_hists(df_loose_fg, results, "stst_prfg")

    else:  ### this is for real data

        stst = df
        dtst = stst.Filter(
            "Muon_isGoodMedium[Muon_isGoodGlobal][0] == 1 && Muon_isGoodMedium[Muon_isGoodGlobal][1] == 1"
        )  ## have already defined that this passes the looseId
        dtdt = dtst.Filter(
            "Muon_isGoodTrigger[Muon_isGoodGlobal][0] == 1 && Muon_isGoodTrigger[Muon_isGoodGlobal][1] == 1"
        )  ## filter for it.

        hist_time_proj = df.HistoBoost(
            "time_proj",
            [
                axis_date,
                axis_mll,
                axis_pt_low,
                axis_eta,
                axis_pt_high,
                axis_eta_copy,
            ],
            [
                "time",
                "goodLoose_mll",
                "pt_probe",
                "eta_probe",
                "pt_tag",
                "eta_tag",
                "weight",
            ],
        )

        hist_time_mll = dtdt.HistoBoost(
            "time_mll",
            [
                axis_date,
                axis_mll,
                axis_pt_low,
                axis_eta,
                axis_pt_high,
                axis_eta_copy,
            ],
            [
                "time",
                "goodLoose_mll",
                "pt_probe",
                "eta_probe",
                "pt_tag",
                "eta_tag",
                "weight",
            ],
        )
        hist_time_dtst = dtst.HistoBoost(
            "time_dtst",
            [
                axis_date,
                axis_mll,
                axis_pt_low,
                axis_eta,
                axis_pt_high,
                axis_eta_copy,
            ],
            [
                "time",
                "goodLoose_mll",
                "pt_probe",
                "eta_probe",
                "pt_tag",
                "eta_tag",
                "weight",
            ],
        )
        hist_time_stst = stst.HistoBoost(
            "time_stst",
            [
                axis_date,
                axis_mll,
                axis_pt_low,
                axis_eta,
                axis_pt_high,
                axis_eta_copy,
            ],
            [
                "time",
                "goodLoose_mll",
                "pt_probe",
                "eta_probe",
                "pt_tag",
                "eta_tag",
                "weight",
            ],
        )

        # if redo_cdf:
        #     # fine_bin_axis = hist.axis.Regular(400, 15, 120, name="mll_fine_bin")
        #     # fine_bin_axis = hist.axis.Regular(400, 25, 80, name="pt_fine_bin")
        #     fine_bin_axis = hist.axis.Regular(400, -2.4, 2.4, name="eta_fine_bin")

        #     fine_bin_mll = df.HistoBoost(
        #         "fine_bin_axis_gen", [fine_bin_axis], ["eta_tag", "weight"]
        #     )
        #     results.append(fine_bin_mll)

        results.append(hist_time_proj)
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
