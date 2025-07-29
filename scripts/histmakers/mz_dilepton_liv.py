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
from wremnants import theory_tools
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
        f"{name}pass", f"{name}mll >= 60 && {name}mll <= 120 && Sum({filter_name})==2"
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

    dtight_dtrig = dtight.Filter(
        "subleading_muon_passTrigger && leading_muon_passTrigger"
    )
    dtight_strig = dtight.Filter(
        "subleading_muon_passTrigger != leading_muon_passTrigger"
    )

    stight_strig = dataframe.Filter(
        "(subleading_muon_passTrigger && Muon_tightId[1]) != (leading_muon_passTrigger && Muon_tightId[0])"
    )

    return dtight_dtrig, dtight_strig, stight_strig


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


args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
era = args.era
calib_filepaths = common.calib_filepaths

# hoping this can go up top
lumicsv = f"{common.data_dir}/bylsoutput_nBunches.csv"
hfoc_csv = f"{common.data_dir}/bylsoutput_nBunches_HFOC.csv"
pcc_csv = f"{common.data_dir}/bylsoutput_nBunches_PCC.csv"
ramses_csv = f"{common.data_dir}/bylsoutput_nBunches_RAMSES.csv"

brilcalc_helper = make_timehelper(lumicsv)
lumi_no_time = make_lumihelper(lumicsv)
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
axis_mll = hist.axis.Variable(
    [
        60,
        70,
        75,
        78,
        80,
        82,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        100,
        102,
        105,
        110,
        120,
    ],
    name="mll",
)
axis_mll_2 = hist.axis.Variable(
    [
        60,
        70,
        75,
        78,
        80,
        82,
        85,
        86,
        87,
        88,
        89,
        90,
        91,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        100,
        102,
        105,
        110,
        120,
    ],
    name="gen_mll",
)

axis_sbil = hist.axis.Regular(24, 9e-7, 3e-8, name="sbil")
axis_num_muons = hist.axis.Regular(3, -0.5, 2.5, name="num_muons")
axis_mll = hist.axis.Variable(
    [
        60.3,
        85.2298,
        88.1398,
        89.3644,
        90.16,
        90.8102,
        91.428,
        92.1163,
        93.0461,
        94.9463,
        120,
    ],
    name="mll",
)
axis_mll_2 = hist.axis.Variable(
    [
        60.3,
        85.2298,
        88.1398,
        89.3644,
        90.16,
        90.8102,
        91.428,
        92.1163,
        93.0461,
        94.9463,
        120,
    ],
    name="gen_mll",
)


########################################################
def build_graph_lumi(df, dataset):
    df = df.Define("time", brilcalc_helper, ["run", "luminosityBlock"])
    hist_lumi_nom = df.HistoBoost("lumi_nom", [axis_date], ["time", "lumival"])
    df = df.Define("fill_count", lumi_bunch_helper, ["run", "luminosityBlock"])

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
        "veto_muon",
        "vetoMuonsPre && Muon_isGoodGlobal && Muon_pt>=25 && abs(Muon_eta) < 2.4",
    )
    df = df.Define(
        "goodTrigObjs",
        f"wrem::goodMuonTriggerCandidate<wrem::Era::Era_2016PostVFP>(TrigObj_id,TrigObj_filterBits)",
    )
    df = df.Define("sum_veto_muons", "Sum(veto_muon)")

    if not dataset.is_data:
        df = theory_tools.define_postfsr_vars(df)
        df = df.Define(
            "postfsrMuons_inAcc",
            f"postfsrMuons && abs(GenPart_eta) < 2.4 && GenPart_pt > 25",
        )
        df = df.Define("sum_gen_muons", "Sum(postfsrMuons_inAcc)")

        df = mass_extraction(df, "gen_", "GenPart", "postfsrMuons_inAcc")
        df = mass_extraction(df, "", "Muon", "veto_muon")

        ### these are all for the case that there are two veto muons
        ### this is for the case that there are two ve
        ### ones that JUST pass the generator

        df_1 = df.Filter("gen_pass")
        df_2 = df.Filter("!gen_pass")

        df_21 = df_1.Filter("sum_veto_muons == 2")
        df_22 = df_2.Filter("sum_veto_muons == 2")

        hist_pass_gen = df_1.HistoBoost("pass_gen", [axis_mll_2], ["gen_mll", "weight"])

        hist_pass_reco_pass_gen = df_21.HistoBoost(
            "pass_reco_pass_gen", [axis_mll, axis_mll_2], ["mll", "gen_mll", "weight"]
        )
        hist_pass_reco_fail_gen = df_22.HistoBoost(
            "pass_reco_fail_gen", [axis_mll, axis_mll_2], ["mll", "gen_mll", "weight"]
        )

        ##### pass reco, fail generator
        dtight_dtrig_df_22, dtight_strig_df_22, stight_strig_df_22 = (
            trigger_tightID_sep(df_22)
        )

        hist_mll_prfg = dtight_dtrig_df_22.HistoBoost(
            "mll_dtdt_prfg", [axis_mll, axis_mll_2], ["mll", "gen_mll", "weight"]
        )
        hist_mll_dtight_strig_prfg = dtight_strig_df_22.HistoBoost(
            "mll_dtst_prfg",
            [axis_mll, axis_mll_2],
            ["mll", "gen_mll", "weight"],  # double tight single trigger
        )
        hist_mll_stight_strig_prfg = stight_strig_df_22.HistoBoost(
            "mll_stst_prfg", [axis_mll, axis_mll_2], ["mll", "gen_mll", "weight"]
        )

        ##### pass reco, pass generator
        dtight_dtrig_df_21, dtight_strig_df_21, stight_strig_df_21 = (
            trigger_tightID_sep(df_21)
        )

        hist_mll_prpg = dtight_dtrig_df_21.HistoBoost(
            "mll_dtdt_prpg", [axis_mll, axis_mll_2], ["mll", "gen_mll", "weight"]
        )
        hist_mll_dtight_strig_prpg = dtight_strig_df_21.HistoBoost(
            "mll_dtst_prpg",
            [axis_mll, axis_mll_2],
            ["mll", "gen_mll", "weight"],  # double tight single trigger
        )
        hist_mll_stight_strig_prpg = stight_strig_df_21.HistoBoost(
            "mll_stst_prpg", [axis_mll, axis_mll_2], ["mll", "gen_mll", "weight"]
        )
        fine_bin_axis = hist.axis.Regular(200, 60, 120, name="mll_fine_bin")
        fine_bin_mll = dtight_dtrig_df_21.HistoBoost(
            "fine_bin_axis_gen", [fine_bin_axis], ["mll", "weight"]
        )

        results.append(hist_pass_reco_pass_gen)
        results.append(hist_pass_reco_fail_gen)
        results.append(hist_pass_gen)
        results.append(hist_mll_prpg)
        results.append(hist_mll_dtight_strig_prpg)
        results.append(hist_mll_stight_strig_prpg)
        results.append(hist_mll_prfg)
        results.append(hist_mll_dtight_strig_prfg)
        results.append(hist_mll_stight_strig_prfg)
        results.append(fine_bin_mll)

    else:  ### this is for real data
        df = df.Filter("sum_veto_muons == 2")

        df = mass_extraction(df, "", "Muon", "veto_muon")

        dtight_dtrig, dtight_strig, stight_strig = trigger_tightID_sep(df)

        fine_bin_axis = hist.axis.Regular(200, 60, 120, name="mll_fine_bin")
        fine_bin_mll = df.HistoBoost(
            "fine_bin_axis_gen", [fine_bin_axis], ["mll", "weight"]
        )

        hist_time_proj = df.HistoBoost(
            "time_proj", [axis_date, axis_mll, axis_mll_2], ["time", "mll", "mll"]
        )

        hist_time_mll = dtight_dtrig.HistoBoost(
            "time_mll",
            [
                axis_date,
                axis_mll,
            ],
            ["time", "mll"],
        )
        hist_time_mll_dtight_strig = dtight_strig.HistoBoost(
            "time_mll_dtst",
            [
                axis_date,
                axis_mll,
            ],
            ["time", "mll"],
        )
        hist_time_mll_stight_strig = stight_strig.HistoBoost(
            "time_mll_stst",
            [
                axis_date,
                axis_mll,
            ],
            ["time", "mll"],
        )

        results.append(hist_time_proj)
        results.append(hist_time)
        results.append(hist_time_mll)
        results.append(hist_time_mll_dtight_strig)
        results.append(hist_time_mll_stight_strig)
        results.append(fine_bin_mll)
    return results, weightsum


logger.debug(f"Datasets are {[d.name for d in datasets]}")
resultdict = narf.build_and_run(datasets[::-1], build_graph, build_graph_lumi)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)
