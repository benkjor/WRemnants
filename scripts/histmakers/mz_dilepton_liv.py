import os
import hist
import numpy as np
import ROOT
from datetime import datetime
import narf

from utilities import common, parsing
from wremnants import (muon_selections, muon_calibration)
from wremnants.datasets.datagroups import Datagroups
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import (
    aggregate_groups,
    scale_to_data,
    write_analysis_output,
)
from wums import logging
from narf.lumitools import make_brilcalc_helper


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


def make_timehelper(filename):
    def to_time(x):
        timestamp = datetime.strptime(x, "%m/%d/%y %H:%M:%S")
        ## calculate julian date since 01.01.2000
        jd = 367*timestamp.year - np.floor(7*(timestamp.year + np.floor((timestamp.month + 9)/12))/4) + np.floor(275*timestamp.month/9) + timestamp.day- 730531.5 + (timestamp.hour + timestamp.minute/60 + timestamp.second/3600)/24
        #calculate greenwich mst
        gmst = (67310.54841 + (876600 * 3600 + 8640184.812866) * jd  + 0.093104 * jd**2 - 6.2e-6 * jd**3) % 86400
        gmst /= 3600
        
        lst = (gmst + 46.309879/15) %24 ## include longitudinal correction, based on pt 5 at cern, could try to get a more accurate (and precise) number
        return lst
    
    return make_brilcalc_helper(filename, idx=2, action=to_time)



args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
era = args.era

#hoping this can go up top
lumicsv = f"{common.data_dir}/bylsoutput.csv" 
brilcalc_helper = make_timehelper(lumicsv)

datasets = getDatasets(
    maxFiles=args.maxFiles,
    filt=args.filterProcs,
    excl=args.excludeProcs,
    nanoVersion="v9",
    base_path=args.dataPath,
    extended="msht20an3lo" not in args.pdfs,
    era=era,
)

axis_date = hist.axis.Regular(24, 0, 24, name = 'time')


def build_graph_lumi(df, dataset):
    df = df.Define("time", brilcalc_helper, ["run", "luminosityBlock"])
    hist_time = df.HistoBoost("time", [axis_date], ['time', 'lumival'])
    results = [hist_time]
    return results

def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    era = args.era
    calib_filepaths = common.calib_filepaths
    results = []
    
    mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = (
    muon_calibration.make_muon_calibration_helpers(args, era=era)
    )
    (
        mc_jpsi_crctn_helper,
        data_jpsi_crctn_helper,
        mc_jpsi_crctn_unc_helper,
        data_jpsi_crctn_unc_helper,
    ) = muon_calibration.make_jpsi_crctn_helpers(
        args, calib_filepaths, make_uncertainty_helper=True)
    
    
    
    cvh_helper = data_calibration_helper if dataset.is_data else mc_calibration_helper
    jpsi_helper = data_jpsi_crctn_helper if dataset.is_data else mc_jpsi_crctn_helper
    
    smearing_helper, smearing_uncertainty_helper = (
        (None, None) if args.noSmearing else muon_calibration.make_muon_smearing_helpers()
    )

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")
    weightsum = df.SumAndCount("weight")
    
    df = df.Define( "isEvenEvent", f"event % 2 {'!=' if args.flipEventNumberSplitting else '=='} 0" ) ## not sure why i have this
    ## select for two muons, plot their invariant mass
    
    df = df.Define("muon_pass_pt", "Muon_pt>=25") # [1,0,1,0 ...]
    df = df.Filter("std::accumulate(muon_pass_pt.begin(), muon_pass_pt.end(), 0) == 2")

    df = df.Define("muon_leading_pt", "Muon_pt[muon_pass_pt][0]") 
    df = df.Define("muon_subleading_pt", "Muon_pt[muon_pass_pt][1]") 
    df = df.Define("muon_total_pt", "Muon_pt[muon_pass_pt][0] + Muon_pt[muon_pass_pt][1]")
    
    ### attempting to copy this section, really not sure what it means
                
    mass_min, mass_max = common.get_default_mz_window()
    isoBranch = muon_selections.getIsoBranch(args.isolationDefinition)
    era = args.era
  

    bias_helper = muon_calibration.make_muon_bias_helpers(args)
    
    df = df.Filter(muon_selections.hlt_string(era))
    df = muon_selections.veto_electrons(df)
    df = muon_selections.apply_met_filters(df) ## what is a met filter
    df = muon_calibration.define_corrected_muons(
        df, cvh_helper, jpsi_helper, args, dataset, smearing_helper, bias_helper
    )

    df = muon_selections.select_veto_muons(df, nMuons=2)
    isoThreshold = args.isolationThreshold
    passIsoBoth = args.muonIsolation[0] + args.muonIsolation[1] == 2
    df = muon_selections.select_good_muons(
        df,
        args.pt[1],
        args.pt[2],
        dataset.group,
        nMuons=2,
        use_trackerMuons=args.trackerMuons,
        use_isolation=passIsoBoth,
        isoBranch=isoBranch,
        isoThreshold=isoThreshold,
        requirePixelHits=args.requirePixelHits,
    )

    df = muon_selections.define_trigger_muons(
        df, dilepton=args.useDileptonTriggerSelection
    )

    # iso cut applied here, if requested, because it needs the definition of trigMuons and nonTrigMuons from muon_selections.define_trigger_muons
    if not passIsoBoth:
        df = muon_selections.apply_iso_muons(
            df, args.muonIsolation[0], args.muonIsolation[1], isoBranch, isoThreshold
        )

    df = df.Define("trigMuons_passIso0", f"{isoBranch}[trigMuons][0] < {isoThreshold}")
    df = df.Define(
        "nonTrigMuons_passIso0", f"{isoBranch}[nonTrigMuons][0] < {isoThreshold}"
    )
    
    df = muon_selections.select_z_candidate(df, mass_min, mass_max) ## this is selecting the z to two muon candidates

    #### MUONS HAVE ALL BEEN SELECTED AT THIS POINT ###


    ### SHOULD SEE IF I CAN CHANGE THIS INTO A GLOBAL VARIABLE
    axis_pt = hist.axis.Regular(int(args.pt[0]), args.pt[1], args.pt[2], name="pt")
    axis_total_pt = hist.axis.Regular(int(args.pt[0]), 50, 118, name="total_pt")
    
    axis_mll = hist.axis.Variable([60,70,75,78,80,82,85,86,87,88,89,90,91,92,93,94,95,96,97,98,100,102,105,110,120], name = 'mll')
    

    hist_total_pt = df.HistoBoost("total_pt", [axis_total_pt], ['muon_total_pt'])
    hist_leading_pt = df.HistoBoost("leading_pt", [axis_pt], ['muon_leading_pt'])
    hist_subleading_pt = df.HistoBoost("subleading_pt", [axis_pt], ['muon_subleading_pt'])
    hist_mll = df.HistoBoost("mll", [axis_mll], ['mll'])
    
    if dataset.is_data:
        df = df.Define("time", brilcalc_helper, ["run", "luminosityBlock"])

        hist_time = df.HistoBoost("time", [axis_date], ['time'])
        hist_time_mll = df.HistoBoost("time_mll", [axis_date, axis_mll], ['time', "mll"])
        results.append(hist_time)
        results.append(hist_time_mll)
        print("went through loop")

    ### want to make the dimuon object and we need to do that by selecting the muons that also satisfy the trigger criterion. so 

    results.append(hist_total_pt)
    results.append(hist_leading_pt)
    results.append(hist_subleading_pt)
    results.append(hist_mll)

    return results, weightsum


logger.debug(f"Datasets are {[d.name for d in datasets]}")
resultdict = narf.build_and_run(datasets[::-1], build_graph, build_graph_lumi)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)
