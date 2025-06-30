import os
import hist
import numpy as np
import ROOT
from datetime import datetime
import narf

from utilities import common, parsing
from wremnants import (muon_selections, muon_calibration, pileup, vertex)
from wremnants.datasets.datagroups import Datagroups
from wremnants.datasets.dataset_tools import getDatasets
from wremnants.histmaker_tools import (aggregate_groups, scale_to_data,  write_analysis_output)
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
calib_filepaths = common.calib_filepaths

#hoping this can go up top
lumicsv = f"{common.data_dir}/bylsoutput.csv" 
brilcalc_helper = make_timehelper(lumicsv)
mc_calibration_helper, data_calibration_helper, calibration_uncertainty_helper = ( muon_calibration.make_muon_calibration_helpers(args, era=era))

(
    mc_jpsi_crctn_helper,
    data_jpsi_crctn_helper,
    mc_jpsi_crctn_unc_helper,
    data_jpsi_crctn_unc_helper,
) = muon_calibration.make_jpsi_crctn_helpers(
    args, calib_filepaths, make_uncertainty_helper=True)

smearing_helper, smearing_uncertainty_helper = (
    (None, None) if args.noSmearing else muon_calibration.make_muon_smearing_helpers()
)
pileup_helper = pileup.make_pileup_helper(era=era)
vertex_helper = vertex.make_vertex_helper(era=era)
# muon_prefiring_helper, muon_prefiring_helper_stat, muon_prefiring_helper_syst = (   muon_prefiring.make_muon_prefiring_helpers(era=era))
    
    
# if args.binnedScaleFactors:

#     muon_efficiency_helper, muon_efficiency_helper_syst, muon_efficiency_helper_stat = (
#         muon_efficiencies_binned.make_muon_efficiency_helpers_binned(
#             filename=args.sfFile, era=era, max_pt=args.pt[2], is_w_like=True
        # )
    # )
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




########################################################
def build_graph_lumi(df, dataset):
    df = df.Define("time", brilcalc_helper, ["run", "luminosityBlock"])
    hist_time = df.HistoBoost("time", [axis_date], ['time', 'lumival'])
    results = [hist_time]
    return results

def build_graph(df, dataset):
    logger.info(f"build graph for dataset: {dataset.name}")
    era = args.era
    results = []

    if dataset.is_data:
        df = df.DefinePerSample("weight", "1.0")
        df = df.Define("time", brilcalc_helper, ["run", "luminosityBlock"])
        hist_time = df.HistoBoost("time", [axis_date], ['time'])
    else:
        df = df.Define("weight", "std::copysign(1.0, genWeight)")
    weightsum = df.SumAndCount("weight")
    
    
    ### basically do the exact same thing as below but using the GenPart stettings(for generator particle)
    
    
    
    df = df.Define( "isEvenEvent", f"event % 2 {'!=' if args.flipEventNumberSplitting else '=='} 0" ) ## not sure why i have this
    df = df.Define("muon_pass_pt", "Muon_pt>=25") # [1,0,1,0 ...]
    df = df.Filter("std::accumulate(muon_pass_pt.begin(), muon_pass_pt.end(), 0) == 2") 

    df = df.Define(
        "goodTrigObjs",
        f"wrem::goodMuonTriggerCandidate<wrem::Era::Era_2016PostVFP>(TrigObj_id,TrigObj_filterBits)",
    )
    
    df = df.Define("muon_leading_pt", "Muon_pt[muon_pass_pt][0]") 
    df = df.Define("muon_leading_eta", "Muon_eta[muon_pass_pt][0]") 
    df = df.Define("muon_leading_phi", "Muon_phi[muon_pass_pt][0]") 

    df = df.Define("muon_subleading_pt", "Muon_pt[muon_pass_pt][1]") 
    df = df.Define("muon_subleading_eta", "Muon_eta[muon_pass_pt][1]") 
    df = df.Define("muon_subleading_phi", "Muon_phi[muon_pass_pt][1]")
    
    df = df.Define("leading_muon_passTrigger", "wrem::hasTriggerMatch(muon_leading_eta,muon_leading_eta,TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])")
    df = df.Define("subleading_muon_passTrigger", "wrem::hasTriggerMatch(muon_subleading_eta,muon_subleading_eta,TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])")
    
    df = df.Define(
        "mu_mom4",
        "ROOT::Math::PtEtaPhiMVector(muon_leading_pt, muon_leading_eta, muon_leading_phi, wrem::muon_mass)")
    df = df.Define(
        "smu_mom4",
        "ROOT::Math::PtEtaPhiMVector(muon_subleading_pt, muon_subleading_eta, muon_subleading_phi, wrem::muon_mass)")
    df = df.Define(
        "ll_mom4",
        f"ROOT::Math::PxPyPzEVector(mu_mom4)+ROOT::Math::PxPyPzEVector(smu_mom4)")
  
    df = df.Define("mll", "ll_mom4.mass()")
    df = df.Filter("mll >= 60 && mll < 120")
    
    
   ### detects two muons but only one has the right momentum
    dtight = df.Filter("std::accumulate(Muon_tightId.begin(), Muon_tightId.end(), 0) == 2")

    dtight_dtrig = dtight.Filter("subleading_muon_passTrigger && leading_muon_passTrigger" )  
    dtight_strig = dtight.Filter("subleading_muon_passTrigger != leading_muon_passTrigger" ) 
    
    stight_strig = df.Filter("(subleading_muon_passTrigger && Muon_tightId[1]) != (leading_muon_passTrigger && Muon_tightId[0])" ) 
    

    ### SHOULD SEE IF I CAN CHANGE THIS INTO A GLOBAL VARIABLE
    axis_pt = hist.axis.Regular(int(args.pt[0]), args.pt[1], args.pt[2], name="pt")
    
    axis_mll = hist.axis.Variable([60,70,75,78,80,82,85,86,87,88,89,90,91,92,93,94,95,96,97,98,100,102,105,110,120], name = 'mll')
    
    hist_leading_pt = df.HistoBoost("leading_pt", [axis_pt], ['muon_leading_pt'])
    hist_subleading_pt = df.HistoBoost("subleading_pt", [axis_pt], ['muon_subleading_pt'])
    
    if dataset.is_data:
        hist_time_mll = dtight_dtrig.HistoBoost("time_mll", [axis_date, axis_mll], ['time', "mll"])
        hist_time_mll_dtight_strig = dtight_strig.HistoBoost("time_mll_dtight_strig", [axis_date, axis_mll], ['time', "mll"])
        hist_time_mll_stight_strig = stight_strig.HistoBoost("time_mll_stight_strig", [axis_date, axis_mll], ['time', "mll"])
        
        results.append(hist_time)
        results.append(hist_time_mll)
        results.append(hist_time_mll_dtight_strig)
        results.append(hist_time_mll_stight_strig)
    else:
        hist_mll = dtight_dtrig.HistoBoost("mll", [axis_mll], ['mll', 'weight'])
        hist_mll_dtight_strig = dtight_strig.HistoBoost("mll_dtight_strig", [axis_mll], ['mll', 'weight'])
        hist_mll_stight_strig = stight_strig.HistoBoost("mll_stight_strig", [axis_mll], ['mll', 'weight'])

        results.append(hist_mll)
        results.append(hist_mll_dtight_strig)
        results.append(hist_mll_stight_strig)


    results.append(hist_leading_pt)
    results.append(hist_subleading_pt)

    return results, weightsum


logger.debug(f"Datasets are {[d.name for d in datasets]}")
resultdict = narf.build_and_run(datasets[::-1], build_graph, build_graph_lumi)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)
