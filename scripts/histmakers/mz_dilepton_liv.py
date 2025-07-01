import os
import hist
import numpy as np
import ROOT
from datetime import datetime
import narf

from utilities import common, parsing
from wremnants import (muon_calibration, muon_prefiring, muon_selections, pileup, theory_tools, vertex)
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
        jd = 367*timestamp.year - np.floor(7*(timestamp.year + np.floor((timestamp.month + 9)/12))/4) + np.floor(275*timestamp.month/9) + timestamp.day- 730531.5 + (timestamp.hour + timestamp.minute/60 + timestamp.second/3600)/24
        #calculate greenwich mst
        gmst = (67310.54841 + (876600 * 3600 + 8640184.812866) * jd  + 0.093104 * jd**2 - 6.2e-6 * jd**3) % 86400
        gmst /= 3600
        
        lst = (gmst + 46.309879/15) %24 ## include longitudinal correction, based on pt 5 at cern, could try to get a more accurate (and precise) number
        return lst
    
    return make_brilcalc_helper(filename, idx=2, action=to_time)

def mass_extraction(dataframe, name, root_dataype, filter_name):
    # if len(name) != 0:
    #     new_df = name
    
    new_df = dataframe.Define(f"{name}muon_leading_pt", f"{root_dataype}_pt[{filter_name}][0]")    
    new_df = new_df.Define(f"{name}muon_leading_eta", f"{root_dataype}_eta[{filter_name}][0]") 
    new_df = new_df.Define(f"{name}muon_leading_phi", f"{root_dataype}_phi[{filter_name}][0]") 

    new_df = new_df.Define(f"{name}muon_subleading_pt", f"{root_dataype}_pt[{filter_name}][1]") 
    new_df = new_df.Define(f"{name}muon_subleading_eta", f"{root_dataype}_eta[{filter_name}][1]") 
    new_df = new_df.Define(f"{name}muon_subleading_phi", f"{root_dataype}_phi[{filter_name}][1]")

    
    new_df = new_df.Define(
    f"{name}mu_mom4",
    f"ROOT::Math::PtEtaPhiMVector({name}muon_leading_pt, {name}muon_leading_eta, {name}muon_leading_phi, wrem::muon_mass)")
    new_df = new_df.Define(
        f"{name}smu_mom4",
        f"ROOT::Math::PtEtaPhiMVector({name}muon_subleading_pt, {name}muon_subleading_eta, {name}muon_subleading_phi, wrem::muon_mass)")
    new_df = new_df.Define(
        f"{name}ll_mom4",
        f"ROOT::Math::PxPyPzEVector({name}mu_mom4)+ROOT::Math::PxPyPzEVector({name}smu_mom4)")
    new_df = new_df.Define(f"{name}mll", f"{name}ll_mom4.mass()")
    new_df = new_df.Filter(f"{name}mll >= 60 && {name}mll < 120")
       
    return new_df
    
args = parser.parse_args()
logger = logging.setup_logger(__file__, args.verbose, args.noColorLogger)
era = args.era
calib_filepaths = common.calib_filepaths

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

########################################################
def build_graph_lumi(df, dataset):
    df = df.Define("time", brilcalc_helper, ["run", "luminosityBlock"])
    hist_time = df.HistoBoost("time", [axis_date], ['time', 'lumival'])
    results = [hist_time]
    return results

def build_graph(df, dataset):
    
    axis_pt = hist.axis.Regular(int(args.pt[0]), args.pt[1], args.pt[2], name="pt")
    
    axis_mll = hist.axis.Variable([60,70,75,78,80,82,85,86,87,88,89,90,91,92,93,94,95,96,97,98,100,102,105,110,120], name = 'mll')
    
    
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
    
    
    
###### NEED TO GENERATE THE RIGHT VARIABLES ####
    ## need the original pt and eta and whatnot, this 

    if not dataset.is_data:
    # ### lines highlighted by david
        df_gen = theory_tools.define_postfsr_vars(df)

        df_gen = df_gen.Define(
            "postfsrMuons_inAcc",
            f"postfsrMuons && abs(GenPart_eta) < 2.4 && GenPart_pt > 25")# {args.vetoGenPartPt}",
               
        df_gen = df_gen.Filter("Sum(postfsrMuons_inAcc) == 2")

        ### might need to use the corrected goodMuon versions, not sure
        df_gen = mass_extraction(df_gen, 'gen_', 'GenPart', 'postfsrMuons_inAcc')
       
        gen_counts = df_gen.HistoBoost("gen_mll", [axis_mll], ["gen_mll", 'weight'])
        results.append(gen_counts)
   
    ###################################
    # real muons
    df = df.Define(
        "vetoMuonsPre",
        "Muon_looseId && abs(Muon_dxybs) < 0.05 && Muon_charge != -99",
    )
    df = df.Define(
        "Muon_isGoodGlobal",
        "Muon_isGlobal && Muon_highPurity",
    )
    df = df.Define("veto_muon", "vetoMuonsPre && Muon_isGoodGlobal && Muon_pt>=25 && abs(Muon_eta) < 2.4") # [1,0,1,0 ...]

    df = df.Filter("Sum(veto_muon) == 2") 
    df = df.Define(
        "goodTrigObjs",
        f"wrem::goodMuonTriggerCandidate<wrem::Era::Era_2016PostVFP>(TrigObj_id,TrigObj_filterBits)",
    )
    df = mass_extraction(df, "", "Muon", "veto_muon")

    df = df.Define("leading_muon_passTrigger", "wrem::hasTriggerMatch(muon_leading_eta,muon_leading_phi,TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])")
    df = df.Define("subleading_muon_passTrigger", "wrem::hasTriggerMatch(muon_subleading_eta,muon_subleading_phi,TrigObj_eta[goodTrigObjs],TrigObj_phi[goodTrigObjs])")
    
   ### detects two muons but only one has the right momentum
    dtight = df.Filter("Sum(Muon_tightId) == 2")

    dtight_dtrig = dtight.Filter("subleading_muon_passTrigger && leading_muon_passTrigger" )  
    dtight_strig = dtight.Filter("subleading_muon_passTrigger != leading_muon_passTrigger" ) 
    
    stight_strig = df.Filter("(subleading_muon_passTrigger && Muon_tightId[1]) != (leading_muon_passTrigger && Muon_tightId[0])" ) 

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

    return results, weightsum


logger.debug(f"Datasets are {[d.name for d in datasets]}")
resultdict = narf.build_and_run(datasets[::-1], build_graph, build_graph_lumi)

if not args.noScaleToData:
    scale_to_data(resultdict)
    aggregate_groups(datasets, resultdict, args.aggregateGroups)

write_analysis_output(
    resultdict, f"{os.path.basename(__file__).replace('py', 'hdf5')}", args
)
