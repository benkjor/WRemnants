import argparse
import hist
import numpy as np
from utilities.io_tools import input_tools
import h5py
from wums.boostHistHelpers import broadcastSystHist, multiplyHists, divideHists

from rabbit import tensorwriter

parser = argparse.ArgumentParser()
### if i need additonal arguments add them here

args = parser.parse_args()


file_in = '/work/submit/jbenke/WRemnants/scripts/histmakers/'
file_in_name = file_in + 'mz_dilepton_liv_scetlib_dyturboCorr.hdf5'
h5file = h5py.File(file_in_name, 'r')
results = input_tools.load_results_h5py(h5file)

reco_data = results['dataPostVFP']['output']['time_veto'].get()
reco_mc = results['ZmumuPostVFP']['output']['veto_muons'].get()
gen_mc = results['ZmumuPostVFP']['output']['gen_muons'].get()
pass_reco_pass_gen_veto = results['ZmumuPostVFP']['output']['veto_muons_prpg'].get()
pass_reco_pass_gen_gen = results['ZmumuPostVFP']['output']['gen_muons_prpg'].get()
pass_reco_fail_gen_veto = results['ZmumuPostVFP']['output']['veto_muons_prfg'].get()
pass_reco_fail_gen_gen = results['ZmumuPostVFP']['output']['gen_muons_prfg'].get()

# background_processes = ### NOT SURE WHAT GOES HERE YET

lumi_scaling = results['dataPostVFP']['lumi_outout']['time'].get()

weightsum = results['ZmumuPostVFP']['weight_sum']
cross_sec = results["ZmumuPostVFP"]["dataset"]["xsec"]

def mc_corrections(mc_results):
    mc_results /= weightsum
    mc_results *= cross_sec
    mc_results *= 1000
    return mc_results

reco_mc = mc_corrections(reco_mc)
gen_mc = mc_corrections(gen_mc)
pass_reco_fail_gen_veto = mc_corrections(pass_reco_fail_gen_veto)
pass_reco_fail_gen_gen = mc_corrections(pass_reco_fail_gen_gen)
pass_reco_pass_gen_veto = mc_corrections(pass_reco_pass_gen_veto)
pass_reco_pass_gen_gen = mc_corrections(pass_reco_pass_gen_gen)


reco_mc_2d = broadcastSystHist(reco_mc, reco_data)
gen_mc_2d = broadcastSystHist(gen_mc, reco_data)
pass_reco_fail_gen_veto_2d = broadcastSystHist(pass_reco_fail_gen_veto, reco_data)
pass_reco_fail_gen_gen_2d = broadcastSystHist(pass_reco_fail_gen_gen, reco_data)
pass_reco_pass_gen_veto_2d = broadcastSystHist(pass_reco_pass_gen_veto, reco_data)
pass_reco_pass_gen_gen_2d = broadcastSystHist(pass_reco_pass_gen_gen, reco_data)

reco_mc_2d = multiplyHists(reco_mc_2d, lumi_scaling)
gen_mc_2d = multiplyHists(gen_mc_2d, lumi_scaling)
pass_reco_fail_gen_veto_2d = multiplyHists(pass_reco_fail_gen_veto_2d, lumi_scaling)
pass_reco_fail_gen_gen_2d = multiplyHists(pass_reco_fail_gen_gen_2d, lumi_scaling)
pass_reco_pass_gen_veto_2d = multiplyHists(pass_reco_pass_gen_veto_2d, lumi_scaling)
pass_reco_pass_gen_gen_2d = multiplyHists(pass_reco_pass_gen_gen_2d, lumi_scaling)

writer = tensorwriter.TensorWriter(
    sparse=args.sparse,
    systematic_type=args.systematicType,
)

writer.add_channel(reco_data.axes, "time")
writer.add_channel(reco_data.axes, "num_muons")

writer.add_data(reco_data, "time")
writer.add_data(reco_data, "num_muons")
writer.add_data(reco_mc_2d, "time")
writer.add_data(reco_mc_2d, "num_muons")
writer.add_data(gen_mc_2d, "time")
writer.add_data(gen_mc_2d, "num_muons")


writer.add_process(pass_reco_pass_gen_2d, "bkg", "time")
writer.add_process(pass_reco_fail_gen_2d, "bkg", "time")


