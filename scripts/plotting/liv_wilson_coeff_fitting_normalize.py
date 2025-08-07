import argparse

import h5py
import hist
import numpy as np

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    scaleHist,
)

variation_list = [1, -1, 1.22, 1.26]
# variation_list = [0.0002, -0.0002, 0.0002, 0.0002]

coeff_num = 2
channel_name = f"1d_coeff_{coeff_num}_norm"
var_scale_factor = variation_list[coeff_num]

parser = argparse.ArgumentParser()
args = parser.parse_args()
quark_amp_scaling = np.load(
    "/work/submit/jbenke/WRemnants/scripts/corrections/quark_liv_scalings.npy"
)

indir_data = "/work/submit/jbenke/WRemnants/scripts/plotting/"
infile_data = indir_data + "fitresults.hdf5"
h5file = h5py.File(infile_data, "r")
results_data = input_tools.load_results_h5py(h5file)

data = results_data["results_asimov"]["physics_models"]["Select"]["channels"][
    "ch_masked"
]["hist_postfit_inclusive"].get()
data_cov = results_data["results_asimov"]["physics_models"]["Select"][
    "hist_postfit_inclusive_cov"
].get()

# pdb.set_trace()
indir_liv_model = "/home/submit/jbenke/LIV/coupling_models/"
mass_dependence_loc = indir_liv_model + "mass_dependence_down.npy"
mass_dependence = np.load(mass_dependence_loc)
infile_liv_model = indir_liv_model + f"coupling_before_{coeff_num+1}.npy"
print(infile_liv_model)
vals = np.load(infile_liv_model)  # (SM+LV)/SM = 1 + LV/SM
# var = hist.Hist(
#     hist.axis.Regular(24, 0, 24, metadata="time", underflow=False, overflow=False),
#     data=vals[:, 1] - 1,
# )

##g# enerator channel
writer = tensorwriter.TensorWriter()


num_mass_bins = len(data[0, :].values())
#### ONLY USED THIS ONCE
## structure is mass bin, coeff #, u/d, value
amplitudes_in = np.load(
    "/work/submit/jbenke/WRemnants/scripts/corrections/liv_fit_final_amplitudes.npy"
)
# amplitudes_in = np.zeros([num_mass_bins, 4, 2])

for j in range(num_mass_bins):
    channel = f"ch{channel_name}_mass_{j}"
    process = f"liv_fit_mass_{j}"
    this_data = data[:, j]

    data_int = np.sum(this_data.values())
    flat_line = hist.Hist(
        hist.axis.Regular(24, 0, 24, metadata="time", overflow=False, underflow=False),
        data=np.ones(24) * data_int / 24,
    )

    up_quark_var = hist.Hist(
        hist.axis.Regular(24, 0, 24, metadata="time", underflow=False, overflow=False),
        data=quark_amp_scaling[j, coeff_num, 0] - 1,
    )

    down_quark_var = hist.Hist(
        hist.axis.Regular(24, 0, 24, metadata="time", underflow=False, overflow=False),
        data=quark_amp_scaling[j, coeff_num, 1] - 1,
    )
    up_quark_var_scaled = scaleHist(up_quark_var, data_int / 24)  # LV
    down_quark_var_scaled = scaleHist(down_quark_var, data_int / 24)  # LV

    # var_scaled = scaleHist(var, data_int / 24)  # LV

    writer.add_channel(this_data.axes, channel)
    writer.add_data(this_data, channel)
    writer.add_process(flat_line, process, channel, signal=True)
    up_variation = up_quark_var_scaled * (mass_dependence[j + 1] + 2001.9) / 2001.9
    writer.add_systematic(
        addHists(flat_line, up_variation),
        f"coeff_{coeff_num+1}_u",
        process,
        channel,
        constrained=False,
        noi=True,
    )
    # pdb.set_trace()

    down_variation = down_quark_var_scaled * (mass_dependence[j + 1] + 2001.9) / 2001.9
    writer.add_systematic(
        addHists(flat_line, down_variation),
        f"coeff_{coeff_num+1}_d",
        process,
        channel,
        constrained=False,
        noi=True,
    )
    amplitudes_in[j, coeff_num, 0] = (
        np.max(up_variation.values()) / flat_line.values()[0]
    )
    amplitudes_in[j, coeff_num, 1] = (
        np.max(down_variation.values()) / flat_line.values()[0]
    )

    # writer.add_systematic(
    #     addHists(flat_line, var_scaled * (mass_dependence[j + 1] + 2001.9) / 2001.9),
    #     f"coeff_{coeff_num+1}",
    #     process,
    #     channel,
    #     constrained=False,
    #     noi=True,
    # )


#### ONLY USED THIS ONCE, DON'T WANT TO OVERWRITE EVERY TIME SO
np.save(
    "/work/submit/jbenke/WRemnants/scripts/corrections/liv_fit_final_amplitudes.npy",
    amplitudes_in,
)
writer.add_data_covariance(data_cov)
writer.write(outfolder="./", outfilename="wilson_coeff_norm")
