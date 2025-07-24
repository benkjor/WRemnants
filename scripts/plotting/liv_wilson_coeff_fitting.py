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

fit_type = 4  ### number of dimensions

if fit_type == 4:
    start_range = 0
    end_range = 4
    channel_name = "4d"

else:
    start_range = 3
    end_range = start_range + 1
    channel_name = f"1d_coeff_{start_range}"


parser = argparse.ArgumentParser()
args = parser.parse_args()

indir_data = "/work/submit/jbenke/WRemnants/scripts/plotting/"
infile_data = indir_data + "fitresults.hdf5"
h5file = h5py.File(infile_data, "r")
results_data = input_tools.load_results_h5py(h5file)

# pdb.set_trace()
data = results_data["results_asimov"]["physics_models"]["Project ch_masked time"][
    "channels"
]["ch_masked"]["hist_postfit_inclusive"].get()
data_cov = results_data["results_asimov"]["physics_models"]["Project ch_masked time"][
    "hist_postfit_inclusive_cov"
].get()

indir_liv_model = "/home/submit/jbenke/LIV/coupling_models/"

data_int = np.sum(data.values())
flat_line = hist.Hist(
    hist.axis.Regular(24, 0, 24, metadata="time", overflow=False, underflow=False),
    data=np.ones(24) * data_int / 24,
)

##g# enerator channel
writer = tensorwriter.TensorWriter()
writer.add_channel(data.axes, f"ch{channel_name}")
writer.add_data(data, f"ch{channel_name}")
writer.add_data_covariance(data_cov)
writer.add_process(flat_line, "liv_fit", f"ch{channel_name}", signal=True)


for i in range(start_range, end_range):
    infile_liv_model = indir_liv_model + f"coupling_before_{i+1}.npy"
    print(infile_liv_model)
    vals = np.load(infile_liv_model)  # (SM+LV)/SM = 1 + LV/SM
    var = hist.Hist(
        hist.axis.Regular(24, 0, 24, metadata="time", underflow=False, overflow=False),
        data=vals[:, 1] - 1,
    )
    var = scaleHist(var, data_int / 24)  # LV

    # pdb.set_trace()

    writer.add_systematic(
        addHists(flat_line, var * 0.1),
        f"coeff_{i+1}",
        "liv_fit",
        f"ch{channel_name}",
        constrained=False,
        noi=True,
    )


writer.write(outfolder="./", outfilename="wilson_coeff")
