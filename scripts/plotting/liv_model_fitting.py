import argparse

import boost_histogram as bh
import h5py
import numpy as np

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    divideHists,
    scaleHist,
)

parser = argparse.ArgumentParser()

args = parser.parse_args()

i = 0


indir_liv_model = "/home/submit/jbenke/LIV/"
infile_liv_model = indir_liv_model + f"coupling_{i}.npy"

vals = np.load(infile_liv_model)
liv_model = bh.Histogram(
    bh.axis.Regular(24, 0, 24, metadata="time"),
)
liv_model.fill(vals[:, 0], weight=vals[:, 1])

### i want to normalize this as well i think

liv_model = scaleHist(liv_model, 1 / np.sum(vals[:, 1]))

flat_line = divideHists(liv_model, liv_model)
flat_line = scaleHist(flat_line, 1 / 24)

liv_model = addHists(liv_model, scaleHist(flat_line.copy(), -1))


indir_data = "/work/submit/jbenke/WRemnants/scripts/plotting/"
infile_data = indir_data + "fitresults.hdf5"
h5file = h5py.File(infile_data, "r")
results_data = input_tools.load_results_h5py(h5file)

data = results_data["results_asimov"]["physics_models"]["Project ch_masked time"][
    "channels"
]["ch_masked"]["hist_postfit_inclusive"].get()
data_cov = results_data["results_asimov"]["physics_models"]["Project ch_masked time"][
    "hist_postfit_inclusive_cov"
].get()

writer = tensorwriter.TensorWriter()
data_int = np.sum(data.values())
flat_line = scaleHist(flat_line, data_int)
liv_model = scaleHist(liv_model, data_int)

##g# enerator channel
writer.add_channel(data.axes, f"ch{i}")
writer.add_data(data, f"ch{i}")
writer.add_data_covariance(data_cov)
writer.add_process(flat_line, "liv_fit", f"ch{i}", signal=True)


# pdb.set_trace()
#
writer.add_systematic(
    addHists(flat_line, liv_model * 0.1),
    f"coeff_{i}",
    "liv_fit",
    f"ch{i}",
    groups=["test_fit"],
    constrained=False,
    noi=True,
)

writer.write(outfolder="./", outfilename="liv_model_fit")

### constraints:
# coeff_0:  -0.69077 +/-    0.66875
# coeff_1: 0.85109 +/-    1.34131
# coeff_2:  -2.7774 +/-    0.64059
# coeff_3: 1.22031 +/-    0.63882


# liv_fit: 1.30391 +/-    0.00112
