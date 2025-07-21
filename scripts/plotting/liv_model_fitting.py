import argparse

import boost_histogram as bh
import h5py
import numpy as np

from rabbit import tensorwriter
from utilities.io_tools import input_tools
from wums.boostHistHelpers import (
    addHists,
    scaleHist,
)

parser = argparse.ArgumentParser()
args = parser.parse_args()

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

indir_liv_model = "/home/submit/jbenke/LIV/coupling_models/"

data_int = np.sum(data.values())
flat_line = bh.Histogram(
    bh.axis.Regular(24, 0, 24, metadata="time"),
)
flat_line.fill(np.linspace(0, 24, 25), weight=np.array(25 * [data_int]))
flat_line[bh.underflow] = data_int  # [-inf, 0)
flat_line[bh.overflow] = data_int  # [24, inf)
flat_line = scaleHist(flat_line, 1 / 24)

##g# enerator channel
writer = tensorwriter.TensorWriter()
writer.add_channel(data.axes, "ch0")
writer.add_data(data, "ch0")
writer.add_data_covariance(data_cov)
writer.add_process(flat_line, "liv_fit", "ch0", signal=True)


for i in range(0, 4):
    infile_liv_model = indir_liv_model + f"coupling_before_{i+1}.npy"
    print(infile_liv_model)
    vals = np.load(infile_liv_model)
    var = bh.Histogram(
        bh.axis.Regular(24, 0, 24, metadata="time"),
    )
    var.fill(vals[:, 0], weight=vals[:, 1])

    var = scaleHist(var, 1 / np.sum(vals[:, 1]))
    var = addHists(var, scaleHist(flat_line.copy(), -1 / data_int))
    var = scaleHist(var, data_int)

    writer.add_systematic(
        addHists(flat_line, var * 0.1),
        f"coeff_{i+1}",
        "liv_fit",
        "ch0",
        constrained=False,
        noi=True,
    )

writer.write(outfolder="./", outfilename="liv_model_fit")

### constraints:
# coeff_1:  -0.69077 +/-    0.66875
# coeff_2: 0.85109 +/-    1.34131
# coeff_3:  -2.7774 +/-    0.64059
# coeff_4: 1.22031 +/-    0.63882


# liv_fit: 1.30391 +/-    0.00112


#### some other ideas -- not in use


# indir_data = "/work/submit/jbenke/WRemnants/scripts/plotting/"
# infile_data = indir_data + "fitresults.hdf5"
# h5file = h5py.File(infile_data, "r")
# results_data = input_tools.load_results_h5py(h5file)

# data = results_data["results_asimov"]["physics_models"]["Project ch_masked time"]["channels"]["ch_masked"]["hist_postfit_inclusive"].get()
# data_cov = results_data["results_asimov"]["physics_models"]["Project ch_masked time"]["hist_postfit_inclusive_cov"].get()

# liv_models = []
# indir_liv_models = "/home/submit/jbenke/LIV/coupling_models/"
# for path in os.listdir(indir_liv_models):
#     if os.path.isfile(os.path.join(indir_liv_models, path)):
#        liv_models.append(os.path.join(indir_liv_models, path))
# # pdb.set_trace()
# liv_models = [i for i in liv_models if "before" in i]
# # liv_models = [liv_models[0], liv_models[4], liv_models[7], liv_models[9]]

# ## making the basline model
# data_int = np.sum(data.values())
# flat_line = bh.Histogram(
#     bh.axis.Regular(24, 0, 24, metadata="time", underflow = False, overflow = False),
# )
# flat_line.fill(np.linspace(0,24, 24), weight=np.array(24*[data_int]))
# flat_line = scaleHist(flat_line, 1/24)

# ones = divideHists(flat_line, flat_line)

# writer = tensorwriter.TensorWriter()
# writer.add_channel(data.axes, "ch0")
# writer.add_data(data, "ch0")
# writer.add_data_covariance(data_cov)

# writer.add_process(flat_line, "liv_fit", "ch0", signal=True)

# ### iterating through the various models
# idx = 0

# for i in range(0, 4):
#     # for j in range(i, 4):
#     if 1 == 1:
#         mod = liv_models[idx]
#         print(mod)
#         idx += 1
#         vals = np.load(mod)

#         var = bh.Histogram(
#             bh.axis.Regular(24, 0, 24, metadata="time", underflow = False, overflow = False),
#         )
#         var.fill(vals[:, 0], weight=vals[:, 1])
#         # pdb.set_trace()
#         # var = addHists(var, ones)
#         print(var)

#         var = scaleHist(var, flat_line)
#         var = addHists(var, scaleHist(flat_line, -1))
#         # var = addHists(var, scaleHist(flat_line, -1))


#         writer.add_systematic(
#             addHists(flat_line, var * 0.1),
#             f"coeff_{i+1}",
#             "liv_fit",
#             "ch0",
#             groups=["test_fit"],
#             constrained=False,
#             noi=True,
#         )
#         # if j != i:
#         #     writer.add_systematic(
#         #         addHists(flat_line, var * 0.1),
#         #         f"coeff_{j+1}",
#         #         "liv_fit",
#         #         "ch0",
#         #         groups=["test_fit"],
#         #         constrained=False,
#         #         noi=True,
#         #     )
#         # pdb.set_trace()

# writer.write(outfolder="./", outfilename="liv_model_fit")
