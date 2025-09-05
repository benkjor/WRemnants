import pdb

import h5py
import hist
import numpy as np

from utilities.io_tools import input_tools


def get_cdf(hist_in):
    arr = hist_in.copy().to_numpy()
    data_arr = arr[0]
    mass_data = np.sum(data_arr, axis=0)
    cdf_arr = np.cumsum(mass_data)
    cdf_arr /= cdf_arr[-1]
    return cdf_arr, data_arr.shape


indir_data = "/work/submit/jbenke/WRemnants/scripts/histmakers/"
infile_data = indir_data + "mz_dilepton_liv_scetlib_dyturboCorr.hdf5"

h5file = h5py.File(infile_data, "r")
results = input_tools.load_results_h5py(h5file)
hist_in = results["ZmumuPostVFP"]["output"]["fine_bin_axis_gen"].get()


def get_cdf(hist_in):
    arr = hist_in.copy().to_numpy()
    data_arr = arr[0]
    cdf_arr = np.cumsum(data_arr)
    cdf_arr /= cdf_arr[-1]

    return cdf_arr, data_arr.shape


def make_quantiles(hist_in, n_quantiles, axis_name):
    cdf_output, hist_shape = get_cdf(hist_in)

    if ((hist_shape[0]) % n_quantiles) != 0:
        print(
            "wrong number of quantiles. choose something that factors into %s"
            % hist_shape[0]
        )
    else:
        cdf_vals_in = np.linspace(0, 1, n_quantiles + 1)
        x_vals = hist_in.axes[0].edges[1:]
        new_edges = np.interp(cdf_vals_in, cdf_output, x_vals)
        pdb.set_trace()
        new_edges = [*new_edges]
        new_axis = hist.axis.Variable(new_edges, name=axis_name)
        return new_axis


k = make_quantiles(hist_in, 10, "eta_sublead")
print(k)
