"""
Copyright (c) 2021, Martin Bienert
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

__version__ = "0.10.1"

# backgroundCorrection_V0.9.5.py
# https://github.com/byTreneib/BackgroundCorrection
# 2021-10-01
# Python 3.7.1 script for background substraction of PXRD (*.xy) and Raman (*.txt or *.spc) data by Martin Bienert.
# Tested with PyCharm and Spyder from Anaconda.
# Script must be edited. "START OF EDITING"
# Background substracted data will be saved in the same folder as the unprocessed data.
# There will be no changes of the X parameters; input=output
# Output: Background corrected data
# Output file extension: dat
# One has to play around with the lambda values. (Raman 1E5; Mill experiments <1E3)
# If nothing happens: stop the script, restart PYTHON
# One can change the order of the file extension for the file selection dialogue in line 99
#
# Format of raman data: as spc-file or as one file with many columns
# RamanShift (cm-1)	00:00:27	00:01:30	00:02:00	..:..:..	00:21:00
# 3400.0000	3622.37918602	5294.52406611	3867.40889509	****.********	4243.28501402
#
# Format of PXRD data: many files with one X and one Y column separated with two spaces)
# Image file: Z:\2019_11\HDF5\2019-11-28_Example_SN\2019-11-28_Example_SN_000001_data_000001.h5
# ImageIntegration1d_chi_output; Output_Directory= Z:\2019_11\Processed\2019-11-28_Example_SN
# Profile = Radial, StartAzimuth = 0.0, EndAzimuth = 360.0, Sam_Det_distance = 228.644337, InnerRadius = 1.0, OuterRadius = 2400.0
# 2400
# 5.6440117e-02  0.0000000e+00
# .............  .............
# 1.0347264e-01  0.0000000e+00

############## BEGIN OF IMPORTS ##############

from tkinter.filedialog import askopenfilenames, askopenfilename, askdirectory
from tkinter.messagebox import askyesno
from tkinter import Tk
from os import getcwd
import os

from warnings import warn

import numpy.linalg
from pyspectra.readers.read_spc import spc
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
from scipy import sparse
import pandas as pd
import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
import time

############### END OF IMPORTS ###############


############## BEGIN OF OPTIONS ##############

# Select the index of the column a test run should be executed on. Leave empty ('' or "") if no testing is required.
# If set, the Background will only run on this column to test the fit of the parameters chosen
test_row_index = '10'

# Set to True to show every processed dataset in a plot.
# ATTENTION: When running with this option turned on, execution will be paused while the plot is being displayed
plot_data = False

dat_file_separator = '\t'  # Separator (as string), that will be used to separate the values when writing to .dat file
include_header = True  # When true, will read the headers of the input files, and extend them with the parameters
header_row_count = 4  # DEFAULT: 4

# When jar_correction is set to True the user will be asked to provide a file containing reference intensities for the
# jar in a second prompt. The program will then scale the reference intensities within the provided jar_scaling_range
# and subtract the reference intensities from the scan data.
# ATTENTION: At current state of development, when enabled, this will consume significantly more time than without it.
jar_correction = False
bkg_before_jar = True
jar_scaling_range = (1704, 1758)  # Set range as (start_value, end_value, "(-np.infty, np.infty)"), e.g. (0, 100)
jar_debug = True

# Set to true to norm the final corrected result to the area under the curve. The additional time taken is minimal.
norm_final = True

# Minimum and maximum value for the x column
wave_min = 0.51
wave_max = 6

# Select the algorithm that will do the background correction (as integer)
# 0 -> arpls, slow, I guess it is better for Raman
# 1 -> als, fast, for PXRD
do_correction = False
algorithm = 1

# Parameters for the baseline algorithm
baseline_itermax = 500  # number of iterations the algorithm will perform
baseline_lambda = 500  # the larger lambda is, the smoother the resulting background, 500 for PXRD with little PMMA jars
baseline_ratio = 0.0007  # wheighting deviations: 0 < baseline_ratio < 1, smaller values allow less negative values, 0.0007 for PXRD with little PMMA jars

# Peak ROI Selection
# WARNING: This option currently only generates expected behaviour for input files with one y column!
get_rois = True
roi_ranges = [
    (0.5, 0.9, "red"),
    (1.8, 2, ""),
]
normalize_integration = True
time_step = 1
plot_flip_y = True

# Select Unit for x-Scale, where x_scale_output_unit is the unit you want the data in, and x_scale_input_unit is the unit the input data is in
# 0 -> q_nm
# 1 -> q_A
# 2 -> 2Theta
x_scale_output_unit = 1
x_scale_input_unit = 1
lam = 0.207

# List of file types that will be available in the file input prompt
readfile_ui_file_types = [("txt raman files", "*.txt"),
                          ("spc files", "*.spc"),
                          ("xy files", "*.xy"),
                          ("dat files", "*.dat"),
                          ("raman files", "*.raman")]


############### END OF OPTIONS ###############


def timeit(function):  # decorator function to record execution time of a function
    def timed(*args, **kwargs):
        start_time = time.time()
        return_value = function(*args, **kwargs)
        end_time = time.time()

        print(f'[TIMEIT: {function.__name__}]: {round(end_time - start_time, 6)} seconds')

        return return_value

    return timed


def sanitize_test_row_index():
    global test_row_index

    if test_row_index == "":
        return

    try:
        test_row_index = int(test_row_index)
    except ValueError:
        raise ValueError("Test row index could not be converted to int. Make sure to use a valid column index.")


def range_to_io_string(start, stop):
    return f"{str(start).replace('.', '_')}_to_{str(stop).replace('.', '_')}"


def deg_to_rad(val: float) -> float:
    return (2 * np.pi / 360) * val


def rad_to_deg(val: float) -> float:
    return val / (2 * np.pi / 360)


def q_nm_to_q_A(val: float) -> float:
    return val / 10


def q_A_to_q_nm(val: float) -> float:
    return val * 10


def two_theta_to_q_A(val: float) -> float:
    val_rad = deg_to_rad(val)
    return (4 * np.pi * np.sin(val_rad / 2)) / lam


def q_A_to_two_theta(val: float) -> float:
    ret_rad = 2 * np.arcsin(lam * val / (4 * np.pi))
    return rad_to_deg(ret_rad)


def convert_x(val: float, from_unit: int, to_unit: int) -> float:
    try:
        if int(from_unit) == int(to_unit):
            return val
    except ValueError:
        raise ValueError("from_unit and to_unit must be of type Integer!")

    to_q_A = [q_nm_to_q_A, lambda x: x, two_theta_to_q_A]
    from_q_A = [q_A_to_q_nm, lambda x: x, q_A_to_two_theta]

    to_q_A_func = to_q_A[from_unit]
    from_q_A_func = from_q_A[to_unit]

    q_A_val = to_q_A_func(val)
    return from_q_A_func(q_A_val)


def unit_x_str(to_unit: int) -> str:
    unit_strs = ["q [1/nm]", "q [1/A]", "q [2θ]"]

    return unit_strs[to_unit]


############### BEGIN OF IO CLASSES ###############

class WriteDFToFile:  # Class to write pandas DataFrames into a chi/dat file
    def __init__(self, df, filename, head="", sep="  ", include_head=True):
        self.df = df.transpose()
        self.filename = filename
        self.sep = sep
        self.head = head if include_head else ''

        self.nr_rows = len(df)
        self.nr_columns = len(df.columns)

        # Thread(self.df_to_string()).start()
        self.df_to_string()

    def df_to_string(self):
        output_list = []
        for column in self.df.columns:
            output_list.append(list(self.df[column]))

        # joins all columns in rows
        output_list = map(lambda x: self.sep.join(map(lambda y: "%2.7e" % y, x)), output_list)
        body = "\n".join(output_list)

        with open(self.filename, "w+") as writefile:
            if self.head is None:
                head = ""
            else:
                head = self.head
            writefile.write(head + "\n" + body)


class ReadRamanToDF:  # Class to read Raman file into pandas DataFrames
    def __init__(self, filename, include_head=True):
        self.filename = filename
        self.include_head = include_head

        if __name__ == '__main__':  # only shows debug output when this program is directly executed
            print(f"[ReadRamanToDF] Reading {self.filename}")

        with open(self.filename) as readfile:  # reads out content of the specified file
            self.file_content = readfile.read()

        if __name__ == '__main__':  # only shows debug output when this program is directly executed
            print(f"[ReadRamanToDF] Reading {self.filename} successfully finished")

    def file_content_to_df(self):
        lines_raw = self.file_content.split("\n")  # splits up lines and removes the header
        if self.include_head:
            lines = list(map(lambda x: x.split(), lines_raw[header_row_count:-1]))  # splits up the columns of each line
        else:
            lines = list(map(lambda x: x.split(), lines_raw[:-1]))  # splits up the columns of each line

        num_columns = len(lines[0])
        try:
            column_names = ['RamanShift (cm-1)', *lines_raw[0].split()[1:]]
        except IndexError:
            column_names = ['RamanShift (cm-1)', *list('I_' + str(x) for x in range(num_columns - 1))]

        content_df = pd.DataFrame(columns=('RamanShift (cm-1)', *column_names[1:]))

        x_column = list(map(lambda x: float(x[0]), lines))
        content_df['RamanShift (cm-1)'] = x_column

        # enumerates through all columns and writes them into the content_df dataFrame
        for column_index, column_name in enumerate(column_names):
            if column_name == 'RamanShift (cm-1)':
                continue
            y_column = list(map(lambda x: float(x[column_index]), lines))

            content_df[column_name] = y_column

        return content_df, column_names  # returns the df and head (column titles in this case)


class ReadChiToDF:  # class for reading chi file into a pandas DataFrame
    def __init__(self, filename, i_column_name, include_head=True):
        self.filename = filename
        self.i_column_name = i_column_name
        self.include_head = include_head

        if __name__ == '__main__':  # only show debug output if this file is directly executed
            print(f"[ReadChiToDF] Reading {self.filename}")

        with open(filename) as file:  # open specified file and read out content
            self.file_content = file.read()

        if __name__ == '__main__':  # only show debug output if this file is directly executed
            print(f"[ReadChiToDF] Reading {self.filename} successfully finished")

    def file_content_to_df(self):
        lines_raw = self.file_content.split("\n")  # splits up lines and removes the header
        if self.include_head:
            lines = list(map(lambda x: x.split(), lines_raw[header_row_count:-1]))  # splits up the columns of each line
        else:
            lines = list(map(lambda x: x.split(), lines_raw[:-1]))  # splits up the columns of each line

        x_column = list(map(lambda x: float(x[0]), lines))  # creates list out of first column (angles)
        y_column = list(map(lambda x: float(x[1]), lines))  # creates list out of second column (intensity)

        content_df = pd.DataFrame(columns=('q', self.i_column_name))  # creates empty dataframe to store x and y column
        content_df['q'] = x_column  # writes x_column list into 'q' column
        content_df[self.i_column_name] = y_column  # writes y_column list into 'I' (or whatever was specified) column

        return content_df, lines_raw[:header_row_count]  # returns the dataframe and the first 4 lines (head)


def read_raman_to_df(filename: str, header=True):
    return ReadRamanToDF(filename, include_head=header).file_content_to_df()


def read_chi_to_df(filename: str, header=True, column_name="I"):
    return ReadChiToDF(filename, include_head=header, i_column_name=column_name).file_content_to_df()


def read_spc_to_df(filename: str):
    # This function was taken from the pyspectra library to be modified to this programs needs
    out = pd.DataFrame()

    f = spc.File(filename)  # Read file
    if f.dat_fmt.endswith('-xy'):
        for s in f.sub:
            x = s.x
            y = s.y

            out["RamanShift (cm-1)"] = x
            out[str(round(s.subtime))] = y
    else:
        for s in f.sub:
            x = f.x
            y = s.y

            out["RamanShift (cm-1)"] = x
            out[str(round(s.subtime))] = y

    return out.copy(), "\t".join(map(str, out.columns))


################ END OF IO CLASSES ################


class Algorithms:
    @staticmethod
    def arpls(y, lam=baseline_lambda, ratio=baseline_ratio, itermax=baseline_itermax) -> np.ndarray:
        r"""
        Baseline correction using asymmetrically
        reweighted penalized least squares smoothing
        Sung-June Baek, Aaron Park, Young-Jin Ahna and Jaebum Choo,
        Analyst, 2015, 140, 250 (2015)

        Abstract

        Baseline correction methods based on penalized least squares are successfully
        applied to various spectral analyses. The methods change the weights iteratively
        by estimating a baseline. If a signal is below a previously fitted baseline,
        large weight is given. On the other hand, no weight or small weight is given
        when a signal is above a fitted baseline as it could be assumed to be a part
        of the peak. As noise is distributed above the baseline as well as below the
        baseline, however, it is desirable to give the same or similar weights in
        either case. For the purpose, we propose a new weighting scheme based on the
        generalized logistic function. The proposed method estimates the noise level
        iteratively and adjusts the weights correspondingly. According to the
        experimental results with simulated spectra and measured Raman spectra, the
        proposed method outperforms the existing methods for baseline correction and
        peak height estimation.

        :param y: input data (i.e. chromatogram of spectrum)
        :param lam: parameter that can be adjusted by user. The larger lambda is,
                    the smoother the resulting background, z
        :param ratio: wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        :param itermax: number of iterations to perform
        :return: the fitted background vector

        """
        assert itermax > 0, f"itermax parameter must be greater than 0, but is {itermax}"

        N = len(y)
        D = sp.sparse.eye(N, format='csc')
        D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
        D = D[1:] - D[:-1]

        H = lam * D.T * D
        w = np.ones(N)
        for i in range(itermax):
            W = sp.sparse.diags(w, 0, shape=(N, N))
            WH = sp.sparse.csc_matrix(W + H)
            cholesky_matrix = cholesky(WH.todense())
            C = sparse.csc_matrix(cholesky_matrix)
            fsolve = sparse.linalg.spsolve(C.T, w * y.astype(np.float64))
            z = sparse.linalg.spsolve(C, fsolve)
            d = y - z
            dn = d[d < 0]
            m = np.mean(dn)
            s = np.std(dn)
            wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                break
            w = wt
        return z

    @staticmethod
    def als(y, lam=baseline_lambda, ratio=baseline_ratio, itermax=baseline_itermax) -> np.ndarray:
        r"""
        edit
        Implements an Asymmetric Least Squares Smoothing
        baseline correction algorithm (P. Eilers, H. Boelens 2005)

        Baseline Correction with Asymmetric Least Squares Smoothing
        based on https://github.com/vicngtor/BaySpecPlots

        Baseline Correction with Asymmetric Least Squares Smoothing
        Paul H. C. Eilers and Hans F.M. Boelens
        October 21, 2005

        Description from the original documentation:

        Most baseline problems in instrumental methods are characterized by a smooth
        baseline and a superimposed signal that carries the analytical information: a series
        of peaks that are either all positive or all negative. We combine a smoother
        with asymmetric weighting of deviations from the (smooth) trend get an effective
        baseline estimator. It is easy to use, fast and keeps the analytical peak signal intact.
        No prior information about peak shapes or baseline (polynomial) is needed
        by the method. The performance is illustrated by simulation and applications to
        real data.

        :param y: input data (i.e. chromatogram of spectrum)
        :param lam: parameter that can be adjusted by user. The larger lambda is,
                    the smoother the resulting background, z
        :param ratio: wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        :param itermax: number of iterations to perform
        :return: the fitted background vector

        """
        L = len(y)
        # D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        D = sparse.eye(L, format='csc')
        D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
        D = D[1:] - D[:-1]
        D = D.T
        w = np.ones(L)
        for i in range(itermax):
            W = sparse.diags(w, 0, shape=(L, L))
            Z = W + lam * D.dot(D.T)
            z = spsolve(Z, w * y.astype(np.float64))
            w = ratio * (y > z) + (1 - ratio) * (y < z)
        return z

    @staticmethod
    def algorithm(alg_index: int):
        return [Algorithms.arpls, Algorithms.als][alg_index]

    @staticmethod
    def algorithm_name(alg_index: int):
        return ["arpls", "als"][alg_index]


class BackgroundCorrection:
    readfile_options = {'filetypes': readfile_ui_file_types}

    def __init__(self):
        sanitize_test_row_index()

        data_frames, files, heads = self.read_files()
        heads = self.extend_headers(heads)

        data_per_subdir = {}
        for df, filename, head in zip(data_frames, files, heads):
            out_df, roi_areas = self.process_data(df, filename, head)

            file_dir = os.path.dirname(filename)

            if file_dir in data_per_subdir.keys():
                data_per_subdir[file_dir][0].append(os.path.basename(filename))
                data_per_subdir[file_dir][1].append(out_df)
                data_per_subdir[file_dir][2].append(roi_areas)
            else:
                data_per_subdir[file_dir] = ([os.path.basename(filename)], [out_df], [roi_areas])

        if get_rois:
            self.export_rois(data_per_subdir)

    def export_rois(self, data_per_subdir):
        root = Tk()
        show_plot = askyesno(title="Plotting",
                             message="Would you like to create a plot with corrected intensity and ROI integration values?")
        root.destroy()

        for file_dir, roi_areas in data_per_subdir.items():
            out_dir = os.path.join(file_dir, "out")

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            data = np.array(roi_areas).T

            export_data = np.array([data[:, 0], *np.array([*data[:, 2]]).T]).T
            header = ",".join(["filename",
                               *[f"roi_{i}[{str(float(start)) + ' to ' + str(float(stop))}]" for i, (start, stop, color) in
                                 enumerate(roi_ranges)]])
            np.savetxt(out_dir + f"\\rois.csv", export_data, delimiter=",", fmt="%s", header=header, comments="")

            if show_plot:
                intensity_dfs = data[:, 1]
                roi_area_values = np.array([*data[:, 2]]).astype(np.float16)

                first_df = intensity_dfs[0]

                x_scale_orig = first_df[first_df.columns[0]]
                # x_scale = np.vectorize(convert_x, excluded=["from_unit", "to_unit"])(x_scale_orig, x_scale_input_unit, x_scale_output_unit)
                x_scale = x_scale_orig

                joined_df = pd.DataFrame(x_scale)
                joined_df.reset_index(drop=True, inplace=True)

                # Join intensity dataframes
                for df in intensity_dfs:
                    df.reset_index(drop=True, inplace=True)
                    for col in np.flip(df.to_numpy()[:, 1:].T):
                        joined_df = pd.concat([joined_df, pd.DataFrame(col)], axis=1)

                y_scale = list(np.arange(0, len(joined_df.columns) - 1) * time_step)
                y_scale = np.flip(y_scale) if not plot_flip_y else y_scale

                extent = [np.min(x_scale), np.max(x_scale), np.min(y_scale), np.max(y_scale)]

                joined_df = joined_df.astype(float)
                plot_data = joined_df.to_numpy()[::-1, 1:].T
                plot_data = np.flip(plot_data, 0) if plot_flip_y else plot_data

                fig, (ax1, ax2) = plt.subplots(1, 2, sharey='row', gridspec_kw={'width_ratios': [5, 2]})
                ax1.imshow(plot_data, extent=extent, cmap="hot")
                ax1.set_xlabel(unit_x_str(x_scale_output_unit))
                ax1.set_ylabel("Time [s]")
                ax1.set_xlim(extent[0], extent[1])
                ax1.set_ylim(extent[2], extent[3])
                ax1.set_aspect("auto")

                if not plot_flip_y:
                    ax1.invert_yaxis()

                # roi_area_values_grounded = (roi_area_values - np.min(roi_area_values))
                # roi_area_values_normalized = roi_area_values_grounded / np.max(roi_area_values_grounded)

                # roi_area_values_normalized = roi_area_values / np.sum(roi_area_values)
                # roi_area_values_grounded = roi_area_values_normalized - np.min(roi_area_values_normalized)

                axis_sum = np.sum(roi_area_values, axis=1)
                axis_quot = 1 / axis_sum

                roi_scale = np.mean(axis_quot)
                print(np.mean(axis_sum), roi_scale)

                roi_area_values_scaled = roi_area_values * roi_scale

                ax2.set_xlim(np.min(roi_area_values_scaled), np.max(roi_area_values_scaled))
                # print(np.min(roi_area_values_grounded), np.max(roi_area_values_grounded))

                for roi_area, (_, _, color) in zip(roi_area_values_scaled.T, roi_ranges):
                    y_scale = np.arange(len(roi_area)) * time_step

                    if color not in [None, ""]:
                        if plot_flip_y:
                            ax2.scatter(roi_area, y_scale, s=5, c=color)
                        else:
                            ax2.scatter(roi_area[::-1], y_scale, s=5, c=color)
                    else:
                        if plot_flip_y:
                            ax2.scatter(roi_area, y_scale, s=5)
                        else:
                            ax2.scatter(roi_area[::-1], y_scale, s=5)

                ax2.tick_params(
                    axis='y',
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False
                )
                ax2.set_xlabel("Norm. Intensity")

                fig.tight_layout()
                fig.savefig(out_dir + "\\rois_plot.png")
                plt.close(fig)

    def read_files(self):
        data_frames = []
        heads = []

        main_window = Tk()  # opens tkinter window to let user select files to process
        files = askopenfilenames(**self.readfile_options, title='Select files to process')

        if not files:
            files = []
            # No files selected -> Ask for directory
            files_dir = askdirectory()
            files_subdirs = [subdir.path for subdir in os.scandir(files_dir) if
                             subdir.is_dir() and subdir.name != "out"]
            subdir_files = [[file.path for file in os.scandir(subdir) if os.path.isfile(file)] for subdir in
                            [*files_subdirs, files_dir]]

            for file_list in subdir_files:
                files.extend(file_list)

        main_window.destroy()  # close window after selection is done

        print(f"[ReadFiles] {len(files)} files selected:")
        print("\n".join(files))

        data_files = []

        for file in files:
            file_content, file_head = self.read_file(file)

            if file_content is not None and file_head is not None:
                data_frames.append(file_content)
                heads.append(file_head)
                data_files.append(file)

        return data_frames, files, heads  # returns both the list of dataFrames and the file list

    def read_file(self, file):
        file_ext = file.split('.')[-1]

        if file_ext in ['xy', 'dat']:
            file_content, file_head = read_chi_to_df(file)
        elif file_ext in ['txt', 'raman']:
            file_content, file_head = read_raman_to_df(file)
            if test_row_index == '':
                file_head = ["", "", "", dat_file_separator.join(file_head)]
            else:
                file_head = ["", "", "", dat_file_separator.join([file_head[0], file_head[test_row_index]])]
        elif file_ext == "spc":
            file_content, file_head = read_spc_to_df(file)
            file_head = ["", "", "", file_head]
        else:
            warn(f"File type {file_ext} has not yet been implemented!")
            return None, None

        return file_content, file_head

    def read_jar_reference(self):
        root = Tk()
        file = askopenfilename(**self.readfile_options, title='Select file with reference data')
        root.destroy()

        file_ext = file.split('.')[1]

        if file_ext in ['xy', 'dat']:
            file_content, _ = read_chi_to_df(file)
        elif file_ext in ['txt', 'raman']:
            file_content, _ = read_raman_to_df(file)
        elif file_ext == "spc":
            file_content, _ = read_spc_to_df(file)
        else:
            NotImplementedError(f"The used file type has not yet been implemented!")

        return file_content

    def extend_headers(self, headers):
        for index, head in enumerate(headers):
            header_extention_line2 = f", BackgroundCorrection.py (Version {__version__})"
            header_extention_line3 = f", Lambda = {baseline_lambda}, Ratio = {baseline_ratio}, " \
                                     f"Itermax = {baseline_itermax}, algorithm = {Algorithms.algorithm_name(algorithm)}"

            head[1] += header_extention_line2
            head[2] += header_extention_line3
            headers[index] = "\n".join(head)

        return headers

    def add_baseline_diff(self, df: pd.DataFrame, column_name: str, return_df: pd.DataFrame, x_column_name: str = None,
                          normalize: bool = True, skip_correction: bool = False):
        """
        Function for extending the return_df dataFrame with the processed values from waxs_df for a certain probe

        :param df: dataFrame with density data of the probes
        :param column_name: name of the probe (e.g. 'I_243')
        :param x_column_name: name of x column
        :param return_df: dataFrame that will later be written into the output file
        :return: the return_df with updated values AND baseline
        """

        # print("BASELINE SELECTION", df, type(df))

        column = df[column_name]

        # print("COLUMN", column_name, column, type(column))
        # print("NP COLUMN", column.to_numpy(na_value=0), type(column.to_numpy(na_value=0)))

        if do_correction and not skip_correction:
            baseline = Algorithms.algorithm(algorithm)(column.to_numpy(na_value=0))

            print("BASELINE", baseline, type(baseline))

            intensity_corrected = np.array(column - baseline)
        else:
            intensity_corrected = np.array(column)
            baseline = None

        if x_column_name is not None and (norm_final or get_rois) and normalize:
            # Norm intensities to area under intensity curve if requested and not done after jar correction
            intensity_corrected_area = abs(np.trapz(y=intensity_corrected, x=df[x_column_name].to_numpy()))
            intensity_corrected_normed = intensity_corrected / intensity_corrected_area

            return_df[column_name] = intensity_corrected_normed  # add difference to return df
        else:
            return_df[column_name] = intensity_corrected

        return return_df, baseline, intensity_corrected  # return edited return_df and the baseline

    def apply_wave_range(self, df: pd.DataFrame, column_name: str, min_selection=None, max_selection=None):
        if min_selection is None or max_selection is None:
            min_selection = df[column_name] >= wave_min
            max_selection = df[column_name] <= wave_max

            return df[column_name].loc[(min_selection & max_selection)], min_selection, max_selection
        return df[column_name].loc[(min_selection & max_selection)]

    def get_jar_reference(self, x_column_selection, min_selection, max_selection):
        # Read jar reference data and apply wave range
        jar_read = self.read_jar_reference()

        print(jar_read, type(jar_read))

        jar_data = pd.concat([x_column_selection, self.apply_wave_range(jar_read, jar_read.columns[1],
                                                                        min_selection, max_selection)], axis=1)

        jar_x_column_name = jar_data.columns[0]
        jar_data_column_name = jar_data.columns[1]

        # Calculate baseline and subtract from intensity. Add to jar_data DataFrame
        jar_intensity = jar_data[jar_data_column_name]

        # Apply user-selected range for jar peak to x and intensity arrays and calculate area underneath the curve
        jar_min_selection = jar_data[jar_x_column_name] >= jar_scaling_range[0]
        jar_max_selection = jar_data[jar_x_column_name] <= jar_scaling_range[1]

        # jar_corrected_ranged_x = jar_data[jar_x_column_name].loc[(jar_min_selection & jar_max_selection)].to_numpy()
        # jar_corrected_ranged_y = jar_data[jar_data_column_name].loc[(jar_min_selection & jar_max_selection)].to_numpy()
        # jar_corrected_ranged_area = np.trapz(y=jar_corrected_ranged_y, x=jar_corrected_ranged_x)

        return jar_intensity, jar_min_selection, jar_max_selection

    @timeit
    def get_roi_area(self, df: pd.DataFrame, x_column_name: str, roi_min: float, roi_max: float):
        min_selection = df[x_column_name] >= roi_min
        max_selection = df[x_column_name] <= roi_max

        roi_selection = df.loc[min_selection & max_selection]
        roi_x = roi_selection.loc[:, roi_selection.columns == x_column_name]
        roi_y = roi_selection.loc[:, roi_selection.columns != x_column_name]

        y_sum = roi_y.sum(axis='columns').to_numpy()
        y_area = np.trapz(x=roi_x.to_numpy().T[0], y=y_sum)

        # TODO: How handle multi-column files?
        # y_sum = roi_y.to_numpy()
        # y_area = np.trapz(x=roi_x.to_numpy().T[0], y=y_sum, axis=2)

        current_sum = 0
        # y_area = []
        # for y_sum_val in y_sum:
        #     current_sum += y_sum_val
        #     y_area.append(current_sum)

        # y_area = [np.sum([y_sum[:i]]) for i in range(len(y_sum))]

        # return_df = pd.DataFrame(columns=[x_column_name, "y_area"])
        # return_df[x_column_name] = roi_x
        # return_df["y_area"] = y_area

        # return roi_x.to_numpy().T[0], np.array(y_area)

        return y_area

    def process_data(self, df: pd.DataFrame, current_file: str, head: str):
        x_column_name = df.columns[0]

        x_column_selection, min_selection, max_selection = self.apply_wave_range(df, x_column_name)

        output_df = pd.DataFrame()
        output_df[x_column_name] = np.vectorize(convert_x, excluded=["from_unit", "to_unit"])(x_column_selection,
                                                                                              x_scale_input_unit,
                                                                                              x_scale_output_unit)

        if jar_correction:
            jar_intensity, jar_min_selection, jar_max_selection = \
                self.get_jar_reference(x_column_selection, min_selection, max_selection)
            jar_min_selection.reset_index(drop=True, inplace=True)
            jar_max_selection.reset_index(drop=True, inplace=True)

        jar_df = pd.DataFrame()
        jar_df[x_column_name] = x_column_selection

        for column_name in df.columns:
            if column_name == x_column_name:
                continue
            try:
                if test_row_index != '' and column_name != df.columns[test_row_index]:
                    continue
            except IndexError:
                pass

            intensity = self.apply_wave_range(df, column_name, min_selection, max_selection)

            original_data = intensity

            if do_correction and jar_correction:
                # intensity_baseline_corrected, _, _ = self.add_baseline_diff(pd.DataFrame(intensity.to_numpy()),
                #                                                             0, pd.DataFrame(),
                #                                                             normalize=False)

                # print("JAR INTENSITY", jar_intensity, type(jar_intensity))
                jar_original = jar_intensity

                jar_ranged = jar_intensity.to_numpy()[jar_min_selection & jar_max_selection]
                data_ranged = intensity.to_numpy()[jar_min_selection & jar_max_selection]

                # Background correct jar and spectra
                _, jar_ranged_baseline, jar_ranged_corrected = self.add_baseline_diff(pd.DataFrame(jar_ranged), 0, pd.DataFrame(), normalize=False)
                _, data_ranged_baseline, data_ranged_corrected = self.add_baseline_diff(pd.DataFrame(data_ranged), 0, pd.DataFrame(), normalize=False)

                # Calculate and apply scaling factor for jar curve
                factor, _, _, _ = numpy.linalg.lstsq(jar_ranged_corrected.reshape(-1, 1), data_ranged_corrected)
                jar_scaled = factor * jar_intensity

                if jar_debug:
                    x_values = x_column_selection.to_numpy()
                    jar_x_ranged = x_column_selection.loc[jar_min_selection & jar_max_selection].to_numpy()

                    plt.plot(x_values, original_data, label="intensity (original)")
                    plt.plot(x_values, intensity, label="intensity (pre jar-correction)")

                # Subtract jar reference intensity from intensity
                intensity = intensity - jar_scaled

                # Correct intensity in case of minimal error due to machine inaccuracy
                intensity_min = np.min(intensity)
                intensity_min_pos = np.argmin(intensity)
                if intensity_min < 0:
                    intensity = intensity - intensity_min

                if jar_debug:
                    print("Jar scale factor:", factor)
                    print("Min intensity", intensity_min, "at", x_column_selection[intensity_min_pos])

                    plt.plot(x_values, jar_original, label="jar (original)")
                    plt.plot(x_values, jar_scaled, label="jar (corrected, scaled)")

                    plt.plot(jar_x_ranged, jar_ranged_baseline, label="jar baseline (ranged)")
                    plt.plot(jar_x_ranged, jar_ranged_corrected, label="jar (ranged, corrected)")

                    plt.plot(jar_x_ranged, data_ranged_corrected, label="intensity (ranged, corrected)")
                    plt.plot(jar_x_ranged, data_ranged_baseline, label="intensity baseline (ranged)")

                    plt.plot(x_values, intensity, label="intensity (post jar-correction)")

                    plt.legend()
                    plt.show()

                # if norm_final:
                #     intensity_corrected_area = abs(np.trapz(y=intensity, x=x_column_selection.to_numpy()))
                #     intensity = intensity / intensity_corrected_area
                #
                #     print("INTENSITY AREA", intensity_corrected_area)
                #     print("INTENSITY POST NORM", intensity, type(intensity))

                jar_df[column_name] = intensity.to_numpy()

            data = pd.concat([x_column_selection, intensity.rename(column_name)], axis='columns')
            data = data.reset_index(drop=True)

            output_df, baseline_diff, unscaled_corrected = self.add_baseline_diff(data, data.columns[1],
                                                                                  output_df, data.columns[0])

            if plot_data:  # will plot every set of data if this option is enabled
                x_values = x_column_selection.to_numpy()
                final = output_df[column_name]

                try:
                    plt.plot(x_values, original_data, label="intensity (original)")
                    plt.plot(x_values, baseline_diff, label="intensity baseline")
                    plt.plot(x_values, unscaled_corrected, label="intensity (corrected, not normalized)")
                    plt.plot(x_values, final, label="intensity (final result)")

                    plt.xlabel(x_column_name)
                    plt.ylabel("intensity")
                    plt.legend(loc='upper right')
                    plt.title(column_name)

                    plt.show()
                except KeyError as error:
                    print(f'[{current_file}] failed plotting:\n', error.with_traceback(error.__traceback__))

        if do_correction:
            out_file = current_file[:-4] + '.dat'
            out_filename = os.path.basename(out_file)
            out_dir = os.path.dirname(out_file)
            out_dir = os.path.join(out_dir, "out")
            out_file = os.path.join(out_dir, out_filename)

            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

            WriteDFToFile(output_df, out_file, head=head, sep=dat_file_separator)

            if jar_correction:
                jar_out_file = current_file[:-4] + '_jar.dat'
                jar_out_filename = os.path.basename(jar_out_file)
                jar_out_dir = os.path.join(out_dir, "jar")
                jar_out_file = os.path.join(jar_out_dir, jar_out_filename)

                if not os.path.exists(jar_out_dir):
                    os.mkdir(jar_out_dir)

                WriteDFToFile(jar_df, jar_out_file, head=head, sep=dat_file_separator)

        if get_rois:
            roi_areas = np.array(
                [self.get_roi_area(output_df, x_column_name, roi_min=start, roi_max=stop) for (start, stop, color) in
                 roi_ranges])

            return output_df, roi_areas
        else:
            return output_df, None


if __name__ == '__main__':
    BackgroundCorrection()
