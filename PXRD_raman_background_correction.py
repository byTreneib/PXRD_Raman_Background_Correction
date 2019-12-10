# import modules
from scipy.sparse.linalg import spsolve
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
from scipy import sparse
import pandas as pd
import numpy as np
import threading
import warnings
import time
from pprint import pprint
from tkinter.filedialog import askopenfilenames
from tkinter import Tk
from os import getcwd

warnings.filterwarnings("ignore")
read_file_types = ['chi', 'dat', 'raman', 'txt']
# file extensions recognized by the program: .chi, .dat -> read as chi; .raman, .txt -> read as raman


# the following variables are to be edited by the user to fit their needs and the specifications of the files to be read

dat_file_separator = '\t'  # edit   set to the desired separator, that will be used when writing to .dat files

use_multiple_threads = False  # edit      turn on to process with max. 30 threads (number of threads can be modified)
MAX_THREADS = 2  # edit        exceeding 2 is not recommended since MemoryErrors and performance issues could occur

# scan_number_position = -17  # edit      specifies position of scan number
# image_number_position = -10  # edit     specifies position of image number
# scan_number_length = 6  # edit      specifies length of scan number (with zeroes)
# image_number_length = 6  # edit     specifies length of image number (with zeroes)

wave_min = 0.0  # edit  change value to set minimum value of x to process
wave_max = 20000  # edit    change value to set maximum value of x to process

include_header = True  # edit

# edit  select column index test are run on; type: Integer; set to '' (emptystring) if no testing is required
test_row_index = ''

plot_data = True  # edit  set true to plot all processed datasets; not recommended when multiple datasets are processed

baseline_lambda = 1e4  # edit   lambda for creation of the baseline
baseline_ratio = 0.05  # edit   ratio for creation of the baseline
baseline_itermax = 10  # edit  maximum number of iterations for creation of the baseline


class ReadChiToDF:  # class for reading chi file into a pandas DataFrame
    def __init__(self, filename, i_column_name, include_head=include_header):
        self.filename = filename
        self.i_column_name = i_column_name
        self.include_head = include_header

        if __name__ == '__main__':  # only show debug output if this file is directly executed
            print(f"[ReadChiToDF] Reading {self.filename}")

        with open(filename) as file:  # open specified file and read out content
            self.file_content = file.read()

        if __name__ == '__main__':  # only show debug output if this file is directly executed
            print(f"[ReadChiToDF] Reading {self.filename} successfully finished")

    def file_content_to_df(self):
        lines_raw = self.file_content.split("\n")  # splits up lines and removes the header
        if self.include_head:
            lines = list(map(lambda x: x.split(), lines_raw[4:-1]))  # splits up the columns of each line
        else:
            lines = list(map(lambda x: x.split(), lines_raw[:-1]))  # splits up the columns of each line

        x_column = list(map(lambda x: float(x[0]), lines))  # creates list out of first column (angles)
        y_column = list(map(lambda x: float(x[1]), lines))  # creates list out of second column (intensity)

        content_df = pd.DataFrame(columns=('q', self.i_column_name))  # creates empty dataframe to store x and y column
        content_df['q'] = x_column  # writes x_column list into 'q' column
        content_df[self.i_column_name] = y_column  # writes y_column list into 'I' (or whatever was specified) column

        if __name__ == '__main__':  # only show debug output if this file is directly executed
            pprint(content_df.head())

        return content_df, lines_raw[:4]  # returns the dataframe and the first 4 lines (head)


class WriteDFToFile:  # Class to write pandas DataFrames into a chi/dat file
    def __init__(self, df, filename, head="", sep="  ", include_head=include_header):
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
            writefile.write(self.head + "\n" + body)


class ReadRamanToDF:  # Class to read Raman file into pandas DataFrames
    def __init__(self, filename, include_head=include_header):
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
            lines = list(map(lambda x: x.split(), lines_raw[1:-1]))  # splits up the columns of each line
        else:
            lines = list(map(lambda x: x.split(), lines_raw[:-1]))  # splits up the columns of each line

        num_columns = len(lines[0])
        try:
            column_names = ['RamanShift (cm-1)', *lines_raw[0].split()[2:]]
        except IndexError:
            column_names = ['RamanShift (cm-1)', *list('I_' + str(x) for x in range(num_columns-1))]

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


def read_raman_to_df(filename):
    return ReadRamanToDF(filename).file_content_to_df()


def read_chi_to_df(filename, column_name="I"):
    return ReadChiToDF(filename, column_name).file_content_to_df()


def timeit(function):  # decorator function to record execution time of a function
    def timed(*args, **kwargs):
        start_time = time.time()
        return_value = function(*args, **kwargs)
        end_time = time.time()

        print(f'[TIMEIT] Time required to run {function.__name__}: {round(end_time - start_time, 2)} seconds')

        return return_value
    return timed

# ***********************************#


def als(y, lam=1e6, p=0.1, itermax=10):
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


    Inputs:
        y:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        p:
            wheighting deviations. 0.5 = symmetric, <0.5: negative
            deviations are stronger suppressed
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector

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
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z


def arpls(y, lam=baseline_lambda, ratio=baseline_ratio, itermax=baseline_itermax):
    r"""
    edit
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

    Inputs:
        y:
            input data (i.e. chromatogram of spectrum)
        lam:
            parameter that can be adjusted by user. The larger lambda is,
            the smoother the resulting background, z
        ratio:
            wheighting deviations: 0 < ratio < 1, smaller values allow less negative values
        itermax:
            number of iterations to perform
    Output:
        the fitted background vector

    """
    N = len(y)
#  D = sparse.csc_matrix(np.diff(np.eye(N), 2))
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]  # numpy.diff( ,2) does not work with sparse matrix. This is a workaround.
    D = D[1:] - D[:-1]

    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / s))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z


def residuals(parameters, x_data, y_observed, func):
    '''
    Compute residuals of y_predicted - y_observed
    where:
    y_predicted = func(parameters,x_data)
    '''
    return func(parameters, x_data) - y_observed


# ***************************************#

@timeit
def add_baseline_diff(df, probe_name, return_df):
    '''
    Function for extending the return_df dataFrame with the processed values from waxs_df for a certain probe

    :param df: dataFrame with density data of the probes
    :param probe_name: name of the probe (e.g. 'I_243')
    :param return_df: dataFrame that will later be written into the output file
    :return: the return_df with the added values
    '''

    test = df[probe_name]
    test_baseline_return = arpls(test)

    return_df[probe_name] = np.array(test - test_baseline_return)  # add difference to return df

    return return_df, test_baseline_return  # return edited return_df and the baseline


# ***************************************#


def fill_chars(string: str, max_length: int, char: str = '0'):
    # function to fill up a string up to a specified length with a specific character (standard is '0')
    # Example: fill_chars(string='34', max_length=6, char='0') => '000034'
    if len(string) > max_length:  # validates string and max_length argument
        raise ValueError("max_length must be greater than the length of the string")
    if len(char) != 1:  # validates char argument
        raise ValueError("fill character length must be 1")

    return (max_length - len(string)) * char + string

# ****************************************#


# @timeit  # record time this function takes to execute
# def read_files(read_type=read_file_type):
#     # The following variables are to be edited depending on the input files
#     data_frames = []
#     files = []
#     heads = []
#
#     if file_end == '':  # only one file to read
#         if read_type == 'chi':
#             file_content, file_head = read_chi_to_df(file_start)
#         elif read_type == 'raman':
#             file_content, file_head = read_raman_to_df(file_start)
#             file_head = ["", "", "", dat_file_separator.join(file_head)]  # completes raman head to 4 lines
#         files = [file_start]
#         data_frames = [file_content]
#         heads = [file_head]
#
#     else:  # multiple files to read
#         #  get start and end of scan and image number
#         start_scan_number = int(file_start[scan_number_position: scan_number_position+scan_number_length])
#         start_image_number = int(file_start[image_number_position: image_number_position+image_number_length])
#
#         end_scan_number = int(file_end[scan_number_position: scan_number_position+scan_number_length])
#         end_image_number = int(file_end[image_number_position: image_number_position+image_number_length])
#
#         # iterates through all scan numbers in between of start and end (start < end required)
#         for scan_number in range(start_scan_number, end_scan_number + 1):
#             # iterates through all image numbers from start to 10^image_number_length (max. image number)
#             for image_number in range(start_image_number, 10**image_number_length):
#                 # breaks iteration when end image number is exceeded and end scan number is reached
#                 if scan_number == end_scan_number and image_number > end_image_number:
#                     break
#                 # puts together the filename depending of current value of scan_number and image_number
#                 filename = file_start[:scan_number_position] + fill_chars(str(scan_number), scan_number_length) + \
#                     file_start[scan_number_position+scan_number_length: image_number_position] + \
#                     fill_chars(str(image_number), image_number_length) + '.chi'
#                 #  checks if file exists, then appends filename to files list
#                 #  if file does not exist most probably the max. of existing image numbers is reached -> breaks the loop
#                 try:
#                     open(filename).close()
#                     files.append(filename)
#                 except FileNotFoundError:
#                     print(f'File {filename} not found.. skipping to scan number {scan_number + 1}')
#                     break
#
#         # iterates through all files and creates dataFrames out of the read values, that are saved in df list
#         for file in files:
#             if read_type == 'chi':
#                 file_content, file_head = read_chi_to_df(file)
#             elif read_type == 'raman':
#                 file_content, file_head = read_raman_to_df(file)
#                 file_head = ["", "", "", dat_file_separator.join(file_head)]
#             else:
#                 NotImplementedError(f"The desired file type has not yet been implemented! The following file types"
#                                     f" are available: {', '.join(read_file_types)}")
#             data_frames.append(file_content)
#             heads.append(file_head)
#
#     return data_frames, files, heads  # returns both the list of dataFrames and the file list

def read_files():
    data_frames = []
    heads = []

    main_window = Tk()  # opens tkinter window to let user select files to process
    files = askopenfilenames(initialdir=getcwd(), filetypes=(("chi files", "*.chi"), ("dat files", "*.dat"),
                                                             ("txt raman files", "*.txt"), ("raman files", "*.raman")),
                             title='Select files to process')
    main_window.destroy()  # close window after selection is done

    print("[ReadFiles] Files selected:")
    print("\n".join(files))

    for file in files:
        file_ext = file.split('.')[1]
        read_type = file_ext
        print(read_type)
        if read_type in ['chi', 'dat']:
            file_content, file_head = read_chi_to_df(file)
        elif read_type in ['txt', 'raman']:
            file_content, file_head = read_raman_to_df(file)
            if test_row_index == '':
                file_head = ["", "", "", dat_file_separator.join(file_head)]
            else:
                file_head = ["", "", "", dat_file_separator.join([file_head[0], file_head[test_row_index]])]
        else:
            NotImplementedError(f"The desired file type has not yet been implemented! The following file types"
                                f" are available: {', '.join(read_file_types)}")
        data_frames.append(file_content)
        heads.append(file_head)

    return data_frames, files, heads  # returns both the list of dataFrames and the file list


# ***************************************#

@timeit
def extend_headers(heads):  # add baseline creation parameters to output file header
    for index, head in enumerate(heads):
        head[2] += f", Lambda = {baseline_lambda}, Ratio = {baseline_ratio}, Itermax = {baseline_itermax}"
        heads[index] = "\n".join(head)

    return heads


# ***************************************#

def process_data(df, current_file, head):
    x_column_name = df.columns[0]
    min_selection = df[x_column_name] >= wave_min
    max_selection = df[x_column_name] <= wave_max

    x_column_selection = df[x_column_name].loc[(min_selection & max_selection)]
    # determine whether to reverse the index or not for correct plotting
    to_reverse = True if float(list(x_column_selection)[0]) > float(list(x_column_selection)[-1]) else False

    output_df = pd.DataFrame()
    output_df[x_column_name] = x_column_selection

    # pprint(df[x_column_name].loc[(min_selection & max_selection)])
    # quit()
    # Formats the input DataFrame, then calculates the baseline and difference to the baseline and writes it to a file
    for column_name in df.columns:
        intensity = df[column_name].loc[(min_selection & max_selection)]
        if to_reverse:
            intensity = pd.DataFrame(intensity).set_index(x_column_selection.iloc[::-1])
        else:
            intensity = pd.DataFrame(intensity).set_index(x_column_selection)

        if column_name == x_column_name:
            continue
        try:
            if test_row_index != '' and column_name != df.columns[test_row_index]:
                continue
        except IndexError:
            pass
        print(f"[{current_file}] Started Processing")

        data = pd.concat([df[x_column_name].loc[(min_selection & max_selection)], df.loc[
                                    :, column_name:column_name].loc[(min_selection & max_selection)]], axis='columns')
        data = data.reset_index(drop=True)

        output_df, baseline_diff = add_baseline_diff(data, column_name, output_df)
        if to_reverse:
            output_df = output_df.set_index(x_column_selection.iloc[::-1])
        else:
            output_df = output_df.set_index(x_column_selection)

        baseline = pd.DataFrame()
        baseline['baseline'] = baseline_diff
        if to_reverse:
            baseline = baseline.set_index(x_column_selection.iloc[::-1])
        else:
            baseline = baseline.set_index(x_column_selection)

        if plot_data:  # will plot every set of data if this option is enabled
            try:
                plt.plot(intensity, color="blue", label="intensity")
                plt.plot(output_df[column_name], color="green", label="baseline difference")
                plt.plot(baseline['baseline'], color="red", label="baseline")

                plt.xlabel(x_column_name)
                plt.ylabel("intensity")
                plt.legend(loc='upper right')
                plt.title(column_name)

                plt.show()
            except KeyError:
                print(f'[{current_file}] failed plotting')

    # output_df.to_csv(current_file[:-4] + 'bc.csv', index=False)  # writes the output_df dataFrame into a csv file

    WriteDFToFile(output_df, current_file[:-4] + '.dat', head=head, sep=dat_file_separator)

    print(f"[{current_file}] Finished processing")

# *****************************************#


@timeit  # record time this function takes to execute
def main():
    if use_multiple_threads:
        current_threads = threading.active_count()  # running threads might not be 0 at the beginning

    dfs, files, heads = read_files()  # get all files specified by the user; dfs = DataFrames

    heads = extend_headers(heads)
    for index, df in enumerate(dfs):
        if use_multiple_threads:
            while threading.active_count() >= MAX_THREADS:
                pass
            threading.Thread(target=process_data, args=(df, files[index], heads[index])).start()
        else:
            process_data(df, files[index], heads[index])

    if use_multiple_threads:
        while threading.active_count() != current_threads:  # run program until all processing threads are done
            pass

# *******************************************#


if __name__ == '__main__':  # only executes main code, if this program is directly executed
    main()
