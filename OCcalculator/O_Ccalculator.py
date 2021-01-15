import pandas as pd
from astropy.time import Time
import os
import subprocess
from scipy.optimize import curve_fit
from scipy.optimize import brenth
from PyAstronomy import pyasl
import operator
from astropy.io import fits
from astropy.table import Table
import dateutil
import numpy
import matplotlib.pyplot as plt
import shutil

# Codes for my O-C extraction code
def process_aavso(indir, ra, dec):
    with open(indir + "Phot_aavso.dat", encoding="utf8") as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    time = numpy.empty(0)
    mag = numpy.empty(0)

    i = 0
    for lines in content:
        if not lines.startswith("#"):
            lines = lines.replace("\t", " ")
            if (
                lines.split()[2] == "V"
                or lines.split()[2] == "Vis."
                or lines.split()[3] == "V"
                or lines.split()[3] == "Vis."
            ):
                time = numpy.append(time, float(lines.split()[0]))
                mag = numpy.append(mag, float(lines.split()[1]))
                i += 1
    data = numpy.vstack((time, mag)).T

    helio_data = numpy.zeros((data.shape[0], data.shape[1]))
    for g in range(data.shape[0]):
        helio_data[g, 0] = pyasl.helio_jd(data[g, 0] - 2.4e6, ra, dec) + 2.4e6
        helio_data[g, 1] = data[g, 1]

    numpy.savetxt(
        indir + "feldolg_aavso.dat",
        helio_data,
        fmt="%10f",
        delimiter="\t",
        newline=" \n",
    )


def process_asassn(indir, df):
    temp_hjd = df["HJD"].values
    temp_mag = df["mag"].values
    filts = ["V", "g"]
    for j in range(2):
        hjd = []
        mag = []
        for i in range(len(temp_hjd)):
            if df["Filter"].values[i] == filts[j]:
                hjd.append(temp_hjd[i])
                mag.append(temp_mag[i])
        data = numpy.vstack((hjd, mag)).T
        numpy.savetxt(
            indir + "feldolg_asas_sn_" + str(filts[j]) + ".dat",
            data,
            fmt="%10f",
            delimiter="\t",
            newline=" \n",
        )


def process_asas(indir, ra, dec):
    with open(indir + "Phot_asas.dat") as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    time = numpy.empty(0)
    mag = numpy.empty(0)

    for lines in content:
        if not lines.startswith("#"):
            try:
                if lines.split()[11] == "A" or lines.split()[11] == "B":
                    time = numpy.append(time, float(lines.split()[0]) + 2450000)
                    mag = numpy.append(mag, float(lines.split()[1]))
            except IndexError:
                time = numpy.append(time, float(lines.split()[0]) + 2450000)
                mag = numpy.append(mag, float(lines.split()[1]))
    data = numpy.vstack((time, mag)).T
    numpy.savetxt(
        indir + "feldolg_ASAS.dat", data, fmt="%10f", delimiter="\t", newline=" \n"
    )


def process_k2(indir):
    data = fits.open(indir + "Phot_K2.fits")
    times = []
    fl = []
    for i in range(data[1].data.shape[0]):
        times.append(data[1].data[i][0] + 2454833)
        fl.append(data[1].data[i][1])
    fl = numpy.array(fl)
    times = numpy.array(times)
    mag = -2.5 * numpy.log10(fl)
    numpy.savetxt(
        indir + "feldolg_K2.dat",
        numpy.vstack((times, mag)).T,
        fmt="%10f",
        delimiter="\t",
        newline=" \n",
    )


def process_kws(indir, ra, dec, filt="V"):
    if filt == "Ic":
        with open(indir + "Phot_kws-ic.dat") as f:
            content = f.readlines()
    else:
        with open(indir + "Phot_kws.dat") as f:
            content = f.readlines()
    content = [x.strip("[\n]") for x in content]
    times = []
    mags = []
    for i in range(len(content)):
        line = content[i].split("\t")
        if line[4] == "V" or line[4] == "Ic":
            tempmag = float(line[2])
            if tempmag < 20:
                times.append(line[1])
                mags.append(float(line[2]))
    t = Time(times)
    jdtimes = numpy.zeros((len(t)))
    for i in range(len(t)):
        jdtimes[i] = float(t[i].jd)
    data = numpy.vstack((jdtimes, mags)).T
    helio_data = numpy.zeros((data.shape[0], data.shape[1]))
    for g in range(data.shape[0]):
        helio_data[g, 0] = pyasl.helio_jd(data[g, 0] - 2.4e6, ra, dec) + 2.4e6
        helio_data[g, 1] = data[g, 1]
    if filt == "Ic":
        numpy.savetxt(
            indir + "feldolg_KWS-ic.dat",
            helio_data,
            fmt="%10f",
            delimiter="\t",
            newline=" \n",
        )
    else:
        numpy.savetxt(
            indir + "feldolg_KWS.dat",
            helio_data,
            fmt="%10f",
            delimiter="\t",
            newline=" \n",
        )


def process_harvard(indir):
    hd = pd.read_csv(indir + "Phot_Harvard.txt", header=5, sep="\t", skiprows=[6])
    numpy.savetxt(
        indir + "feldolg_harvard.dat",
        numpy.vstack((hd["Date"], hd["magcal_magdep"])).T,
        fmt="%10f",
        delimiter="\t",
        newline=" \n",
    )


def process_iomc(indir):
    data = Table.read(indir + "Phot_IOMC.fits")
    time = numpy.array(data["BARYTIME"]) + 2451544.5
    mag = numpy.array(data["MAG_V"])
    numpy.savetxt(
        indir + "feldolg_iomc.dat",
        numpy.vstack((time, mag)).T,
        fmt="%10f",
        delimiter="\t",
        newline=" \n",
    )


def process_smei(indir, ra, dec):
    with open(indir + "Phot_SMEI.dat", "r") as myfile:
        data = myfile.readlines()

    time_arr = []
    mag_arr = []

    j = 0
    i = 0
    while j < len(data):
        line_split = data[j].split(",")
        try:
            if float(line_split[1][:-2]) > 0:
                dt = dateutil.parser.parse(line_split[0])
                time = Time(dt)
                time_arr.append(time.jd)
                mag_arr.append(float(line_split[1][:-2]))
            i += 1
            j += 2
        except ValueError:  # The SMEI data series can contain invalid values (INF-s, Nan-s)
            i += 1
            j += 2

    data = numpy.vstack((time_arr, mag_arr)).T

    helio_data = numpy.zeros((data.shape[0], data.shape[1]))
    for g in range(data.shape[0]):
        helio_data[g, 0] = pyasl.helio_jd(data[g, 0] - 2.4e6, ra, dec) + 2.4e6
        helio_data[g, 1] = data[g, 1]

    numpy.savetxt(
        indir + "feldolg_smei.dat",
        helio_data,
        fmt="%10f",
        delimiter="\t",
        newline=" \n",
    )


def process_tess(indir):
    hdul = fits.open(indir + "/Phot_tess.fits")
    time = []
    flux = []

    for i in range(hdul[1].data.shape[0]):
        temp = hdul[1].data[i][3]
        if not numpy.isnan(temp):
            time.append(hdul[1].data[i][0] + 2.457e6)
            flux.append(hdul[1].data[i][3])
    numpy.savetxt(
        indir + "feldolg_tess.dat",
        numpy.vstack((time, -2.5 * numpy.log10(flux))).T,
        fmt="%10f",
        delimiter="\t",
        newline="\n",
    )


def process_wasp(indir):
    df = pd.read_csv(indir + "Phot_swasp.csv")
    temp_hjd = df["HJD"].values
    temp_mag = df["magnitude"].values
    hjd = []
    mag = []
    for i in range(len(temp_hjd)):
        hjd.append(temp_hjd[i])
        mag.append(temp_mag[i])
    data = numpy.vstack((hjd, mag)).T
    numpy.savetxt(
        indir + "feldolg_swasp.dat", data, fmt="%10f", delimiter="\t", newline=" \n"
    )


def process_other(indir, file, ra, dec):
    data = pd.read_csv(indir + file, sep=" ", index_col=False)
    column_names = list(data.columns.values)
    if "JD" in column_names:
        temp_hjd = numpy.zeros((data.shape[0]))
        for k in range(data.shape[0]):
            temp_hjd[k] = pyasl.helio_jd(data["JD"].values[k] - 2.4e6, ra, dec) + 2.4e6
        outp = numpy.vstack((temp_hjd, data["V"])).T
    elif "HJD" in column_names:
        outp = numpy.vstack((data["HJD"], data["V"])).T
    elif "MJD" in column_names:
        temp_hjd = numpy.zeros((data.shape[0]))
        for k in range(data.shape[0]):
            temp_hjd[k] = pyasl.helio_jd(data["MJD"].values[k] + 0.5, ra, dec) + 2.4e6
        outp = numpy.vstack((temp_hjd, data["V"])).T
    else:
        result = [
            i
            for i in column_names
            if (i.startswith("JD-") or i.startswith("HJD-") or i.startswith("MJD-"))
        ]

        if result[0].startswith("JD"):
            templs = data[result[0]] + float(result[0][3:])
            temp_hjd = numpy.zeros((templs.shape[0]))
            for k in range(templs.shape[0]):
                temp_hjd[k] = pyasl.helio_jd(templs[k] - 2.4e6, ra, dec) + 2.4e6
            outp = numpy.vstack((temp_hjd, data["V"])).T

        elif result[0].startswith("HJD"):
            outp = numpy.vstack((data[result[0]] + float(result[0][4:]), data["V"])).T

        elif result[0].startswith("MJD"):
            templs = data[result[0]] + float(result[0][4:])
            temp_hjd = numpy.zeros((templs.shape[0]))
            for k in range(templs.shape[0]):
                temp_hjd[k] = pyasl.helio_jd(templs[k] + 0.5, ra, dec) + 2.4e6
            outp = numpy.vstack((temp_hjd, data["V"])).T

        else:
            print("Something went wrong")
    numpy.savetxt(
        indir + "feldolg_" + file[5:], outp, fmt="%10f", delimiter="\t", newline="\n"
    )


def process_photometry(indir, ra, dec):
    files = []
    for i in os.listdir(indir):
        if os.path.isfile(os.path.join(indir, i)) and i.startswith("Phot"):
            files.append(i)

    surveys = []

    for j in range(len(files)):

        if files[j].endswith("aavso.dat"):
            process_aavso(indir, ra, dec)
            surveys.append("AAVSO")

        elif files[j].endswith("asas.dat"):
            process_asas(indir, ra, dec)
            surveys.append("ASAS")

        elif files[j].endswith("asassn.csv"):
            df = pd.read_csv(indir + "Phot_asassn.csv", sep=",")
            process_asassn(indir, df)
            surveys.append("ASASSN-g")
            surveys.append("ASASSN-V")

        elif files[j].endswith("Harvard.txt"):
            process_harvard(indir)
            surveys.append("Harvard")

        elif files[j].endswith("IOMC.fits"):
            process_iomc(indir)
            surveys.append("IOMC")

        elif files[j].endswith("K2.fits"):
            process_k2(indir)
            surveys.append("K2")

        elif files[j].endswith("kws.dat"):
            process_kws(indir, ra, dec)
            surveys.append("KWS")

        elif files[j].endswith("kws-ic.dat"):
            process_kws(indir, ra, dec, filt="Ic")
            surveys.append("KWS-Ic")

        elif files[j].endswith("SMEI.dat"):
            process_smei(indir, ra, dec)
            surveys.append("SMEI")

        elif files[j].endswith("tess.fits"):
            process_tess(indir)
            surveys.append("TESS")

        elif files[j].endswith("swasp.csv"):
            process_wasp(indir)
            surveys.append("SWASP")

        else:
            process_other(indir, files[j], ra, dec)
            surveys.append(files[j][5:-4])

    return surveys


def apply_filter(indir, filename, lms):
    data = numpy.loadtxt(indir + filename)
    n_data = data[(data[:, 1] > lms[0]) & (data[:, 1] < lms[1])]
    numpy.savetxt(indir + filename, n_data, fmt="%10f", delimiter="\t", newline="\n")


def get_surveys(indir):
    lst = os.listdir(indir)
    surveys = [(i.split("_")[1])[:-4] for i in lst if i.startswith("period")]
    return surveys


def master_period(indir, surveys):
    if "master_period.per" in os.listdir(indir):
        for i in range(len(surveys)):
            shutil.copyfile(
                indir + "master_period.per", indir + "period_" + surveys[i] + ".per"
            )


def manage_files(detail, mode, indir, P04dir="../../Period04"):
    files = []
    for i in os.listdir(P04dir):
        if os.path.isfile(os.path.join(P04dir, i)) and i.startswith(detail):
            files.append(i)

    if mode == "Clean":
        for i in range(len(files)):
            os.remove(P04dir + "/" + files[i])
    elif mode == "Move":
        for i in range(len(files)):
            shutil.move(P04dir + "/" + files[i], indir + files[i])
    else:
        print("Wrong mode flag")


def sin_fit_p04(x, F, A, PH, c, ps):
    outp = c
    for i in range(F.shape[0]):
        outp += A[i] * numpy.sin(2 * numpy.pi * (F[i] * x + PH[i] + ps))
    return outp


def derivate_p04(x, F, A, PH, ps):
    outp = 0
    for i in range(F.shape[0]):
        outp += (
            A[i]
            * 2
            * numpy.pi
            * F[i]
            * numpy.cos(2 * numpy.pi * (F[i] * x + PH[i] + ps))
        )
    return outp


def load_P04_file(indir, filename):

    with open(indir + filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    F_all = []
    A_all = []
    PH_all = []
    for lines in content:
        lines = lines.replace("\t", " ")
        F_all = numpy.append(F_all, float(lines.split()[1]))
        A_all = numpy.append(A_all, float(lines.split()[2]))
        PH_all = numpy.append(PH_all, float(lines.split()[3]))

    return F_all, A_all, PH_all


def find_zero_point_mod(mm, F, A, PH, popt_n, mode="Incr"):
    zerop = popt_n[0]
    middle = numpy.average(mm)
    if mode == "Incr":
        limited = mm[
            (mm > middle - 1 / 2 * 1 / F[0]) & (mm < middle + 1 / 2 * 1 / F[0])
        ]
        l_curve = sin_fit_p04(limited, F, A, PH, *popt_n)
        min_ind = list(l_curve).index(max(l_curve))
        max_ind = list(l_curve).index(min(l_curve))
    elif mode == "Decr":
        limited = mm[
            (mm > middle + 1 / 4 * 1 / F[0]) & (mm < middle + 3 / 4 * 1 / F[0])
        ]
        l_curve = sin_fit_p04(limited, F, A, PH, *popt_n)
        min_ind = list(l_curve).index(min(l_curve))
        max_ind = list(l_curve).index(
            max(l_curve)
        )  # Mivel más a fénygörbe menete, ezt bele kell tenni ide
    else:
        raise NameError("Wrong epoch finding mode given")

    transformed = abs(l_curve[min_ind:max_ind] - zerop)
    return limited[min_ind:max_ind][list(transformed).index(min(transformed))], zerop


def find_maximum(mm, F, A, PH, popt_n, e_zp):
    cv = sin_fit_p04(mm, F, A, PH, *popt_n)
    nvl = numpy.sort(cv)  # Don't forget here, that the magnitude scale is reversed
    i = 0
    while True:
        if mm[list(cv).index(nvl[i])] > e_zp:
            return mm[list(cv).index(nvl[i])], nvl[i]
        else:
            i += 1


def find_zero_point(mm, F, A, PH, popt_n, mode="Incr"):
    j = 0
    zerop = popt_n[0]
    min_places = numpy.zeros((5))
    temp_min = []

    diffr = abs(sin_fit_p04(mm, F, A, PH, *popt_n) - zerop)

    for i in diffr:
        if i < 0.01:
            temp_min.append(mm[j])
        j += 1

    summa = 0
    count = 0
    pl_ind = 0

    for k in range(len(temp_min)):
        if pl_ind < min_places.shape[0]:
            if summa == 0:
                summa = temp_min[k]
                count += 1

            elif k < len(temp_min) - 1:
                if temp_min[k + 1] - temp_min[k] < 0.1:
                    summa += temp_min[k + 1]
                    count += 1

                else:
                    min_places[pl_ind] = summa / count
                    summa = 0
                    count = 0
                    pl_ind += 1

            else:
                min_places[pl_ind] = summa / count
                summa = 0
                count = 0
                pl_ind += 1
    if mode == "Incr":
        if derivate_p04(min_places[1], F, A, PH, popt_n[1]) < 0:
            return min_places[1], zerop
        else:
            return min_places[2], zerop
    elif mode == "Decr":
        if derivate_p04(min_places[1], F, A, PH, popt_n[1]) < 0:
            return min_places[2], zerop
        else:
            return min_places[3], zerop
    else:
        raise NameError("Wrong epoch finding mode given")


def find_epochs(
    indir, surveys, stack_interval=350, drop_num=5, ep_method="Zerop", submode="Incr"
):
    fitter = []
    nms = []
    zp_files = []
    per_files = []
    f_index = 0

    for i in os.listdir(indir):
        if os.path.isfile(os.path.join(indir, i)) and i.startswith(
            "feldolg"
        ):  # pick the files whick contain "feldolg"
            zp_files.append(i)

    for i in os.listdir(indir):
        if os.path.isfile(os.path.join(indir, i)) and i.startswith(
            "period"
        ):  # pick the files whick contain "period"
            per_files.append(i)

    for t_filename in per_files:
        print(t_filename)
        dropped_points = 0
        F, A, PH = load_P04_file(indir, t_filename)

        Per = 1 / F[0]

        indata_t = numpy.loadtxt(indir + zp_files[f_index])
        indata = numpy.array(
            sorted(indata_t, key=lambda x: x[0])
        )  # sort the time string data into increasing order by the time

        start_date = min(indata[:, 0])

        if (
            sin_fit_p04(start_date, F, A, PH, 0, 0) > 0
        ):  # for error handling, happened sometimes that it found the wrong
            start_date = start_date - 1 / (2 * F[0])  # maximum

        t_int = numpy.linspace(
            start_date - 1 / (2 * F[0]), start_date + 1 / (2 * F[0]), 2000
        )  # create a dataset for the first measured peak
        t_list = sin_fit_p04(t_int, F, A, PH, 0, 0)
        min_index, min_value = min(
            enumerate(t_list), key=operator.itemgetter(1)
        )  # find the time of the max brightness (min mg)
        max_index, max_value = max(
            enumerate(t_list[0:min_index]), key=operator.itemgetter(1)
        )  # find min from the left of the max
        try:
            t_ref_ep = brenth(
                sin_fit_p04, t_int[min_index], t_int[max_index], args=(F, A, PH, 0, 0)
            )  # calculate the place where it reaches 0
        except ValueError:
            f_index += 1
            print("Error encountered at Brenth")
            continue
        # it happens sometimes, that this one gives an error; delete first few datapoints

        refmax_k = t_ref_ep  # from here on, its the former light curve separator, which creates the phase-peaks

        num_peaks_to_stack = int(stack_interval * F[0])
        peaks = {}
        avg_times = []
        i = 0
        l = 0
        cycle = 0
        drop = 0
        while i < indata.shape[0]:
            while refmax_k < indata[i, 0] - Per:
                refmax_k += Per
            sett = numpy.empty((0))
            setm = numpy.empty((0))
            j = 0
            sett = indata[i, 0]
            time_store = []
            time_store.append(indata[i, 0])
            setm = indata[i, 1]
            i += 1
            k = i
            numPoints = 1
            while (
                j < num_peaks_to_stack
            ):  # number of light curve peaks we want to stack to create the phase curve
                while True:
                    if k == indata.shape[0]:  # if we reached the end of the file, exit
                        j += num_peaks_to_stack + 1
                        break
                    elif (
                        indata[k, 0] > refmax_k - (1 / 2) * Per
                        and indata[k, 0] < refmax_k + (1 / 2) * Per
                    ):  # otherwise substract cycle*Period, add it
                        time_store.append(
                            indata[k, 0]
                        )  # for the fitting we will use the phase curves, but on the O-C
                        redtime = (
                            indata[k, 0] - (j) * Per
                        )  # diagram the time will be the average of the used points
                        sett = numpy.append(sett, redtime)
                        setm = numpy.append(setm, indata[k, 1])
                        k += 1
                        numPoints += 1
                    else:  # in case the reference is too far from the next point, make a Period long step
                        j += 1  # (this still counts as a cycle)
                        refmax_k += Per
                        break
            cycle += 1
            i = k
            if (
                numPoints < drop_num
            ):  # if the created phase peak is not populated well enough, then just drop it
                dropped_points += numPoints
                drop += 1
                continue
            nms.append(numPoints)
            data = numpy.vstack((sett, setm)).T
            cycle_prot = 0
            while (
                True
            ):  # BUG: sometimes somehow the time data of the first point in the new peak array
                if (
                    abs(data[0, 0] - numpy.mean(data[1:, 0])) > 1 / 2 * Per
                ):  # try to force it back to its right place
                    data[0, 0] -= Per
                else:  # sometimes it succeds, but sometimes we need to throw the first datapoint
                    break
                cycle_prot += 1
                if (
                    cycle_prot > 100
                ):  # it happens sometimes, that the program creates the right phase curve
                    print(
                        "Infinite loop break \n"
                    )  # but the loop fails to stop, so I implemented an artificial break
                    break
            peaks[l] = data
            l += 1
            avg_times.append(numpy.mean(time_store))
        print(cycle, drop)
        print(indata.shape[0], dropped_points)

        output = numpy.zeros((len(peaks), 4))

        fig = plt.figure(figsize=(20, int(len(peaks)) + 1))

        for g in range(len(peaks)):
            t_tempt = peaks[g][:, 0]
            t_tempm = peaks[g][:, 1]

            tempt = []
            tempm = []
            for check in range(t_tempt.shape[0]):
                if t_tempt[check] > numpy.mean(t_tempt) - 10 * Per:
                    tempt.append(t_tempt[check])
                    tempm.append(t_tempm[check])

            popt_n, pcov_n = curve_fit(
                lambda x, c, ps: sin_fit_p04(x, F, A, PH, c, ps), tempt, tempm
            )
            fitter.append([popt_n, pcov_n, Per])
            mm = numpy.linspace(
                min(tempt) - 1 / 2 * Per, max(tempt) + 1 / 2 * Per, 2000
            )

            if ep_method == "Zerop":
                try:
                    output[g, 0], output[g, 1] = find_zero_point_mod(
                        mm, F, A, PH, popt_n, submode
                    )
                except ValueError:
                    output[g, 0], output[g, 1] = find_zero_point(
                        mm, F, A, PH, popt_n, submode
                    )
            elif ep_method == "Maximum":
                try:
                    zp_ep, zp_val = find_zero_point_mod(mm, F, A, PH, popt_n, submode)
                except ValueError:
                    zp_ep, zp_val = find_zero_point(mm, F, A, PH, popt_n, submode)

                output[g, 0], output[g, 1] = find_maximum(mm, F, A, PH, popt_n, zp_ep)
            else:
                raise NameError("Wrong epoch finding mode given")

            output[g, 3] = estim_avg_khi_sq(popt_n, tempt, tempm, F, A, PH)
            boots_fits = []
            boots_zp = []
            for boots_i in range(100):
                random_indices = numpy.random.choice(
                    len(tempt), int(len(tempt) * 0.75), replace=False
                )
                try:
                    popt_r, pcov_r = curve_fit(
                        lambda x, c, ps: sin_fit_p04(x, F, A, PH, c, ps),
                        numpy.array(tempt)[random_indices],
                        numpy.array(tempm)[random_indices],
                    )
                except RuntimeError:
                    continue
                boots_fits.append(popt_r)
                try:
                    if ep_method == "Zerop":
                        boots_zp.append(
                            find_zero_point_mod(mm, F, A, PH, popt_r, submode)[0]
                        )
                    else:
                        boots_eps = find_zero_point_mod(mm, F, A, PH, popt_r, submode)[
                            0
                        ]
                        boots_zp.append(
                            find_maximum(mm, F, A, PH, popt_r, boots_eps)[0]
                        )
                except ValueError:
                    continue
            output[g, 2] = numpy.sqrt(
                numpy.std(numpy.array(boots_zp)[numpy.array(boots_zp) > 0]) ** 2
                + (numpy.std(numpy.array(boots_fits)[:, 0]) / popt_n[0] * Per) ** 2
            )

            fig.add_subplot(int(len(peaks) / 4) + 1, 4, g + 1)
            plt.gca().invert_yaxis()
            plt.plot(tempt, tempm, "bo")
            plt.plot(mm, sin_fit_p04(mm, F, A, PH, *popt_n), "r")
            for p_i in range(len(boots_fits)):
                plt.plot(
                    mm,
                    sin_fit_p04(mm, F, A, PH, *(boots_fits[p_i])),
                    "r",
                    lw=0.8,
                    alpha=0.2,
                )
            plt.grid(True)
            plt.plot(output[g, 0], output[g, 1], "o", color="black")
        plt.show()
        print(output.shape[0])

        with open(indir + "/peaks_" + surveys[f_index] + ".dat", "a") as filee:
            for i in range(output.shape[0]):
                filee.write("%f %f %f\n" % (output[i, 0], output[i, 1], output[i, 2]))
            filee.close()

        f_index += 1
    return fitter, nms


def estim_avg_khi_sq(popt, tempt, tempm, F, A, PH):
    khis = []
    for i in range(len(tempt)):
        khis.append((tempm[i] - sin_fit_p04(tempt[i], F, A, PH, *popt)) ** 2)
    return numpy.average(khis) / popt[0] ** 2


def calc_OC(indir, refepoch, freqv, surveys=None, fn=None):
    deltat = 1 / (2 * freqv)
    program = "./OCCalc_mod.exe"  # will call an external program written in C
    if fn is None:
        for peak_file_index in range(len(surveys)):
            peak_data = numpy.loadtxt(
                indir + "peaks_" + surveys[peak_file_index] + ".dat"
            )

            try:
                peak_data.shape[1]
                dpoints = peak_data.shape[0]
                minbound = min(peak_data[:, 0]) - 100
                maxbound = max(peak_data[:, 0]) + 100
            except IndexError:
                dpoints = peak_data.shape[0]
                minbound = peak_data[0] - 100
                maxbound = peak_data[0] + 100

            flags = (
                str(refepoch)
                + " "
                + str(freqv)
                + " "
                + str(minbound)
                + " "
                + str(maxbound)
                + " "
                + str(dpoints)
                + " "
                + str(deltat)
                + " "
                + indir
                + "OC_f_"
                + surveys[peak_file_index]
                + ".dat"
            )
            argument = [indir + "peaks_" + surveys[peak_file_index] + ".dat " + flags]
            subprocess.Popen([program, argument])

    else:
        # if fn.startswith('Data'):
        # program = './OCCalc.exe'
        data = numpy.loadtxt(indir + fn, comments="#")
        dpoints = data.shape[0]
        minbound = min(data[:, 0]) - 100
        maxbound = max(data[:, 0]) + 100
        flags = (
            str(refepoch)
            + " "
            + str(freqv)
            + " "
            + str(minbound)
            + " "
            + str(maxbound)
            + " "
            + str(dpoints)
            + " "
            + str(deltat)
            + " "
            + indir
            + "OC_"
            + fn[5:-4]
            + "_comp.dat"
        )
        argument = [indir + fn + " " + flags]
        subprocess.Popen([program, argument])


def remove_errors(indir, surveys, P):
    for i in range(len(surveys)):
        data = numpy.loadtxt(indir + "OC_f_" + surveys[i] + ".dat", comments="#")
        try:
            numpy.savetxt(
                indir + "OC_f_" + surveys[i] + ".dat",
                data[
                    (data[:, 0] > 0) & (data[:, 3] < P) & (data[:, 3] != 0)
                ],
                fmt="%10f",
                delimiter="\t",
                newline="\n",
            )
        except IndexError:
            continue


def set_edgewidth(sizes):
    width_array = numpy.zeros(sizes.shape)
    width_array[sizes < 10] = 0.1
    width_array[(sizes > 10) & (sizes < 20)] = 0.25
    width_array[(sizes > 20)] = 0.5
    return width_array


def change_h_e(x, all_data):
    change = (
        (max(all_data[:, 1]) - min(all_data[:, 1]))
        / (max(all_data[:, 0]) - min(all_data[:, 0]))
        * (x - min(all_data[:, 0]))
    ) + min(all_data[:, 1])
    return change


def create_all_datafile(indir, offset, P, prefix=""):
    OC_all_data = []

    oc_files = []
    oc_all_files = []
    for i in os.listdir(indir):
        if os.path.isfile(os.path.join(indir, i)) and i.startswith(prefix + "OC_f"):
            oc_files.append(i)

    for plot_index in range(len(oc_files)):
        temp_n = numpy.loadtxt(indir + oc_files[plot_index], comments="#")
        try:
            temp = numpy.zeros((temp_n.shape[0], temp_n.shape[1] + 1))
            temp = temp_n  # temp[:,:-1] = temp_n
            # temp[:,-1]=size[plot_index]
            for i in range(temp_n.shape[0]):
                OC_all_data.append(temp[i])
        except IndexError:
            temp = numpy.zeros((temp_n.shape[0]))
            temp = temp_n  # temp[:-1] = temp_n
            # temp[-1] = size[plot_index]
            OC_all_data.append(temp)

    for i in os.listdir(indir):
        if (
            os.path.isfile(os.path.join(indir, i))
            and i.startswith(prefix + "OC")
            and (i.endswith("weights.dat") is False)
        ):
            oc_all_files.append(i)
    old_files = list(set(oc_all_files) - set(oc_files))
    print(old_files)

    for i in range(len(old_files)):
        temp_n = numpy.loadtxt(indir + old_files[i])
        if temp_n.shape[1] < 4:
            temp = numpy.zeros((temp_n.shape[0], temp_n.shape[1] + 1))
            temp[:, :-1] = temp_n
            if os.path.isfile(indir + "Errvect_" + old_files[i][3:-9] + ".dat"):
                temp[:, -1] = numpy.loadtxt(
                    indir + "Errvect_" + old_files[i][3:-9] + ".dat", comments="#"
                )
            else:
                temp[:, -1] = 0.015 * P
        else:
            temp = temp_n
        for j in range(temp_n.shape[0]):
            OC_all_data.append(temp[j])

    return numpy.array(OC_all_data)


def create_OC_plot(
    mode,
    surveys,
    P,
    name,
    y_limits,
    indir,
    refepoch,
    old_data=None,
    offs=0,
    save=False,
    prefix="",
    SizeLabelPlot=True,
    trunc_date=False,
    onec=False,
    gridss=False,
    subplt=None,
    f=None,
    xll=None,
    sp_n=None,
):
    OC_datas = {}
    err_datas = {}
    if mode == "HJD":
        m = 0
    elif mode == "Epoch":
        m = 1
    else:
        print("Wrong mode flag given")
        return ()

    if trunc_date and mode == "HJD":
        trunc = 2400000
    else:
        trunc = 0

    f_index = len(surveys)
    if gridss:
        if sp_n is not None:
            ax = f.add_subplot(subplt[sp_n])
        else:
            ax = f.add_subplot(subplt[0])
    else:
        f = plt.figure(figsize=(12, 6))
        ax = f.add_subplot(1, 1, 1)
    cust = "o"
    for plot_index in range(f_index):
        OC_datas[plot_index] = numpy.loadtxt(
            indir + prefix + "OC_f_" + surveys[plot_index] + ".dat", comments="#"
        )
        if OC_datas[plot_index].ndim == 1:
            try:
                err_datas[plot_index] = OC_datas[plot_index][3]
            except IndexError:
                err_datas[plot_index] = numpy.loadtxt(
                    indir + prefix + "peaks_" + surveys[plot_index] + ".dat",
                    comments="#",
                )[2]
        else:
            try:
                err_datas[plot_index] = OC_datas[plot_index][:, 3]
            except IndexError:
                err_datas[plot_index] = numpy.loadtxt(
                    indir + prefix + "peaks_" + surveys[plot_index] + ".dat",
                    comments="#",
                )[:, 2]

        sizes = give_size(err_datas[plot_index] / P)

        if plot_index > 9:
            cust = "^"

        try:
            if surveys[plot_index] == "Harvard":
                plt.scatter(
                    OC_datas[plot_index][:, m] - trunc,
                    OC_datas[plot_index][:, 2],
                    marker=cust,
                    s=sizes,
                    edgecolor="black",
                    label="DASCH",
                    linewidths=0.4,
                )

            elif surveys[plot_index] == "Piszkes":
                plt.scatter(
                    OC_datas[plot_index][:, m] - trunc,
                    OC_datas[plot_index][:, 2],
                    marker=cust,
                    s=sizes,
                    edgecolor="black",
                    label="Piszkesteto",
                    linewidths=0.4,
                )
            else:
                plt.scatter(
                    OC_datas[plot_index][:, m] - trunc,
                    OC_datas[plot_index][:, 2],
                    marker=cust,
                    s=sizes,
                    edgecolor="black",
                    label=surveys[plot_index],
                    linewidths=0.4,
                )

        except IndexError:
            plt.scatter(
                OC_datas[plot_index][m] - trunc,
                OC_datas[plot_index][2],
                marker=cust,
                s=sizes,
                edgecolor="black",
                label=surveys[plot_index],
                linewidths=0.4,
            )

    dl = 0
    if old_data is not None:
        if os.path.isfile(indir + old_data[0][:-9] + "_weights.dat"):
            OC_old_data2 = numpy.loadtxt(indir + prefix + old_data[0], comments="#")
            old_data_weight = numpy.loadtxt(
                indir + old_data[0][:-9] + "_weights.dat", comments="#"
            )
            slit = 0
            for i in range(OC_old_data2.shape[0]):
                if old_data_weight[i] != 0:
                    plt.scatter(
                        OC_old_data2[i, m] - trunc,
                        OC_old_data2[i, 2] + offs,
                        s=10 / 3 * old_data_weight[i],
                        color="navy",
                    )
                    if slit == 0:
                        plt.scatter(
                            OC_old_data2[i, m] - trunc,
                            OC_old_data2[i, 2] + offs,
                            s=10 / 3 * old_data_weight[i],
                            color="navy",
                            label=old_data[0][3:-9] + " (Arc.)",
                        )
                        slit += 1
                else:
                    plt.scatter(
                        OC_old_data2[i, m] - trunc,
                        OC_old_data2[i, 2] + offs,
                        s=10,
                        facecolor="none",
                        edgecolor="navy",
                    )

            dl = 1
        for i in range(dl, len(old_data)):
            OC_old_data = numpy.loadtxt(indir + prefix + old_data[i], comments="#")
            try:
                sizes = give_size(OC_old_data[:, 3] / P)
            except IndexError:
                sizes = 10
            plt.scatter(
                OC_old_data[:, m] - trunc,
                OC_old_data[:, 2] + offs,
                marker="^",
                label=old_data[i][3:-9] + " (Arc.)",
                s=sizes,
                edgecolor="black",
                linewidths=0.4,
            )

    if onec:
        ncl = 1
    else:
        ncl = 2
    lgnd = plt.legend(ncol=ncl, loc=2)
    for lgnd_ind in range(f_index):
        lgnd.legendHandles[lgnd_ind]._sizes = [40]

    temp_d = create_all_datafile(indir, 0, P)
    differ = numpy.max(temp_d[:, m]) - numpy.min(temp_d[:, m])
    if xll is None:
        plt.xlim(
            numpy.min(temp_d[:, m]) - 0.05 * differ - trunc,
            numpy.max(temp_d[:, m]) + 0.05 * differ - trunc,
        )
    else:
        plt.xlim(xll[0], xll[1])
    if not gridss:
        if trunc_date and mode == "HJD":
            plt.xlabel(mode + "-2400000", fontsize=15)
        else:
            plt.xlabel(mode, fontsize=15)
    plt.ylabel("$O-C$ (days)", fontsize=15)
    ax.tick_params(axis="y", labelsize=12)
    if gridss:
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.tick_params(axis="x", labelsize=12)
    plt.grid(True, alpha=0.3)  # linewidth=0.2)
    plt.ylim(y_limits[0], y_limits[1])

    if (
        SizeLabelPlot
    ):  # Small plot for showing the size-error relation of the datapoints
        if gridss:
            a = plt.axes([0.58, 0.38, 0.3, 0.1], facecolor="white")
        else:
            a = plt.axes([0.58, 0.15, 0.3, 0.1], facecolor="white")
        errs = temp_d[:, -1]
        plt.scatter(
            numpy.logspace(numpy.log10(max(errs)), numpy.log10(min(errs)), 10),
            numpy.ones((10)),
            s=give_size(
                numpy.logspace(numpy.log10(max(errs)), numpy.log10(min(errs)), 10)
            ),
            color="red",
            edgecolor="black",
            linewidths=0.6,
        )
        plt.xscale("log")
        a.tick_params(axis="y", left=False, labelleft=False)
        a.tick_params(
            which="both",
            axis="x",
            direction="in",
            labelsize=10,
            pad=-17,
            bottom=False,
            labelbottom=False,
            top=True,
            labeltop=True,
        )
        plt.xlabel(r"$O-C$ error", fontsize=12)
        a.xaxis.set_label_position("top")
        plt.xlim(min(errs) * 0.6, max(errs) * 1.4)

    textstr = "\n".join(
        (
            r"Cepheid: " + name.replace("_", " "),
            r"$P=%.4f$ d" % (P,),
            r"HJD$_{ref}=%.3f$" % (refepoch,),
        )
    )
    props = dict(boxstyle="round", facecolor="lightblue", alpha=0.7)

    f.text(0.52, 0.86, textstr, verticalalignment="top", fontsize=13, bbox=props)

    if save:
        plt.savefig(indir + name + "_O-C_" + mode + ".png", bbox_inches="tight")

    return ax


def remove_OC_breaks(indir, o_data, P, size):
    all_dat = create_all_datafile(indir, 0, size)
    n_data = all_dat[:, 0:3]
    for i in range(len(o_data)):
        n_data = numpy.append(n_data, numpy.loadtxt(indir + o_data[i]), axis=0)
    sn_data = numpy.array(sorted(n_data, key=lambda a_entry: a_entry[0]))

    new_data = sn_data
    begin_sl = 0
    end_sl = 0
    j = 0
    while j < (sn_data.shape[0] - 1):

        if (
            numpy.ediff1d(sn_data[:, 2])[j] > P / 2 and begin_sl == 0
        ):  # this works for the first few
            new_data[:j, 2] += P
            begin_sl = 1

        elif (
            numpy.ediff1d(sn_data[:, 2])[j] < 0
            and abs(numpy.ediff1d(sn_data[:, 2])[j]) > P / 2
            and all(numpy.ediff1d(abs(sn_data[:, 2])[j : j + 6]) < P / 2)
            and end_sl < 2
        ):

            new_data[j + 1 :, 2] += P
            end_sl += 1

            j += 15
            continue

        j += 1

    for j in range(new_data.shape[0]):
        if (
            abs(new_data[j, 2] - new_data[j - 4, 2]) > P / 2
            and abs(new_data[j, 2] - new_data[j + 4, 2]) > P / 2
        ):
            new_data[j, 2] += -1 * numpy.sign(new_data[j, 2] - new_data[j - 4, 2]) * P

    return new_data


def update_files(indir, all_data):

    for i in os.listdir(indir):
        if (
            os.path.isfile(os.path.join(indir, i))
            and i.startswith("OC")
            and (i.endswith("weights.dat") is False)
        ):
            print(i)
            data = numpy.loadtxt(indir + i)
            k = 0
            j = 0
            try:
                while j < (all_data.shape[0]) and k < (data.shape[0]):
                    if all_data[j, 0] == data[k, 0]:
                        data[k, 2] = all_data[j, 2]
                        k += 1
                    j += 1
            except IndexError:
                while j < (all_data.shape[0]):
                    if all_data[j, 0] == data[0]:
                        data[2] = all_data[j, 2]
                    j += 1
            numpy.savetxt(
                indir + "filt_" + i, data, fmt="%10f", delimiter="\t", newline=" \n"
            )


def give_size(x):
    s = 45 * numpy.exp(-70 * x)
    if isinstance(s, float):
        if s <= 0.5:
            s = 0.5
    else:
        s[~(s > 0.5)] = 0.5
    return s


def create_str(z):
    string = ""
    for i in range(z.shape[0]):
        if z[i] > 0 or z[i] == 0:
            if i == 0:
                pass
            else:
                string += " + "
        else:
            string += " - "

        if z.shape[0] - i - 1 > 1:
            string += (
                str(abs(z[i]) * 10 ** int((-numpy.floor(numpy.log10(abs(z[i]))))))[0:5]
                + " $\cdot$ "
                + r"$10^{"
                + str(int((numpy.floor(numpy.log10(abs(z[i]))))))
                + "}$"
                + r"$x^{"
                + str(z.shape[0] - i - 1)
                + "}$"
            )
        elif z.shape[0] - i - 1 == 1:
            string += (
                str(abs(z[i]) * 10 ** int((-numpy.floor(numpy.log10(abs(z[i]))))))[0:5]
                + " $\cdot$ "
                + r"$10^{"
                + str(int((numpy.floor(numpy.log10(abs(z[i]))))))
                + "}$"
                + r"$x$"
            )
        else:
            string += (
                str(abs(z[i]) * 10 ** int((-numpy.floor(numpy.log10(abs(z[i]))))))[0:5]
                + " $\cdot$ "
                + r"$10^{"
                + str(int((numpy.floor(numpy.log10(abs(z[i]))))))
                + "}$"
            )

        if i == 2:
            string += " \n                 "
    return string


def sin_lin(x, a, b, c, d, e):
    return a * numpy.sin(x * b + c) + d * x + e


def sin_par(x, a, b, c, d, e, f):
    return a * numpy.sin(x * b + c) + d * x ** 2 + e * x + f


def create_OC_with_fit(
    mode,
    surveys,
    P,
    csillag,
    y_limits,
    indir,
    refepoch,
    old_data=None,
    offs=0,
    save=False,
    prefix="",
    order=4,
    modif=None,
    txt="OFF",
    special=None,
    NoErr=False,
    sinit=None,
    f=None,
    gridss=False,
    onec=False,
    sp_n=None,
    SizeLabelPlot=True,
):

    all_data = create_all_datafile(indir, offs, P, prefix)

    if mode == "HJD":
        m = 0
    elif mode == "Epoch":
        m = 1
    else:
        print("Wrong mode flag given")
        return ()
    if gridss:
        ax = create_OC_plot(
            mode,
            surveys,
            P,
            csillag,
            y_limits,
            indir,
            refepoch,
            old_data,
            offs,
            save=False,
            prefix=prefix,
            onec=onec,
            gridss=gridss,
            f=f,
            sp_n=sp_n,
            SizeLabelPlot=SizeLabelPlot,
        )
    else:
        ax = create_OC_plot(
            mode,
            surveys,
            P,
            csillag,
            y_limits,
            indir,
            refepoch,
            old_data,
            offs,
            save=False,
            prefix=prefix,
            onec=onec,
            SizeLabelPlot=SizeLabelPlot,
        )
    ax.ticklabel_format(style="plain")
    if special == "Sine":
        if sinit is not None:
            popt, pcov = curve_fit(sin_lin, all_data[:, m], all_data[:, 2], p0=sinit)
        else:
            popt, pcov = curve_fit(
                sin_lin,
                all_data[:, m],
                all_data[:, 2],
                p0=[
                    0.2,
                    2 * numpy.pi * 2 / (max(all_data[:, m]) - min(all_data[:, m])),
                    0,
                    0,
                    0,
                ],
            )
        timett = numpy.linspace(min(all_data[:, m]), max(all_data[:, m]), 6000)
        ax.plot(timett, sin_lin(timett, *popt), "k:", lw=1.2, label="Lin+Sine")
        print(popt)

        if save:
            plt.savefig(
                indir + csillag + "_O-C_" + mode + "_fitted_sine.png",
                bbox_inches="tight",
            )

    elif special is not None and "SinePar" in special:
        if sinit is not None:
            popt, pcov = curve_fit(
                sin_par, all_data[:, m], all_data[:, 2], p0=sinit, sigma=all_data[:, 3]
            )
        else:
            popt, pcov = curve_fit(
                sin_par,
                all_data[:, m],
                all_data[:, 2],
                p0=[
                    0.2,
                    2 * numpy.pi * 2 / (max(all_data[:, m]) - min(all_data[:, m])),
                    0,
                    0,
                    0,
                    0,
                ],
                sigma=all_data[:, 3],
            )
        timett = numpy.linspace(
            min(all_data[:, m]) - 3000, max(all_data[:, m]) + 3000, 7000
        )
        if "Evo" in special:
            ax.plot(timett, numpy.polyval(popt[-3:], timett), "k-", lw=1.5)
        else:
            ax.plot(timett, sin_par(timett, *popt), "k:", lw=1.2, label="Par+Sine")
        print(popt)

        if save:
            plt.savefig(
                indir + csillag + "_O-C_" + mode + "_fitted_sinepar.png",
                bbox_inches="tight",
            )
        if mode == "HJD":
            ax.set_xlim(modif[0]["Range"][0], modif[0]["Range"][1])
        else:
            ax.set_xlim(
                change_h_e(modif[0]["Range"][0], all_data),
                change_h_e(modif[0]["Range"][1], all_data),
            )

    else:
        if NoErr:
            z = numpy.polyfit(all_data[:, m], all_data[:, 2], order)
        else:
            z = numpy.polyfit(
                all_data[:, m], all_data[:, 2], order, w=give_size(all_data[:, 3] / P)
            )
        print(z)
        if modif is None:
            timett = numpy.linspace(min(all_data[:, m]), max(all_data[:, m]), 6000)
            ax.plot(
                timett, numpy.polyval(z, timett), "k:", lw=1.2, label="4th order fit"
            )

        else:
            if mode == "HJD":
                for i in range(len(modif) - 1):
                    timett = numpy.linspace(
                        modif[i]["Range"][0], modif[i]["Range"][1], 6000
                    )
                    ax.plot(
                        timett, numpy.polyval(z, timett), modif[i]["LineType"], lw=1.2
                    )
                    if i == 0:
                        ax.plot(
                            timett,
                            numpy.polyval(z, timett),
                            modif[i]["LineType"],
                            lw=1.2,
                            label="4th order fit",
                        )

                if txt == "ON":
                    ax.text(
                        modif["Text"][0],
                        modif["Text"][1],
                        "$p_{\mathrm{fitted}}(x)=$" + create_str(z),
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        color="black",
                        fontsize=12,
                        bbox=dict(ec="w", fc="w"),
                    )
                ax.set_xlim(modif[0]["Range"][0], modif[0]["Range"][1])

            else:
                for i in range(len(modif) - 1):
                    timett = numpy.linspace(
                        change_h_e(modif[i]["Range"][0], all_data),
                        change_h_e(modif[i]["Range"][1], all_data),
                        6000,
                    )
                    ax.plot(
                        timett, numpy.polyval(z, timett), modif[i]["LineType"], lw=1.2
                    )
                    if i == 0:
                        ax.plot(
                            timett,
                            numpy.polyval(z, timett),
                            modif[i]["LineType"],
                            lw=1.2,
                            label="4th order fit",
                        )

                if txt == "ON":
                    ax.text(
                        change_h_e(modif["Text"][0], all_data),
                        modif["Text"][1],
                        "$p_{\mathrm{fitted}}(x)=$" + create_str(z),
                        verticalalignment="bottom",
                        horizontalalignment="left",
                        color="black",
                        fontsize=12,
                        bbox=dict(ec="w", fc="w"),
                    )
                ax.set_xlim(
                    change_h_e(modif[0]["Range"][0], all_data),
                    change_h_e(modif[0]["Range"][1], all_data),
                )
        if save:
            plt.savefig(
                indir + csillag + "_O-C_" + mode + "_fitted.png", bbox_inches="tight"
            )

    try:
        return popt
    except UnboundLocalError:
        return z


def setup_modif_dict(ranges, types, text_coords):
    modif_dict = {}
    temp = {}
    try:
        tv = ranges.shape[1]
        for i in range(ranges.shape[0]):
            temp = {"Range": ranges[i], "LineType": types[i]}
            modif_dict[i] = temp
    except IndexError:
        modif_dict[0] = {"Range": ranges, "LineType": types}
    modif_dict["Text"] = text_coords
    return modif_dict


def create_OC_residual_plot(
    mode,
    surveys,
    P,
    name,
    y_limits,
    indir,
    resid_polin,
    o_data=None,
    errs=None,
    old_data=None,
    trunc_date=False,
    offs=0,
    save=False,
    prefix="",
    gridss=False,
    subplt=None,
    f=None,
    sp_n=None,
):
    OC_datas = {}
    if mode == "HJD":
        m = 0
    elif mode == "Epoch":
        m = 1
    else:
        print("Wrong mode flag given")
        return ()

    f_index = len(surveys)

    if trunc_date and mode == "HJD":
        trunc = 2400000
    else:
        trunc = 0

    if gridss:
        if sp_n is not None:
            ax = f.add_subplot(subplt[sp_n])
        else:
            ax = f.add_subplot(subplt[1])
    else:
        f = plt.figure(figsize=(12, 8))

    cust = "o"
    for plot_index in range(f_index):

        OC_datas[plot_index] = numpy.loadtxt(
            indir + prefix + "OC_f_" + surveys[plot_index] + ".dat", comments="#"
        )

        if plot_index > 9:
            cust = "^"

        try:
            plt.errorbar(
                OC_datas[plot_index][:, m] - trunc,
                OC_datas[plot_index][:, 2]
                - numpy.polyval(resid_polin, OC_datas[plot_index][:, m]),
                OC_datas[plot_index][:, -1],
                marker=cust,
                markeredgecolor="black",
                markeredgewidth=0.4,
                ls="",
                label=surveys[plot_index],
                ms=3.5,
                capsize=2,
                capthick=0.5,
                linewidth=0.5,
            )

        except IndexError:

            plt.errorbar(
                OC_datas[plot_index][m] - trunc,
                OC_datas[plot_index][2]
                - numpy.polyval(resid_polin, OC_datas[plot_index][m]),
                OC_datas[plot_index][-1],
                marker=cust,
                markeredgecolor="black",
                markeredgewidth=0.4,
                ls="",
                label=surveys[plot_index],
                ms=3.5,
                capsize=2,
                capthick=0.5,
                linewidth=0.5,
            )

    dl = 0
    if old_data is not None:
        if os.path.isfile(
            indir + old_data[0][:-9] + "_weights.dat"
        ):  # old_data[0][3:11]=='Szabados':
            OC_old_data2 = numpy.loadtxt(indir + prefix + old_data[0], comments="#")
            old_data_weight = numpy.loadtxt(
                indir + o_data[0][:-9] + "_weights.dat", comments="#"
            )
            slit = 0
            for i in range(OC_old_data2.shape[0]):
                if old_data_weight[i] != 0:

                    plt.errorbar(
                        OC_old_data2[i, m] - trunc,
                        OC_old_data2[i, 2]
                        - numpy.polyval(resid_polin, OC_old_data2[i, m])
                        + offs,
                        OC_old_data2[i, -1],
                        marker=cust,
                        color="navy",
                        markeredgecolor="black",
                        markeredgewidth=0.4,
                        ls="",
                        ms=3.5,
                        capsize=2,
                        capthick=0.5,
                        linewidth=0.5,
                    )
                    if slit == 0:

                        plt.errorbar(
                            OC_old_data2[i, m] - trunc,
                            OC_old_data2[i, 2]
                            - numpy.polyval(resid_polin, OC_old_data2[i, m])
                            + offs,
                            OC_old_data2[i, -1],
                            marker=cust,
                            color="navy",
                            markeredgecolor="black",
                            markeredgewidth=0.4,
                            ls="",
                            ms=3.5,
                            capsize=2,
                            capthick=0.5,
                            linewidth=0.5,
                        )
                        slit += 1
                else:
                    plt.errorbar(
                        OC_old_data2[i, m] - trunc,
                        OC_old_data2[i, 2]
                        - numpy.polyval(resid_polin, OC_old_data2[i, m])
                        + offs,
                        OC_old_data2[i, -1],
                        marker=cust,
                        color="navy",
                        markeredgecolor="black",
                        markeredgewidth=0.4,
                        ls="",
                        ms=3.5,
                        capsize=2,
                        capthick=0.5,
                        linewidth=0.5,
                    )

            dl = 1
        elif os.path.isfile(indir + "Errvect_" + old_data[0][3:-9] + ".dat"):
            OC_old_data2 = numpy.loadtxt(indir + prefix + old_data[0], comments="#")
            old_data_err = numpy.loadtxt(
                indir + "Errvect_" + o_data[0][3:-9] + ".dat", comments="#"
            )
            plt.errorbar(
                OC_old_data2[:, m] - trunc,
                OC_old_data2[:, 2] - numpy.polyval(resid_polin, OC_old_data2[:, m]),
                old_data_err,
                marker=cust,
                ls="",
                label=surveys[plot_index],
                ms=3.5,
                capsize=2,
                capthick=0.5,
                linewidth=0.5,
            )
        else:
            for i in range(dl, len(old_data)):
                OC_old_data = numpy.loadtxt(indir + prefix + old_data[i], comments="#")
                plt.errorbar(
                    OC_old_data[:, m] - trunc,
                    OC_old_data[:, 2]
                    - numpy.polyval(resid_polin, OC_old_data[:, m])
                    + offs,
                    OC_old_data[:, -1],
                    marker=cust,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                    ls="",
                    ms=3.5,
                    capsize=2,
                    capthick=0.5,
                    linewidth=0.5,
                )

    temp_d = create_all_datafile(indir, 0, P)
    differ = numpy.max(temp_d[:, m]) - numpy.min(temp_d[:, m])
    plt.xlim(
        numpy.min(temp_d[:, m]) - 0.05 * differ - trunc,
        numpy.max(temp_d[:, m]) + 0.05 * differ - trunc,
    )

    if trunc_date and mode == "HJD":
        plt.xlabel(mode + "-2400000", fontsize=15)
    else:
        plt.xlabel(mode, fontsize=15)
    plt.ylabel("$O-C_{res}$ (days)", fontsize=15)
    if gridss:
        ax.tick_params(axis="y", labelsize=12)
        ax.tick_params(axis="x", labelsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(y_limits[0], y_limits[1])
    # xlim(2425000,2460000)
    plt.ticklabel_format(style="plain")
    if save:
        plt.savefig(indir + name + "_O-C_residual" + mode + ".png", bbox_inches="tight")


def calc_phase(data, P, ref_ep):
    phi = numpy.zeros((data.shape[0], 1))
    for i in range(data.shape[0]):
        phi[i] = (data[i, 0] - ref_ep) / P - int((data[i, 0] - ref_ep) / P)
    return phi
