import sys
import csv
import math

def get_percentile(data, percentile):
    index = len(data) * percentile / 100
    if float(index).is_integer():
        index = int(index)
        return ((data[index] + data[min(index + 1, len(data) - 1)]) / 2)
    else:
        index = int(math.ceil(index))
        return data[index]

def convert_to_num(i):
    try:
        return int(i)
    except ValueError:
        return float(i)

def main():
    args = len(sys.argv) - 1
    if args > 0:
        matrices = []
        percentiles = []
        header = ["Matrix"]
        runtimes = []
        average_runtimes = []
        percentile_runtimes = {}
        delimiter = ";"
        decimal_separator = ","

        f1 = open(sys.argv[1],"r")

        for percentile in sys.argv[2:]:
            if percentile.isdigit():
                num = convert_to_num(percentile)
                percentiles.append(num)
                percentile_runtimes[num] = []

        if args == 1:
            f1_lines = f1.readlines()
            for line in f1_lines:
                if "LOADING INPUT FILE" in line:
                    tmp_line = line.split("/")
                    tmp_line = (tmp_line[len(tmp_line) - 1].split("."))[0]
                    matrices.append(tmp_line)
                    average_runtimes.append([])
                elif "-- STARTING " in line:
                    tmp_line = line.replace("-- STARTING ", "").replace(" OPERATION --", "").strip()
                    if not tmp_line in header:
                        header.append(tmp_line)
                elif "Average time" in line:
                    tmp_line = line.split()
                    average_runtimes[len(average_runtimes) - 1].append(tmp_line[2].replace(".",decimal_separator))
            f1.close()
        elif args > 1 and len(percentiles) > 0:
            f1_lines = f1.readlines()
            for line in f1_lines:
                if "LOADING INPUT FILE" in line:
                    tmp_line = line.split("/")
                    tmp_line = (tmp_line[len(tmp_line) - 1].split("."))[0]
                    matrices.append(tmp_line)
                    average_runtimes.append([])

                    for percentile in percentiles:
                        percentile_runtimes[percentile].append([])
                elif "-- STARTING " in line:
                    tmp_line = line.replace("-- STARTING ", "").replace(" OPERATION --", "").strip()
                    if not tmp_line in header:
                        header.append(tmp_line)
                elif "Run:" in line:
                    tmp_line = line.split()
                    runtimes.append(convert_to_num(tmp_line[5]))
                elif "Average time" in line:
                    runtimes.sort()
                    for percentile in percentiles:
                        percentile_runtimes[percentile][len(percentile_runtimes[percentile]) - 1].append(get_percentile(runtimes, percentile))

                    tmp_line = line.split()
                    average_runtimes[len(average_runtimes) - 1].append(tmp_line[2].replace(".",decimal_separator))

                    runtimes.clear()
            f1.close()

        with open(sys.argv[1] + ".average_runtimes.csv", "w", newline="") as f2:
            writer = csv.writer(f2, delimiter=delimiter)
            writer.writerow(header)
            for i in range(len(matrices)):
                line = average_runtimes[i]
                line.insert(0, matrices[i])
                writer.writerow(line)

        if args > 1:
            for percentile in percentiles:
                with open(sys.argv[1] + ".percentile_" + str(percentile) + "_runtimes.csv", "w", newline="") as f2:
                    writer = csv.writer(f2, delimiter=delimiter)
                    writer.writerow(header)
                    for i in range(len(matrices)):
                        line = list(str(x).replace(".",decimal_separator) for x in percentile_runtimes[percentile][i])
                        line.insert(0, matrices[i])
                        writer.writerow(line)
    else:
        print("!!!IMPORTANT: Should only be used in WARP_SIZE tests!!!\n\nPlease specify filename (including path) as first argument.\nRemaining arguments should only be percentiles to calculate (can be any amount of integers).\n\nEx: ./SUITE/GLOBAL/NORMAL/output.txt 5 50 95\nThis opens file './SUITE/GLOBAL/NORMAL/output.txt' and calculates 5th, 50th and 95th percentiles.")
        return

if __name__== "__main__":
  main()
