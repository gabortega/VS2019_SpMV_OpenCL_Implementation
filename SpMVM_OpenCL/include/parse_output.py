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
        try:
            return float(i)
        except ValueError:
            return 0

def main():
    args = len(sys.argv) - 1
    if args > 0:
        matrices = []
        percentiles = []
        header = ["Matrix"]
        runtimes = []
        gflops = []
        gbps = []
        cpwi = []
        average_runtimes = []
        average_gflops = []
        average_gbps = []
        average_cpwi = []
        instruction_count = []
        percentile_runtimes = {}
        percentile_gflops = {}
        percentile_gbps = {}
        percentile_cpwi = {}
        delimiter = ";"
        decimal_separator = ","

        f1 = open(sys.argv[1],"r")

        for percentile in sys.argv[2:]:
            if percentile.isdigit():
                num = convert_to_num(percentile)
                percentiles.append(num)
                percentile_runtimes[num] = []
                percentile_gflops[num] = []
                percentile_gbps[num] = []
                percentile_cpwi[num] = []

        if args == 1:
            f1_lines = f1.readlines()
            for line in f1_lines:
                if "LOADING INPUT FILE" in line:
                    tmp_line = line.split("/")
                    tmp_line = (tmp_line[len(tmp_line) - 1].split("."))[0]
                    matrices.append(tmp_line)
                    average_runtimes.append([])
                    average_gflops.append([])
                    average_gbps.append([])
                    average_cpwi.append([])
                    instruction_count.append([])
                elif "Total kernel instructions:" in line:
                    tmp_line = line.split()
                    instruction_count[-1].append(tmp_line[-1].replace(".",decimal_separator))
                elif "-- STARTING " in line:
                    tmp_line = line.replace("-- STARTING ", "").replace(" OPERATION --", "").strip()
                    if not tmp_line in header:
                        header.append(tmp_line)
                elif "Average time" in line:
                    tmp_line = line.split()
                    average_runtimes[-1].append(tmp_line[2].replace(".",decimal_separator))
                    average_gflops[-1].append(tmp_line[8].replace(".",decimal_separator))
                    average_gbps[-1].append(tmp_line[14].replace(".",decimal_separator))
                    average_cpwi[-1].append(tmp_line[-1].replace(".",decimal_separator))
            f1.close()
        elif args > 1 and len(percentiles) > 0:
            f1_lines = f1.readlines()
            for line in f1_lines:
                if "LOADING INPUT FILE" in line:
                    tmp_line = line.split("/")
                    tmp_line = (tmp_line[len(tmp_line) - 1].split("."))[0]
                    matrices.append(tmp_line)
                    average_runtimes.append([])
                    average_gflops.append([])
                    average_gbps.append([])
                    average_cpwi.append([])

                    for percentile in percentiles:
                        percentile_runtimes[percentile].append([])
                        percentile_gflops[percentile].append([])
                        percentile_gbps[percentile].append([])
                        percentile_cpwi[percentile].append([])

                    instruction_count.append([])
                elif "Total kernel instructions:" in line:
                    tmp_line = line.split()
                    instruction_count[-1].append(tmp_line[-1].replace(".",decimal_separator))
                elif "-- STARTING " in line:
                    tmp_line = line.replace("-- STARTING ", "").replace(" OPERATION --", "").strip()
                    if not tmp_line in header:
                        header.append(tmp_line)
                elif "Run:" in line:
                    tmp_line = line.split()
                    runtimes.append(convert_to_num(tmp_line[5]))
                    gflops.append(convert_to_num(tmp_line[10]))
                    gbps.append(convert_to_num(tmp_line[15]))
                    cpwi.append(convert_to_num(tmp_line[-1]))
                elif "Average time" in line:
                    runtimes.sort()
                    gflops.sort()
                    gbps.sort()
                    cpwi.sort()
                    for percentile in percentiles:
                        percentile_runtimes[percentile][-1].append(get_percentile(runtimes, percentile))
                        percentile_gflops[percentile][-1].append(get_percentile(gflops, percentile))
                        percentile_gbps[percentile][-1].append(get_percentile(gbps, percentile))
                        percentile_cpwi[percentile][-1].append(get_percentile(cpwi, percentile))

                    tmp_line = line.split()
                    average_runtimes[-1].append(tmp_line[2].replace(".",decimal_separator))
                    average_gflops[-1].append(tmp_line[8].replace(".",decimal_separator))
                    average_gbps[-1].append(tmp_line[14].replace(".",decimal_separator))
                    average_cpwi[-1].append(tmp_line[14].replace(".",decimal_separator))

                    runtimes.clear()
                    gflops.clear()
                    gbps.clear()
                    cpwi.clear()
            f1.close()

        with open(sys.argv[1] + ".average_runtimes.csv", "w", newline="") as f2:
            writer = csv.writer(f2, delimiter=delimiter)
            writer.writerow(header)
            for i in range(len(matrices)):
                line = average_runtimes[i]
                line.insert(0, matrices[i])
                writer.writerow(line)
        with open(sys.argv[1] + ".average_gflops.csv", "w", newline="") as f3:
            writer = csv.writer(f3, delimiter=delimiter)
            writer.writerow(header)
            for i in range(len(matrices)):
                line = average_gflops[i]
                line.insert(0, matrices[i])
                writer.writerow(line)
        with open(sys.argv[1] + ".average_gbps.csv", "w", newline="") as f4:
            writer = csv.writer(f4, delimiter=delimiter)
            writer.writerow(header)
            for i in range(len(matrices)):
                line = average_gbps[i]
                line.insert(0, matrices[i])
                writer.writerow(line)
        with open(sys.argv[1] + ".average_cpwi.csv", "w", newline="") as f5:
            writer = csv.writer(f5, delimiter=delimiter)
            writer.writerow(header)
            for i in range(len(matrices)):
                line = average_cpwi[i]
                line.insert(0, matrices[i])
                writer.writerow(line)
        with open(sys.argv[1] + ".instr_count.csv", "w", newline="") as f6:
            writer = csv.writer(f6, delimiter=delimiter)
            writer.writerow(header)
            for i in range(len(matrices)):
                line = instruction_count[i]
                line.insert(0, matrices[i])
                writer.writerow(line)

        if args > 1:
            for percentile in percentiles:
                with open(sys.argv[1] + ".percentile_" + str(percentile) + "_runtimes.csv", "w", newline="") as f1:
                    writer = csv.writer(f1, delimiter=delimiter)
                    writer.writerow(header)
                    for i in range(len(matrices)):
                        line = list(str(x).replace(".",decimal_separator) for x in percentile_runtimes[percentile][i])
                        line.insert(0, matrices[i])
                        writer.writerow(line)
                with open(sys.argv[1] + ".percentile_" + str(percentile) + "_gflops.csv", "w", newline="") as f2:
                    writer = csv.writer(f2, delimiter=delimiter)
                    writer.writerow(header)
                    for i in range(len(matrices)):
                        line = list(str(x).replace(".",decimal_separator) for x in percentile_gflops[percentile][i])
                        line.insert(0, matrices[i])
                        writer.writerow(line)
                with open(sys.argv[1] + ".percentile_" + str(percentile) + "_gbps.csv", "w", newline="") as f3:
                    writer = csv.writer(f3, delimiter=delimiter)
                    writer.writerow(header)
                    for i in range(len(matrices)):
                        line = list(str(x).replace(".",decimal_separator) for x in percentile_gbps[percentile][i])
                        line.insert(0, matrices[i])
                        writer.writerow(line)
                with open(sys.argv[1] + ".percentile_" + str(percentile) + "_cpwi.csv", "w", newline="") as f4:
                    writer = csv.writer(f4, delimiter=delimiter)
                    writer.writerow(header)
                    for i in range(len(matrices)):
                        line = list(str(x).replace(".",decimal_separator) for x in percentile_cpwi[percentile][i])
                        line.insert(0, matrices[i])
                        writer.writerow(line)
    else:
        print("Please specify filename (including path) as first argument.\nRemaining arguments should only be percentiles to calculate (can be any amount of integers).\n\nEx: ./SUITE/GLOBAL/WARP SIZE/output.txt 5 50 95\nThis opens file './SUITE/GLOBAL/NORMAL/output.txt' and calculates 5th, 50th and 95th percentiles.")
        return

if __name__== "__main__":
  main()
