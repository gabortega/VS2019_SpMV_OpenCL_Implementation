import sys
import csv
import math
import parser_config as cfg

def convert_to_num(i):
    try:
        return int(i)
    except ValueError:
        return float(i)

def main():
    args = len(sys.argv) - 1
    if args > 0:
        local_mem = []
        real_local_mem = []
        header = ["Local mem.\\Workgroup size"]
        kernels = []
        average_runtimes = []
        average_gflops = []
        average_gbps = []
        average_cpwi = []
        delimiter = cfg.delimiter
        decimal_separator = cfg.decimal_separator

        f1 = open(sys.argv[1],"r")

        f1_lines = f1.readlines()
        for line in f1_lines:
            if "OCCUPANCY_WORKGROUP_SIZE" in line:
                tmp_line = line.split()[2]
                if not tmp_line in header:
                    header.append(tmp_line)
                    average_runtimes.append([])
                    average_gflops.append([])
                    average_gbps.append([])
                    average_cpwi.append([])
                    real_local_mem.append([])

            elif "OCCUPANCY_LOCAL_MEM_SIZE" in line:
                tmp_line = line.split()[2]
                if not tmp_line in local_mem:
                    local_mem.append(tmp_line)

                average_runtimes[-1].append([])
                average_gflops[-1].append([])
                average_gbps[-1].append([])
                average_cpwi[-1].append([])
                real_local_mem[-1].append([])
            elif "-- STARTING " in line:
                tmp_line = line.replace("-- STARTING ", "").replace(" OPERATION --", "").strip()
                if not tmp_line in kernels:
                    kernels.append(tmp_line)
            elif "A work-group uses" in line:
                tmp_line = line.split()[4]
                real_local_mem[-1][-1].append(tmp_line)
            elif "Average time" in line:
                tmp_line = line.split()
                average_runtimes[-1][-1].append(tmp_line[2].replace(".",decimal_separator))
                average_gflops[-1][-1].append(tmp_line[8].replace(".",decimal_separator))
                average_gbps[-1][-1].append(tmp_line[14].replace(".",decimal_separator))
                average_cpwi[-1][-1].append(tmp_line[-1].replace(".",decimal_separator))
        f1.close()


        for kernel in kernels:
            kernel_index = kernels.index(kernel)
            with open(sys.argv[1] + "." + kernel.replace(" ", "_") + ".average_runtimes.csv", "w", newline="") as f2:
                writer = csv.writer(f2, delimiter=delimiter)
                writer.writerow(header)
                for i in range(len(local_mem)):
                    line = [local_mem[i]]
                    for j in range(len(header) - 1):
                        line.append(average_runtimes[j][i][kernel_index])
                    writer.writerow(line)
            with open(sys.argv[1] + "." + kernel.replace(" ", "_") + ".average_gflops.csv", "w", newline="") as f3:
                writer = csv.writer(f3, delimiter=delimiter)
                writer.writerow(header)
                for i in range(len(local_mem)):
                    line = [local_mem[i]]
                    for j in range(len(header) - 1):
                        line.append(average_gflops[j][i][kernel_index])
                    writer.writerow(line)
            with open(sys.argv[1] + "." + kernel.replace(" ", "_") + ".average_gbps.csv", "w", newline="") as f4:
                writer = csv.writer(f4, delimiter=delimiter)
                writer.writerow(header)
                for i in range(len(local_mem)):
                    line = [local_mem[i]]
                    for j in range(len(header) - 1):
                        line.append(average_gbps[j][i][kernel_index])
                    writer.writerow(line)
            with open(sys.argv[1] + "." + kernel.replace(" ", "_") + ".average_cpwi.csv", "w", newline="") as f5:
                writer = csv.writer(f5, delimiter=delimiter)
                writer.writerow(header)
                for i in range(len(local_mem)):
                    line = [local_mem[i]]
                    for j in range(len(header) - 1):
                        line.append(average_cpwi[j][i][kernel_index])
                    writer.writerow(line)
            with open(sys.argv[1] + "." + kernel.replace(" ", "_") + ".local_mem_sizes.csv", "w", newline="") as f6:
                writer = csv.writer(f6, delimiter=delimiter)
                writer.writerow(header)
                for i in range(len(local_mem)):
                    line = [local_mem[i]]
                    for j in range(len(header) - 1):
                        line.append(real_local_mem[j][i][kernel_index])
                    writer.writerow(line)
    else:
        print("Please specify filename (including path) as first argument.\n\nEx: ./SUITE/GLOBAL/WARP SIZE/output.txt\nThis opens file './SUITE/GLOBAL/NORMAL/output.txt'.")
        return

if __name__== "__main__":
  main()
