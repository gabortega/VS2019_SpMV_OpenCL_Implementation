import sys
import csv

def main():
	args = len(sys.argv) - 1
	if args > 0:
		matrices = []
		header = ["","GMVM_SEQ","GMVM","CSR_SEQ","CSR","DIA_SEQ","DIA","HDIA_SEQ","HDIA","ELL_SEQ","ELL","ELLG_SEQ","ELLG","HLL_SEQ","HLL","HYB-ELL_SEQ","HYB-ELL","HYB-ELLG_SEQ","HYB-ELLG","HYB-HLL_SEQ","HYB-HLL","JAD_SEQ","JAD"]
		runtimes = []
		gflops = []
		gbps = []
		delimiter = ";"
		decimal_separator = ","

		f1 = open(sys.argv[1],"r")

		f1_lines = f1.readlines()
		for line in f1_lines:
			if "LOADING INPUT FILE" in line:
				tmp_line = line.split("/")
				tmp_line = (tmp_line[len(tmp_line) - 1].split("."))[0]
				matrices.append(tmp_line)
				runtimes.append([])
				gflops.append([])
				gbps.append([])
			if "Average time" in line:
				tmp_line = line.split()
				runtimes[len(runtimes) - 1].append(tmp_line[2].replace(".",decimal_separator))
				gflops[len(gflops) - 1].append(tmp_line[8].replace(".",decimal_separator))
				gbps[len(gbps) - 1].append(tmp_line[14].replace(".",decimal_separator))
		f1.close()

		with open(sys.argv[1] + ".runtimes.csv", "w", newline="") as f2:
			writer = csv.writer(f2, delimiter=delimiter)
			writer.writerow(header)
			for i in range(len(matrices)):
				line = runtimes[i]
				line.insert(0, matrices[i])
				writer.writerow(line)
		with open(sys.argv[1] + ".gflops.csv", "w", newline="") as f3:
			writer = csv.writer(f3, delimiter=delimiter)
			writer.writerow(header)
			for i in range(len(matrices)):
				line = gflops[i]
				line.insert(0, matrices[i])
				writer.writerow(line)
		with open(sys.argv[1] + ".gbps.csv", "w", newline="") as f4:
			writer = csv.writer(f4, delimiter=delimiter)
			writer.writerow(header)
			for i in range(len(matrices)):
				line = gbps[i]
				line.insert(0, matrices[i])
				writer.writerow(line)

if __name__== "__main__":
  main()
