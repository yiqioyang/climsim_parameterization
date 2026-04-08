import glob
import csv

inp_files_raw = glob.glob("/glade/work/qingyuany/Climsim/train/*/E3SM*mli*.nc")
out_files_raw = glob.glob("/glade/work/qingyuany/Climsim/train/*/E3SM*mlo*.nc")


inp_files = []
count = 0
for f in inp_files_raw:
    temp = f.replace("mli", "mlo")
    if temp in out_files_raw:
        inp_files.append(f)

    else:
        count += 1
        print(count)

print('Finished checking for input and output matching')

with open("/glade/work/qingyuany/Climsim/aux/train_input_files.txt", "w") as f:
    for path in inp_files:
        f.write(path + "\n")

print('Finished saving the matched input file paths to train_input_files.txt')