import argparse
from glob import glob
import os
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_name", required=True, help="name of your output file")
parser.add_argument("-d", "--data_folder", required=True, help="path to folder containing wav files")
args = parser.parse_args()

print (args.output_name)
print (args.data_folder)
assert os.path.isdir(args.data_folder)


# print (os.listdir(args.data_folder))
print (glob(args.data_folder + "/*.wav"))
# for file in os.listdir(args.data_folder):
#     if file.endswith(".wav"):

#         print(os.path.join( os.getcwd(), args.data_folder, file))
#         # print(os.getcwd())