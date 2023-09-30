from test import generate_single_sequence
from tqdm import tqdm
import glob

test_list = glob.glob('Data_test/with_detr_test/*')
# test_list = glob.glob('Data_test/rawData_test_DHM2/*')
for name in tqdm(test_list):
    print(name)
    generate_single_sequence(name)
