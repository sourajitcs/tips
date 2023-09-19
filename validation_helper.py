import shutil, os, time
from tqdm import tqdm
import sys, argparse

def change_class_indices(data_directory):
    
    parent_dir = data_directory
    already_found_data_path = parent_dir + "val"
    existing_data_path = parent_dir + "val_original"
    new_data_path = parent_dir + "val"
    start = time.time()
    os.rename(already_found_data_path, existing_data_path)
    with open(existing_data_path+"/val_annotations.txt", 'r') as file:
        for line in tqdm(file):
            line_splitted = line.split()
            destination_path = new_data_path+"/"+str(line_splitted[1])
            source_path = existing_data_path+"/images/"+str(line_splitted[0])
            if not os.path.exists(destination_path):
                os.makedirs(destination_path)
            shutil.copyfile(source_path, destination_path+"/"+str(line_splitted[0]))
    end = time.time()
    lapse = end-start
    print ("####        ...      Finished     ...        ####")
    print ("  >>>> File backed up at: ", existing_data_path)
    print ("####    Job completed in ", lapse, " sec.    ####")
    


if __name__ == "__main__":
    change_class_indices(str(sys.argv[-1:][0]))

