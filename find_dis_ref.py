# those functions are used to find all the references and distortions files for Light_GraphicsLPIPS_csv.py
import os
def find_ref_csvfiles(root_refPatches): 
    # Root_ref_patches = '..../out/dataset_ref_method_XVP_NNNN/'
    # The objective is to find all the .csv files of the patchified reference images
    # We return a list of the paths of the .csv files
    # The architecture of the folders is as follows:
    
    # root_refPatches
    # ├── obj_1
    # │   ├── views
    # │   │   ├── view_1.jpg
    # │   │   ├── view_2.jpg
    # │   │   ├── ...
    # │   ├── masks
    # |   ├── |── view_1.jpg
    # │   ├── |── view_2.jpg
    # │   ├── |── ...
    # │   ├── patches
    # │   │   ├── obj_1_patchlist.csv
    # │   │   ├── view_1_patchified.jpg
    # │   │   ├── view_2_patchified.jpg
    # │   │   ├── ...
    # ├── obj_2
    #...

    ref_csv_files = []
    for root, dirs, files in os.walk(root_refPatches):
        for file in files:
            if file.endswith('.csv'):
                ref_csv_files.append(os.path.join(root, file).replace('\\', '/'))
    return ref_csv_files
def find_dis_files(root_disPatches, ref_obj_name):

    # find all the distorted obj names related to ref_obj_name
    # Returns a list of the names of the distorted obj
    
    
    # The architecture of the folders is as follows:
    # root_distPatches
    # ├── obj_1_dis_1
    # │   ├── views
    # │   │   ├── view_1.jpg
    # │   │   ├── view_2.jpg
    # │   │   ├── ...
    # │   ├── masks
    # |   ├── |── view_1.jpg
    # │   ├── |── view_2.jpg
    # │   ├── |── ...
    # ├── obj_1_dis_2
    #...

    # We get the list of the names of the objects in the root_distPatches folder
    # So we find the name of the distorted objects by searching the name of the reference object in the root_distPatches folder.
    # e.g. :
    # ref_img : Orbiter_Space_Shuttle_OV-103_Discovery-150k-4096
    # dis_img : Orbiter_Space_Shuttle_OV-103_Discovery-150k-4096_simpL1_qp10_qt8_decompJPEG_512x512_Q75
 
    # We get the name of the reference object
    # We search for the name of the distorted objects in the root_distPatches folder
    # We take any folder that contains the name of the reference object and is a directory
    dis_files = [f for f in os.listdir(root_disPatches) if ref_obj_name in f and os.path.isdir(os.path.join(root_disPatches, f))]
    # dis_files might be a list of lists, really big size. around 3000 elements for Yana's Database.
    return dis_files

def find_ref_files(root_refPatches):
    # find all the reference obj names related to ref_obj_name
    # Returns a list of the names of the reference obj
        
    
    # The architecture of the folders is as follows:
    # root_refPatches
    # ├── obj_1
    # │   ├── views
    # │   │   ├── view_1.jpg
    # │   │   ├── view_2.jpg
    # │   │   ├── ...
    # │   ├── masks
    # |   ├── |── view_1.jpg
    # │   ├── |── view_2.jpg
    # │   ├── |── ...
    # ├── obj_2
    #...

    # We only need to get the folder names of the root_refPatches folder

    ref_files = []
    for root, dirs, files in os.walk(root_refPatches):
        for dir in dirs:
            # Only take first subdirectories
            if os.path.dirname(root) != root_refPatches:
                ref_files.append(dir)
    return ref_files
