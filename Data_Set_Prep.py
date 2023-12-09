# This script is for training /test set preparation;
# set the path "./xxx" of the folder that contains all the data you want to pickle 
# put all the data in the folders with the measured beta/defocus value (label)

from NPSmethods2 import *
from RealDataPrep_V6 import *
from Mask import *
import random
import shutil

# read the original data from "path" and then pack them into "path/result" folder
def data_pickle(path,final_imgsize=(49,49),train_test_ratio = 0.8,numOfImgsInEachRun = 2,trapRegion = (slice(50, 250), slice(50, 250)),noiseRegion = (slice(0, 300), slice(0, 300))):
    #Default settings
    parameter = 0
    imgSysData = {
        "CCDPixelSize": 13,  # pixel size of the CCD, in micron
        "magnification": 27,  # 799.943 / 29.9099, # magnification of the imaging system
        "wavelen": 0.852,  # wavelength of the imaging beam, in micron
        "NA": 0.6,  # numerical aperture of the objective
        "ODtoAtom": 13
    }

    # -------------------codes
    print(os.listdir('.'))
    raw_folder_list = []

    subdir_folders = []
    for sub_dir in os.listdir(path):
            if sub_dir.isdigit():
                subdir_folders.append(sub_dir)


    data_input = []
    data_output = []
    with open(path + "/beta.txt") as file:
        for beta in file.readlines():
            data_output.append(float(beta))
    data_output = cont_to_discrete(data_output)


    for subdir in subdir_folders:
        temp_path = path + '/' + subdir
        print(temp_path)
        M2k_Exp,_,_,_,_,_,_,_ = \
            calcNPS(temp_path, numOfImgsInEachRun, parameter,trapRegion,noiseRegion,imgSysData=imgSysData)
        if 1:
            #M2k_Exp = apply_mask(mask_dark_center(M2k_Exp.shape[0],17),M2k_Exp)  
            #M2k_Exp = apply_mask(mask_grad(M2k_Exp.shape[0], decay_rate=0.6),M2k_Exp)
            M2k_Exp = apply_mask(mask_poly(M2k_Exp.shape[0], N=1.1),M2k_Exp) 
            cutting_ratio = 0.5
            cx = M2k_Exp.shape[1]/2
            edg_x = cutting_ratio*M2k_Exp.shape[1]
            x1 = int(cx - edg_x/2)
            x2 = int(cx + edg_x/2)
            cy = M2k_Exp.shape[0]/2
            edg_y = cutting_ratio*M2k_Exp.shape[0]
            y1 = int(cy - edg_x/2)
            y2 = int(cy + edg_x/2)
            M2k_Exp = M2k_Exp[y1:y2,x1:x2]
        data_input.append(M2k_Exp)

    temp = list(zip(data_input,data_output))
    random.shuffle(temp)
    data_input,data_output = zip(*temp)
    L = len(data_input)
    train_in = np.array(data_input[:int(train_test_ratio*L)])
    test_in = np.array(data_input[int(train_test_ratio*L):])
    train_out = np.array(data_output[:int(train_test_ratio*L)])
    test_out = np.array(data_output[int(train_test_ratio*L):])
    curr_time_str = datetime.today().strftime("%b_%d_%H%M")

    try:
        os.makedirs(path+'/result_%s' % curr_time_str)
    except:
        print("folder exists")
    else:   
        pickle.dump(train_in,open(path + "/result_%s/train_in_%s" % (curr_time_str,curr_time_str), 'wb+'))
        pickle.dump(train_out,open(path + "/result_%s/train_out_%s" % (curr_time_str,curr_time_str), 'wb+'))
        pickle.dump(test_in,open(path + "/result_%s/test_in_%s" % (curr_time_str,curr_time_str), 'wb+'))
        pickle.dump(test_out,open(path + "/result_%s/test_out_%s" % (curr_time_str,curr_time_str), 'wb+'))
        print('Data loading sucessfully')


def NPS_show(path,numOfImgsInEachRun = 1,trapRegion= (slice(50, 250), slice(50, 250)),noiseRegion= (slice(0, 300), slice(0, 300)),Lable = 'NaN'):
        M2k_Exp,_,_,_,_,atomODAvg ,_,_ = \
        calcNPS(path, numOfImgsInEachRun,parameter=0,trapRegion=trapRegion,noiseRegion=noiseRegion,imgSysData=imgSysData)
        plt.subplot(121)
        img = plt.imshow(atomODAvg,vmax=1.2)
        plt.colorbar(img)
        plt.subplot(122)
        img = plt.imshow(M2k_Exp,vmax=0.2)
        plt.colorbar(img)
        plt.title('file ='+Lable)
        plt.show()

# Show all the OD and NPS figures at "path" one by one
def NPS_show_all(path,numOfImgsInEachRun = 1,trapRegion= (slice(50, 250), slice(50, 250)),noiseRegion= (slice(0, 300), slice(0, 300)),Lable = 'NaN'):
    for sub_dir in os.listdir(path):
        if sub_dir.isdigit():
            NPS_show(path+ '/' + sub_dir,numOfImgsInEachRun = numOfImgsInEachRun,trapRegion= trapRegion,noiseRegion=noiseRegion,Lable = sub_dir)

def file_copy(path,newpath,identifier):
    if os.path.exists(newpath):
        pass
    else:
        os.mkdir(newpath)
        try:
            para_file = open(path+'/parameters.txt')
            lines = para_file.readlines()
            lines_new = [lines[0]]
            for line in lines[1:]:
                if line.split()[0] in identifier:
                    lines_new.append(line)
            for id in identifier:
                try:
                    shutil.copy(path+'/'+id,newpath+'/'+id)
                    shutil.copy(path+'/procedure_'+id,newpath+'/procedure_'+id)
                    shutil.copy(path+'/rawimg_'+id,newpath+'/rawimg_'+id)
                except Exception as exc:
                    print(exc)
        finally:
            if para_file:
                para_file.close()
        para_file_new = open(newpath+'/parameters.txt','w')
        para_file_new.writelines(lines_new)
        para_file_new.close()

# seperate all the data under "path" into "newpath"
def file_split(path,newpath,min_grouping_number=1,extra_dataset=0):
    identifier = []
    beta_file = open(path+'/beta.txt')
    beta = beta_file.readlines()
    beta_file.close()
    if os.path.exists(newpath):
        pass
    else:
        os.mkdir(newpath)
    beta_file_new = open(newpath+'/beta.txt','w')
    new_beta = []
    for sub_dir in os.listdir(path):
        if sub_dir.isdigit():
            identifier.append(sub_dir)
    pos = -1
    for sub_dir in identifier:
        pos = pos + 1
        list_of_id = os.listdir(path+'/'+sub_dir)
        true_id = []
        for id in list_of_id:
            if id.isdigit():
                true_id.append(id)
        if len(true_id) < min_grouping_number:
            print('warning:number of data is not enough in folder ',sub_dir)
        else:
            for i in np.arange(int(len(true_id)/min_grouping_number)):
                file_copy(path+'/'+sub_dir,
                          newpath+'/'+sub_dir+str(i).zfill(4),
                          true_id[i*min_grouping_number:(i+1)*min_grouping_number]
                          )
                new_beta.append(beta[pos])
            for i in int(len(true_id)/min_grouping_number)+np.arange(extra_dataset)+1:
                file_copy(path+'/'+sub_dir,
                newpath+'/'+sub_dir+str(i).zfill(4),
                np.random.choice(true_id,min_grouping_number,replace=False)
                )
                new_beta.append(beta[pos])           
    beta_file_new.writelines(new_beta)
    beta_file_new.close()          


