import numpy as np
import cv2
import os
from tqdm import tqdm
from einops import rearrange, reduce, repeat

class Data:
    def __init__(self, workPath, imagePath):
        self.dataPath = os.path.join(workPath[0],imagePath) #image path
        self.imagePath = imagePath


    def unison_shuffled_copies_4(self,a , b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]
    
    def load(self,trainRatio=0.8,testRatio=0.2):
        arr_img = []
        arr_label = []
        arr_Namelabel = []
        self.count = 0
        for i, (dirpath, dirnames, filenames) in tqdm(enumerate(os.walk(self.imagePath)), desc= "Loading Image Data"):
            #print('{} {} {}'.format(repr(dirpath), repr(dirnames), repr(filenames)))
            #print(i)
            if dirpath is not self.imagePath:
                dirpath_components = dirpath.split("/")
                arr_img = []
                arr_label = []
                semantic_label = dirpath_components[-1]
                
                _, label = os.path.split(semantic_label)

                #print("\nProcessing {}, {}".format(semantic_label,i))
                arr_Namelabel.append(label)
                self.count = 0

                for f in filenames:
                    #load images
                    file_path = os.path.join(dirpath, f)
                    #print(file_path)
                    img = cv2.imread(file_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = rearrange(img, ' h w c ->  c h w ')
                    #print("---",img.shape,file_path)

                    arr_img.append(img)
                    arr_label.append(i-1)
                    self.count+=1
                    
                dataset = np.array(arr_img)
                #print(np.array(arr_img).shape)
                label = np.array(arr_label)
                dataset, label = self.unison_shuffled_copies_4(dataset, label)

        
                self.labelName = np.array(arr_Namelabel)
                #print(self.labelName)
                self.jum_kelas = len(self.labelName)
                n_train = (int(self.count * trainRatio) ) 
                n_test = (int(self.count * testRatio ) ) - 1

                if not hasattr(self, 'trainSet'):
                    self.trainSet = dataset[0:n_train,:,:,:]
                    self.trainLabel = label[0:n_train,]
                    self.testSet = dataset[n_train:,:,:,:]
                    self.testLabel = label[n_train:,]
                else:
                    self.trainSet = np.concatenate((self.trainSet, dataset[0:n_train,:,:,:]), axis = 0)
                    self.trainLabel = np.concatenate((self.trainLabel, label[0:n_train,]), axis = 0)
                    self.testSet = np.concatenate((self.testSet, dataset[n_train:,:,:,:]), axis = 0)
                    self.testLabel = np.concatenate((self.testLabel, label[n_train:,]), axis = 0)

        self.trainSet, self.trainLabel = self.unison_shuffled_copies_4(self.trainSet, self.trainLabel)
        self.testSet, self.testLabel  = self.unison_shuffled_copies_4(self.testSet, self.testLabel)
            
        return self.trainSet, self.trainLabel, self.testSet, self.testLabel
                

#mainPath = os.path.dirname(os.path.abspath(__file__)) #file path main.py
#workPath = os.path.split(mainPath) #path working folder (whole file project)
#imagePath = "data_jepun"
#data = Data(workPath,imagePath)
#trainSet, trainLabel, testSet, testLabel = data.load(trainRatio=0.8,testRatio=0.2)

#print("ts",trainSet.shape)
#print("tl",trainLabel.shape)
#print("tts",testSet.shape)
#print("ttl",testLabel.shape)
