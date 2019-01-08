import os
from skimage import io
import numpy as np
from sklearn.model_selection import train_test_split
benign_path='D:\\Rishal\\BreaKHis_v1\\BreaKHis_v1\\histology_slides\\breast\\benign\\SOB'
benign_path_final='D:\\Rishal\\PycharmProjects\\sop\\binary_classification'
io.use_plugin('pil')
#print('lol')
for roots,dirs,files in os.walk(benign_path):
    if len(files) > 0:
        #print('lol')
        file_train, file_val=train_test_split(files, test_size=0.4, random_state=42)
        file_val, file_test=train_test_split(file_val, test_size=0.5, random_state=42)
        for file in files:
            try :
                #print('lol')
                if file in file_train:
                    if file.endswith('_norm.tif'):
                        print('lol')
                        root=roots.split('\\')[-1]
                        img = io.imread(os.path.join(roots, file))

                        io.imsave(os.path.join(benign_path_final,root,'training\\benign',file), img)
                elif file in file_test:
                    if file.endswith('_norm.tif'):
                        print('lol')
                        root = roots.split('\\')[-1]
                        img = io.imread(os.path.join(roots, file))

                        io.imsave(os.path.join(benign_path_final, root, 'test\\benign', file), img)
                else:
                    if file.endswith('_norm.tif'):
                        root=roots.split('\\')[-1]
                        img = io.imread(os.path.join(roots, file))
                        io.imsave(os.path.join(benign_path_final,root,'validation\\benign',file), img)
            except:
                pass

