"""
Created on Mon Apr 29 05:21:13 2019
@author: saif
"""
import numpy as np
from dipy.viz import window, actor
from dipy.tracking.streamline import transform_streamlines
import vtk.util.colors as colors
from dipy.tracking import utils
from dipy.tracking.streamline import set_number_of_points
from sklearn import svm
import nibabel as nib
from dipy.tracking.vox2track import streamline_mapping
import time

   
def show_tract(segmented_tract, color_positive ,segmented_tract_negative, color_negative, out_path):
    """Visualization of the segmented tract.
    """ 
    affine=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
    bundle_native = transform_streamlines(segmented_tract, np.linalg.inv(affine))
    
    bundle_nativeNeg = transform_streamlines(segmented_tract_negative, np.linalg.inv(affine))

    renderer = window.Renderer()
    stream_actor2 = actor.line(bundle_native,
                           colors=color_positive, linewidth=0.1)
    
    stream_actorNeg = actor.line(bundle_nativeNeg, colors=color_negative,
                           opacity=0.01, linewidth=0.1)
    renderer.set_camera(position=(408.85, -26.23, 92.12),
                    focal_point=(0.42, -14.03, 0.82),
                    view_up=(-0.09, 0.85, 0.51))
    
    bar = actor.scalar_bar()
    renderer.add(stream_actor2)
    
    renderer.add(stream_actorNeg)
    renderer.add(bar)
    window.show(renderer, size=(1920, 1039), reset_camera=False)
    renderer.camera_info()
        
    """Take a snapshot of the window and save it
    """
    window.record(renderer, out_path = out_path, size=(1920, 1039))

def compute_dsc(estimated_tract, true_tract):
    """Compute the overlap between the segmented tract and ground truth tract
    """
    aff=np.array([[-1.25, 0, 0, 90],[0, 1.25, 0, -126],[0, 0, 1.25, -72],[0, 0, 0, 1]])
    #aff=utils.affine_for_trackvis(voxel_size=np.array([1.25,1.25,1.25]))
    voxel_list_estimated_tract = streamline_mapping(estimated_tract, affine=aff).keys()
    voxel_list_true_tract = streamline_mapping(true_tract, affine=aff).keys()
    TP = len(set(voxel_list_estimated_tract).intersection(set(voxel_list_true_tract)))
    vol_A = len(set(voxel_list_estimated_tract))
    vol_B = len(set(voxel_list_true_tract))
    DSC = 2.0 * float(TP) / float(vol_A + vol_B)
    return DSC      
          
def load(filename):
    """Load tractogram from TRK file 
    """
    wholeTract= nib.streamlines.load(filename)  
    wholeTract = wholeTract.streamlines
    return  wholeTract    
def embedding(streamlines, no_of_points):
    """Resample streamlines using 12 points and also flatten the streamlines
    """
    return np.array([set_number_of_points(s, no_of_points).ravel() for s in streamlines]) 
 
def create_train_data_set(train_subjectList,tract):
    
    train_data=[]
    for sub in train_subjectList:  
        print (sub)        
        T_filename=sub+tract
        wholeTract = load (T_filename)       
        train_data=np.concatenate((train_data, wholeTract),axis=0) 
    
    print ("train data Shape") 
    resample_tract=embedding(train_data,no_of_points=no_of_points)
   
    return resample_tract, train_data
    
def create_test_data_set(testTarget_brain):    
    
    print ("Preparing Test Data")    
    t_filename=testTarget_brain #"124422_af.left.trk"    
    test_data=load(t_filename)  
    resample_tractogram=embedding(test_data,no_of_points=no_of_points)

    return resample_tractogram, test_data  
      
if __name__ == '__main__':
    
    train_subjectList =[ "124422", "111312",  "100408", "100307", "856766"]
    tract = "_cg.right.trk"
    no_of_points=12    
    leafsize=10
        
    ################################ Train Data ######################################
    print ("Preparing Train Data")
    resample_tract_train, train_data= create_train_data_set(train_subjectList, tract)    
    
    ###################### Test Data################################
    testTarget = "161731"
    testTarget_brain = "full1M_"+testTarget+".trk"
    t0=time.time()
    resample_tract_test, test_data= create_test_data_set(testTarget_brain)
    trueTract=load(testTarget + tract)  
    t1=t0-time.time()
    """########################### one class SVM Linear ######################"""
    gamma_value = 0.001
    clf = svm.OneClassSVM(nu=0.1, kernel="linear", gamma=gamma_value)
    clf.fit(resample_tract_train)
    """ linear poly rbf """
    """#########################################"""
    
    x_pred_train = clf.predict(resample_tract_train.tolist())
    n_error_test = x_pred_train[x_pred_train==-1].size
    print('number of error for training =', n_error_test)    
    
    
    x_pred_test=clf.predict(resample_tract_test.tolist())    
    n_error_test = x_pred_test[x_pred_test==-1].size
    print('number of error for testing=',n_error_test)
    
    
    ########################### visualize tract ######################
    test_data=np.array(test_data)
    segmented_tract_positive= test_data[np.where(x_pred_test==1)]
    segmented_tract_negative= test_data[np.where(x_pred_test==-1)]
    dsc=compute_dsc(segmented_tract_positive,trueTract)
    print("Accuracy for linear: ",dsc)
    
    """########################### one class SVM Poly ######################"""
    gamma_value = 0.001
    clf = svm.OneClassSVM(nu=0.1, kernel="poly", gamma=gamma_value)
    clf.fit(resample_tract_train)
    """ linear poly rbf """
    """#########################################"""
    
    x_pred_train = clf.predict(resample_tract_train.tolist())
    n_error_test = x_pred_train[x_pred_train==-1].size
    print('number of error for training =', n_error_test)    
    
    
    x_pred_test=clf.predict(resample_tract_test.tolist())    
    n_error_test = x_pred_test[x_pred_test==-1].size
    print('number of error for testing=',n_error_test)
    
    
    ########################### visualize tract ######################
    test_data=np.array(test_data)
    segmented_tract_positive= test_data[np.where(x_pred_test==1)]
    segmented_tract_negative= test_data[np.where(x_pred_test==-1)]
    dsc=compute_dsc(segmented_tract_positive,trueTract)
    print("Accuracy for poly: ",dsc)
    
    """########################### one class SVM RBF######################"""
    t2=time.time()
    gamma_value = 0.001
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=gamma_value)
    clf.fit(resample_tract_train)
    
    x_pred_train = clf.predict(resample_tract_train.tolist())
    n_error_test = x_pred_train[x_pred_train==-1].size
    print('number of error for training =', n_error_test)    
    
    
    x_pred_test=clf.predict(resample_tract_test.tolist())    
    n_error_test = x_pred_test[x_pred_test==-1].size
    print('number of error for testing=',n_error_test)
    
    
    ########################### visualize tract ######################
    test_data=np.array(test_data)
    segmented_tract_positive= test_data[np.where(x_pred_test==1)]
    segmented_tract_negative= test_data[np.where(x_pred_test==-1)]
    ###########################Calculating Dice Similarity Co-efficient########################### 
    
    dsc=compute_dsc(segmented_tract_positive,trueTract)
    print("Accuracy for rbf: ",dsc)
    
    print("Total amount of time to compute svm is %f seconds" % ((time.time()-t2)+t1)) 
    
    print("Show the tract")
    
    out_path="images\\svm\\"+str(len(train_subjectList) )+"_sub_"+testTarget+"_"+tract+"_SVMResult.png" # Save image in this path
    color_positive= colors.green
    color_negative=colors.red
    show_tract(segmented_tract_positive, color_positive, segmented_tract_negative, color_negative, out_path)#,color_negative)#,segmented_tract_negative) 
    
    
    
    
    
    
    
    
    
    
    
    
    
