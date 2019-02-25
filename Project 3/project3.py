# Hikmet Orhun Güley / Department of Computer Science

import cv2
import numpy as np
import os
import time
#from numba import jit


#@jit
def ReadVideos(yourpath):
    yourpath = yourpath + '\\'
    array = []
    for root, dirs, files in os.walk(yourpath, topdown=False):
        for name in files:
            array.append(yourpath + name)
    return array


#@jit
def GetDataSet(videos):

    test_set = []
    train_labels = []
    test_labels = []
    for i in range(len(videos)):
        rand = np.random.randint(0, len(videos[i]))
        test_set.append(videos[i].pop(rand))
        test_labels.append(i+1)
    label = 1
    for video in videos:
        for i in range(len(video)):
            train_labels.append(label)
        label += 1
    return np.array(videos), np.array(test_set), np.array(train_labels), np.array(test_labels)


#@jit
def CalculateBin(B, ang):

    pi = 180
    if ang >= 90 and ang < 180:
        angle = 180 - ang
    elif ang >= 180 and ang < 270:
        angle = ang - 270
    elif ang > 270 and ang < 360:
        angle = ang - 360
    else:
        angle = ang
    for b in range(B):
        if ang >= (-(pi / 2)) + ((b+1-1) / B) * pi and angle < (-(pi / 2)) + (pi * (b+1) / B):
            return b+1
    return 0


#@jit
def GetBins(angles, B):
    X = angles.shape[0]
    Y = angles.shape[1]
    binMatrix = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            binMatrix[x, y] = CalculateBin(B, angles[x, y])
    return binMatrix


#@jit
def CalculateHOOF(B, ang, mag):
    hoof = []
    binMatrix = GetBins(ang, B)
#    m = np.zeros((binMatrix.shape[0], binMatrix.shape[1]))
    for b in range(B):
        sum_error = 0
#        hoof.append(binMatrix[binMatrix == b+1])
        positions = np.argwhere(binMatrix == b+1)
        for pos in positions:
            sum_error += mag[pos[0]][pos[1]]
        hoof.append(sum_error)

    return np.array(hoof) / np.sum(np.array(hoof))


#@jit
def PCA(data):
    # for the PCA I got help from the link
    #(https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/)
    mean = np.mean(data, axis=0)
    cov = np.cov(data - mean, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    return eigenvalues, eigenvectors


def DimentionalityReduction(eigenvalues, eigenvectors, data, threshold):

    if np.min(eigenvalues) < 0:
        normalized_eigenvalues = eigenvalues + np.abs(np.min(eigenvalues))
        normalized_eigenvalues = normalized_eigenvalues / \
            np.sum(normalized_eigenvalues)
    else:
        normalized_eigenvalues = eigenvalues / np.sum(eigenvalues)

    normalized_eigenvalues = np.sort(normalized_eigenvalues)
    sortedindex = np.argsort(eigenvalues)

    tot_variance = 0
    e = 0
    while (tot_variance + normalized_eigenvalues[e] <= 1 - threshold):
        tot_variance += normalized_eigenvalues[e]
        e += 1
    feature_vector = np.delete(eigenvectors, sortedindex[:e], 1)
    dimred = np.matmul(data, feature_vector)

    return feature_vector, dimred



#@jit
def OpticalFlow(capture):
    cap = cv2.VideoCapture(capture)
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    hsv_array = []
    flow_array = []
    ang_array = []
    mag_array = []
    hoof_array = []
    while(1):
        ret, frame2 = cap.read()
        if ret:
            nxt = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_array.append(flow)
            mag_array.append(mag)
            hoof_array.append(CalculateHOOF(B, ang*360/np.pi/2, mag))
            ang_array.append(ang*360/np.pi/2)
            hsv[..., 0] = ang*360/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv_array.append(hsv.copy())
#            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#            cv2.imshow('frame2', bgr)
#            i += 1
#            k = cv2.waitKey(1) & 0xff
#            if k == ord('q') or k == 27:
#                print("Escape")
#                break
#            elif k == ord('s'):
#                cv2.imwrite('opticalfb.png', frame2)
#                cv2.imwrite('opticalhsv.png', bgr)
            prvs = nxt
        else:
            #            print(i)
            break

    cap.release()
    cv2.destroyAllWindows()
    return hsv_array, np.matrix(hoof_array)


videos = []
videos.append(ReadVideos('bend'))
videos.append(ReadVideos('jack'))
videos.append(ReadVideos('jump'))
videos.append(ReadVideos('pjump'))
videos.append(ReadVideos('run'))
videos.append(ReadVideos('side'))
videos.append(ReadVideos('skip'))
videos.append(ReadVideos('walk'))
videos.append(ReadVideos('wave1'))
videos.append(ReadVideos('wave2'))




B_array = np.arange(20,41,4)
accuracy_array_PCA = []
accuracy_array_nonPCA = []
tour = 1




for b in B_array:
    print('Tour ',tour,' started for B = ', b)
    train_set, test_set, train_labels, test_labels = GetDataSet(videos)
    B = b

    start = time.time()
    B = 30
    hoof_mean_list = []
    
    video_n = 1
    for video in test_set:
        print('Test Video Type: ', video_n)
        tourtime = time.time()
        hsv_data, hoof_data = OpticalFlow(video)
        hoof_mean_list.append(np.copy(np.mean(hoof_data, axis=0)))
        print('time: ', time.time()-tourtime)
    
        video_n += 1
    end = time.time()
    print('total time: ', end - start)
    
    h_test_data = np.array(hoof_mean_list).reshape((len(hoof_mean_list), B))
    hoof_mean_list = []
    
    video_n = 1
    for video in train_set:
        print('Train Video Type: ', video_n)
        for i in range(len(video)):
            tourtime = time.time()
            hsv_data, hoof_data = OpticalFlow(video[i])
            hoof_mean_list.append(np.copy(np.mean(hoof_data, axis=0)))
            print('time: ', time.time()-tourtime)
    
        video_n += 1
    end = time.time()
    
    
    
    h_train_data = np.array(hoof_mean_list).reshape((len(hoof_mean_list), B))
    eigenvalues, eigenvectors = PCA(h_train_data)
    featurevector, dimred = DimentionalityReduction(eigenvalues, eigenvectors,
                                                    h_train_data, 0.9)
    test_dimred = np.matmul(h_test_data, featurevector)
        
    
    train_dimred = np.ndarray.astype(dimred, np.float32)
    test_dimred = np.ndarray.astype(test_dimred, np.float32)
    test_labels = np.ndarray.astype(test_labels, np.float32)
    train_labels = np.ndarray.astype(train_labels, np.float32)
    
    h_train_data = np.ndarray.astype(h_train_data, np.float32)
    h_test_data = np.ndarray.astype(h_test_data, np.float32)
    
    knn = cv2.ml.KNearest_create()
    knn.train(train_dimred, cv2.ml.ROW_SAMPLE, train_labels)
    ret, results, neighbours ,dist = knn.findNearest(test_dimred, cv2.ml.KNearest_BRUTE_FORCE)
    
    accuracy = np.sum(results == test_labels.reshape((10,1))) / len(test_labels)
    print('Accuracy PCA: ', accuracy)
    accuracy_array_PCA.append(accuracy)
    
    knn = cv2.ml.KNearest_create()
    knn.train(h_train_data, cv2.ml.ROW_SAMPLE, train_labels)
    ret, results, neighbours ,dist = knn.findNearest(h_test_data, cv2.ml.KNearest_BRUTE_FORCE)
    
    accuracy = np.sum(results == test_labels.reshape((10,1))) / len(test_labels)
    print('Accuracy non-PCA: ', accuracy)
    accuracy_array_nonPCA.append(accuracy)
    
    print('total time: ', end - start)




