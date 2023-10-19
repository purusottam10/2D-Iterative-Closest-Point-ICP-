import numpy as np
from math import *
import matplotlib.pyplot as plt

def ICP(reference_pt_cld,measured_pt_cld):
    P = np.array(measured_pt_cld)                                       # This is the measured point cloud
    Q = np.array(reference_pt_cld)                                      # This is the reference point cloud

    index_Q = np.array([])
    min_dist = np.array([])

    for points_q in P:
        distance = np.array([])
        for points_p in Q:
            dist = ((points_q[0]-points_p[0])**2 + (points_q[1]-points_p[1])**2 )**0.5
            distance = np.append(distance,np.array([dist]),axis=0)
        arguments_Q = np.array([np.argmin(distance)])
        index_Q = np.append(index_Q,arguments_Q,axis=0)
        min_dist_from_P = np.array([distance[np.argmin(distance)]])
        min_dist = np.append(min_dist,min_dist_from_P,axis=0)
    #print(index_Q)
    #print(min_dist)
    #print(len(np.unique(index_Q)))                                  # This prints the no of correspondence points between the pointclouds. 
    unique_indices_in_measured_data = np.array([])

    for i in np.unique(index_Q):
        unique_indices_in_Q = np.where(index_Q == i)[0]
        #print("The {} index apperas {} times ".format(i,len(unique_indices_in_Q)))
        #print(unique_indices_in_Q)

        if len(unique_indices_in_Q) != 1:
            #print(unique_indices_in_Q)
            #print(min_dist[unique_indices_in_Q])
            #print(np.argmin(min_dist[unique_indices_in_Q]))
            #print(unique_indices_in_Q[np.argmin(min_dist[unique_indices_in_Q])])
            #print(index_Q[unique_indices_in_Q[np.argmin(min_dist[unique_indices_in_Q])]])
            unique_indices_in_P = np.array([unique_indices_in_Q[np.argmin(min_dist[unique_indices_in_Q])]])
            #print(unique_indices_in_P)
            unique_indices_in_measured_data = np.append(unique_indices_in_measured_data,unique_indices_in_P,axis=0)

        else:
            #print(unique_indices_in_Q)
            #print(index_Q[unique_indices_in_Q[np.argmin(min_dist[unique_indices_in_Q])]])
            unique_indices_in_P = np.array([unique_indices_in_Q[np.argmin(min_dist[unique_indices_in_Q])]])
            #print(unique_indices_in_P)
            unique_indices_in_measured_data = np.append(unique_indices_in_measured_data,unique_indices_in_P,axis=0)

    unique_indices_in_measured_data = np.int64(unique_indices_in_measured_data)
    #print(unique_indices_in_measured_data)

    unique_indices_in_reference_data = np.int64(np.unique(index_Q))

    #print(unique_indices_in_reference_data)

    #plt.scatter(P[unique_indices_in_measured_data,0],P[unique_indices_in_measured_data,1])
    #plt.plot(P[:,0],P[:,1])
    #plt.scatter(Q[unique_indices_in_reference_data,0],Q[unique_indices_in_reference_data,1])
    #plt.plot(Q[:,0],Q[:,1])
    #plt.show()

    # Computing the Covarianace matrix

    filtered_measured_points = P[unique_indices_in_measured_data]
    #print(np.shape(filtered_measured_points))
    #print(np.sum(filtered_measured_points[:,0])/len(filtered_measured_points))
    filtered_refrence_points = Q[unique_indices_in_reference_data]

    P_mean = np.mean(filtered_measured_points,axis=0)                                 #     mean of the filtered measued data 
    #print(P_mean)
    Q_mean = np.mean(filtered_refrence_points,axis=0)                                 #     mean of the filtered reference data
    #print(Q_mean)

#    P_mean = np.mean(P,axis=0)
#    Q_mean = np.mean(Q,axis=0)

    # Cross-covariance matrix calculation
    k=np.dot((filtered_measured_points-P_mean).T,(filtered_refrence_points-Q_mean))/(filtered_refrence_points.shape[0]-1)

    # Performing singular value decomposition to get the rotation matrix and translation matrix.
    U, S, V = np.linalg.svd(k)

    # Rotation matrix calculation
    Rotation = np.dot(U,V)
    #print(Rotation)

    #Translation matrix calculation
    Translation = Q_mean - np.dot(P_mean,Rotation)
    #print(Translation)

    P_new = np.array([])
    #print(np.shape(P))
    for points in P:
        p_new = np.dot(points,Rotation) + Translation
        P_new = np.append(P_new,p_new,axis=0)
    P_new = P_new.reshape(100,3)
    #print(P_new)

    # Calculating the RMS error of the two data sets
    P_new_mean = np.mean(P_new,axis=0)
    #print(P_new_mean)
    Q_new_mean = np.mean(Q,axis=0)
    #print(Q_new_mean)

    #RMS_error = ((Q_new_mean[0]-P_new_mean[0])**2 + (Q_new_mean[1]-P_new_mean[1])**2)**0.5
    RMS_error = ((Translation[0])**2 + (Translation[1])**2)**0.5

    plt.scatter(P[unique_indices_in_measured_data,0],P[unique_indices_in_measured_data,1])
    plt.plot(P[:,0],P[:,1])
    plt.scatter(Q[unique_indices_in_reference_data,0],Q[unique_indices_in_reference_data,1])
    plt.plot(Q[:,0],Q[:,1])
    plt.plot(P_new[:,0],P_new[:,1])
    plt.show()
    
    return RMS_error,P_new,Rotation



x = np.arange(0,10,0.1)
y = np.sin(x)
Q = np.array([(x,y,1) for x,y in zip(x,y)])                                           # For example we have taken a pointcloud with a sine wave. This is the actual pointcloud. REFERENCE POINTS/DATA
T = np.array([[0.7071,0.7071,0],[-0.7071,0.7071,0],[5,5,1]])                          # This will rotate the actual point cloud in -ve(anticlock-wise) 45 degrees.
#T = np.array([[1,0,0],[0,1,0],[0,0,1]]) 
P = np.array([])                                                                      
for points in Q:
    q = np.dot(points,T)
    P = np.append(P,q,axis=0)
P = P.reshape(100,3)  

#ICP(Q,P)

error =100
while error>=0.0003:
    icp = ICP(Q,P)
    error = icp[0]
    P = icp[1]
    print(icp[0])       # Printing the rms error.




