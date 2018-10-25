import numpy as np
import cv2
from io import StringIO
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mylibrary as mylib
from mylibrary import euclidean_distance
import os

class Center:
    """
    Purpose:
        Each Center object save the location of center and the records that belong to this center.
        pts save the id of each record that belongs to the Center object.
        The id of recrod is the index of record in given data matrix, start from 0.
    """
    def __init__(self, center, pts):
        self.center = center
        self.pts = pts
        
    def set_center(self, center):
        self.center = center
        
    def set_pts(self, pts):
        self.pts = pts
    
    def __eq__(self, other): 
        return np.all(self.center == other.center)
    
    def __repr__(self):
         return "cluster: " + str(self.center) 


def init_given(data,k,seed):
    """
    Purpose:
        Init the list of Center objects by the data given in task3 part A.
    """
    centers = np.array([[6.2,3.2],
              [6.6,3.7],
              [6.5,3.0]])
    centers = list(map(lambda x: Center(x,np.array([])), centers))
    centers = assign_to_center(data, centers,k)
    return centers

def init_centers_random(data, k, seed=20):
    """
    Purpose:
        randomly shuffle the data, and pick the top k recrods to updates the list of centers.
    Input:
        data: a two dimension matrix
        k: int, the number of centers
        seed: the seed of random number generator.
    Output:
        centers: a list of Center objects. 
    """
    centers = data[np.random.RandomState(seed=seed).permutation(data.shape[0])[0:k]]
    centers = list(map(lambda x: Center(x,np.array([])), centers))
    centers = assign_to_center(data, centers,k)
    return centers

def assign_to_center(data, centers, k):
    """
    Purpose:
        Assign records in data to centers using Euclidean distance. The record id is the index
        of record in data matrix.
    Input:
        data: a two dimension matrix of real number.
        centers: a list of Center objects.
        k: int, the number of centers. equal to the length of centers.
    Output:
        centers: a list of Center objects, the pts list has been updated.
    """
    dis_matrix = np.empty((0,data.shape[0]))
    for center in centers:
        dis_matrix = np.vstack((dis_matrix, np.sum(np.square(data - center.center), axis=1)))
    belongs = np.argmin(dis_matrix, axis=0)
    for i in range(k):
        centers[i].pts = np.where(belongs == i)[0]
    return centers

def update_centers(data, centers, k):
    """
    Purpose:
        Generate a new list of Center object given the information from the previous centers.
    Input:
        data: a two dimension matrix
        centers: a list of Center objects. previous.
        k: int, the number of centers.
    Output:
        the new list of Center objects.
    """
    not_updated = True
    new_centers = []
    for center in centers:
        new_centers.append(Center(np.mean(data[center.pts],axis=0), np.array([])))
    return assign_to_center(data,new_centers, k)   

def plot_k_mean(data, centers, itr, save_path="../task3_img/k_mean"):
    """
    Purpose:
        General pupose plot function to plot k_mean funcion.
    """
    try:
        os.makedirs(save_path)
    except FileExistsError:
        print("use existing folder:", save_path)
        
    label_set = set(np.arange(len(centers)))
    color_map = dict(zip(label_set, cm.rainbow(np.linspace(0, 1, len(label_set)))))
    for label in label_set:
        index = centers[label].pts
        plt.scatter(data[index][:,0], data[index][:,1], s=20, c=color_map[label],
                    alpha=0.3, label=label)
        plt.scatter(centers[label].center[0], centers[label].center[1], s=100, c=color_map[label],
                    alpha=1.0, marker='x')
    plt.title("iteration: "+ str(itr))
    plt.legend(loc='best')
    #plt.show()
    plt.savefig(save_path+"/iteration_" + str(itr) + ".png")
    plt.close()

def plot_k_mean_a(data, centers, itr, save_path="../task3_img/k_mean"):
    """
    Purpose:
        The plot funcion specific for the cv project2 task3 k-mean part.
    """
    try:
        os.makedirs(save_path)
    except FileExistsError:
        print("use existing folder:", save_path)
    
    label_set = set(np.arange(len(centers)))
    #color_map = dict(zip(label_set, cm.rainbow(np.linspace(0, 1, len(label_set)))))
    color_map = dict(zip(label_set, ['red','green','blue']))
    for label in label_set:
        index = centers[label].pts
        plt.scatter(data[index][:,0], data[index][:,1], s=50, c=color_map[label],
                    alpha=0.3, label=label,marker='^')
        plt.scatter(centers[label].center[0], centers[label].center[1], s=50, color=color_map[label],
                    alpha=1.0, marker='o')
    plt.title("iteration: "+ str(itr))
    plt.legend(loc='best')
    plt.savefig(save_path+"/iteration_" + str(itr) + ".png")
    plt.close()

def k_mean(data, k, init_fun, max_itr=50, seed=20, need_plot=False, plot_fun=None):
    """
    Purpose:
        Main funcion for k mean algorithm.
    Input:
        data: a two dimension matrix
        k: the number of centers
        init_fun: the funcion used to generate the intial list of Center objects.
        max_tr: the maximum number of iterations.
        need_plot: If set true, the k_mean funcion will save plot for each iteration.
        plot_fun: function, the plot function.
    Output:
        centers: a list of Center objects. The result for the final iteration.
    """
    itr = 0
    centers = init_fun(data,k,seed)
    while itr <= max_itr:
        if need_plot:
            plot_fun(data, centers, itr)
        if itr % 5 == 0:
            print("iteration :", itr)
        new_centers = update_centers(data,centers,k)
        centers_1 = np.asarray(list(map(lambda x: x.center ,centers)))
        centers_2 = np.asarray(list(map(lambda x: x.center ,new_centers)))
        if np.all(centers_1 == centers_2):
            break
        centers = new_centers 
        itr += 1
    print("total iteration", itr)
    return centers


if __name__ == "__main__":
	data_given = np.array([[5.9,3.2],
	                 [4.6,2.9],
	                 [6.2,2.8],
	                 [4.7,3.2],
	                 [5.5,4.2],
	                 [5.0,3.0],
	                 [4.9,3.1],
	                 [6.7,3.1],
	                 [5.1,3.8],
	                 [6.0,3.0]])

	data_a = data_given
	seed = 20
	k = 3
	max_itr = 100
	itr = 0
	centers = k_mean(data_a,k,init_fun=init_given ,need_plot=True,plot_fun=plot_k_mean_a)






