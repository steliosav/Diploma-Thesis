import torch
import numpy as np
from scipy.cluster.vq import kmeans2
from utils.spectrum import SpectrumClustering

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Implementation of Spectral and Kmeans clustering for use on ResNet50's feature maps.
Clustering of feature maps is used to reduce feature space redundancy. 
Each feature map cluster will be represented by a single feature map in the output.
"""

def spectral_clustering_torch_2048(x): # feature map size specifically for ResNet50
    x = x.squeeze(0)
    features = 2048
    dataset = torch.empty((features, 8100))
    for i in range(features):
        dataset[i,:] = x[i,:,:].flatten()
    n_clusters = 40
    cluster = SpectrumClustering(n_clusters)
    print("clustering start.")
    clustering, _ = cluster(dataset)
    print("clustering done.")

    cluster_means = torch.empty((n_clusters,8100))
    for i in range(n_clusters):
        idx_mask = clustering == i
        indices = idx_mask.nonzero().squeeze(1)
        cluster_grp = torch.index_select(dataset, 0, indices)
        cluster_means[i,:] = torch.nan_to_num(torch.mean(cluster_grp, 0), 0.0)
    del dataset

    new_dataset = torch.empty((features,8100))
    for i in range(features):
        new_dataset[i,:] = cluster_means[clustering[i], :]

    result = torch.empty((1, features, 90, 90))
    for i in range(features):
        stack_mean = torch.stack(torch.split(new_dataset[i], 90))
        result[:,i,:,:] = stack_mean

    del cluster_means

    torch.cuda.empty_cache()
    result = result.to("cuda")
    return result

def spectral_clustering_torch(x): # Same function as spectral_clustering_torch_2048, but with batched input support
    batch_num=1
    for n in range(batch_num):
        batch_item = x[n,:,:,:]
        batch_item = batch_item.squeeze(0)
        features = 2048
        dim = 8100
        dim_tuple = 90
        dataset = torch.empty((features, dim))
        # breakpoint()
        for i in range(features):
            dataset[i,:] = batch_item[i,:,:].flatten()
        n_clusters = 40
        cluster = SpectrumClustering(n_clusters)
        print("clustering start.")
        clustering, _ = cluster(dataset)
        print("clustering done.")

        cluster_means = torch.empty((n_clusters,dim))
        need_nonzero = True
        for i in range(n_clusters):
            idx_mask = clustering == i
            indices = idx_mask.nonzero().squeeze(1)
            cluster_grp = torch.index_select(dataset, 0, indices)
            cluster_means[i,:] = torch.nan_to_num(torch.mean(cluster_grp, 0), 0.0)
            if need_nonzero:
                zero = (torch.sum(cluster_means[i]) == 0)
                if not(zero):
                    nonzero_mean = cluster_means[i]
                    need_nonzero = False
        del dataset

        for i in range(n_clusters):
            all_zeroes = (torch.sum(cluster_means[i]) == 0)
            if all_zeroes:
                cluster_means[i] = nonzero_mean

        result = torch.empty((batch_num, n_clusters, dim_tuple, dim_tuple))
        for i in range(n_clusters):
            stack_mean = torch.stack(torch.split(cluster_means[i], dim_tuple))
            result[n,i,:,:] = stack_mean

        del cluster_means

    torch.cuda.empty_cache()
    result = result.to("cuda")
    return result

def kmeans_clustering(x): # Non-accelerated CPU code for Kmeans clustering on ResNet50's feature maps 
    x_arr = x.squeeze(0)
    x_arr = x_arr.to("cpu")

    z = []
    for i in range(2048):
        flat_seg = x_arr[i, :, :].detach().flatten().numpy()
        z.append(flat_seg)

    z = np.array(z)

    centroid, label = kmeans2(z, 40, minit='points')

    centroid_arr = []
    for i in range(2048):
        centroid_arr.append(centroid[label[i], :])

    centroid_arr = np.array(centroid_arr)

    tmp = np.empty((1, 2048, 90, 90), dtype=np.double)
    for k in range(2048):
        for i in range(90):
            for j in range(90):
                tmp[:, k, i, j] = centroid_arr[k, (90*i + j)]

    tensor = torch.from_numpy(tmp)
    tensor = tensor.type(torch.FloatTensor)
    tensor = tensor.to("cuda")
    return tensor