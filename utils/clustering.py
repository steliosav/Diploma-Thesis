import torch
import numpy as np
from scipy.cluster.vq import kmeans2
from sklearn.cluster import SpectralClustering
from torchcluster.zoo.spectrum import SpectrumClustering

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def spectral_clustering_torch_2048(x):
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

def spectral_clustering_torch(x):
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

def spectral_clustering40(x):
    features = 2048
    x_arr = x.squeeze(0)
    # x_arr = x_arr.to("cpu")

    z = []
    for i in range(features):
        flat_seg = x_arr[i, :, :].detach().cpu().flatten().numpy()
        z.append(flat_seg)

    z = np.array(z)
    z = np.transpose(z)
    del x_arr
    # print(z.shape)
    # breakpoint()

    n_clusters = 40
    print('clustering start.')
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans').fit_predict(z)
    print('clustering done.')
    # labels = clustering.labels_
    # matrix = clustering.affinity_matrix_
    # features = clustering.n_features_in_
    # print(clustering)
    # print(labels.shape)
    # print(matrix.shape)
    # print(features)

    cluster_segments = []
    centroids = np.empty((40, 8100))
    cluster_pop = []
    cluster_size = 0
    count = 0
    for i in range(n_clusters):
        for j in range(features):
             # print(clustering[j])
             # print(i)
             # breakpoint()
            if clustering[j] == i:
                cluster_segments.append(z[:, j])
                cluster_size += 1

        if cluster_size != 0:
            cluster_segments = np.array(cluster_segments)
            centroid = np.mean(cluster_segments, axis=0)
            centroids[i,:] = centroid[:]
            index = i
        else:
            count += 1
            centroid = np.zeros((8100))
            centroids[i,:] = centroid[:]

        cluster_pop.append(cluster_size)

        cluster_segments = []
        cluster_size = 0

        # print(cluster_segments)
        # print(cluster_segments.shape)
        # print(centroid)
        # print(centroid.shape)
        # print(type(centroid))
        # breakpoint()

    del z

    # print(cluster_pop)
    # centroids = np.array(centroids)
    for i in range(n_clusters):
        all_zeroes = not np.any(centroids[i])
        if all_zeroes:
            centroids[i] = centroids[index]

    # print(centroids.shape)
    # print(count)
    # breakpoint()

    t_out = np.empty((1, n_clusters, 90, 90), dtype=np.double)
    for k in range(n_clusters):
        for i in range(90):
            for j in range(90):
                t_out[:, k, i, j] = centroids[k, (90*i + j)]

    del centroids

    t_out = torch.from_numpy(t_out)
    t_out = t_out.type(torch.FloatTensor)
    # print(t_out.size())
    t_out = t_out.to("cuda")
    return t_out

# x = torch.load('resnet_out.pt', map_location=torch.device('cuda'))
# # x = spectral_clustering40(x)
# x = spectral_clustering_torch(x)
# print(x)
# print(x.size())

def spectral_clustering(x):
    x_arr = x.squeeze(0)
    x_arr = x_arr.to("cpu")

    z = []
    for i in range(2048):
        flat_seg = x_arr[i, :, :].detach().flatten().numpy()
        z.append(flat_seg)

    z = np.array(z)
    del x_arr
    # print(z.shape)

    n_clusters = 40
    clustering = SpectralClustering(n_clusters=n_clusters, assign_labels='kmeans', random_state=0).fit_predict(z)
    # labels = clustering.labels_
    # matrix = clustering.affinity_matrix_
    # features = clustering.n_features_in_
    # print(clustering)
    # print(labels.shape)
    # print(matrix.shape)
    # print(features)

    cluster_segments = []
    centroids = []
    cluster_pop = []
    cluster_size = 0
    count = 0
    for i in range(n_clusters):
        for j in range(2048):
             # print(clustering[j])
             # print(i)
             # breakpoint()
            if clustering[j] == i:
                cluster_segments.append(z[j, :])
                cluster_size += 1

        if cluster_size != 0:
            cluster_segments = np.array(cluster_segments)
            centroid = np.mean(cluster_segments, axis=0)
            centroids.append(centroid[:])
        else:
            count += 1
            centroid = np.zeros((8100))
            centroids.append(centroid[:])

        cluster_pop.append(cluster_size)

        cluster_segments = []
        cluster_size = 0

        # print(cluster_segments)
        # print(cluster_segments.shape)
        # print(centroid)
        # print(centroid.shape)
        # print(type(centroid))
        # breakpoint()

    # print(cluster_pop)
    centroids = np.array(centroids)
    # print(centroids)
    # print(centroids.shape)
    # print(count)

    centroid_arr = []
    for i in range(2048):
        centroid_arr.append(centroids[clustering[i], :])

    centroid_arr = np.array(centroid_arr)

    # print(centroid_arr)
    # print(centroid_arr.shape)

    t_out = np.empty((1, 2048, 90, 90), dtype=np.double)
    for k in range(2048):
        for i in range(90):
            for j in range(90):
                t_out[:, k, i, j] = centroid_arr[k, (90*i + j)]

    t_out = torch.from_numpy(t_out)
    t_out = t_out.type(torch.FloatTensor)
    t_out = t_out.to("cuda")
    return t_out


def kmeans_clustering(x):
    x_arr = x.squeeze(0)
    x_arr = x_arr.to("cpu")

    z = []
    for i in range(2048):
        flat_seg = x_arr[i, :, :].detach().flatten().numpy()
        z.append(flat_seg)

    z = np.array(z)

    centroid, label = kmeans2(z, 40, minit='points')

    # print(x_arr.shape)
    # print(z.shape)
    # print(z)

    # print(centroid.shape)
    # print(label.shape)

    centroid_arr = []
    for i in range(2048):
        centroid_arr.append(centroid[label[i], :])

    centroid_arr = np.array(centroid_arr)

    # print(centroid_arr)

    tmp = np.empty((1, 2048, 90, 90), dtype=np.double)
    for k in range(2048):
        for i in range(90):
            for j in range(90):
                tmp[:, k, i, j] = centroid_arr[k, (90*i + j)]

    tensor = torch.from_numpy(tmp)
    tensor = tensor.type(torch.FloatTensor)
    tensor = tensor.to("cuda")
    return tensor

# print(tensor.size())
# print(tensor)
# print(x)
