# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:49:21 2022

@author: melte
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor

#%%
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
import numpy as np
from mcnn_model import MCNN
from my_dataloader import CrowdDataset
from sklearn.preprocessing import minmax_scale



def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("cpu")
    mcnn=MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    mae=0
    E = []
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(dataloader):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img)
            
            err = 100*(abs(et_dmap.data.sum()-gt_dmap.data.sum())/gt_dmap.data.sum()).item()
            E.append(err)
            
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            # print(dataset.img_names[i], err, "%")
            
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" MAE:"+str(mae/len(dataloader)))
    return E, mae/len(dataloader)


def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    # device=torch.device("cpu")
    mcnn=MCNN()#.to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    for i,(img,gt_dmap) in enumerate(dataloader):
        global out, nodes, feature
        if i==index:
            img=img#.to(device)
            gt_dmap=gt_dmap#.to(device)
            # forward propagation
            et_dmap=mcnn(img)#.detach()
            #et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            
            nodes, _ = get_graph_node_names(mcnn)
            print(nodes)
            feature_extractor = create_feature_extractor(
             mcnn, return_nodes=[ 'branch1.0', 'branch1.1', 'branch1.2', 'branch1.3', 'branch1.4', 'branch1.5', 'branch1.6', 'branch1.7', 'branch1.8', 'branch1.9', 'branch2.0', 'branch2.1', 'branch2.2', 'branch2.3', 'branch2.4', 'branch2.5', 'branch2.6', 'branch2.7', 'branch2.8', 'branch2.9', 'branch3.0', 'branch3.1', 'branch3.2', 'branch3.3', 'branch3.4', 'branch3.5', 'branch3.6', 'branch3.7', 'branch3.8', 'branch3.9','cat', 'fuse.0'])
            # global out, nodes
            out = feature_extractor(img)
            # plt.imshow(out['img_tensor'].detach().numpy()[0,2])
            # print(out)
            # for feature in out['cat'].detach().numpy()[0]:
            #     feature = minmax_scale(feature)
            #     plt.figure()
            #     plt.imshow(feature, cmap=CM.jet)
            #     plt.show()
            
            
            # feature = np.mean(out['cat'].detach().numpy()[0],axis=0)
            # feature = minmax_scale(feature)
            # plt.figure()
            # plt.imshow(feature, cmap=CM.jet)
            # plt.show()
            i= 0
            fig, axs = plt.subplots(10,3, figsize=(12, 25))
            
            for node in out.keys():
                feature = np.mean(out[node].detach().numpy()[0],axis=0)
                # feature = out[node].detach().numpy()[0,1]
                feature = minmax_scale(feature)
                axs[i%10, i//10].imshow(feature, cmap=CM.jet)
                axs[i%10, i//10].set_title(node)
                axs[i%10, i//10].axis('off')
                
                i+= 1
            fig.show()    
            
            # for i in range(out["branch1.9"].detach().numpy().shape[1]):
            #     img = out["branch1.9"].detach().numpy()[0,i]
            #     plt.figure()
            #     plt.imshow(img, cmap=CM.jet)
            #     plt.show()
            # print(et_dmap.shape)
            # plt.imshow(et_dmap,cmap=CM.jet)
            break


if __name__=="__main__":
    try:
        torch.backends.cudnn.enabled=False
        img_root='C:\\Users\\melte\\Desktop\\bitirme test\\ShanghaiTech\\part_A\\train_data\\images'
        gt_dmap_root='C:\\Users\\melte\\Desktop\\bitirme test\\ShanghaiTech\\part_A\\train_data\\ground-truth'
        mae_list = []
        for i in range(113, 114):
            model_param_path=f'C:\\Users\\melte\\Desktop\\bitirme test\\MCNN-pytorch\\checkpoints_1\\epoch_{i}.param'
            # E, mae = cal_mae(img_root,gt_dmap_root,model_param_path)
            # mae_list.append(mae)
            # E = np.array(E)
            # plt.hist(E, bins = 10)
            # plt.show()
            # plt.hist(E[E<100], bins = 10)
            # plt.show()
        estimate_density_map(img_root,gt_dmap_root,model_param_path,1)
    except:
        pass
    
    #plt.hist(E, bins=40)
    



    
    