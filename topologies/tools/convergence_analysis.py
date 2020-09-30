import torch
import glob
import sys
import os
import csv
import numpy as np
import re

def ca_tensor_error_stats(a,b):
    d = np.subtract(a,b)
    ad = np.abs(d)
    maxabs = np.amax(ad)
    minabs = np.amin(ad)

    h,_ = np.histogram(ad,5)
    n = ad.size
    h = np.divide(h,n)
    h = np.multiply(h,100.0)
    h = np.around(h,2)

    dsq = np.square(d)
    mse = np.mean(dsq)
    rmse = np.sqrt(mse)
    return maxabs.item(), minabs.item(),mse.item(), h, rmse.item()

def ca_cosine_similarity(a, b, cos_sim_thld):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if (na.item() == 0.0) and (nb.item() == 0.0):
        nr = np.array(1.0, dtype=np.float64)
        angle = np.array(0.0, dtype=np.float64)
        cos_sim_ok=True
        return na.item(),nb.item(),nr.item(), angle.item(), cos_sim_ok
    else:
        nr =  np.divide(na,nb)
        angle = np.arccos(min(np.dot(a, b) / na / nb, 1.0))/np.pi*180
        angle = np.around(angle,2)
        cos_sim_ok =  np.greater(cos_sim_thld , angle)
        return na.item(),nb.item(),nr.item(), angle.item(), cos_sim_ok

# Keys for individual tensor stats
ca_base_key_list = ['min', 'max','mean', 'std', 'norm']
def ca_get_header_keys(dev1,dev2, b_key_list):
    header_keys = dict()
    for b_key in b_key_list:
        header_keys[b_key] = {
                                dev1: b_key+'_'+dev1+'_t',
                                dev2: b_key+'_'+dev2+'_t'}
    return header_keys

#returns id(0, 1 or 2) of tensor to permute; 0 - no permute; 1 - dev1 tensor to permute; 2 - dev2 tensor to permute
def tensor_to_permute(dev1, dev2, tensor_name, t_dev1_torch, t_dev2_torch, same_device, topology):
    tid = 0
    if 'resnet' in topology and same_device is False and t_dev1_torch.ndim == 4:
        if 'habana' in dev1 or 'habana' in dev2:
            head, tail = os.path.split(tensor_name)
            if not (tail == "input.pt"):
                if 'habana' in dev1:
                    tid = 1
                elif 'habana' in dev2:
                        tid = 2
    return tid

def do_tensor_permute(t_dev1_torch, t_dev2_torch, tid):
    tensor_to_perm = None
    if tid == 1:
        tensor_to_perm = t_dev1_torch
    elif tid == 2:
        tensor_to_perm = t_dev2_torch

    if tensor_to_perm is not None:
        tensor_to_perm = tensor_to_perm.permute((3,2,0,1)) # permute RSCK to KCRS
        if tid == 1:
            return tensor_to_perm, t_dev2_torch
        elif tid == 2:
            return t_dev1_torch, tensor_to_perm
    else:
        return t_dev1_torch, t_dev2_torch

def ca_get_tensor_comparison_stats(dev1, dev2, tensor_name, t_dev1_torch, t_dev2_torch):
        if t_dev1_torch.is_floating_point() is not True:
            t_dev1_torch = t_dev1_torch.float()
            t_dev2_torch = t_dev2_torch.float()

        dim = list(t_dev1_torch.shape)
        num_els = t_dev1_torch.numel()

        t_dev1 = t_dev1_torch.numpy().flatten().astype(np.float64)
        t_dev2 = t_dev2_torch.numpy().flatten().astype(np.float64)

        cos_sim_thld = 1.0 # 1 degree threshold for cosine ca_cosine_similarity
        maxabs, minabs, mse, dist, rmse = ca_tensor_error_stats(t_dev1, t_dev2)
        norm_dev1, norm_dev2, norm_r, angle, cos_sim_ok = ca_cosine_similarity(t_dev1, t_dev2, cos_sim_thld)

        hk = ca_get_header_keys(dev1, dev2, ca_base_key_list)

        tensor_cmp_stat_dict = dict()
        tensor_cmp_stat_dict['tensor_name'] = tensor_name
        tensor_cmp_stat_dict['dim'] = dim
        tensor_cmp_stat_dict['size_elems'] = num_els
        tensor_cmp_stat_dict[hk['min'][dev1]] = np.amin(t_dev1).item()
        tensor_cmp_stat_dict[hk['min'][dev2]] = np.amin(t_dev2).item()
        tensor_cmp_stat_dict[hk['max'][dev1]] = np.amax(t_dev1).item()
        tensor_cmp_stat_dict[hk['max'][dev2]] = np.amax(t_dev2).item()
        tensor_cmp_stat_dict[hk['mean'][dev1]] = np.mean(t_dev1).item()
        tensor_cmp_stat_dict[hk['mean'][dev2]] = np.mean(t_dev2).item()
        tensor_cmp_stat_dict[hk['std'][dev1]] = np.std(t_dev1).item()
        tensor_cmp_stat_dict[hk['std'][dev2]] = np.std(t_dev2).item()


        tensor_cmp_stat_dict[hk['norm'][dev1]] = norm_dev1
        tensor_cmp_stat_dict[hk['norm'][dev2]] = norm_dev2
        tensor_cmp_stat_dict['norm_ratio_t'] = norm_r
        tensor_cmp_stat_dict['maxabs_e'] = maxabs
        tensor_cmp_stat_dict['minabs_e'] = minabs
        tensor_cmp_stat_dict['distribution%_abs_e'] = dist
        tensor_cmp_stat_dict['ms_e'] = mse
        tensor_cmp_stat_dict['rms_e'] = rmse
        tensor_cmp_stat_dict['angle'] = angle
        tensor_cmp_stat_dict['cosine_sim_ok'] = cos_sim_ok

        return tensor_cmp_stat_dict


def ca_make_file_pair_list(dev1, dev2, path1, path2):
    path1_m = os.path.join(path1, dev1)
    if path2 is None: # no separate path for dev2, use dev1's toplevel path
        path2_m = os.path.join(path1, dev2)
    else:
        path2_m = os.path.join(path2, dev2)

    print("Comparing tensors between: ",  path1_m, " and ", path2_m)

    files_dev1 = [f for f in glob.glob(path1_m + "/**/*.pt", recursive=True)]
    #print(files_dev1)
    files_dev2 = [f.replace(path1_m, path2_m) for f in files_dev1 ]

    #print(files_dev2)
    return zip(files_dev1,files_dev2)

def ca_compare_tensor_files(dev1, dev2, file_pair_list, base_path=None, rtol=1e-3, atol=1e-3, topology=None,skip_pattern='None'):
    #If we are comparing the tensors on same device, say, habana, rename the devices as
    # habana1 and 2 for the csv file. Else the dictionary key for dev1 and 2 will be same
    #causing an overwriting
    same_device = False
    if dev1 == dev2:  #e.g. habana
        dev1=dev1+'1' #e.g. habana1
        dev2=dev2+'2' #e.g. habana2
        same_device = True

    print("Using Tolerances rtol = ", rtol, " atol =", atol, "for comparing", dev1,  "and ", dev2)
    print('Applying skip_pattern:',skip_pattern)
    hk = ca_get_header_keys(dev1, dev2, ca_base_key_list)
    tcs_csv = open('tensor_cmp_stats.csv', 'w', newline='')
    header = ['tensor_name', 'dim','size_elems', hk['min'][dev1], hk['min'][dev2], hk['mean'][dev1], hk['mean'][dev2],
                hk['max'][dev1], hk['max'][dev2], hk['std'][dev1], hk['std'][dev2],hk['norm'][dev1], hk['norm'][dev2],
                'norm_ratio_t', 'minabs_e','maxabs_e','distribution%_abs_e', 'ms_e', 'rms_e', 'angle', 'cosine_sim_ok']
    writer = csv.DictWriter(tcs_csv, fieldnames=header)
    writer.writeheader()
    max_angle=0.0
    for file_dev1,file_dev2 in file_pair_list:
        if re.search(skip_pattern,file_dev1) is not None:
            print('Skipping comparison for :',file_dev1)
            continue
        tensor_info = file_dev1
        if base_path is not None:
            tensor_info = file_dev1.replace(base_path, 'base_dir')
        t_dev1 = torch.load(file_dev1)
        t_dev2 = torch.load(file_dev2)

        #Some tensors like convolution weights need permutation when comparing habana tensors with GPU or CPU
        tid = tensor_to_permute(dev1, dev2, tensor_info, t_dev1, t_dev2, same_device, topology)
        if tid != 0 : # Need permute
            t_dev1, t_dev2 = do_tensor_permute(t_dev1, t_dev2, tid)

        tensor_cmp_stat_dict = ca_get_tensor_comparison_stats(dev1,dev2,tensor_info, t_dev1, t_dev2)
        max_angle = max(tensor_cmp_stat_dict['angle'], max_angle)
        writer.writerow(tensor_cmp_stat_dict)

        equal = torch.allclose(t_dev1, t_dev2, rtol=rtol,atol=atol)

        if equal is False:
            error = torch.isclose(t_dev1, t_dev2, rtol=rtol, atol=atol)
            max_diff = torch.max(torch.abs(t_dev1[error.logical_not()] - t_dev2[error.logical_not()]))
            print("MISMATCH: max_diff : ", max_diff,  "   \tfor tensor: ", tensor_info, " with  rtol : ", rtol, "atol : ", atol)
            if 'loss' in tensor_info:
                print("device1 loss = ", t_dev1.item(), "device2 loss = ", t_dev2.item())
        else:
            print("NO-DIFF : for tensor: ", tensor_info,  " with rtol : ", rtol, " atol : ", atol)
            if 'loss' in tensor_info:
                print("device1 loss = ", t_dev1.item(), "device2 loss = ", t_dev2.item())
    tcs_csv.close()
    print("Cosine Similarity: Max angle =", max_angle)

