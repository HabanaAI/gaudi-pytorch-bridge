import torch
import glob
import sys
import os
import csv
import numpy as np
import re

def find_max_angle_per_iter(angle_dict, data_dict):
    regex = r"(e\d+)([/])(i\d+)"
    match = re.search(regex, data_dict['tensor_name'])
    if match is not None:
        key = match.group()
        print(key)
        if  angle_dict.get(key) is None :
            angle_dict[key] = data_dict['angle']
        else:
            angle_dict[key] = max (angle_dict[key], data_dict['angle'])

def print_max_angle_dict(angle_dict):
    for key,value in angle_dict.items():
        print(key, "\t:",value)

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

def ca_cosine_similarity(a, b, cos_sim_thld,rms_threshold):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if (na.item() == 0.0) or (nb.item() == 0.0):
        l2_norm = np.linalg.norm(np.array(a)-np.array(b))
        cos_sim_ok = np.greater(rms_threshold, l2_norm/np.sqrt(a.size))
        angle = 0 if cos_sim_ok else 90
        nr = 1.0 if cos_sim_ok else 100
        return na.item(),nb.item(),nr, angle, cos_sim_ok
    else:
        nr =  np.divide(na,nb)
        angle = np.arccos(min(np.dot(a, b) / na / nb, 1.0))/np.pi*180
        angle = np.around(angle,2)
        cos_sim_ok =  np.greater(cos_sim_thld , angle) or np.greater(rms_threshold,na/np.sqrt(a.size))
        if np.greater(angle,cos_sim_thld) and cos_sim_ok:
            angle = np.float32(0.99)
            print(f'Cosine similarity marked True as RMS was below threshold, calculated angle is {angle.item()} set to 0.99')
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
    permute_required = True
    if ('unet3d' in topology or 'unet2d' in topology) and t_dev1_torch.size() == t_dev2_torch.size() and 'bkwd' in tensor_name:
        print(f"not permuting as same shape - {tensor_name}")
        permute_required = False
    if ('resnet' in topology or 'mobilenetv2' in topology or 'googlenet' in topology or 'maskrcnn' in topology or 'mlpmixer' in topology) and same_device is False and t_dev1_torch.ndim == 4:
        if 'hpu' in dev1 or 'hpu' in dev2:
            head, tail = os.path.split(tensor_name)
            if not (tail == "input.pt"):
                if 'hpu' in dev1:
                    tid = 1
                elif 'hpu' in dev2:
                    tid = 2
    if same_device is False and (('unet2d' in topology and t_dev1_torch.ndim == 4) or ('unet3d' in topology and t_dev1_torch.ndim == 5)):
        if 'hpu' in dev1 or 'hpu' in dev2:
            head, tail = os.path.split(tensor_name)
            if (tail != "target.pt")  and (tail != "input.pt") and (tail != "output.pt") and not 'frwd' in head and permute_required:
                if 'hpu' in dev1:
                    tid = 1
                elif 'hpu' in dev2:
                    tid = 2
    return tid

def do_tensor_permute(t_dev1_torch, t_dev2_torch, tid):
    tensor_to_perm = None
    if tid == 1:
        tensor_to_perm = t_dev1_torch
    elif tid == 2:
        tensor_to_perm = t_dev2_torch

    if tensor_to_perm is not None:
        if tensor_to_perm.ndim == 4:
            tensor_to_perm = tensor_to_perm.permute((3, 2, 0, 1)) # permute RSCK to KCRS
        elif tensor_to_perm.ndim == 5:
            tensor_to_perm = tensor_to_perm.permute((4, 3, 0, 1, 2)) # permute RSTCK to KCRST
        if tid == 1:
            return tensor_to_perm, t_dev2_torch
        elif tid == 2:
            return t_dev1_torch, tensor_to_perm
    else:
        return t_dev1_torch, t_dev2_torch

def ca_get_tensor_comparison_stats(dev1, dev2, tensor_name, t_dev1_torch, t_dev2_torch,rms_threshold):
        if t_dev1_torch.is_floating_point() is not True:
            t_dev1_torch = t_dev1_torch.float()
            t_dev2_torch = t_dev2_torch.float()

        dim = list(t_dev1_torch.shape)
        num_els = t_dev1_torch.numel()

        t_dev1 = t_dev1_torch.numpy().flatten().astype(np.float64)
        t_dev2 = t_dev2_torch.numpy().flatten().astype(np.float64)

        cos_sim_thld = 1.0 # 1 degree threshold for cosine ca_cosine_similarity
        maxabs, minabs, mse, dist, rmse = ca_tensor_error_stats(t_dev1, t_dev2)
        norm_dev1, norm_dev2, norm_r, angle, cos_sim_ok = ca_cosine_similarity(t_dev1, t_dev2, cos_sim_thld,rms_threshold)

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

def ca_compare_tensor_files(dev1, dev2, file_pair_list, base_path=None, rtol=1e-3, atol=1e-3, topology=None,skip_pattern='None',rms_threshold=1e-10):
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
    print('Applying rms_threshold:',rms_threshold)
    hk = ca_get_header_keys(dev1, dev2, ca_base_key_list)
    tcs_csv = open('tensor_cmp_stats.csv', 'w', newline='')
    header = ['tensor_name', 'dim','size_elems', hk['min'][dev1], hk['min'][dev2], hk['mean'][dev1], hk['mean'][dev2],
                hk['max'][dev1], hk['max'][dev2], hk['std'][dev1], hk['std'][dev2],hk['norm'][dev1], hk['norm'][dev2],
                'norm_ratio_t', 'minabs_e','maxabs_e','distribution%_abs_e', 'ms_e', 'rms_e', 'angle', 'cosine_sim_ok']
    writer = csv.DictWriter(tcs_csv, fieldnames=header)
    writer.writeheader()
    max_angle=0.0 #Max angle over all iterations
    max_angle_per_iter = dict()
    for file_dev1,file_dev2 in file_pair_list:
        if re.search(skip_pattern,file_dev1) is not None:
            print('Skipping comparison for :',file_dev1)
            continue
        tensor_info = file_dev1
        if base_path is not None:
            tensor_info = file_dev1.replace(base_path, 'base_dir')
        t_dev1 = torch.load(file_dev1)
        t_dev2 = torch.load(file_dev2)

        if (t_dev1.dtype == torch.bfloat16):
            t_dev1 = t_dev1.float()
            print(f'Casting{file_dev1}')

        if (t_dev2.dtype == torch.bfloat16):
            t_dev2 = t_dev2.float()
            print(f'Casting{file_dev2}')

        if (t_dev1.dtype == torch.int):
            t_dev1 = t_dev1.long()
        if (t_dev2.dtype == torch.int):
            t_dev2 = t_dev2.long()

        # Tensor permute not needed for vision based topologies where channel_last feature is disabled and this will be done for various topologies in steps.
        # Currently done for 'resnet','mobilenetv2','googlenet','unet2d', 'unet3d', 'maskrcnn' so these are being skipped.
        if topology not in ['resnet', 'mobilenetv2', 'googlenet', 'unet3d', 'unet2d', 'maskrcnn']:
            #Some tensors like convolution weights need permutation when comparing habana tensors with GPU or CPU
            tid = tensor_to_permute(dev1, dev2, tensor_info, t_dev1, t_dev2, same_device, topology)
            if tid != 0 : # Need permute
                t_dev1, t_dev2 = do_tensor_permute(t_dev1, t_dev2, tid)
                if t_dev1.size() != t_dev2.size() and ('unet3d' in topology or 'unet2d' in topology or 'maskrcnn' in topology) and ('bkwd' in tensor_info or 'frwd' in tensor_info):
                    print(f"because of view, after permute also shape didn't match.. {t_dev1.size()}, {t_dev2.size()} ....\n permute back and do reshape with cpu size")
                    tensor_to_perm = None
                    if tid == 1:
                        tensor_to_perm = t_dev1
                    elif tid == 2:
                        tensor_to_perm = t_dev2
                    if tensor_to_perm.ndim == 4:
                        tensor_to_perm = tensor_to_perm.permute((2, 3, 1, 0)) # permute KCRS to RSCK
                    elif tensor_to_perm.ndim == 5:
                        tensor_to_perm = tensor_to_perm.permute((2, 3, 4, 1, 0)) # permute KCRST to RSTCK
                    if tid == 1:
                        t_dev1 = tensor_to_perm.reshape(t_dev2.size())
                    else:
                        t_dev2 = tensor_to_perm.reshape(t_dev1.size())

        if ('bkwd' in tensor_info or 'frwd' in tensor_info) and t_dev1.size() != t_dev2.size():
            print(f"because of view, shape didn't match.., {file_dev1} {t_dev1.size()}, {t_dev2.size()} ....\n do reshape with cpu size")
            t_dev2 = t_dev2.reshape(t_dev1.size())

        if  t_dev1.numel() == 0:
            equal = True
        else:
            tensor_cmp_stat_dict = ca_get_tensor_comparison_stats(dev1,dev2,tensor_info, t_dev1, t_dev2,rms_threshold)
            max_angle = max(tensor_cmp_stat_dict['angle'], max_angle)
            find_max_angle_per_iter(max_angle_per_iter, tensor_cmp_stat_dict)
            writer.writerow(tensor_cmp_stat_dict)

            equal = torch.allclose(t_dev1, t_dev2, rtol=rtol,atol=atol)

        if equal is False:
            error = torch.isclose(t_dev1, t_dev2, rtol=rtol, atol=atol)
            if t_dev1.ndim ==0 :
                max_diff = abs(t_dev1.item()-t_dev2.item())
            else:
                max_diff = torch.max(torch.abs(t_dev1[error.logical_not()] - t_dev2[error.logical_not()]))

            print("MISMATCH: max_diff : ", max_diff, "angle : ",tensor_cmp_stat_dict['angle'],  "   \tfor tensor: ", tensor_info, " with  rtol : ", rtol, "atol : ", atol)
            if 'loss' in tensor_info:
                print("device1 loss = ", t_dev1.item(), "device2 loss = ", t_dev2.item())
        else:
            if  t_dev1.numel() == 0:
                print("SKIPPING DIFF : ZERO SIZE tensor", tensor_info,  " with rtol : ", rtol, " atol : ", atol)
            print("NO-DIFF : for tensor: ", tensor_info,  " with rtol : ", rtol, " atol : ", atol)
            if 'loss' in tensor_info:
                print("device1 loss = ", t_dev1.item(), "device2 loss = ", t_dev2.item())
    tcs_csv.close()
    print("Cosine Similarity: Overall Max angle (across all iterations)=", max_angle)
    print("Cosine Similarity: Max angle Per iteration=")
    print_max_angle_dict(max_angle_per_iter)

