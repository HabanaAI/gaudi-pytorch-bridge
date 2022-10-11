
#!/usr/bin/python3
import argparse
import os
import csv
import glob
import yaml

def match_any(l, match):
    for m in match:
        if m in l:
            return True
    return False
    
def extract_signature_list(lst):
    sl = []
    for l in lst:
        sl.append(extract_signature(l))
    return sl

def write_consolidated_op_list(name, op_d):

    #tcs_csv = open(name+'.csv', 'w', newline='')

    #header = ['tensor_name', 'dim','size_elems', hk['min'][dev1], hk['min'][dev2], hk['mean'][dev1], hk['mean'][dev2],
    #            hk['max'][dev1], hk['max'][dev2], hk['std'][dev1], hk['std'][dev2],hk['norm'][dev1], hk['norm'][dev2],
    #            'norm_ratio_t', 'minabs_e','maxabs_e','distribution%_abs_e', 'ms_e', 'rms_e', 'angle', 'cosine_sim_ok']
    #writer = csv.DictWriter(tcs_csv, fieldnames=header)
    #writer.writeheader()


    #writer.writerow(tensor_cmp_stat_dict)
    #tcs_csv.close()
    op_csv = open(name, 'w', newline='')
    header = ["OpName", "Relevant", "OpType", "OpType2", "Implemented","ImplMethod", "OpSignature"]
    writer = csv.DictWriter(op_csv, fieldnames=header, delimiter="|")
    writer.writeheader()


    Total = 0
    rv = 0
    nrv = 0
    rv_df_dt_t = 0
    rv_df_dt_impl = 0
    rv_dt_dt_t =0
    rv_dt_dt_impl =0
    rv_dt_dt_nimpl =0
    rv_dt_df_t = 0
    rv_dt_df_impl =0
    rv_dt_df_nimpl =0

    rv_dt_dt_impl_a = 0
    rv_dt_dt_impl_m = 0
    rv_dt_df_impl_a = 0
    rv_dt_df_impl_m = 0

    with open(name,"w") as f:
        #for k,v in op_d.items():
        for k in sorted(op_d.keys()):
            v = op_d.get(k)
            row = {}
            row["OpName"] = k
            row["OpSignature"] = v["op_with_sig"]
            row["Relevant"] = v["relevant"]
            row["OpType"] = v["type"]
            row["OpType2"] = v["type2"]
            row["Implemented"] = v["impld"]
            row["ImplMethod"] = v["impl_method"]
            writer.writerow(row)

            Total += 1
            if row["Relevant"] == "yes" :
                rv = rv +1
                if row["OpType2"] == "df_dt":
                    rv_df_dt_t += 1
                    if row["Implemented"] == "yes":
                        rv_df_dt_impl += 1
                elif row["OpType2"] == "dt_dt":
                    rv_dt_dt_t += 1
                    if row["Implemented"] == "yes":
                        rv_dt_dt_impl += 1
                        if row["ImplMethod"] == "auto":
                            rv_dt_dt_impl_a += 1
                        elif row["ImplMethod"] == "manual":
                            rv_dt_dt_impl_m += 1
                elif row["OpType2"] == "dt_df":
                    rv_dt_df_t += 1
                    if row["Implemented"] == "yes":
                        rv_dt_df_impl += 1
                        if row["ImplMethod"] == "auto":
                            rv_dt_df_impl_a += 1
                        elif row["ImplMethod"] == "manual":
                            rv_dt_df_impl_m += 1
    nrv = Total - rv
    rv_dt_dt_nimpl = rv_dt_dt_t - rv_dt_dt_impl
    rv_dt_df_nimpl = rv_dt_df_t - rv_dt_df_impl

    rv_non_mandat_t = rv_df_dt_t + rv_dt_dt_t

    with open("summary.txt", "w") as f:
        f.write("Total , " + str(Total))
        f.write("\nRelevant on HPU , "+ str(rv))
        f.write("\nNot Relevant on HPU , "+ str(nrv))

        f.write("\nNon-Mandatory Total , "+ str(rv_non_mandat_t))
        f.write("\ndf_dt Total , "+ str(rv_df_dt_t))
        f.write("\ndf_dt implemented , "+ str(rv_df_dt_impl))

        f.write("\ndt_dt Total , "+ str(rv_dt_dt_t))
        f.write("\ndt_dt implemented , "+ str(rv_dt_dt_impl))
        f.write("\ndt_dt implemented auto, "+ str(rv_dt_dt_impl_a))
        f.write("\ndt_dt implemented manual, "+ str(rv_dt_dt_impl_m))

        f.write("\ndt_dt remaining , "+ str(rv_dt_dt_nimpl))


        f.write("\nMandatory Total , "+ str(rv_dt_df_t))
        f.write("\ndt_df implemented , "+ str(rv_dt_df_impl))
        f.write("\ndt_df implemented auto , "+ str(rv_dt_df_impl_a))
        f.write("\ndt_df implemented manual , "+ str(rv_dt_df_impl_m))
        f.write("\ndt_df remaining , "+ str(rv_dt_df_nimpl))

        op_csv.close()
        """
        with open(name,"w") as f:
            for k,v in op_d.items():
                f.write( k + '|')
                for k1,v1 in v.items():
                    f.write(v1 + '|')
                #f.write(v["relevant"] + '|')
                #f.write(v["type"] + '|')
                #f.write(v["impld"] + '|')
                #f.write(v["impl_method"] + '|')
                #f.write(v["impl_method"] + '|')
                f.write('\n')
        """

def write_unique_op_list_v1(name, op_d):
    op_csv = open(name, 'w', newline='')
    header = ["Unique Op Name", "Relevant", "Variants", "Non-Cmpnd Not Implmntd", "Non-Cmpnd Implmntd", "Cmpnd Implmntd", "Cmpnd Not Implmntd", "Total Non-Cmpnd", "Total Cmpnd"]
    writer = csv.DictWriter(op_csv, fieldnames=header, delimiter="|")
    writer.writeheader()
#total_variants" : 0, "cvi" : 0, "cvni": 0, "ncvi": 0, "ncvni":0

    with open(name,"w") as f:
        #for k,v in op_d.items():
        for k in sorted(op_d.keys()):
            v = op_d.get(k)
            row = {}
            row["Unique Op Name"] = k
            row["Relevant"] = v["rlv"]
            row["Variants"] = v["total_variants"]
            row["Non-Cmpnd Not Implmntd"] = v["ncvni"]
            row["Non-Cmpnd Implmntd"] = v["ncvi"]
            row["Cmpnd Implmntd"] = v["cvi"]
            row["Cmpnd Not Implmntd"] = v["cvni"]
            row["Total Non-Cmpnd"] = v["tnc"]
            row["Total Cmpnd"] = v["tc"]
            writer.writerow(row)
        op_csv.close()
    """
    with open(name,"w") as f:
        for k,v in op_d.items():
            f.write( k + '|')
            for k1,v1 in v.items():
                f.write(str(v1) + '|')
            f.write('\n')
    """

def write_unique_op_list_v2(name, op_d):
    op_csv = open(name, 'w', newline='')
    header = ["Unique Op Name", "Relevant", "Variants", "df_df Not Impl", "df_df Impl", "df_dt Not Impl", "df_dt Impl", "dt_df Not Impl", "dt_df Impl", "dt_dt Not Impl", "dt_dt Impl","Total df_df", "Total df_dt", "Total dt_df", "Total dt_dt"]
    writer = csv.DictWriter(op_csv, fieldnames=header, delimiter="|")
    writer.writeheader()
#total_variants" : 0, "cvi" : 0, "cvni": 0, "ncvi": 0, "ncvni":0

    with open(name,"w") as f:
        #for k,v in op_d.items():
        for k in sorted(op_d.keys()):
            v = op_d.get(k)
            row = {}
            row["Unique Op Name"] = k
            row["Relevant"] = v["rlv"]
            row["Variants"] = v["total_variants"]
            row["df_df Not Impl"] = v["df_df_ni"]
            row["df_df Impl"] = v["df_df_i"]
            row["df_dt Not Impl"] = v["df_dt_ni"]
            row["df_dt Impl"] = v["df_dt_i"]
            row["dt_df Not Impl"] = v["dt_df_ni"]
            row["dt_df Impl"] = v["dt_df_i"]
            row["dt_dt Not Impl"] = v["dt_dt_ni"]
            row["dt_dt Impl"] = v["dt_dt_i"]
            row["Total df_df"] = v["t_df_df"]
            row["Total df_dt"] = v["t_df_dt"]
            row["Total dt_df"] = v["t_dt_df"]
            row["Total dt_dt"] = v["t_dt_dt"]
            writer.writerow(row)
        op_csv.close()

def get_unique_op_name(n):
    s = n.split(".")[0]

    if s.endswith("_"):
        return s[:-1]
    else:
        return s

def unique_ops_stats_v1(unique_ops, pt_op_dict):
    unique_op_dict = {}
    for uop in unique_ops:
        for k,v in pt_op_dict.items():
            p_uop = get_unique_op_name(k)
            def_dict = {"total_variants" : 0, "cvi" : 0, "cvni": 0, "ncvi": 0, "ncvni":0, "rlv":"no", "tc":0, "tnc":0}
            if uop == p_uop:
                unique_op_dict[uop] = unique_op_dict.get(uop, def_dict)
                unique_op_dict[uop]["total_variants"] = unique_op_dict[uop].get("total_variants", 0) +1
                #print("XXXXXX: ", pt_op_dict[k])
                if pt_op_dict[k]["relevant"] == "yes":
                    unique_op_dict[uop]["rlv"] = "yes"
                else:
                    unique_op_dict[uop]["rlv"] = "no"

                if pt_op_dict[k]["type"] == "compound":
                    unique_op_dict[uop]["tc"] = unique_op_dict[uop].get("tc", 0) +1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["cvi"] = unique_op_dict[uop].get("cvi", 0) +1
                    else:
                        unique_op_dict[uop]["cvni"] = unique_op_dict[uop].get("cvni", 0) +1
                else:
                    unique_op_dict[uop]["tnc"] = unique_op_dict[uop].get("tnc", 0) +1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["ncvi"] = unique_op_dict[uop].get("ncvi", 0) +1
                    else:
                        unique_op_dict[uop]["ncvni"] = unique_op_dict[uop].get("ncvni", 0) +1

    return unique_op_dict

def unique_ops_stats_v2(unique_ops, pt_op_dict):
    unique_op_dict = {}
    for uop in unique_ops:
        for k,v in pt_op_dict.items():
            p_uop = get_unique_op_name(k)
            def_dict = {"total_variants" : 0, "df_df_i" : 0, "df_df_ni": 0, "df_dt_i": 0, "df_dt_ni":0,"dt_df_i": 0, "dt_df_ni":0, "dt_dt_i": 0, "dt_dt_ni":0, "rlv":"no", "t_df_df":0, "t_df_dt":0, "t_dt_df":0,  "t_dt_dt":0}
            if uop == p_uop:
                unique_op_dict[uop] = unique_op_dict.get(uop, def_dict)
                unique_op_dict[uop]["total_variants"] = unique_op_dict[uop].get("total_variants", 0) +1
                #print("XXXXXX: ", pt_op_dict[k])
                if pt_op_dict[k]["relevant"] == "yes":
                    unique_op_dict[uop]["rlv"] = "yes"
                else:
                    unique_op_dict[uop]["rlv"] = "no"

                if pt_op_dict[k]["type2"] == "df_df":
                    unique_op_dict[uop]["t_df_df"] = unique_op_dict[uop].get("t_df_df", 0) +1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["df_df_i"] = unique_op_dict[uop].get("df_df_i", 0) +1
                    else:
                        unique_op_dict[uop]["df_df_ni"] = unique_op_dict[uop].get("df_df_ni", 0) +1
                elif pt_op_dict[k]["type2"] == "df_dt":
                    unique_op_dict[uop]["t_df_dt"] = unique_op_dict[uop].get("t_df_dt", 0) +1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["df_dt_i"] = unique_op_dict[uop].get("df_dt_i", 0) +1
                    else:
                        unique_op_dict[uop]["df_dt_ni"] = unique_op_dict[uop].get("df_dt_ni", 0) +1
                elif pt_op_dict[k]["type2"] == "dt_df":
                    unique_op_dict[uop]["t_dt_df"] = unique_op_dict[uop].get("t_dt_df", 0) +1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["dt_df_i"] = unique_op_dict[uop].get("dt_df_i", 0) +1
                    else:
                        unique_op_dict[uop]["dt_df_ni"] = unique_op_dict[uop].get("dt_df_ni", 0) +1
                elif pt_op_dict[k]["type2"] == "dt_dt":
                    unique_op_dict[uop]["t_dt_dt"] = unique_op_dict[uop].get("t_dt_dt", 0) +1
                    if pt_op_dict[k]["impld"] == "yes":
                        unique_op_dict[uop]["dt_dt_i"] = unique_op_dict[uop].get("dt_dt_i", 0) +1
                    else:
                        unique_op_dict[uop]["dt_dt_ni"] = unique_op_dict[uop].get("dt_dt_ni", 0) +1
                else:
                    print("got type2 as ",pt_op_dict[k]["type2"])
                    assert 0, "invalid type2"

    return unique_op_dict

#Auto generated files are split into multiple files like hpu_op0.cpp, hpu_op1.cpp ...
#Combine them(read the lines in each file and return combined set of lines)
def combine_auto_generated_files(p):
    px = p +'/hpu_op*[0-9].cpp'
    print(px)
    files = glob.glob(px)
    print(files)
    l_auto_ops_decl = []
    for f in files:
        f_auto_ops_decl = open(f, 'r')
        l = f_auto_ops_decl.readlines()
        l_auto_ops_decl.extend(l)
        f_auto_ops_decl.close()
    return l_auto_ops_decl

def get_manual_ops_with_overrides_in_yaml(f_yaml):
    manual_ops_override = []
    with open(f_yaml, "r") as stream:
        try:
            yaml_dict = yaml.safe_load(stream)
            #print(yaml_dict)
            for k,v in yaml_dict.items():
                if "override_fn" in v.keys():
                    #print("kkkkkk" ,k)
                    manual_ops_override.append(k)
        except yaml.YAMLError as exc:
            print(exc)
    return manual_ops_override


def is_op_overridden_in_yaml(op_name, ops_override_list):
    for op in ops_override_list:
        if op_name == op:
            return True
    return False 

def main(args):

    f_op_decl =open(args.ops_decl, 'r')
    l_op_decl = f_op_decl.readlines()
    f_op_decl.close()
    p1 = os.path.join(args.gen_files_path, 'wrap_kernels_registrations.cpp')
    p2 = args.gen_files_path
    f_manual_ops_decl =open(p1, 'r')
    l_manual_ops_decl = f_manual_ops_decl.readlines()
    f_manual_ops_decl.close()
    l_auto_ops_decl = combine_auto_generated_files(p2)

    #exclude1 =['"compound": "True"', 'mkldnn', 'cudnn', 'miopen', 'nnpack', 'thnn']
    exclude1 =['mkldnn', 'cudnn', 'miopen', 'nnpack', 'thnn', 'mkl']
    exclude2 =['_sparse', 'slow', 'quantiz', 'fbgemm', '_cufft_', 'vulkan']
    exclude3 = ['_cast_Byte', '_cast_Char', '_cast_Double', '_cast_Float','_cast_Half', '_cast_Int', '_cast_Long', '_cast_Short']
    exclude = []
    exclude.extend(exclude1)
    exclude.extend(exclude2)
    exclude.extend(exclude3)
    valid_op_decl =[]
    valid_op_decl_compound_op =[]
    valid_op_decl_prop =[]
    j = v = m =0
    pt_op_dict = {}
    unique_ops = set()
    for line in l_op_decl:
        if "aten::" in line:
            op_name = line.split("aten::")[1].split("(")[0]
            op_with_sig = line.split(");")[0]
            prop = {}
            valid_op_decl.append(op_name)
            if not any([x in op_name for x in exclude]):
        #if not match_any(line, exclude):
                prop["relevant"] = "yes"
            else:
                prop["relevant"] = "no"
            
        
            if '"dispatch": "True"' in line:
                #valid_op_decl_compound_op.append('compound')
                prop["type"] = "non-compound"
                if '"default": "True"' in line:
                    prop["type2"] = "dt_dt"
                else:
                    prop["type2"] = "dt_df"

                
            else:
                #valid_op_decl_compound_op.append('non-compound')
                prop["type"] = "compound"
                #prop["type2"] = "df_xx"
                if '"default": "True"' in line:
                    prop["type2"] = "df_dt"
                else:
                    prop["type2"] = "df_df"
            prop["op_with_sig"] = op_with_sig + ")";
            valid_op_decl_prop.append(prop)
            pt_op_dict[op_name] = prop
            unique_ops.add(get_unique_op_name(op_name))


    print("len valid_op_decl = ", len(valid_op_decl))
    #print(valid_op_decl)


    valid_manual_ops_decl =[]
    manual_op_dict = {}
    for line in l_manual_ops_decl:
        if "m.impl(" in line and "hpu_wrap" in line:
            op_name = line.split('m.impl("')[1].split('",')[0]
            valid_manual_ops_decl.append(op_name)
            manual_op_dict[op_name] = "manual"

    #Get and append Manual ops with override in yaml
    yaml_file = os.path.join(args.pt_integ_path, 'scripts/hpu_op.yaml')
    manual_ops_override_list = get_manual_ops_with_overrides_in_yaml(yaml_file)
    print("\n\nManual ops with override in yaml = ", manual_ops_override_list)
    for op_name in manual_ops_override_list:
        valid_manual_ops_decl.append(op_name)
        manual_op_dict[op_name] = "manual" 
    print("len valid_manual_ops_decl = ", len(valid_manual_ops_decl))
    print(valid_manual_ops_decl)

    valid_auto_ops_decl =[]
    auto_op_dict = {}
    for line in l_auto_ops_decl:
        #if "m.impl(" in line and "HpuOp" in line:
        if "m.impl(" in line:
            op_name = line.split('m.impl("')[1].split('",')[0]
            # Skip adding to "auto" ops list if op is registered as part of auto code,
            #  but is actually manual op overridden in yaml
            if is_op_overridden_in_yaml(op_name, manual_ops_override_list):
                continue
            valid_auto_ops_decl.append(op_name)
            auto_op_dict[op_name] = "auto"
    print("len valid_auto_ops_decl = ", len(valid_auto_ops_decl))
    print(valid_auto_ops_decl)
    impl_ops_list = []
    impl_ops_list.extend(valid_manual_ops_decl)
    impl_ops_list.extend(valid_auto_ops_decl)
    impl_op_dict = {}
    impl_op_dict.update(manual_op_dict)
    impl_op_dict.update(auto_op_dict)
    print("len  impl_ops_decl = ", len(impl_ops_list))
    print(impl_ops_list)

    print(impl_op_dict)
    op_names = list(pt_op_dict.keys())
    #for pop in pt_op_dict.keys():
    for pop in op_names:
        if pop in impl_op_dict.keys():
            pt_op_dict[pop].update({"impld" : "yes"})
            pt_op_dict[pop].update({"impl_method" : impl_op_dict[pop]})
        else:
            pt_op_dict[pop].update({"impld" : "no"})
            pt_op_dict[pop].update({"impl_method" : "NA"})

    #print("\n\n ptopdict = ",pt_op_dict)
    write_consolidated_op_list("consolidate_ops_list.csv", pt_op_dict)

    print(impl_ops_list)
    print("********************************") 
    print("len  unique ops = ", len(unique_ops))
    print("********************************") 
    print(unique_ops)
 #test_errors_freq[l] = test_errors_freq.get(l, 0) + 1
    unique_op_dict_v1 = unique_ops_stats_v1(unique_ops, pt_op_dict)
    unique_op_dict_v2 = unique_ops_stats_v2(unique_ops, pt_op_dict)
    print(unique_op_dict_v1)
    print(unique_op_dict_v2)

    print(unique_op_dict_v1["remainder"])
    write_unique_op_list_v1("unique_ops_list.csv",unique_op_dict_v1)
    write_unique_op_list_v2("unique_ops_list2.csv",unique_op_dict_v2)
    print("len valid_op_decl = ", len(valid_op_decl))
    print("len  impl_ops_decl = ", len(impl_ops_list))
    print("len valid_manual_ops_decl = ", len(valid_manual_ops_decl))
    print("len valid_auto_ops_decl = ", len(valid_auto_ops_decl))
    print("\nOps with following strings in their names are considered not relevant on HPU", exclude)
    return
    


if __name__ == '__main__':
    # for command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ops_decl', default='',
                        help='ops declarations file')
    parser.add_argument('--pt_integ_path', default='',
                        help='path of pytorch integration git')
    parser.add_argument('--gen_files_path', default='',
                        help='path of auto generated op files and wrap declarations file for manual ops')
    args = parser.parse_args()
    main(args)
