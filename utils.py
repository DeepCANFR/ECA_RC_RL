import os
import numpy as np
import random
import shutil
import subprocess as sp

def unique_eca_rules(): 
    return [
        0  ,1  ,2  ,3  ,4  ,5  ,6  ,7  ,8  ,9  ,
        10 ,11 ,12 ,13 ,14 ,15 ,18 ,19 ,22 ,23 ,
        24 ,25 ,26 ,27 ,28 ,29 ,30 ,32 ,33 ,34 ,
        35 ,36 ,37 ,38 ,40 ,41 ,42 ,43 ,44 ,45 ,
        46 ,50 ,51 ,54 ,56 ,57 ,58 ,60 ,62 ,72 ,
        73 ,74 ,76 ,77 ,78 ,90 ,94 ,104,105,106,
        108,110,122,126,128,130,132,134,136,138,
        140,142,146,150,152,154,156,160,162,164,
        168,170,172,178,184,200,204,232,
    ]

def eca_classes():
    return {
        '1': [0, 8, 32, 40, 128, 136, 160, 168],
        '2': [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23,
              24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 42,
              43, 44, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 76,
              77, 78, 94, 104, 108, 130, 132, 134, 138, 140, 142,
              152, 154, 156, 162, 164, 170, 172, 178, 184, 200,
              204, 232],
        '3': [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150],
        '4': [41, 54, 106, 110]
    }

def xor(a, b):
    if (a == 1 or a == 0) and (b == 0 or b == 1):
        return 1 if bool(a) != bool(b) else 0
    else:
        raise Exception(f"Can't xor non 0 or 1 values ({a}, {b})")

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)

def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def delete_temp_log_folders():
    a = 'logs'
    for b in os.listdir(a):
        ab = os.path.join(a, b)
        for c in os.listdir(ab):
            abc = os.path.join(ab, c)
            for d in os.listdir(abc):
                abcd = os.path.join(abc, d)

                if d == 'train':
                    shutil.rmtree(abcd)
                elif os.path.isdir(abcd):
                    for e in os.listdir(abcd):
                        abcde = os.path.join(abcd, e)
                        if e == 'train':
                            shutil.rmtree(abcde)

# delete_temp_log_folders()

def print_memory_usage(process):
    B = process.memory_info().rss
    MB = B / (1<<20)
    print(f'{MB} MB')

def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values[0]

def write_run_file(file_name, rule_string):
    with open(file_name, 'a') as file:
        file.writelines(rule_string + '\n')



