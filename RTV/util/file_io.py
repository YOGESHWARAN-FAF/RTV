import os


def get_file_path_list(dir_path:str,type_list=None)->list:
    filelist = os.listdir(dir_path)
    filelist.sort()
    file_list = []



    for item in filelist:
        if check_type(item,type_list):
            if item.startswith('.'):
                continue
            # print(item)
            # print(item.split('.')[0])
            file_list.append(item)
    file_path_list=[os.path.join(dir_path,item) for item in file_list]
    return file_path_list


def check_type(path,type_list):
    if type_list is None:
        return True
    if not isinstance(type_list, list):
        type_list = [type_list]
    result=False
    for i in range(len(type_list)):
        if path.endswith(type_list[i]):
            result=True
            break
    return result