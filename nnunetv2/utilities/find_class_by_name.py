import importlib
import pkgutil

from batchgenerators.utilities.file_and_folder_operations import *


# 递归地从指定的文件夹中查找Python类
# 它接收三个参数：文件夹路径，类名和当前模块的名称
def recursive_find_python_class(folder: str, class_name: str, current_module: str):
    tr = None
    # 调用 pkgutil.iter_modules([folder]) 函数，该函数返回一个可迭代对象，用于迭代指定文件夹中的模块
    # iter_modules() 函数被调用时传入了一个参数 [folder]，该参数是一个包含一个目录路径的列表。iter_modules() 函数将会迭代该目录下的所有模块，并返回一个可迭代对象
    for importer, modname, ispkg in pkgutil.iter_modules([folder]):
        # print(modname, ispkg)
        # 如果模块不是一个包，则尝试导入该模块
        if not ispkg:
            # 使用 importlib.import_module() 函数导入该模块。
            m = importlib.import_module(current_module + "." + modname)
            # 检查该模块是否包含特定的类名
            if hasattr(m, class_name):
                # 如果找到符合条件的类，则返回该类的引用并退出循环
                tr = getattr(m, class_name)
                break

    # 如果在指定的文件夹下没有找到符合条件的类，则递归地查找子目录中的模块
    if tr is None:
        for importer, modname, ispkg in pkgutil.iter_modules([folder]):
            if ispkg:
                next_current_module = current_module + "." + modname
                tr = recursive_find_python_class(join(folder, modname), class_name, current_module=next_current_module)
            if tr is not None:
                break
    return tr