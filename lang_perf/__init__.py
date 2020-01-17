import os
import shutil
import platform

try:
    import lang_perf.cython_class
except:
    try:
        dir_path = os.path.dirname(__file__)
        bash_path = os.path.join(dir_path, "compile.sh")
        if platform.system() == "Linux":
            os.system("chmod 777 " + bash_path)
        os.system(bash_path)
        inner_dir_path = os.path.join(dir_path, "lang_perf")
        if os.path.isdir(inner_dir_path):
            for f in os.listdir(inner_dir_path):
                shutil.move(os.path.join(inner_dir_path, f), os.path.join(dir_path, f))
        import lang_perf.cython_class
    except:
        raise ImportError("Couldn't compile Cython lang-perf")
