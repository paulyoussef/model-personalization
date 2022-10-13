import os
import sys
os.system("nohup sh -c '" +
          # sys.executable + " wo_static.py 0 > wo_static.out &&" +
          # sys.executable + " wo_static.py 1 > wo_static.out &&" +
          # sys.executable + " wo_static.py 2 > wo_static.out &&" +
          # sys.executable + " wo_static.py 3 > wo_static.out &&" +
          sys.executable + " eval_train.py > eval_train.out" +
          "' &")
