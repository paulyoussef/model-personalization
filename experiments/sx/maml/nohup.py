import os
import sys
os.system("nohup sh -c '" +
          # sys.executable + " w_static.py 0 > w_static.out &&" +
          # sys.executable + " w_static.py 1 > w_static.out &&" +
          # sys.executable + " w_static.py 2 > w_static.out &&" +
          # sys.executable + " w_static.py 3 > w_static.out &&" +
          sys.executable + " eval_train.py > eval_train.out" +
          "' &")
