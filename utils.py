import inspect
import sys

def raiseNotDefined():
    filename = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]
    
    print(f"*** Method not implemented: {method} at line {line} of {filename} ***")
    sys.exit()