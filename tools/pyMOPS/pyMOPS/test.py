import sys
sys.path.append("./")
import pyMOPS


print("Available API:", dir(pyMOPS))
print("\n\n")
    
pyMOPS.MOPS_Init("gpu")
