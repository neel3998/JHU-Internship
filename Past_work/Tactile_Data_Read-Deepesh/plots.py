import matplotlib.pyplot as plt
import time
import numpy as np
filee=open("./tactile_test.txt","r")
a=[]
for line in filee:
    a.append(line.strip().split())
#print(a)
b=[]
for i in a:
    for j in i:
        b.append(float(j))
print("Minimum value "+str(min(b)))
print("Maximum value " +str(max(b)))
print("Mean Value "+str(np.mean(b)))
print("Standard Deviation "+str(np.std(b)))
#Plot
plt.plot(b)
plt.ylim((2,3))
plt.show()
'''
21 May 2021
Untouched:
Expected outcome for port 4:
Minimum value 2.3638183593749997
Maximum value 2.44886962890625
Mean Value 2.436410742295696
Standard Deviation 0.018984020214603742

Expected outcome for port 3:
Minimum value 2.4403564453124997
Maximum value 2.450830078125
Mean Value 2.4442794346628487
Standard Deviation 0.0023577556727929303

Expected outcome for port 2:
Minimum value 2.3420318603515624
Maximum value 2.4410251464843746
Mean Value 2.426483423039951
Standard Deviation 0.022608812294732383

Expected outcome for port 1:
Minimum value 2.3444260253906246
Maximum value 2.441637451171875
Mean Value 2.426094382075945
Standard Deviation 0.02165582221318424

Expected outcome for port 0:
Minimum value 2.35577783203125
Maximum value 2.447357666015625
Mean Value 2.434048443312799
Standard Deviation 0.02089434145474797

Touched:
Expected outcome for port 0:
Minimum value 2.0909999999999997
Maximum value 2.4500244140625
Mean Value 2.4210151724273525
Standard Deviation 0.05328579688182751
'''