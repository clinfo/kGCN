import numpy as np
filename="protein.fa"
cnt=0
temp_lines=[]
for i,line in enumerate(open(filename)):
    if len(line.strip())==0:
        if cnt!=2:
            for l in temp_lines:
                print(l)
            print(cnt)
        cnt=0
        temp_lines=[]
    elif line[0]==">":
        temp_lines.append(line.strip())
        cnt+=1
    else:
        pass

