import math

if __name__ == '__main__':
    f = "train600k.txt"
    count_map = {}
    with open(f) as rels:
        for rel in rels:
            a = rel.split("\t")
            if a[0] not in count_map.keys():
                count_map[a[0]]=0
            count_map[a[0]]+=1
    tups = []
    for key in count_map.keys():
        tups.append((key,count_map[key]))
    s = sorted(tups,key=lambda x: x[1])
    s_r = sorted(tups,key=lambda x: x[1],reverse=True)
    print("Sorted:")
    print(s_r)
    train_tot = 0
    tot = 0
    for v in s:
        tot+=v[1]
    for idx, v in enumerate(s):
        if (idx*1.0)/len(s)<0.63:
            print(v)
            train_tot+=v[1]
        else:
            print(idx,len(s))
            print(train_tot,tot)
            print(s[idx-1][1],s[idx-1][1]*(len(s)-idx))
            print(s[idx-1][1]*(len(s)-idx)+train_tot)
            import numpy as np
            t=[s[x][1] for x in range(idx)]+[s[idx-1][1] for x in range(len(s)-idx)]
            print(len(t))
            average_samples = np.average(t)
            std_dev_samples = np.std([s[x][1] for x in range(idx)]+[s[idx-1][1] for x in range(len(s)-idx)])
            print(average_samples,std_dev_samples)
            break