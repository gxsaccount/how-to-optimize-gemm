f = open("log") 
m = {} #dict{b,t:{range}}
for i in f.readlines():
    if i[0]!='b':
        continue 
    items = i.split(' ') 
    if items[0] not in m:
        # def update_range(old_range,new_range_str:str):
        #     new_range = new_range_str.removeprefix("from_a:(").removesuffix(")").split(":")
        #     if int(new_range[0]) < old_range[0]
        m[items[0]] = {} 
    if items[1] not in m[items[0]]:
        m[items[0]][items[1]] = []
    m[items[0]][items[1]].append(items[2]) 

from IPython import embed 
embed()
print(m)

