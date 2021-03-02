fd = open("Donut.csv",'r')
lines = fd.readlines()
l=len(lines)
for i in range(len(lines)):
    if i < l -6:
        print(lines[i].strip()+",0")
    else:
        print(lines[i].strip()+",1")