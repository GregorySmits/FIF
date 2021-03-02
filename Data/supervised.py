fd=open('Data8.csv','r')

for i in fd.readlines():
    i=i.strip()
    d = i.split(',')
    if len(d) == 2:
        if float(d[1]) < 14 or float(d[1]) > 16.2:
            print(str(d[0])+","+str(d[1])+",1")
        else:
            print(str(d[0])+","+str(d[1])+",0")
