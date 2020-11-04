
f = open('./deu.txt','r')
fwde = open('./deu.de','w')
fwen = open('./deu.en','w')

lines = [line for line in f.read().split('\n') if line]
for line in lines:
    items = line.split('\t')
    assert len(items)==3
    fwen.write(items[0] + '\n')
    fwde.write(items[1] + '\n')

fwde.close()
fwen.close()
