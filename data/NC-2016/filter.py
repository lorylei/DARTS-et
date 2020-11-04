base_path_source = './'
base_path_target = '../NC2016/'

files = ['concatenated_en2de_dev_de.txt', 'concatenated_en2de_dev_en.txt', 'concatenated_en2de_test_de.txt', 'concatenated_en2de_test_en.txt', 'concatenated_en2de_train_de.txt', 'concatenated_en2de_train_en.txt']

for file_path in files:
    source_path = base_path_source + file_path 
    target_path = base_path_target + file_path
    fr = open(source_path, 'r')
    items = [item for item in fr.read().split('<d>\n') if item]
    fr.close()
    fw = open(target_path, 'w')
    fw.write(''.join(items))
    fw.close()
    