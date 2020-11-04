source_paths = ['./concatenated_en2de_train_de.txt', './concatenated_en2de_train_en.txt']
target_paths = ['./concatenated_en2de_train_sample2_de.txt', './concatenated_en2de_train_sample2_en.txt']

for source_path, target_path in zip(source_paths, target_paths):
    fr = open(source_path, 'r')
    sents = [sent for sent in fr.read().split('\n') if sent]
    fr.close()
    sents = sents[4000:5000]
    fw = open(target_path, 'w')
    fw.write('\n'.join(sents))
    fw.close()
    