import json
import numpy as np

from preprocess import recover_vocabulary
from sklearn import svm, naive_bayes

if __name__ == "__main__":
    partial_mapping_path = 'partial_mnemonic_to_index.csv'
    train_path = 'datasets/train_dataset.jsonl'

    partial_mapping = recover_vocabulary(partial_mapping_path)

    train_input_file = open(train_path, mode='r')
    train_x = list()
    train_y_opt = list()
    train_y_cmp = list()

    for json_line in train_input_file:
        json_data = json.loads(json_line)
        current_features = np.zeros(len(partial_mapping))
        
        for instr in json_data['instructions']:
            mnemonic = instr.split(" ")[0].rstrip()

            current_features[partial_mapping[mnemonic] if mnemonic in partial_mapping else partial_mapping['<UNK>']] += 1

        train_x.append(current_features)
        train_y_opt.append(json_data['opt'])
        train_y_cmp.append(json_data['compiler'])

    train_input_file.close()
    
    opt_model = naive_bayes.MultinomialNB()
    opt_model.fit(train_x, train_y_opt)

    cmp_model = naive_bayes.MultinomialNB()
    cmp_model.fit(train_x, train_y_cmp)

    test_path = 'datasets/test_dataset_blind.jsonl'
    output_path = '1711234.csv'

    test_input_file = open(test_path, mode='r')
    output_file = open(output_path, mode='w')

    for json_line in test_input_file:
        json_data = json.loads(json_line)
        current_features = np.zeros(len(partial_mapping))

        for instr in json_data['instructions']:
            mnemonic = instr.split(" ")[0].rstrip()

            current_features[partial_mapping[mnemonic] if mnemonic in partial_mapping else partial_mapping['<UNK>']] += 1

        output_file.write(str((cmp_model.predict([current_features]))[0]) + "," + str((opt_model.predict([current_features]))[0]) + "\n")

    test_input_file.close()
    output_file.close()