import json
import numpy as np


def parse_jsonl_file(jsonl_path: str, training_output_path: str, testing_output_path: str, mapping: dict, features_style: int,
                     ratio: int = 0.85, sep=',', n_files=-1) -> None:
    """
        parse_jsonl_file(jsonl_path: str, output_path: str, mapping: dict, features_style: int, sep=',', n_files=-1) -> None:
        takes as INPUTS:
            -jsonl_path: the path to the jsonl file which we need to parse
            -training_output_path: the path to the output file on which the parsed jsonl file will be written
            -testing_output_path: the path to the output file on which the parsed jsonl file will be written
            -mapping: the mapping from mnemonics to unique indices
            -features_style: -if 0: the values in the output file will be a list of integers, 
                                    where the integer in position i will be the number of occurrences of the instruction 
                                    associated to it(accordingly with the output dictionary) in the function.
                             -if positive: the values in the output file will be a list of integers of length features_style,
                                    where the integer in position i will be the number associated to the instruction executed as number i-th in that function
                                    and a series of zeros if the function is shorter than features_style.
                             -if negative: the values in the output file will be a list of integers of length abs(features_style)*2+1,
                                    where the numbers in the first abs(features_style) positions will be associated to the instructions executed as number i-th
                                        the numbers in the following abs(features_style) positions will be associated to the last abs(features_style) executed instructions
                                        the last number will represent the number of instructions in the function
            -[DEFAULT = 0.85] ratio: the ratio between the size of the training set and the testing one
            -[DEFAULT = ','] sep: separator to use in the output file, which should be of an appropriate format; e.g. '.csv' if sep=','
            -[DEFAULT = -1] n_files: number of json_files to parse, 
                            if this number is negative or greater than the number of rows in jsonl_path, the whole file will be parsed.
        DOES:
            Parses the first n_files lines in jsonl_path considering each one of them as a different json file and 
            writes them either in the training_output_path or in the testing_output_file according to ratio using the following format:
                value1*sep*value2*sep*...*sep*valueN\n
            for every of the N keys in the corresponding json file
        and OUTPUTS:
            Nothing
    """

    jsonl_file = open(jsonl_path, mode='r')
    training_file = open(training_output_path, mode='w')
    testing_file = open(testing_output_path, mode='w')

    for json_string in jsonl_file:
        # every line basically is a json file of his own

        if n_files == 0:
            break

        ###
        # Actual operations begin
        ###

        loaded_json = json.loads(json_string)

        optimized_value = '0'
        compiler_value = '0'

        if loaded_json['opt'] == 'H':
            optimized_value = '1'

        if loaded_json['compiler'] == 'gcc':
            compiler_value = '1'
        elif loaded_json['compiler'] == 'icc':
            compiler_value = '2'

        if features_style == 0:  #  number of occurrences
            current_features = np.zeros(len(mapping))

            for instr in loaded_json['instructions']:
                mnemonic = instr.split(" ")[0].rstrip()

                current_features[mapping[mnemonic] if mnemonic in mapping else mapping['<UNK>']] += 1

            if (np.random.uniform() < ratio):
                training_file.write(' '.join(str(int(el)) for el in current_features) +
                                    sep + optimized_value + sep + compiler_value + '\n')
            else:
                testing_file.write(' '.join(str(int(el)) for el in current_features) +
                                   sep + optimized_value + sep + compiler_value + '\n')

        elif features_style > 0:  # ordered instructions
            current_features = np.zeros(features_style)
            random_value = np.random.uniform()
            i = 0

            for instr in loaded_json['instructions']:
                mnemonic = instr.split(" ")[0].rstrip()

                current_features[i] = mapping[mnemonic] if mnemonic in mapping else mapping['<UNK>']

                i += 1
                if (i == features_style):
                    if (random_value < ratio):
                        training_file.write(' '.join(str(int(
                            el)) for el in current_features) + sep + optimized_value + sep + compiler_value + '\n')
                    else:
                        testing_file.write(' '.join(str(int(
                            el)) for el in current_features) + sep + optimized_value + sep + compiler_value + '\n')

                    current_features = np.zeros(features_style)
                    i = 0

            if (i != 0):
                if (random_value < ratio):
                    training_file.write(' '.join(str(int(
                        el)) for el in current_features) + sep + optimized_value + sep + compiler_value + '\n')
                else:
                    testing_file.write(' '.join(str(int(
                        el)) for el in current_features) + sep + optimized_value + sep + compiler_value + '\n')

        else:  # first abs(features_style) instructions, last abs(features_style) ones and the number of instructions in the function

            n_instructions = len(loaded_json['instructions'])
            absolute_features_style = abs(features_style)

            current_features = np.zeros(absolute_features_style*2+1)
            current_features[-1] = n_instructions

            mnemonics = (instr.split(" ")[0].rstrip() for instr in loaded_json['instructions'])
            mapped_mnemonics = list(mapping[mnemonic] if mnemonic in mapping else mapping['<UNK>'] for mnemonic in mnemonics)

            if n_instructions >= absolute_features_style:
                current_features[:absolute_features_style] = mapped_mnemonics[:absolute_features_style]
                current_features[absolute_features_style:-1] = mapped_mnemonics[-absolute_features_style:]
            else:
                current_features[:n_instructions] = mapped_mnemonics[:]
                current_features[absolute_features_style:absolute_features_style+n_instructions] = mapped_mnemonics[:]

            if (np.random.uniform() < ratio):
                training_file.write(' '.join(str(int(el)) for el in current_features) +
                                    sep + optimized_value + sep + compiler_value + '\n')
            else:
                testing_file.write(' '.join(str(int(el)) for el in current_features) +
                                   sep + optimized_value + sep + compiler_value + '\n')

        ###
        # Actual operations end
        ###

        try:
            # just parsed one more json file
            n_files -= 1

        except OverflowError as _:
            # almost impossible to occur
            # still, better safe than sorry

            print("Wow, you're file is huge! Probably you are from the future(or you've fed me a very big negative number as n_files)\n")
            print(
                "I've encountered an OverflowError, I need an input to decide what to do next.\n")
            inp = input("Enter 'STOP' if you want me to save the output file as it is and return.\nEnter 'GO' if you want me to continue parsing.\n(Both options are equally safe)\n").lower()

            while(inp != 'stop' and inp != 'go'):
                inp = input(
                    "Input string should either be 'STOP' or 'GO', nothing else is handled. Retry:\n").lower()

            if(inp == 'stop'):
                # closing opened files to avoid any issues
                jsonl_file.close()
                training_file.close()
                testing_file.close()

                return

            elif(inp == 'go'):  #  silly check
                n_files = -1

    # closing opened files to avoid any issues
    jsonl_file.close()
    training_file.close()
    testing_file.close()


def save_vocabulary(voc: dict, output_path: str, n: int = 1, m: int = 1, sep: str = ',') -> None:
    """
        save_vocabulary(voc: dict, output_path: str, n: int = 1, m: int = 1, sep: str = ',') -> None:
        takes as INPUTS:
            -voc: The dictionary to save
            -output_path: The path where to save the vocabulary
            -[DEFAULT = 1]n : the number of elements in the keys
            -[DEFAULT = 1]m : the number of elements in the values
            -[DEFAULT = ',']sep: the separator to print between keys and values
        DOES
            Takes as input a vocabulary and saves it in the following format:
                key[0] + sep + key[1] + ... + key[n] + sep + value[0] + sep + value[1] + ... + value[m].
            The first row will be written as:
                n + sep + m

            Notice: Both key and value elements __repr__ function should be implemented.
        and OUTPUTS:
            Nothing
    """

    f = open(output_path, mode='w')

    f.write(str(n) + sep + str(m) + '\n')

    for key, value in voc.items():
        if n == 1:
            f.write(str(key) + sep)
        elif n != 0:
            for i in range(n):
                f.write(str(key[i]) + sep)

        if m == 1:
            f.write(str(value) + '\n')
        elif m != 0:
            for i in range(m):
                f.write(str(value[i]) + sep)
            f.write('\n')

    # Closing the file to avoid any issues
    f.close()


def recover_vocabulary(voc_path: str, sep: str = ',') -> dict:
    """
        recover_vocabulary(voc_path: str, sep: str = ',') -> dict
        takes as INPUTS:
            -voc_path: The path to the file where the dictionary has been saved
            -[DEFAULT = ',']sep: the separator used in the file to separate values
        DOES:
            Takes as input a path to a file and returns the dictionary encoded into the file according to the following format:
                key[0] + sep + key[1] + ... + key[n] + sep + value[0] + sep + value[1] + ... + value[m].
            The first row must be written as:
                n + sep + m
        and OUTPUTS:
            A dictionary mapping every tuple of the first n elements to a list containing the last m elements of every row.
            Actually if n == 1 the keys will just be value, likewise if m == 1, value won't be lists but values.
    """

    mapping = dict()

    f = open(voc_path, mode='r')

    n, m = f.readline().rstrip().split(sep)

    n = int(n)
    m = int(m)

    for line in f:

        splitted = line.rstrip().split(sep)

        if n == 1:
            try:
                key = int(splitted[0])
            except:
                key = splitted[0]
        else:
            temp = list()

            # This function was supposed to be as general as possible but to avoid useless complications I've tweaked it a little bit for my use-case
            # First element in my case must always be a string
            temp.append(splitted[0])

            for i in range(1, n):
                try:
                    temp.append(int(splitted[i]))
                except:
                    temp.append(splitted[i])

            # The key of a dictionary must be hashable, lists are not
            key = tuple(temp)

        if m == 1:
            try:
                value = int(splitted[-1])
            except:
                value = splitted[-1]
        else:
            value = list()
            for i in range(m):
                try:
                    value.append(int(splitted[n+i]))
                except:
                    value.append(splitted[n+1])

        mapping[key] = value

    # Closing the file to avoid any issues
    f.close()

    return mapping


def get_mapping(jsonl_path: str, min_occurrences: int = 0) -> dict:
    """
        get_mapping(jsonl_path: str, min_occurrences: int) -> dict:
        takes as INPUTS:
            -jsonl_path: the path to the jsonl file
            -min_occurrences: the number of occurrences in jsonl_path to be saved in the mapping
        DOES:
            Associates one unique index to each unique mnemonic which appears at least min_occurrences times in jsonl_path
        and OUTPUTS:
            The dictionary containing the mapping between every mnemonic and the associated index and two special mappings:
                '<PAD>' -> 0 needed for padding fixed-length sequences
                '<UNK>' -> 1 needed for mnemonics which appear less min_occurrences times in jsonl_path
    """

    input_file = open(jsonl_path, mode='r')
    count = dict()

    for json_text in input_file:
        json_data = json.loads(json_text)

        for instr in json_data['instructions']:
            mnemonic = instr.split(" ")[0].rstrip()

            if mnemonic not in count:
                count[mnemonic] = 1
            else:
                count[mnemonic] += 1

    mapping = dict()

    mapping['<PAD>'] = 0
    mapping['<UNK>'] = 1

    for key, value in count.items():
        if value > min_occurrences:
            mapping[key] = len(mapping)

    return mapping


if __name__ == "__main__":

    # hyperaparameters:
    # number of minimum occurrences needed to actually assign a label to a mnemonic
    min_occurrences = 1000
    # length of fixed-length sequences for ordered instructions
    fixed_length = 50
    # number of instructions in the beginning and in the end to get from every function
    n_instructions = 5

    # storing the mappings
    full_mapping = get_mapping('datasets/train_dataset.jsonl', 0)
    full_mapping_path = 'full_mnemonic_to_index.csv'
    save_vocabulary(full_mapping, full_mapping_path, n=1, m=1, sep=',')

    partial_mapping = get_mapping('datasets/train_dataset.jsonl', min_occurrences)
    partial_mapping_path = 'partial_mnemonic_to_index.csv'
    save_vocabulary(partial_mapping, partial_mapping_path, n=1, m=1, sep=',')

    # if you already have the mappings run these instead
    #full_mapping_path = 'mappings/full_mnemonic_to_index.csv'
    #full_mapping = recover_vocabulary(full_mapping_path)
    #partial_mapping_path = 'mappings/partial_mnemonic_to_index.csv'
    #partial_mapping = recover_vocabulary(partial_mapping_path)

    # generating the parsed datasets
    # using number of occurrences of every mnemonic in the mappings as features
    parse_jsonl_file('datasets/train_dataset.jsonl', 'datasets/parsed/train_full_number_occurrences.csv',
                     'datasets/parsed/test_full_number_occurrences.csv', full_mapping, 0, ratio=0.9)
    parse_jsonl_file('datasets/train_dataset.jsonl', 'datasets/parsed/train_partial_number_occurrences.csv',
                     'datasets/parsed/test_partial_number_occurrences.csv', partial_mapping, 0, ratio=0.9)
    # using an ordered list of every mnemonic, in the mappings, in the functions
    parse_jsonl_file('datasets/train_dataset.jsonl', 'datasets/parsed/train_full_ordered_instructions.csv',
                     'datasets/parsed/test_full_ordered_instructions.csv', full_mapping, fixed_length, ratio=0.9)
    parse_jsonl_file('datasets/train_dataset.jsonl', 'datasets/parsed/train_partial_ordered_instructions.csv',
                     'datasets/parsed/test_partial_ordered_instructions.csv', partial_mapping, fixed_length, ratio=0.9)
    # using a mixed feature: structured as: first n_instructions instructions + last n_instructions instructions + number of instructions
    parse_jsonl_file('datasets/train_dataset.jsonl', 'datasets/parsed/train_full_mixed_features.csv',
                     'datasets/parsed/test_full_mixed_features.csv', full_mapping, -n_instructions, ratio=0.9)
    parse_jsonl_file('datasets/train_dataset.jsonl', 'datasets/parsed/train_partial_mixed_features.csv',
                     'datasets/parsed/test_partial_mixed_features.csv', partial_mapping, -n_instructions, ratio=0.9)
