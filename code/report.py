import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels
import json

def check_data():

    train_dataset = 'datasets/train_dataset.jsonl'

    fd = open(train_dataset, mode='r')

    count = 0

    count_high = 0
    count_low = 0

    count_gcc = 0
    count_clang = 0
    count_icc = 0

    for line in fd:
        count += 1
        
        json_data = json.loads(line)

        for key in json_data:

            if key == 'instructions':
                
                continue

            elif key == 'compiler':
                if json_data[key] == 'gcc':
                    count_gcc += 1
                elif json_data[key] == 'icc':
                    count_icc += 1
                elif json_data[key] == 'clang':
                    count_clang += 1
                else:
                    print('ho')
                    return

            elif key == 'opt':
                if json_data[key] == 'H':
                    count_high += 1
                elif json_data[key] == 'L':
                    count_low += 1
                else:
                    print("hu?")
                    return

            else:
                print("hi")
                return

    fd.close()

    print("{} number of json files!".format(count))
    print("{} number of json files optimized!".format(count_high))
    print("{} number of json files unoptimized!".format(count_low))
    print("{} number of json files compiled with gcc!".format(count_gcc))
    print("{} number of json files compiled with icc!".format(count_icc))
    print("{} number of json files compiled with clang!".format(count_clang))


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig('test.png')
    return ax


def plot_length():

    train_dataset_path = 'datasets/train_dataset.jsonl'
    train_file = open(train_dataset_path, mode='r')

    bigger = 0
    smaller = 0
    biggests = 0

    # containing the length of the functions
    lengths = list()

    for json_file in train_file:

        json_data = json.loads(json_file)

        n_instr = len(json_data['instructions'])
        lengths.append(n_instr)

        if n_instr > 250:
            bigger += 1
            if n_instr > 2000:
                biggests += 1
        else:
            smaller += 1

    train_file.close()

    # An "interface" to matplotlib.axes.Axes.hist() method
    plt.hist(lengths, bins='auto', color='#0504aa')
    plt.style.use('ggplot')
    plt.xlabel('Length')
    plt.ylabel('# of samples')
    #plt.title('Distribution of functions lengths')

    #plt.ylim(ymax=3000)
    plt.xlim(xmin=0, xmax=2000)

    plt.savefig('ml_hw1_report/images/lengths_plot.png')

    print("Percentage of functions shorter than 250: {}".format(smaller/(bigger+smaller)))
    print("Longest function is long: {}".format(max(lengths)))
    print("Shortest function is long: {}".format(min(lengths)))
    print("There are {} functions longer than 2000 instructions".format(biggests))
    

    return


if __name__ == "__main__":

    check_data()
    plot_length()
    #plot_confusion_matrix(test_y_number_occurrences, y_pred, classes=np.array(['L','H']), normalize=True)
    pass