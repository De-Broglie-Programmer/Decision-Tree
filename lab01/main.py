from PointSet import PointSet
from Tree import Tree
from read_write import load_data, write_results
import csv
import sys
import evaluation

files_eval =\
[
    ['../input_data/eval_data1.csv', '../input_data/eval_data2.csv'],
    ['../input_data/eval_data1.csv', '../input_data/eval_data2.csv'],
    ['../input_data/exo4_eval1.csv', '../input_data/exo4_eval2.csv'],
    ['../input_data/eval_data1.csv', '../input_data/eval_data2.csv'],
    ['../input_data/eval_data1.csv', '../input_data/eval_data2.csv'],
    ['../input_data/cat_eval_data1.csv', '../input_data/cat_eval_data2.csv', '../input_data/cat_eval_data3.csv'],
]

files_debug =\
[
    ['../input_data/debug_data1.csv', '../input_data/debug_data2.csv', '../input_data/debug_data3.csv'],
    ['../input_data/debug_data1.csv', '../input_data/debug_data2.csv', '../input_data/debug_data3.csv'],
    ['../input_data/exo4_debug1.csv', '../input_data/exo4_debug2.csv'],
    ['../input_data/debug_data1.csv', '../input_data/debug_data2.csv', '../input_data/debug_data3.csv'],
    ['../input_data/debug_data1.csv', '../input_data/debug_data2.csv', '../input_data/debug_data3.csv'],
    ['../input_data/cat_debug_data1.csv', '../input_data/cat_debug_data2.csv', '../input_data/cat_debug_data3.csv'],
]

def exercice1(files_to_study):
    results = []
    for file in files_to_study:
        features, labels, types = load_data(file)
        ps_instance = PointSet(features, labels, types)
        results += [[ps_instance.get_gini()]]
    return results

def exercice2(files_to_study):
    results = []
    for file in files_to_study:
        features, labels, types = load_data(file)
        ps_instance = PointSet(features, labels, types)
        results += [ps_instance.get_best_gain()]
    return results

def exercice3(files_to_study):
    results = [];
    for file in files_to_study:
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            expected_res = []
            true_res = []
            for line in csv_reader:
                expected_res += [line[0] == '1']
                true_res += [line[1] == '1']
        precision, recall = evaluation.precision_recall(expected_res, true_res)
        results += [[precision, recall, evaluation.F1_score(expected_res, true_res)]]
    return results
    
def exercice4(files_to_study):
    results = []
    training_proportion = .8
    for file in files_to_study:
        features, labels, types = load_data(file)
        training_nb = int(len(features)*training_proportion)
        current_tree = Tree(features[:training_nb], labels[:training_nb], types)
        expected_results = labels[training_nb:]
        actual_results = []
        for point_features in features[training_nb:]:
            actual_results += [current_tree.decide(point_features)]
        results += [[evaluation.F1_score(expected_results, actual_results)]]
    return results

def exercice5(files_to_study):
    results = []
    training_proportion = .8
    for file in files_to_study:
        features, labels, types = load_data(file)
        training_nb = int(len(features)*training_proportion)
        current_tree = Tree(features[:training_nb], labels[:training_nb], types, h=2)
        expected_results = labels[training_nb:]
        actual_results = []
        for point_features in features[training_nb:]:
            actual_results += [current_tree.decide(point_features)]
        results += [[evaluation.F1_score(expected_results, actual_results)]]
    return results

if __name__ == '__main__':
    exercice = int(sys.argv[2])
    if sys.argv[1] == 'eval':
        files_to_study = files_eval[exercice-1]
        dest_file = f'results/achieved/exercice{exercice}_eval.csv'        
    else:
        files_to_study = files_debug[exercice-1]
        dest_file = f'results/achieved/exercice{exercice}_debug.csv'
    if exercice == 1:
        results = exercice1(files_to_study)
    elif exercice == 2:
        results = exercice2(files_to_study)
    elif exercice == 3:
        results = exercice3(files_to_study)
    elif exercice == 4:
        results = exercice4(files_to_study)
    elif exercice == 5:
        results = exercice5(files_to_study)
    elif exercice == 6:
        results = exercice5(files_to_study)
    write_results(results, dest_file)
