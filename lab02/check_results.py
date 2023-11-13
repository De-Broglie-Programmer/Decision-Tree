import sys
import hashlib
import csv
import itertools

if sys.argv[1] == 'eval':
    hasher = hashlib.sha256()
    with open(f'results/achieved/exercice{sys.argv[2]}_eval.csv', 'rb') as file:
        buf = file.read()
    hasher.update(buf)
    achieved_hash = hasher.hexdigest()
    
    with open(f'results/expected/exercice{sys.argv[2]}_eval.hash') as hashfile:
        expected_hash = hashfile.read()
    if expected_hash == achieved_hash:
        print('Congratulations, your results for this exercice match the expected results!\n'
              'Please make sure that this result still hold in the code you will submit to get the points for this question.')
    else:
        print('Sorry, your results for this exercice don\'t match the expected results.\n'
              'Please check and correct your code and results to get this exercice\'s points.\n'
              'You can use the debug data to test your results against known expected results.\n'
              'If all your results seem good but you still get this message, please check the formatting of your result\'s file.')
else:
    all_correct = True
    with open(f'results/achieved/exercice{sys.argv[2]}_debug.csv') as achieved,\
         open(f'results/expected/exercice{sys.argv[2]}_debug.csv') as expected:
        exp_read = csv.reader(expected)
        ach_read = csv.reader(achieved)
        results_ziped = list(itertools.zip_longest(ach_read, exp_read))
        for idx_line, (ach_line, exp_line) in enumerate(results_ziped):
            if ach_line is None:
                print(f'Warning : the expected results are related to {len(results_ziped)} samples, but your achieved results are only related to {idx_line} samples. This would not be considered a correct answer in evaluation')
                all_correct = False
                break
            elif exp_line is None:
                print(f'Warning : the expected results are only related to {idx_line} samples, but your achieved results are related to {len(results_ziped)} samples. This would not be considered a correct answer in evaluation')
                all_correct = False
                break
            else:
                current_correct = True
                print(f'Sample {idx_line+1} : ')
                line_ziped = list(itertools.zip_longest(ach_line, exp_line))
                for idx_elem, (ach_elem, exp_elem) in enumerate(line_ziped):
                    convertible = False
                    try:
                        float_ach = float(ach_elem)
                        float_exp = float(exp_elem)
                    except ValueError:
                        convertible = False
                    if ach_elem is None:
                        print(f'    Warning : the expected line contains {len(line_ziped)} elements, but your achieved result line only contains {idx_elem} element. This would not be considered a correct answer in evaluation')
                        current_correct = False
                        break
                    elif exp_elem is None:
                        print(f'    Warning : the expected line only contains {idx_elem} elements, but your achieved result line contains {len(line_ziped)} element. This would not be considered a correct answer in evaluation')
                        current_correct = False
                        break
                    elif (convertible and float(ach_elem) != float(exp_elem))\
                        or (not convertible and ach_elem != exp_elem):
                        print(f'    The elements number {idx_elem} for this sample don\'t match. The expected element is {exp_elem} and the achieved one is {ach_elem}')
                        current_correct = False
                if current_correct:
                    print('    All results match for this sample')
                all_correct = all_correct and current_correct
    if all_correct:
        with open(f'results/achieved/exercice{sys.argv[2]}_debug.csv') as achieved,\
             open(f'results/expected/exercice{sys.argv[2]}_debug.csv') as expected:
            if achieved.read() == expected.read():
                print('Congratulations, all your results match the expected results')
            else:
                print('Your results match the expected results but the files are not strictly identical. Please check the formatting of your file')
                    
