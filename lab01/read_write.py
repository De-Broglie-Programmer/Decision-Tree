from typing import Tuple, List

import csv
from PointSet import FeaturesTypes

def load_data(file_name: str) -> Tuple[List[List[float]], List[bool], List[FeaturesTypes]]:
    """Read the content of the file.

    The file should be formatted as a csv delimited by ',',
    with a first line containing the types of the features
    and the following lines containing each the data of a
    point.

    Each column of the first line should contain only one
    of the following letters:
    - 'l' : if the column contains the points labels.
            Exactly one column should have this type.
    - 'b' : if the column contains a boolean feature.
    - 'c' : if the column contains a categorial feature.
    - 'r' : if the column contains a continuous feature.

    Parameters
    ----------
        file_name : str
            The name or path of the file to read

    Returns
    -------
        List[List[float]]
            The features of the points. Each of the sublist
            is related to a single point. This list does not
            contain the labels.
        List[bool]
            The labels of the points.
        List[FeaturesTypes]
            The types of the features.
    """
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        labels = []
        features = []
        features_types = []
        label_id = -1
        for i, datum in enumerate(next(csv_reader)):
            if datum=='l':
                label_id = i
            elif datum=='b':
                features_types += [FeaturesTypes.BOOLEAN]
            elif datum=='c':
                features_types += [FeaturesTypes.CLASSES]
            elif datum=='r':
                features_types += [FeaturesTypes.REAL]
            else:
                raise NotImplementedError(f'Unknown data type header : {datum}')
        if label_id < 0:
            raise Exception('Label ID not found in file header')
        for line in csv_reader:
            labels += [line[label_id]=='1']
            features += [[float(val) for i, val in enumerate(line) if i != label_id]]
    return features, labels, features_types

def format_result(result) -> str:
    """Format a result into an unambiguous string

    Parameters
    ----------
        result : Any type that can be casted to a str
            The result to convert

    Returns
    -------
        str
            The formatted result
    """
    if isinstance(result, float):
        return "{:.6f}".format(result)
    else:
        return str(result)

def write_results(results: List[List], file_name: str) -> None:
    """Write the results into a file in an unambiguous manner

    Parameters
    ----------
        results : List[List]
            The results to write. Each sublist correspond
            to a given experiment, of which result will be
            printed to a separated line of the file.
        file : str
            The name or path of the file into which the result
            should be printed.
    """
    results = [[format_result(val) for val in row] for row in results]
    with open(file_name, 'w', newline='') as dest_file:
        csv_writer = csv.writer(dest_file, delimiter=',', lineterminator='\r\n')
        csv_writer.writerows(results)

