import spacy
import pandas as pd
from spacy.util import decaying, env_opt


def get_product_position(x, y):
    """Find the product's position in the recall announcment based on the label.
    Otherwise return 0,0, '' to signify that the product does not exist."""
    if str(y) in str(x):
        result_start = x.index(y)
        result_end = result_start + len(y)
        return result_start, result_end, 'PRODUCT'
    else:
        return 0, 0, ""


def set_dropout():
    """Set the dropout range supported by Spacy"""
    dropout_rates = decaying(
        env_opt("dropout_from", 0.6),
        env_opt("dropout_to", 0.2),
        env_opt("dropout_decay", 1e-4),
    )
    return dropout_rates


def set_data_list(data, indexes):
    """ For each recall announcment from the given indexes creates the following tuples:
    tuple('recall announcment',{'entities': ["(Start, End, 'PRODUCT')"]})
    """
    data_list = [
        (
            data['processed'].iloc[index],
            dict(entities=[data['annotation'].iloc[index]])
        )
        for index in indexes
        # if str(data['annotation'].iloc[index]) != str((0, 0, ''))
    ]
    return data_list


def set_data_dict(data, indexes):
    """ Creates dictionary with the recall announcment being the key and the annotation being the value.
    {'recall announcment': {'entities': ["(Start, End, 'PRODUCT')"]}}
    """
    data_dict = {
        data['processed'].iloc[index]: dict(entities=[data['annotation'].iloc[index]])
        for index in indexes
        # if str(row['annotation']) != str((0, 0, ''))
    }
    return data_dict


def init_dictionaries():
    """ Initialize dictionaries used for Evaluation Strategy.
        We need to check type, partial, exact and strict matches.
    """
    categories = ['correct', 'incorrect', 'partial', 'missed', 'spurious', 'possible', 'actual', 'precision', 'recall']
    type_metrics_results = {k: 0 for k in categories}
    partial_metrics_results = {k: 0 for k in categories}
    exact_metrics_results = {k: 0 for k in categories}
    strict_metrics_results = {k: 0 for k in categories}
    return type_metrics_results, partial_metrics_results, exact_metrics_results, strict_metrics_results


def set_metrics(dicts):
    """ Helper method for calculating evaluation metrics"""
    for dictionary in dicts:
        dictionary['possible'] = dictionary['correct'] + dictionary['incorrect'] + dictionary['partial'] + dictionary[
            'missed']
        dictionary['actual'] = dictionary['correct'] + dictionary['incorrect'] + dictionary['partial'] + dictionary[
            'spurious']
        dictionary['precision'] = dictionary['correct'] / dictionary['actual']
        dictionary['recall'] = dictionary['correct'] / dictionary['possible']


def eval_model(model_path, testing_dict, testing_data):
    """ Implementation of Evaluation Strategy.
        https://www.davidsbatista.net/blog/2018/05/09/Named_Entity_Evaluation/
        SemEval'13
    """
    type_metrics_results, partial_metrics_results, exact_metrics_results, strict_metrics_results = init_dictionaries()
    nlp = spacy.load(model_path)
    for text, _ in testing_data:
        doc = nlp(text)  # model-based ner
        pred_ents = [(ent.text, ent.label_) for ent in doc.ents]  # list of predicted entities
        pred_string = ""
        golden_string = ""
        if len(pred_ents) != 0:
            pred_string = pred_ents[0][0]
        # find corresponding golden standard entities
        golden_ents_dict = testing_dict.get(text)  # returns a dict of entities (or None)
        if golden_ents_dict is not None and golden_ents_dict.get('entities')[0] != "(0, 0, '')":
            golden_ents = golden_ents_dict.get('entities')  # returns [(start,end,'PRODUCT')]
            start_point = golden_ents[0][0]
            end_point = golden_ents[0][1]
            golden_string = text[start_point:end_point]

        # HYPOTHESIS: each pred_ents contains one or no entries (what if there were more?)
        if golden_string == pred_string:
            type_metrics_results['correct'] += 1
            partial_metrics_results['correct'] += 1
            exact_metrics_results['correct'] += 1
            strict_metrics_results['correct'] += 1
        elif pred_string == "":
            type_metrics_results['missed'] += 1
            partial_metrics_results['missed'] += 1
            exact_metrics_results['missed'] += 1
            strict_metrics_results['missed'] += 1
        elif golden_string == "":
            type_metrics_results['spurious'] += 1
            partial_metrics_results['spurious'] += 1
            exact_metrics_results['spurious'] += 1
            strict_metrics_results['spurious'] += 1
        elif pred_string in golden_string:
            type_metrics_results['correct'] += 1
            partial_metrics_results['partial'] += 1
            exact_metrics_results['partial'] += 1
            strict_metrics_results['incorrect'] += 1
        else:
            type_metrics_results['correct'] += 1
            partial_metrics_results['incorrect'] += 1
            exact_metrics_results['incorrect'] += 1
            strict_metrics_results['incorrect'] += 1

        set_metrics(
            [type_metrics_results, partial_metrics_results, exact_metrics_results, strict_metrics_results]
        )

    s1 = pd.DataFrame.from_dict(type_metrics_results, orient='index', columns=['type'])
    s2 = pd.DataFrame.from_dict(partial_metrics_results, orient='index', columns=['partial'])
    s3 = pd.DataFrame.from_dict(exact_metrics_results, orient='index', columns=['exact'])
    s4 = pd.DataFrame.from_dict(strict_metrics_results, orient='index', columns=['strict'])

    testing_output = pd.concat([s1, s2, s3, s4], axis=1)
    return testing_output
