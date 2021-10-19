import pandas as pd
import spacy
import numpy as np
import random
import utils as ut
from tqdm import tqdm

def train_model(training_data, epochs):
    nlp = spacy.blank("en")  # create blank Language class
    print("Created blank 'en' model")

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe('ner')

    dropout_rates = ut.set_dropout()
    for _, annotations in training_data:
        # train data ["str",{"entities": tag}]
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        optimizer = nlp.begin_training()
        for itn in range(epochs):
            random.shuffle(training_data)
            losses = {}
            for text, annotations in tqdm(training_data):
                nlp.update(
                    [text],
                    [annotations],
                    drop=next(dropout_rates),
                    sgd=optimizer,
                    losses=losses)
            print(losses)
    return nlp
