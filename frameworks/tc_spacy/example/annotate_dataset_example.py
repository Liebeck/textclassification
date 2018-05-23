import itertools
import json
from document_example import DocumentExample
from frameworks.tc_spacy.spacy_wrapper import SpacyWrapper


def jdefault(o):
    return o.__dict__


if __name__ == '__main__':
    examples_train = []
    examples_test = []
    examples_train.append(DocumentExample(unique_id='item1', label='Class 1', text='Dies ist ein kurzer Text.'))
    examples_train.append(DocumentExample(unique_id='item2', label='Class 1', text='Mein Auto ist blau.'))
    examples_train.append(DocumentExample(unique_id='item3', label='Class 2', text='Das Wetter soll morgen schön sein.'))
    examples_test.append(DocumentExample(unique_id='item4', label='Class 1', text='Heute gibt es Pizza zum Abendessen.'))
    examples_test.append(DocumentExample(unique_id='item5', label='Class 2', text='Zu viel Schlaf ist nicht gut.'))
    spacy_wrapper = SpacyWrapper()

    for document in itertools.chain(examples_train, examples_test):
        result = spacy_wrapper.process_document(document.text)
        document.tokens = result['tokens']
        document.dependencies = result['dependencies']

    with open('frameworks/tc_spacy/example/train.json', 'w') as outfile:
        json.dump(examples_train, outfile, indent=2, default=jdefault)
    with open('frameworks/tc_spacy/example/test.json', 'w') as outfile:
        json.dump(examples_test, outfile, indent=2, default=jdefault)
