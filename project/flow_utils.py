"""

    This script collects utility functions to make the flow file smaller and more readable.

"""


from datetime import datetime


def get_finance_sentiment_dataset(split: str='sentences_allagree') -> list:
    """
        Load financial dataset from HF: https://huggingface.co/datasets/financial_phrasebank

        Note that there's no train/validation/test split: the dataset is available in four possible 
        configurations depending on the percentage of agreement of annotators. By default, load just 
        sentences for which all annotators agree.
    """
    from datasets import load_dataset
    dataset = load_dataset("financial_phrasebank", split)
    
    return dataset['train']


def get_finance_sentences():
    """
        Load and clean up sentences from the dataset.
    """
    dataset = get_finance_sentiment_dataset()
    cleaned_dataset = [[pre_process_sentence(_['sentence']), _['label']] for _ in dataset]
    
    return cleaned_dataset


def pre_process_sentence(sentence: str) -> str:
    """
        Given a sentence, return a new one all lower-cased and without punctuation.
    """
    import string
    # lower case
    lower_sentence = sentence.lower()
    # remove punctuation
    exclude = set(string.punctuation)
    return ''.join(ch for ch in lower_sentence if ch not in exclude)


def tf_idf_vectorizer(X_train: list, X_test: list) -> tuple:
    """
        Given a list of sentences, return a list of vectors based on TF-IDF weighting scheme
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(analyzer='word', stop_words='english')
    _X_train = vectorizer.fit_transform(X_train)
    _X_test = vectorizer.transform(X_test)
    
    return  vectorizer, _X_train, _X_test


def get_classification_model():
    """
        Returns a scikit model with the usual fit / predict interface. For now, we always return naive bayes:

        See: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB

    """
    from sklearn.naive_bayes import MultinomialNB
    
    return MultinomialNB()


def evaluate_model_performance(y_test: list, y_predicted):
    """
        Returns a quantitative assessment of model performances. For now, we just re-use
        the built-in classification_report from sklearn

    """
    from sklearn.metrics import classification_report

    return classification_report(y_test, y_predicted)


def back_translate(sentences: list):
    """
        Use BackTranslation to perform back-translation.
    """

    from BackTranslation import BackTranslation

    trans = BackTranslation(url=[
        'translate.google.com',
        'translate.google.co.kr',
        ])

    translated_sentences = []
    for t in sentences:
        result = trans.translate(t, src='en', tmp = 'zh-cn')
        translated_sentences.append(result.result_text)

    assert len(translated_sentences) == len(sentences)

    return translated_sentences


def create_perturbated_sentences(sentences: list):
    """
        Given original financial use, return sentences with the same meaning but different phrasing.

        For now, we return with no further processing the BackTranslation package  
    """
    return back_translate(sentences)


def test_on_quarterly_info(X_test: list, predicted: list, y_test, quarter_keyword: str='quarter'):
    """
        Run quant. metrics only on test sentences referring to quarterly info.
    """
    return report_metrics_on_subset(
        X_test,
        predicted,
        y_test,
        slice_function=lambda x : quarter_keyword in x  
    )

def test_on_company(X_test: list, predicted: list, y_test, target_company: str):
    """
        Run quant. metrics only on test sentences mentioning specific companies.
    """
    return report_metrics_on_subset(
        X_test,
        predicted,
        y_test,
        slice_function=lambda x : target_company in x  
    )


def report_metrics_on_subset(X_test: list, predicted: list, y_test: list, slice_function):
    """
        Generic function that runs on the test set and produce a quant. report only if the
        supplied function returns true on the input, i.e. only if the test case contains the word
        'quarter'.
    """
    target_golden = []
    target_predicted = []
    for x, p, y in zip(X_test, predicted, y_test):
        if slice_function(x):
           target_golden.append(y)
           target_predicted.append(p) 

    return evaluate_model_performance(target_golden, target_predicted)