"""

This script is a Metaflow-based refactoring of the text classification pipeline in the notebook. Its purpose is to
show a more realistic DAG as an example for the final project: note that we skip some steps for brevity, so refer 
to the project checklist for a more complete set of features/requirements.

This script has been created for pedagogical purposes, and it does NOT necessarely reflect all best practices.

"""


from metaflow import FlowSpec, step, Parameter, current
from datetime import datetime


class FinanceNewsFlow(FlowSpec):
    """
    FinanceNewsFlow is a minimal DAG showcasing a scikit text classification pipeline for sentiment
    analysis over financial news.

    Data from: https://huggingface.co/datasets/financial_phrasebank
    """
    
    # this parameter tells the DAG where to dump the final model, so that the Flask app can load it
    # and serve predictions. It defaults to my own path ;-) 
    FINAL_FOLDER = Parameter(
        name='final_folder',
        help='Determining the folder in which to store the pickled model, after training',
        default='/Users/jacopotagliabue/Documents/repos/FREE_7773-1/project'
    )

    @step
    def start(self):
        """
        Start up and print out some info to make sure everything is ok metaflow-side
        """
        print("Starting up at {}".format(datetime.utcnow()))
        print("flow name: {}".format(current.flow_name))
        print("run id: {}".format(current.run_id))
        print("username: {}".format(current.username))
        print("Final folder: {}".format(self.FINAL_FOLDER))

        self.next(self.load_data)

    @step
    def load_data(self): 
        """
        Read the data in using the HF API.
        """
        from flow_utils import get_finance_sentences

        # get the dataset and use self to version it
        self.finance_dataset = get_finance_sentences()
        # get sentences and labels to simplify downstream vectorization
        self.raw_sentences = [_[0] for _ in self.finance_dataset]
        self.raw_labels = [_[1] for _ in self.finance_dataset]
        # debug / info
        print("Total # of sentences loaded is: {}".format(len(self.finance_dataset)))
        # go to the next step
        self.next(self.check_dataset)

    @step
    def check_dataset(self):
        """
        Check data for anomalous data points and weird labels
        """
        # first, check all sentences are "long enough", > 20 chars, otherwise flag them
        self.labels = []
        self.sentences = []
        for s, l in zip(self.raw_sentences, self.raw_labels):
            if len(s) < 20:
                print("====> Sentence '{}' seems too short, ignoring it for now".format(s))
                continue
            self.labels.append(l)
            self.sentences.append(s)

        # make sure # labels and sentences is the same
        assert len(self.labels) == len(self.sentences)
        # check we actually have only 3 target classes, as expected 
        all_labels = set(self.labels)
        assert len(all_labels) == 3
        print("All labels are: {}".format(all_labels))
        # if data is all good, let's go to training
        self.next(self.prepare_train_and_test_dataset)

    @step
    def prepare_train_and_test_dataset(self):
        """
        Train / test split 

        TODO: add a Flow parameter to make test_size configurable at each run
        """
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.sentences,
            self.labels,
            test_size=0.2, 
            random_state=42)

        # debug / info
        print("# train sentences: {},  # test: {}".format(len(self.X_train), len(self.X_test)))

        self.next(self.prepare_features)

    @step
    def prepare_features(self):
        """
        Transform our Xs (the sentences) using TF-IDF 

        TODO: once we identify TF-IDF hyper-params, train multiple classifier to tune the model! See for example
        the analysis here: https://www.highonscience.com/blog/2021/05/24/ml-model-selection-with-metaflow/
        """
        from flow_utils import tf_idf_vectorizer

        self.vectorizer, self.X_train_vectorized, self.X_test_vectorized = tf_idf_vectorizer(self.X_train, self.X_test)
        # train a model now that we have the features
        self.next(self.train_classifier)

    @step
    def train_classifier(self):
        """
        Get a scikit model and train it on the vectorized text
        """
        from flow_utils import get_classification_model

        model = get_classification_model()
        model.fit(self.X_train_vectorized, self.y_train)
        # versioned the trained model using self
        self.trained_model = model
        # go to the testing phase
        self.next(self.test_model)

    @step 
    def test_model(self):
        """
        Test the model on the held out sample

        TODO: add confusion matrix and a plot!
        """
        from flow_utils import evaluate_model_performance

        self.predicted = self.trained_model.predict(self.X_test_vectorized)
        self.report = evaluate_model_performance(self.y_test, self.predicted)
        # print out the report
        print("!!!!! Classification Report !!!!!")
        print(self.report)
        # all is done go to the end
        self.next(self.beahvioral_tests)

    @step
    def beahvioral_tests(self):
        """
        As we learned in the course, it is very important to not just test quantitave behavior, but diving 
        deep into model performances on cases of interest, slices of data, perturbed input.continue

        Note: we don't make the Flow fail here, but just flag when a test does not have the desired result.
        Other choices are possible of course.
        """
        from random import randint
        from flow_utils import test_on_company, test_on_quarterly_info, create_perturbated_sentences

        # slice data by quarter
        self.quarterly_report = test_on_quarterly_info(self.X_test, self.predicted, self.y_test)
        # print out the report
        print("\n$$$ Classification Report on Quarterly News Only $$$")
        print(self.quarterly_report)
        # report performances on, say, https://en.wikipedia.org/wiki/Comptel
        self.company_report = test_on_company(self.X_test, self.predicted, self.y_test, target_company='comptel')
        print("\n$$$ Classification Report on Quarterly News Only $$$")
        print(self.company_report)
        # finally, some perturbation tests over 2 randomly sampled cases
        rnd_index = [randint(0, len(self.X_test)) for _ in range(2)]
        self.test_sentences = [self.X_test[_] for _ in rnd_index]
        self.test_predictions = [self.predicted[_] for _ in rnd_index]
        self.perturbated_test_sentences = create_perturbated_sentences(self.test_sentences)
        # run  predictions on perturbated inputs and compare the output
        self.new_Xs = self.vectorizer.transform(self.perturbated_test_sentences)
        self.new_Ys = self.trained_model.predict(self.new_Xs)
        print("\n@@@@ Perturbation tests @@@@\n")
        for original, pert, pred, y in zip(self.test_sentences, self.perturbated_test_sentences, self.test_predictions, self.new_Ys):
                print("\n\nOriginal: '{}', Perturbated: '{}'\n".format(original, pert))
                print("Original Y: '{}', Perturbated Y: '{}'\n".format(pred, y))
                if y != pred:
                    print("ATTENTION: label changed after perturbation!\n")
            
        # all is done, dump the model
        self.next(self.dump_for_serving)

    @step
    def dump_for_serving(self):
        """
        Make sure we pickled the artifacts necessary for the Flask app to work

        Hint: is there a better way of doing this than pickling feature prep and model in two files? ;-)
        """
        import pickle
        import os

        pickle.dump(self.vectorizer, open(os.path.join(self.FINAL_FOLDER, 'vectorizer.pkl'), 'wb+'))
        pickle.dump(self.trained_model, open(os.path.join(self.FINAL_FOLDER, 'model.pkl'), 'wb+'))
        # go to the end
        self.next(self.end)

    @step
    def end(self):
        # all done, just print goodbye
        print("All done at {}!\n See you, space cowboys!".format(datetime.utcnow()))


if __name__ == '__main__':
    FinanceNewsFlow()
