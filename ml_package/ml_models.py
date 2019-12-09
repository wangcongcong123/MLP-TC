import sys
from ml_package.this_logging import *
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
from ml_package.data_processor import *
from sklearn.metrics import classification_report
# binary relevance: now this method is implemented for multi-label classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix

class MLModel():
    def __init__(self, configs):
        self.model = None
        self.configs = configs
        logging.info("configs:" + str(configs))
        self.data_processor = DataProcessor(configs)
        if self.configs["type"] == "multi":
            self.mlb = MultiLabelBinarizer()
        self.train_data, self.dev_data, self.test_data = self.data_processor.getData()

    def get_model_name(self):
        """
        get model name
        :return:
        """
        return type(self.configs["model"]).__name__

    def get_multi_label_binarizer(self):
        """
        get multi label binarizer when it is avialable
        :return:
        """
        if self.mlb is not None:
            return self.mlb
        else:
            logger.error(
                "multi label binarizer does not exist, please specify configs with configs['type']='multi' if you are for a multi-label classification problem")
            sys.exit(0)

    def get_data_processor(self):
        """
        get data processor
        :return:
        """
        return self.data_processor

    def get_fit_dataset(self):
        """
        get splits of dataset after being preprocessed to fit the model
        :return: train, dev, and test
        """
        return self.train_data, self.dev_data, self.test_data

    def get_model_to_train(self, data):
        model_to_train = self.configs["model"]
        labels_to_train = data["labels"]
        # if it is multi-label classification
        if self.configs["type"] == "multi":
            model_to_train = OneVsRestClassifier(model_to_train)
            labels_to_train = self.mlb.fit_transform(labels_to_train)
        return model_to_train, labels_to_train

    def train(self):
        """
        train model
        :return:
        """
        if self.model is None:
            logging.info("start training model " + type(self.configs["model"]).__name__)
            model_to_train, labels_to_train = self.get_model_to_train(self.train_data)
            self.model = model_to_train.fit(
                self.train_data["features"].toarray() if hasattr(self.train_data["features"], "toarray") else
                self.train_data["features"], labels_to_train)
            logging.info("model is done training...")
        else:
            logger.info("the trained model is loaded externally")

    def study_model_selection(self, params={"C": [0.001, 0.01, 0.1, 1, 10, 100]}):
        """
        TODO: not finished: not support for multi-label classification
        Here just provides an example of how to tune C for a model
        :return:
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import StratifiedKFold
        param_grid = dict(C=params["C"])
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

        model_to_train, labels_to_train = self.get_model_to_train(self.dev_data)
        grid_search = GridSearchCV(model_to_train, param_grid, n_jobs=-1, cv=kfold)
        grid_result = grid_search.fit(
            self.dev_data["features"].toarray() if hasattr(self.dev_data["features"], "toarray") else
            self.dev_data["features"], labels_to_train)

        print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
        means = grid_result.cv_results_['mean_test_score']
        params = grid_result.cv_results_['params']
        for mean, param in zip(means, params):
            print("%f  with: %r" % (mean, param))

    def predict_in_batch(self, batch):
        """
        predict batch
        :param batch: matrix of batch features, shape(batch_size, feature_dim)
        :return: predicted labels for the batch
        """
        return self.model.predict(batch)

    def get_confusion_matrix(self, labels, pred, tag=""):
        """
        TODO: not finished: not support for multi-label classification
        :param labels:
        :param pred:
        :param tag:
        :return:
        """
        cm = confusion_matrix(labels, pred)
        logger.info("confusion matrix for " + tag + " predictions:")
        logger.info(cm)
        df_cm = pd.DataFrame(cm, [i for i in self.model.classes_],
                             [i for i in self.model.classes_])
        plt.figure (figsize = (10,7))
        sn.set(font_scale=1.4)  # for label size
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
        plt.show()

    def eval(self, name, confusion_matrix=False):
        """
        evaluate given name and log the classification report
        :param name: train, dev or test
        """
        if self.model == None:
            logging.error("train the model first, please...")
            sys.exit(0)
        logging.info(
            "evaluation results for " + name + " set" + " [model name: " + type(self.configs["model"]).__name__ + "]")
        if name == "train":
            self._report_eval(self.train_data, tag="train", confusion_matrix=confusion_matrix)
        elif name == "dev":
            self._report_eval(self.dev_data, tag="dev", confusion_matrix=confusion_matrix)
        elif name == "test":
            self._report_eval(self.test_data, tag="test", confusion_matrix=confusion_matrix)
        else:
            logging.error("Give the correct name")
            sys.exit(0)

    def _report_eval(self, data, tag="", confusion_matrix=False):
        batch_features = data["features"].toarray() if hasattr(data["features"], "toarray") else \
            data["features"]
        labels = self.mlb.fit_transform(data["labels"]) if self.configs["type"] == "multi" else \
            data["labels"]
        pred = self.predict_in_batch(batch_features)
        logging.info(classification_report(
            labels, pred, digits=4,
            target_names=self.mlb.classes_ if self.configs["type"] == "multi" else None))
        if confusion_matrix:
            self.get_confusion_matrix(labels, pred, tag=tag)
