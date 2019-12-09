import re
from tqdm import tqdm
from ml_package.this_logging import *
import json
from nltk.tokenize import TweetTokenizer
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from collections import Counter
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)
import gzip
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords

enStop = stopwords.words('english')
enStop_dict = {e: 0 for e in enStop}


class DataProcessor():
    def __init__(self, configs={}):
        self.configs = configs
        self.stemmer = PorterStemmer()
        self.train_path = configs["relative_path"] + "dataset/" + self.configs["data"] + "/train.json"
        # self.dev_path = configs["relative_path"]+"dataset/" + configs["data"] + "/json/dev.json"
        self.test_path = configs["relative_path"] + "dataset/" + self.configs["data"] + "/test.json"
        if configs["tokenizer"] == "tweet":
            self.tokenizer = TweetTokenizer()

        logger.info("building vocabulary...")
        self.vocab, train_samples, train_tokens = self.build_vocab(self.train_path)

        if configs["vectorizer"].endswith(".gz"):
            logging.info("reading pre-trained embedding from " + self.configs["vectorizer"])
            self.embedding_dict = self.build_embedding(self.configs["vectorizer"])
        else:
            if configs["vectorizer"] == "tf-idf":
                self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), use_idf=True, max_features=5000, max_df=0.501)
            elif configs["vectorizer"] == "count":
                self.vectorizer = CountVectorizer(ngram_range=(1, 1), max_features=5000)
            self.vectorizer.fit(train_tokens)

        self.train, self.dev = self.build_train_dev(train_samples, train_tokens)
        self.test = self.build_test(self.test_path)

    def build_test(self, test_path):
        return self._read(test_path, tag="test")

    def _reformat(self, samples, features):
        reformatted = {}
        reformatted["features"] = features
        tokens = []
        content = []
        labels = []
        for each in samples:
            tokens.append(each["tokens"])
            content.append(each[self.configs["data_mapping"]["content"]])
            labels.append(each["labels"])
        reformatted[self.configs["data_mapping"]["content"]] = content
        reformatted["labels"] = labels
        reformatted["tokens"] = tokens
        return reformatted

    def build_train_dev(self, train_dev_samples, corpus):
        logger.info("building train and dev (split and feature construction)")
        features = self._get_features(corpus)
        # actually split the train into train and dev for model training following the ratio 08:0.2.
        train_samples, dev_samples, train_features, dev_features = train_test_split(train_dev_samples, features,
                                                                                    test_size=0.2, random_state=42)

        train = self._reformat(train_samples, train_features)
        dev = self._reformat(dev_samples, dev_features)
        return train, dev

    def build_embedding(self, pretrain_path):
        embedding_dict = {}
        with gzip.open(pretrain_path, 'rt', encoding="utf-8") as f:
            dim = int(pretrain_path.split(".")[-3][:-1])
            for line in tqdm(f, desc="reading pre-trained embedding"):
                line_tokens = line.rstrip().split()
                if len(line_tokens) == dim + 1 and line_tokens[0] in self.vocab:
                    key = line_tokens.pop(0)
                    value = [float(e) for e in line_tokens]
                    embedding_dict[key] = value
        return embedding_dict

    def build_vocab(self, data_path):
        tokenized = []
        train_samples = []
        train_tokens = []
        with open(data_path, encoding="utf-8") as f:
            for each in tqdm(f, desc="building vocabulary..."):
                ins = json.loads(each.strip())
                tokens = self._tokenize(ins[self.configs["data_mapping"]["content"]])
                tokenized.extend(tokens)
                train_tokens.append(" ".join(tokens))
                train_samples.append({self.configs["data_mapping"]["content"]: ins[self.configs["data_mapping"]["content"]], "tokens": tokens,
                                      "labels": ins[self.configs["data_mapping"]["label"]].split(",") if self.configs["type"] == "multi" else ins[
                                          self.configs["data_mapping"]["label"]]})
        return Counter(tokenized), train_samples, train_tokens

    def _preprocess_tweet(self, tweet):
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                           '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        tweet = re.sub(giant_url_regex, "<url>", tweet)
        tweet = re.sub(r"#[A-Za-z0-9]+", "<hashtag>", tweet)
        tweet = re.sub(r"@[A-Za-z0-9]+", "<user>", tweet)
        tweet = re.sub(r"\d+", "<number>", tweet)
        return tweet

    def _tokenize(self, text: str):
        if self.configs["tokenizer"] == "tweet":
            text = self._preprocess_tweet(text)
            tokens = self.tokenizer.tokenize(text)
        else:
            tokens = text.split()
        if self.configs["stemming"] == "true":
            tokens = [self.stemmer.stem(t.lower()) for t in tokens if len(t) >= 3 and t not in enStop_dict]
        else:
            tokens = [t.lower() for t in tokens if len(t) >= 3 and t not in enStop_dict]
        return tokens

    def _vectorize(self, sentence, way="average"):
        """
        vectorize a sentence based on pre-trained embedding in a way of "average" embeddings (by default)
        :param sentence: the sentence to be vectorized
        :param way: average over embeddings of tokens in the sentence
        :return:
        """
        tokenized = sentence.split()
        wvs = []
        for t in tokenized:
            if t in self.embedding_dict:
                v = self.embedding_dict[t]
                norm = np.linalg.norm(v)
                normed_v = v / norm
                wvs.append(normed_v)
        m = np.array(wvs)
        normed_m = np.mean(m, axis=0)
        return normed_m

    def _to_features(self, corpus):
        """
        return features for corpus from pre-trained embedding
        :param corpus:
        :return:
        """
        X_matrix = np.zeros((len(corpus), int(self.configs["vectorizer"].split(".")[-3][:-1])))
        for index, s in enumerate(corpus):
            sv = self._vectorize(s)
            if not np.isnan(sv).any():
                X_matrix[index, :] = sv
            else:
                logging.info(
                    "no tokens in " + s + " are found in the pre-trained embedding, hence it is vectorized to zeros")
        return X_matrix

    def _get_features(self, corpus):
        if self.configs["vectorizer"].endswith(".gz"):
            X = self._to_features(corpus)
        else:
            logger.info("feature matrix is constructing...")
            X = self.vectorizer.transform(corpus)
            logging.info("shape of features:" + str(X.shape))
            # feature_names = self.vectorizer.get_feature_names()
            # with open("tf-idf-vocab.txt", "w") as f:
            #     f.write("\n".join(feature_names))
            # logger.info("vocabulary is written to tf-idf-vocab.txt")
        return X

    def raw2predictable(self, raw):
        """
        convert raw corpus (a list of documents) without ground truth, to a format that can be predicted by model
        :param raw: the raw corpus consisting of documents
        :return: predictable data of the corpus
        """
        data = {}
        data[self.configs["data_mapping"]["content"]] = raw
        tokens_list = []
        tokenized_texts = []
        for each in raw:
            tokens = self._tokenize(each)
            tokens_list.append(tokens)
            tokenized_texts.append(" ".join(tokens))
        data["tokens"] = tokens_list
        data["features"] = self._get_features(tokenized_texts)
        return data

    def _read(self, data_path, tag="train"):
        """
        this is a private method used for reading each set
        :param data_path:
        :param tag:
        :return:
        """
        logger.info("reading " + tag + " from" + data_path)
        data = {}
        with open(data_path, encoding="utf-8") as train_f:
            content = []
            tokens_list = []
            labels = []
            tokenized_texts = []
            for each_sample in tqdm(train_f, desc="reading " + tag + " set:"):
                each_instance = json.loads(each_sample.strip())
                content.append(each_instance[self.configs["data_mapping"]["content"]])
                tokens = self._tokenize(each_instance[self.configs["data_mapping"]["content"]])
                tokens_list.append(tokens)
                tokenized_texts.append(" ".join(tokens))
                labels.append(
                    each_instance[self.configs["data_mapping"]["label"]].split(",") if self.configs["type"] == "multi" else each_instance[self.configs["data_mapping"]["label"]])
        data["tokens"] = tokens_list
        data[self.configs["data_mapping"]["content"]] = content
        data["features"] = self._get_features(tokenized_texts)
        data["labels"] = labels
        return data

    def getData(self):
        """
        get read train, and test respectively
        :return: train {"content":[text1, text2, ...],"tokens":[[TOKEN1,TOKEN2,...],[],....],"features":[[FEATURE1, FEATURE2,...],[]...],"labels":[[LABEL(s)],[LABEL(s)],...]}
        """
        logger.info("shape of train: " + str(self.train["features"].shape))
        logger.info("shape of dev: " + str(self.dev["features"].shape))
        logger.info("shape of test: " + str(self.test["features"].shape))
        return self.train, self.dev, self.test
