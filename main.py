import ml_package.model_handler as mh
import pprint
from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

print("=========configuration for model training======")
configs = {}
configs["relative_path"] = "./"  # the path relative to dataset
configs["data"] = "tweet_sentiment_3/json"  # specify the path of your data that is under the dataset dir
configs["data_mapping"] = {"content": "content",
                           "label": "label"}  # this is the mapping from the package required attribute names to your json dataset attributes

configs["stemming"] = "true"  # specify whether you want to stem or not in preprocessing
configs["tokenizer"] = "tweet"  # if it is a tweet-related dataset, it is suggested to use tweet tokenizer, or "string"
configs["vectorizer"] = "count"  # options: count, tf-idf, embeddings/glove.twitter.27B.100d.txt.gz

configs["type"] = "single"  # single or multi label classification?
configs[
    "model"] =  LinearSVC(C=0.1)  # Options: LinearSVC(C=0.1),SVC, LogisticRegression(solver='ibfgs'),GaussianNB(),RandomForest, etc.

print("=========model training and save======")
model = mh.get_model(
    configs)  # get the specified LinearSVC model from the model handler with configs passed as the parameter
model.train()  # train a model
mh.save_model(model, configs)  # you can save a model after it is trained

print("=========evaluate on train, dev and test set======")
model.eval("train")  # classification report for train set
model.eval("dev")  # classification report for dev set
model.eval("test", confusion_matrix=True)  # we can let confusion_matrix=True so as to report confusion matrix as well

print("=========predict a corpus without ground truth======")
corpus2predict = ["i love you", "i hate you"]  # get ready for two documents
data_processor = model.get_data_processor()
to_predict = data_processor.raw2predictable(["i love you", "i hate you"])
predicted = model.predict_in_batch(
    to_predict["features"].toarray() if hasattr(to_predict["features"], "toarray") else to_predict["features"])
print("Make predictions for:\n ", to_predict["content"])
print("The predicted results are: ", predicted)

print("=========predict a corpus with ground truth======")
train_data, _, test_data = model.get_fit_dataset()
data = test_data
to_predict_first = 0
to_predict_last = 3

if configs["type"] == "multi":
    mlb = model.get_multi_label_binarizer()

predicted = model.predict_in_batch(data["features"][to_predict_first:to_predict_last].toarray() if hasattr(
    data["features"][to_predict_first:to_predict_last], "toarray") else data["features"][
                                                                       to_predict_first:to_predict_last])

print("Make predictions for:\n ", "\n".join(data["content"][to_predict_first:to_predict_last]))
print("Ground truth are:\n ")
pprint.pprint(data["labels"][to_predict_first:to_predict_last])
print("The predicted results are: ")
pprint.pprint(mlb.inverse_transform(predicted) if configs["type"] == "multi" else predicted)