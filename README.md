# Machine Learning Package for Text Classification


This repository contains codes of machine learning algorithms for text classification, abbreviated to MLP-TC (**M**achine **L**earning **P**ackage for **T**ext **C**lassification).
Due to the poor reproducibility of classification modelling on different datasets with different algorithms in Jupyter, and thus this package is borned. This package is designed in a way especially suitable for researchers conducting comparison experiments and benchmarking analysis.
This package empowers to explore the performance difference that different ML techniques have on your specific datasets.  _Updated: 2019/12/09._

## Highlights

- Well logged for the whole process of training a model for text classification 
- Fed different datasets into models quickly as long as they are formatted as required.
- Support single or multi label (only binary relevance at this stage) classification
- Support model save, load, train, predict, eval, etc.
   
## Dependencies
In order to use the package, clone the repository first and then install the following dependencies if you have not got it ready. 

- scikit-learn
- seaborn
- pandas
- numpy
- matplotlib

## Steps of Usage
1. **Data preparation**: format your classification datasets into the following format. For label attribute, labels are separated by "," if multi labels are available for a sample.
   Have a look at the [dataset/tweet_sentiment_3](dataset/tweet_sentiment_3) dataset provided in this package to know the format required. 
   ```python
    {"content":"this is the content of a sample in dataset","label":"label1,label2,..."}
    ```
2. **Configuration for model training**: Below is an example of configuration model training (the script is in [main.py](main.py)). Important stuff are commented below.

    ```python
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
    ```

3. **Train and save**: Below is an example of model training and save (the script is in main.py). Important stuff are commented below.
    ```python
   import ml_package.model_handler as mh
    print("=========model training and save======")
    model = mh.get_model(
        configs)  # get the specified LinearSVC model from the model handler with configs passed as the parameter
    model.train()  # train a model
    mh.save_model(model, configs)  # you can save a model after it is trained
    ```
4. **Eval and predict**: Below is an example of evaluating train,dev, and test, and predict without ground truth (the script is in main.py). Important stuff are commented below.
    ```python
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
    ```
   Predict with ground truth, e.g. the first three examples from test set.
   ```python
    print("=========predict a corpus with ground truth======")
    train_data, _, test_data = model.get_fit_dataset()
    data = test_data
    to_predict_buttom = 0
    to_predict_top = 3
    
    if configs["type"] == "multi":
        mlb = model.get_multi_label_binarizer()
    
    predicted = model.predict_in_batch(data["features"][to_predict_buttom:to_predict_top].toarray() if hasattr(
        data["features"][to_predict_buttom:to_predict_top], "toarray") else data["features"][
                                                                            to_predict_buttom:to_predict_top])
    
    print("Make predictions for:\n ", "\n".join(data["content"][to_predict_buttom:to_predict_top]))
    print("Ground truth are:\n ")
    pprint.pprint(data["labels"][to_predict_buttom:to_predict_top])
    print("The predicted results are: ")
    pprint.pprint(mlb.inverse_transform(predicted) if configs["type"] == "multi" else predicted)
    ```
    * After running [main.py](main.py), you will find `output.log` and `LinearSVCdc26f10760747d1c6d94b3a9679d28cf.pkl` under the root of the repository. When you rerun the experiment, the model will be loaded locally instead of re-training from scratch as long as your configurations keep the same.
    
    
## Others
- More extensions of this package go to [this tutorial](in plan). Feedback is welcome or any errors/bugs reporting is well appreciated.