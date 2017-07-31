import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

def cleanup_text(txt):
    punctuation = '!"#$%&\'()+,-./:;<=>?@[\\]^_`{|}~'
    # remove punctuations from text and convert to upper case
    cleaned_text = txt.translate(str.maketrans('', '', punctuation)).upper()
    cleaned_text = cleaned_text.split(' ')
    # Remove stop words from "words"
    upper_case_stopwords = [w.upper() for w in stopwords.words('english')]
    cleaned_text = [w for w in cleaned_text if not w in upper_case_stopwords]
    cleaned_text = ' '.join(cleaned_text)
    return cleaned_text


def prep_text_features(text_file):
    d = {}
    line_count = 0
    with open(text_file) as f:
        for line in f:
            if line_count > 0:
                line = line.split('||')
                d[int(line[0])] = line[1]
            line_count += 1

    # cleanup text
    cleaned_text_lst = []
    for key, value in d.items():
        cleaned_text = cleanup_text(value)
        cleaned_text_lst.append(cleaned_text)

    # create bag of words
    vectorizer = CountVectorizer(analyzer="word",
                                 tokenizer=None,
                                 preprocessor=None,
                                 stop_words=None,
                                 max_features=50000)
    text_features = vectorizer.fit_transform(cleaned_text_lst)

    return text_features

def main():
    try:
        train_variants_df = pd.read_csv('data/training_variants')
    except Exception as e:
        print(e)
        sys.exit(1)

    train_data_features = prep_text_features('data/training_text')
    train_data_features = train_data_features.toarray()
    train_data_features_df = pd.DataFrame(data=train_data_features)

    # join train variants with the final version of the text
    train_final_df=train_variants_df.join(train_data_features_df, lsuffix='_train_variants_df', rsuffix='_train_data_features_df')

    # *** convert discrete values to categorical variables
    discrete_features = ['Gene', 'Variation']

    for name in discrete_features:
        col = pd.Categorical.from_array(train_final_df[name])
        train_final_df[name] = col.codes


    features = train_final_df.columns.values.tolist()
    features.remove('Class')

    # how many classes of mutations
    unique_classes = train_final_df["Class"].unique()

    models = {}
    # one-versus-all method
    for c in unique_classes:
        model = RandomForestClassifier(n_jobs=500, max_depth=40)

        X_train = train_final_df[features]
        y_train = train_final_df['Class'] == c

        model.fit(X_train, y_train)
        models[c] = model
        accuracy = model.score(X_train, y_train)
        print("Accuracy Score for model of origin {} is {}".format(c, accuracy))

    ##### test
    test_variants_df = pd.read_csv('data/test_variants')


    test_data_features = prep_text_features('data/test_text')
    test_data_features = test_data_features.toarray()

    test_data_features_df = pd.DataFrame(data=test_data_features)

    # join test variants with the final version of the text
    test_final_df = test_variants_df.join(test_data_features_df, lsuffix='_test_variants_df',
                                            rsuffix='_test_data_features_df')

    print(test_final_df.head())
    print(features)

    for name in discrete_features:
        col = pd.Categorical.from_array(test_final_df[name])
        test_final_df[name] = col.codes

    testing_probs = pd.DataFrame(columns=unique_classes)  # this data frame will contain the prob by class
    for c in unique_classes:
        X_test = test_final_df[features]
        # probability of observation being in the origin
        testing_probs[c] = models[c].predict_proba(X_test)[:, 1]
        testing_probs['ID'] = test_final_df['ID']

    print(len(testing_probs))
    print(testing_probs.head())

    submission_df = pd.DataFrame(
        columns=['ID', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9'])
    submission_df['ID'] = testing_probs['ID']
    for i in range(1, 10):
        submission_df['class' + str(i)] = testing_probs[i]

    print(submission_df.head())
    print(len(submission_df))

    submission_df.to_csv('data/tk_submission_2.csv', index=False)


if __name__ == '__main__':
    sys.exit(0 if main() else 1)