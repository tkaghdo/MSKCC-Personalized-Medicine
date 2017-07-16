import sys
import pandas as pd
from sklearn.linear_model import LogisticRegression

def main():
    try:
        train_variants_df = pd.read_csv('data/training_variants')
        test_variants_df = pd.read_csv('data/test_variants')
    except Exception as e:
        print(e)
        sys.exit(1)

    print('training variants length ', len(train_variants_df))
    print(train_variants_df.head())

    print('test variants length ', len(test_variants_df))
    print(test_variants_df.head())

    d = {}
    line_count = 0
    with open('data/training_text') as f:
        for line in f:
            if line_count > 0:
                line = line.split('||')
                d[int(line[0])] = line[1]
            line_count += 1

    '''
    print(len(d))
    line_count = 0
    for key, value in d.items():
        print(key, ': ', value)
        line_count += 1
        if line_count == 16:
            break
    '''

    # how many classes of mutations
    unique_classes = train_variants_df["Class"].unique()
    print(unique_classes)

    # *** convert discrete values to categorical variables
    discrete_features = ['Gene', 'Variation']

    '''
    for i in discrete_features:
        temp_df = pd.get_dummies(train_variants_df[i], prefix=i)
        train_variants_df = pd.concat([train_variants_df, temp_df], axis=1)
        train_variants_df = train_variants_df.drop(i, axis=1)
    '''
    for name in discrete_features:
        col = pd.Categorical.from_array(train_variants_df[name])
        train_variants_df[name] = col.codes

    print(train_variants_df.head())

    features = train_variants_df.columns.values.tolist()
    features.remove('Class')
    print(features)

    models = {}
    # one-versus-all method
    for c in unique_classes:
        model = LogisticRegression()

        X_train = train_variants_df[features]
        y_train = train_variants_df['Class'] == c

        model.fit(X_train, y_train)
        models[c] = model
        accuracy = model.score(X_train, y_train)
        print("Accuracy Score for model of origin {} is {}".format(c, accuracy))

    # predict probabilities
    for name in discrete_features:
        col = pd.Categorical.from_array(test_variants_df[name])
        test_variants_df[name] = col.codes

    testing_probs = pd.DataFrame(columns=unique_classes)  # this data frame will contain the prob by class
    for c in unique_classes:
        X_test = test_variants_df[features]
        # probability of observation being in the origin
        testing_probs[c] = models[c].predict_proba(X_test)[:, 1]
        testing_probs['ID'] = test_variants_df['ID']

    print(len(testing_probs))
    print(testing_probs.head())

    submission_df = pd.DataFrame(columns=['ID', 'class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9'])
    submission_df['ID'] = testing_probs['ID']
    for i in range(1,10):
        submission_df['class' + str(i)] = testing_probs[i]

    print(submission_df.head())
    print(len(submission_df))

    submission_df.to_csv('data/tk_submission.csv', index=False)

if __name__ == '__main__':
    sys.exit(0 if main() else 1)