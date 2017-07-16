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

    print(len(d))
    line_count = 0
    for key, value in d.items():
        print(key, ': ', value)
        line_count += 1
        if line_count == 1:
            break

if __name__ == '__main__':
    sys.exit(0 if main() else 1)