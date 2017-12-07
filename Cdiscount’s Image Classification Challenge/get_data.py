import pandas as pd
from PIL import Image
import io, bson

FILEPATH = '' # File path should be given here.
def read_data():
    print('PREPARING TO READ DATA...')
    print('Category names reading...')
    categories = pd.read_csv(FILEPATH + '/category_names.csv', index_col='category_id')
    print('Category names has been read.')

    # read bson file into pandas DataFrame
    print('Training data reading...')
    with open(FILEPATH + '/train_example.bson', 'rb') as b:
        df = pd.DataFrame(bson.decode_all(b.read()))

    # convert binary image to raw image and store in the imgs column
    df['imgs'] = df['imgs'].apply(lambda rec: rec[0]['picture'])
    df['imgs'] = df['imgs'].apply(lambda img: Image.open(io.BytesIO(img)))
    print('Training data has been read.')
    return categories, df['imgs'].as_matrix(), df['category_id'].as_matrix()

def split_data(X, y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test