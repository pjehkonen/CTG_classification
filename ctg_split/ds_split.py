from sklearn.model_selection import train_test_split

def ctg_split(X, y, split=0.1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, stratify=y)
