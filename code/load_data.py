



def load_data(train_call, train_clin):
    train_arr = train_call[train_call.columns[-100:]].to_numpy()
    train_arr = train_arr.T
    l = train_clin[train_clin.columns[-1:]].to_numpy().T.squeeze()
    labels = np.array(pd.factorize(l)[0].tolist())
    return train_arr.astype(np.float32)

def feature_selection(train_arr)
    clf = ExtraTreesClassifier(n_estimators=50)
    clf = clf.fit(train_arr, labels)
    clf.feature_importances_  
    model = SelectFromModel(clf, prefit=True)
    train_arr = model.transform(train_arr)

    args.num_features = train_arr.shape[1]
    print("Number of features that will be used: ", args.num_features)
    return train_arr