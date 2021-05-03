import shopping

results_load_data_short = ([[0, 0.0, 0, 0.0, 1, 0.0, 0.2, 0.2, 0.0, 0.0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 0.0, 0, 0.0, 2, 64.0, 0.0, 0.1, 0.0, 0.0, 1, 2, 2, 1, 2, 0, 0],
                            [0, 0.0, 0, 0.0, 1, 0.0, 0.2, 0.2, 0.0, 0.0, 1, 4, 1, 9, 3, 0, 0],
                            [0, 0.0, 0, 0.0, 2, 2.666666667, 0.05, 0.14, 0.0, 0.0, 1, 3, 2, 2, 4, 0, 0],
                            [0, 0.0, 0, 0.0, 10, 627.5, 0.02, 0.05, 0.0, 0.0, 1, 3, 3, 1, 4, 0, 0],
                            [0, 0.0,0, 0.0, 19, 154.2166667, 0.015789474, 0.024561404, 0.0, 0.0, 1, 2, 2, 1, 3, 0, 0]],
                           ['FALSE', 'FALSE', 'FALSE', 'FALSE', 'FALSE', 'FALSE'])

results_test_train_data = [[0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0., 0.],
       [1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1.],
       [0., 1., 0., 0., 0., 0.]]


def test_Load_data():
    assert shopping.load_data("short.csv") == results_load_data_short

def test_train_model():
    evidence, labels = shopping.load_data("short.csv")
    model = shopping.train_model(evidence, labels)
    model_graph = model.kneighbors_graph().toarray()
    assert (model_graph == results_test_train_data).all()


