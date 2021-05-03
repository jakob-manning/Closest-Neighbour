import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def month_to_int(month: str):
    month_short_names = ["Jan", "Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return month_short_names.index(month)


def convert_visitor_type(visitor_type: str):
    if visitor_type == "returning":
        return 1
    return 0


def convert_weekend(weekend: str):
    if weekend == "TRUE":
        return 1
    return 0


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        10 - Month, an index from 0 (January) to 11 (December)
        11 - OperatingSystems, an integer
        12 - Browser, an integer
        13 - Region, an integer
        14 - TrafficType, an integer
        15 - VisitorType, an integer 0 (not returning) or 1 (returning)
        16 - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """

    with open(f"{filename}", encoding="utf-8") as f:
        labels = []
        evidence = []
        reader = csv.reader(f)

        next(reader)

        for row in reader:
            evidenceRow = []
            labels.append(int(row.pop() == "TRUE"))
            evidenceRow.append(int(row[0]))
            evidenceRow.append(float(row[1]))
            evidenceRow.append(int(row[2]))
            evidenceRow.append(float(row[3]))
            evidenceRow.append(int(row[4]))
            evidenceRow.append(float(row[5]))
            evidenceRow.append(float(row[6]))
            evidenceRow.append(float(row[7]))
            evidenceRow.append(float(row[8]))
            evidenceRow.append(float(row[9]))
            evidenceRow.append(month_to_int(row[10]))
            evidenceRow.append(int(row[11]))
            evidenceRow.append(int(row[12]))
            evidenceRow.append(int(row[13]))
            evidenceRow.append(int(row[14]))
            evidenceRow.append(convert_visitor_type(row[15]))
            evidenceRow.append(convert_weekend(row[16]))
            evidence.append(evidenceRow)

    return evidence, labels


def train_model(evidence, labels, neighbors = 1):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    neigh_model = KNeighborsClassifier(n_neighbors=neighbors)
    neigh_model.fit(evidence, labels)

    return neigh_model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    total_zeros = labels.count(0)
    total_ones = labels.count(1)

    correct_zeros = 0
    correct_ones = 0

    for index in range(len(labels)):
        if labels[index] == 0:
            # Check if prediction matches - score accordingly
            if predictions[index] == 0:
                correct_zeros += 1
        if labels[index] == 1:
            # Check if prediction matches - score accordingly
            if predictions[index] == 1:
                correct_ones += 1
    if total_zeros > 0 and total_ones > 0:
        return correct_ones/total_ones, correct_zeros/total_zeros
    else:
        raise ValueError


if __name__ == "__main__":
    main()
