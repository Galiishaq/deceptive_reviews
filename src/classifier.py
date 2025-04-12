from sklearn.naive_bayes import MultinomialNB


class TextNaiveBayes:
    def __init__(self, smoothing=1):
        self.model = MultinomialNB(alpha=smoothing)

    def train(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def predict(self, X):
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        from sklearn.metrics import accuracy_score, classification_report
        preds = self.predict(X)
        acc = accuracy_score(y, preds)
        report = classification_report(y, preds)
        return acc, report

