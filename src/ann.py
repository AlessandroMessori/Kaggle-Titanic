from keras import Sequential
from keras.layers import Dense, Dropout


class NetClassifier:

    def __init__(self, epochs=100, batch_size=10):
        self.classifier = Sequential()
        self.epochs = epochs
        self.batch_size = batch_size

        self.build()

    def build(self):
        # First Hidden Layer
        self.classifier.add(Dense(16, activation='relu', kernel_initializer='random_normal', input_dim=12))
        # Second  Hidden Layer
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(32, activation='relu', kernel_initializer='random_normal'))
        # Third Hidden Layer
        self.classifier.add(Dropout(0.5))
        # Output Layer
        self.classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

        self.classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, x, y):
        self.classifier.fit(x, y, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, y):
        preds = self.classifier.predict(y)
        return list(map(lambda x: 1 if (x > 0.5) else 0, preds))
