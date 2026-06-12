class SpamDetector:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))

    def _get_model(self):
        if self.model is None:
            self.model = LogisticRegression()
        return self.model
    
    def _get_vectorizer(self):
        if self.vectorizer is None:
            self.vectorizer = CountVectorizer()
        return self.vectorizer
    
    def _preprocess_text(self, text):
        text = text.lower()
        try:
            import re
            text = re.sub(r'\d+', '', text)
        except ImportError:
            return 'Error: re module not found. Please ensure it is installed.'
        # text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [word for word in tokens if word not in self.stop_words]
        return ' '.join(tokens)
    
    def _load_data(self, file_path):
        if self.data is None:

            self.data = pd.read_csv(file_path)
            self.data.dropna(inplace=True)
        return self.data
    
    def _process_data(self):
        self.data['label'] = self.data['Spam/Ham'].map({'ham': 0, 'spam': 1})
        self.data['processed_text'] = self.data['Message'].apply(self._preprocess_text)
        self.data = self.data[['processed_text', 'label']]
        return self.data
    
    def _train(self):
        self.data = self._process_data()
        X = self.data['processed_text']
        y = self.data['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = self._get_vectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        model = self._get_model()
        model.fit(X_train_vec, y_train)
        self.X_train = X_train_vec
        self.X_test = vectorizer.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
    
    def _evaluate(self):
        model = self._get_model()
        y_pred = model.predict(self.X_test)
        print(classification_report(self.y_test, y_pred))
        print(f'Accuracy: {accuracy_score(self.y_test, y_pred):.4f}')
    
    def _predict(self, text):
        processed_text = self._preprocess_text(str(text))
        vectorizer = self._get_vectorizer()
        text_vec = vectorizer.transform([processed_text])
        self.model = self._get_model()
        prediction = self.model.predict(text_vec)
        perc       = self.model.predict_proba(text_vec)
        designed_prob  = perc[0][1] * 100
        return {'Prediction': 'Spam' if prediction[0] == 1 else 'Ham', 'Probability': designed_prob}
    
    def _format_output(self, prediction):
        return f"The message is classified as: {prediction['Prediction']} with a probability of {prediction['Probability']:.2f}%"