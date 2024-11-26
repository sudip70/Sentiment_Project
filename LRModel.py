import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class SentimentModel:
    def __init__(self, cleaned_file_path):
        self.cleaned_file_path = cleaned_file_path
        self.df_data = None
        
    def load_cleaned_data(self):
        # Load the cleaned data
        self.df_data = pd.read_csv(self.cleaned_file_path)
        print("Cleaned data loaded successfully.")
    
    def train_sentiment_model(self):
        # Replace NaN values in 'cleaned_text' with an empty string
        self.df_data['cleaned_text'] = self.df_data['cleaned_text'].fillna('')
        if self.df_data is not None:
            # Vectorization of the text data using TfidfVectorizer
            vectorizer = TfidfVectorizer()  # max_features=10000)  # Limiting to 5000 features
            X = vectorizer.fit_transform(self.df_data['cleaned_text'])
            y = self.df_data['target']
            
            # Splitting data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Creating and training the Logistic Regression model
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Predicting on test set
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get the probabilities for the positive class
            
            # Model performance evaluation
            print('Accuracy Score:', accuracy_score(y_test, y_pred))
            print('\nPrecision Score:', precision_score(y_test, y_pred))
            print('\nRecall Score:', recall_score(y_test, y_pred))
            print('\nF1 Score:', f1_score(y_test, y_pred))
            print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
            print('\nClassification Report:\n', classification_report(y_test, y_pred))
            
            # Adding AUC score
            auc_score = roc_auc_score(y_test, y_pred_proba)
            print('\nAUC Score:', auc_score)
            
            # Plotting the ROC curve
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='b', label=f'AUC = {auc_score:.2f}')
            plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
            plt.title('ROC Curve')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.grid(True)
            plt.show()
            
            # Plotting Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()
        else:
            print("Cleaned data not loaded yet. Please load the data first.")
            
# Main execution block
if __name__ == "__main__":
    cleaned_file_path = 'D:/Big Data Analytics/Term-2/BDM 1034 - Application Design for Big Data 01/Project_sentiment/training_data/cleaned_data.csv'
    
    # Calling SentimentModel class
    sentiment_model = SentimentModel(cleaned_file_path)
    
    # Loading cleaned data
    sentiment_model.load_cleaned_data()
    
    # Training the sentiment analysis model
    sentiment_model.train_sentiment_model()
