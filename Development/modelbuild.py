import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 

#Class to create the model
class SentimentModel:
    def __init__(self, cleaned_file_path):
        self.cleaned_file_path = cleaned_file_path
        self.df_data = None
        self.model = None 
        
    def load_cleaned_data(self):
        #Loading the cleaned data
        self.df_data = pd.read_csv(self.cleaned_file_path)
        print("Cleaned data loaded successfully.")
    
    def train_sentiment_model(self):
        #Replaceing NaN values in 'cleaned_text' with an empty string
        self.df_data['cleaned_text'] = self.df_data['cleaned_text'].fillna('')
        
        if self.df_data is not None:
            #Splitting data into training and testing sets before vectorization
            X = self.df_data['cleaned_text']
            y = self.df_data['target']
            
            #Spliting into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            #Vectorization of the text data using TfidfVectorizer (only fitting on training data)
            vectorizer = TfidfVectorizer(ngram_range=(1, 3))
            X_train_vect = vectorizer.fit_transform(X_train)
            #Transforming the test data using the fitted vectorizer
            X_test_vect = vectorizer.transform(X_test)  
            
            #Base models if Logistic and Random Forest
            base_learners = [
                ('lr', LogisticRegression(C=2, max_iter=1000, n_jobs=-1)),
                ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1))
            ]
            
            #Stacking Classifier (Logistic Regression as meta-model)
            self.model = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
            
            #Fitting the stacking model on the half training data
            self.model.fit(X_train_vect, y_train)
            
            #Predicting on the test set
            y_pred = self.model.predict(X_test_vect)
            #Getting probabilities for the positive class
            y_pred_proba = self.model.predict_proba(X_test_vect)[:, 1]  
            
            #Model performance evaluation
            print('Accuracy Score:', accuracy_score(y_test, y_pred))
            print('Precision Score:', precision_score(y_test, y_pred))
            print('Recall Score:', recall_score(y_test, y_pred))
            print('F1 Score:', f1_score(y_test, y_pred))
            print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))
            print('\nClassification Report:\n', classification_report(y_test, y_pred))
            
            #Adding AUC score
            auc_score = roc_auc_score(y_test, y_pred_proba)
            print('AUC Score:', auc_score)
            
            #Plotting the ROC curve
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
            
            #Plotting Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.show()

            #Saveing the trained model and vectorizer
            joblib.dump(self.model, 'stacked_sentiment_model_v2.pkl')
            joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
            print("Model and vectorizer saved successfully.")

        else:
            print("Cleaned data not loaded yet. Please load the data first.")
            
#Main execution block
if __name__ == "__main__":
    cleaned_file_path = 'D:/Big Data Analytics/Term-2/BDM 1034 - Application Design for Big Data 01/Project_sentiment/training_data/cleaned_data.csv'
    
    #Calling SentimentModel class
    sentiment_model = SentimentModel(cleaned_file_path)
    
    #Loading cleaned data
    sentiment_model.load_cleaned_data()
    
    #Training the sentiment analysis model with stacking
    sentiment_model.train_sentiment_model()
