import pandas as pd
import os
import csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

print(os.getcwd())
print(os.listdir())

df = pd.read_csv(
    'data_clean.csv',
    encoding='latin1',
    engine='python',
    on_bad_lines='skip',
    sep=',',
    quoting=csv.QUOTE_MINIMAL
)

if 'clean_text' not in df.columns or 'label' not in df.columns:
    raise ValueError("Kolom tidak ditemukan")

df = df.dropna(subset=['clean_text','label'])
df['clean_text'] = df['clean_text'].astype(str)
df['clean_text'] = df['clean_text'].str.strip()
df = df[df['clean_text']!='']

X = df['clean_text']
y = df['label']

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,stratify=y,random_state=42
)

tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

mlflow.set_experiment("Hoax Detection Experiment")
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="SVM Linear Training"):
    model = SVC(kernel='linear')
    model.fit(X_train_tfidf,y_train)
    y_pred = model.predict(X_test_tfidf)

    acc_score = accuracy_score(y_test,y_pred)
    prec_score = precision_score(y_test,y_pred,average='weighted')
    rec_score = recall_score(y_test,y_pred,average='weighted')
    f1 = f1_score(y_test,y_pred,average='weighted')

    print("Accuracy:",acc_score)
    print(classification_report(y_test,y_pred))

    cm = confusion_matrix(y_test,y_pred)

    mlflow.log_metrics({
        "test_accuracy":acc_score,
        "test_precision":prec_score,
        "test_recall":rec_score,
        "test_f1_score":f1
    })

    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model.classes_)
    disp.plot()
    plt.savefig('confusion_matrix_svm.png')
    plt.close()

    mlflow.log_artifact('confusion_matrix_svm.png')