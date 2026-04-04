import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
# 1. Load Dataset
# Menggunakan data_clean.csv sesuai permintaan awal
df = pd.read_csv('data_clean.csv', encoding='latin1')

# Hapus baris yang memiliki nilai NaN pada kolom teks atau label
df = df.dropna(subset=['clean_text', 'label'])

# Pastikan nama kolom sesuai (disesuaikan dengan gambaran Anda)
X = df['clean_text']   # hasil cleaning
y = df['label']

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 3. TF-IDF
tfidf = TfidfVectorizer(max_features=5000)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 4. Inisialisasi MLflow
mlflow.set_experiment("Hoax Detection Experiment")

# Menggunakan autolog dari MLflow untuk Scikit-Learn
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="SVM Linear Training"):
    # 5. SVM
    model = SVC(kernel='linear')
    
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    
    # 6. Evaluasi
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Evaluasi menggunakan Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)
    
    # Evaluasi Score (Accuracy, Precision, Recall, F1-Score)
    acc_score = accuracy_score(y_test, y_pred)
    prec_score = precision_score(y_test, y_pred, average='weighted')
    rec_score = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n=== Evaluation Scores ===")
    print(f"Accuracy  : {acc_score:.4f}")
    print(f"Precision : {prec_score:.4f}")
    print(f"Recall    : {rec_score:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print(f"=========================\n")
    
    # Pencatatan metric testing secara eksplisit ke MLflow
    mlflow.log_metrics({
        "test_accuracy": acc_score,
        "test_precision": prec_score,
        "test_recall": rec_score,
        "test_f1_score": f1
    })
    
    # Visualisasi Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix - SVM")
    plt.savefig('confusion_matrix_svm.png') # Menyimpan hasil plot
    plt.show()
    
    # Menyimpan artifact gambar ke MLflow
    mlflow.log_artifact('confusion_matrix_svm.png')