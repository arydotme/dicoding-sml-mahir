import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn

def main():
    # 1. Load Dataset
    # Menggunakan joblib untuk me-load data_clean.csv karena sesuai format dataset di modelling.py
    try:
        df = joblib.load('data_clean.csv')
    except Exception as e:
        print(f"Gagal memuat dataset: {e}")
        return

    # Pastikan nama kolom sesuai
    X = df['clean_text']
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
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("Hoax_Detection_Experiment")

    # Matikan autologging sementara (optional) karena kita ingin log hasil tuning spesifik secara manual atau menggunakan autolog khusus
    mlflow.sklearn.autolog(disable=True) 

    # 5. Konfigurasi Hyperparameter Tuning (GridSearchCV)
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    
    svc = SVC(random_state=42)
    grid_search = GridSearchCV(
        estimator=svc,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    with mlflow.start_run(run_name="SVM_Fine_Tuning"):
        print("Mulai proses Hyperparameter Tuning (Grid Search)...")
        grid_search.fit(X_train_tfidf, y_train)
        
        # Ekstrak Best Model dan Parameter
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)
        print("Best Cross-Validation Score: {:.4f}".format(grid_search.best_score_))
        
        # Log Hyperparameters
        mlflow.log_params(best_params)
        
        # 6. Prediksi dan Evaluasi dengan Best Model
        y_pred = best_model.predict(X_test_tfidf)
        
        print("\nClassification Report (Tuned Model):\n", classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:\n", cm)
        
        # Evaluasi Score
        acc_score = accuracy_score(y_test, y_pred)
        prec_score = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec_score = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"\n=== Evaluation Scores (Tuned) ===")
        print(f"Accuracy  : {acc_score:.4f}")
        print(f"Precision : {prec_score:.4f}")
        print(f"Recall    : {rec_score:.4f}")
        print(f"F1-Score  : {f1:.4f}")
        print(f"=================================\n")
        
        # Log Metrics
        mlflow.log_metrics({
            "test_accuracy": acc_score,
            "test_precision": prec_score,
            "test_recall": rec_score,
            "test_f1_score": f1,
            "best_cv_score": grid_search.best_score_
        })
        
        # Visualisasi Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - Tuned SVM")
        
        cm_image_path = 'confusion_matrix_svm_tuned.png'
        plt.savefig(cm_image_path)
        plt.close()
        
        # Log Best Model & Artifact Gambar
        mlflow.sklearn.log_model(best_model, "svm_tuned_model")
        mlflow.log_artifact(cm_image_path)

        # Simpan objek vectorizer dan best model jika dibutuhkan untuk deploy
        joblib.dump(best_model, 'best_svm_model.pkl')
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

        print(f"Tuning selesai. Model Tuned telah disimpan di MLflow.")

if __name__ == "__main__":
    main()
