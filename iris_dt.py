import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score , confusion_matrix , ConfusionMatrixDisplay
from sklearn.metrics import ConfusionMatrixDisplay
import mlflow
import matplotlib.pyplot as plt



import dagshub
dagshub.init(repo_owner='Pranay5519', repo_name='mlflow-dagshub-demo', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Pranay5519/mlflow-dagshub-demo.mlflow")  # Set the MLflow tracking URI

#import seaborn as sns
# 1. Load the Iris dataset
iris = load_iris()
X = iris.data  # features
y = iris.target  # target labels
class_names = iris.target_names  # ['setosa', 'versicolor', 'virginica']
# 2. Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#n_estimators = 50  # Number of trees in the forest
max_depth = 2  # Maximum depth of the tree
input_example = X_train[:1]
mlflow.set_experiment("iris-df")  # Set the MLflow experiment
with mlflow.start_run(run_name="first-run"):
    
    model = DecisionTreeClassifier( max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)  # Train the model 
    y_pred = model.predict(X_test)  # Make predictions on the test set`
    accuracy = accuracy_score(y_test, y_pred)  # Calculate accuracy
    mlflow.log_metric("accuracy", accuracy)  # Log the accuracy metric
   # mlflow.log_param("n_estimators", n_estimators)  # Log the number of trees
    mlflow.log_param("max_depth", max_depth)  # Log the maximum depth of the trees  
  #  mlflow.sklearn.log_model(rf, "model")  # Log the trained model  
    
    print("accuracy" , accuracy)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix (counts):\n", cm)

    # 7. Confusion Matrix Heatmap (counts)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    plt.title("Random Forest - Confusion Matrix (Counts)")
    plt.show()
    mlflow.log_figure(fig, "confusion_matrix_counts.png")  # Log the confusion matrix figure
    #mlflow.log_artifact("confusion_matrix_counts.png")  # Save the confusion matrix figure as an artifact
    
    mlflow.log_artifact(__file__)  # Log the current script as an artifact
    mlflow.sklearn.log_model(model, name="my_model",input_example=input_example)
    print("Model logged to MLflow")
        