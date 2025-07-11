import pandas as pd
import joblib
from PIL import Image
from data.config import thresholds



from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score,
                             roc_auc_score)

data = pd.read_csv('datasets/diabetes.csv')
X = data[['Pregnancies', 'Glucose', 'Insulin', 'BMI', 'Age']]
y = data['Outcome']

page_icon = Image.open("image/yuki.png")

model = joblib.load('model.pkl')



y_score = model.predict_proba(X)[:, 1]
y_pred = (y_score >= thresholds).astype(int)

# Accuracy Score
accuracy_result = round(accuracy_score(y, y_pred) * 100, 2)
# F! Score
f1_result = round(f1_score(y, y_pred) * 100, 2)
# Recall Score
recall_result = round(recall_score(y, y_pred) * 100, 2)
# Precision Score
precision_result = round(precision_score(y, y_pred) * 100, 2)
# ROC AUC Score
roc_auc = round(roc_auc_score(y, y_score)*100, 2)