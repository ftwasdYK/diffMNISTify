import numpy as np
from glob import glob
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV,StratifiedKFold

from utils import load_precomputed_features, save_confusion_matrix_sklearn
from ml_models import RandomForest, RidgeClassifierModel, SupportVectorMachine

# print(y.shape)
# y_pred_test = pipeline.predict(x)

# cm = confusion_matrix(y, y_pred_test)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# model_class_name = type(MODEL).__name__ 
# plt.savefig(f'./results/{model_class_name}_confusion_matrix.png')
# plt.show()

# report_test = classification_report(y, y_pred_test, output_dict=False)

# print(report_test)
# joblib.dump(pipeline, f'./models_pkl/{model_class_name}.pkl')
#################################################################


MODELS = [RandomForest(), RidgeClassifierModel(), SupportVectorMachine()]
NORMALAZATION_TECHNIQUE = StandardScaler()
NUM_SAMPLES_PER_CLASS = 4000
NUM_TEST_SAMPLES_PER_CLASS = 3000
LAYERS_EXPORT = 9
DIR_RESULTS = './training/'
    

if __name__ == "__main__":
    
    if not os.path.exists("training/results"):
        os.makedirs("training/results")

    x_train, y_train = load_precomputed_features(NUM_SAMPLES_PER_CLASS, LAYERS_EXPORT, subset='train')
    
    for MODEL in MODELS:

        # Define the pipeline
        pipeline = Pipeline([
            ('norm_tech', NORMALAZATION_TECHNIQUE),
            ('model', MODEL.get_model())
        ])

        # Define the parameter grid for GridSearch
        param_grid = MODEL.param_grid

        # Initialize GridSearchCV
        cv = StratifiedKFold(n_splits=3)
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)

        grid_search.fit(x_train, y_train)
        print("Best parameters found: ", grid_search.best_params_)
        
        # Evaluate and the save the results
        y_pred_tr = grid_search.predict(x_train)
        report_train = classification_report(y_train, y_pred_tr)

        del x_train, y_train
        x_test, y_test= load_precomputed_features(NUM_TEST_SAMPLES_PER_CLASS, LAYERS_EXPORT, subset='test')

        y_pred = grid_search.predict(x_test)
        report_test = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_test) #.rename(columns={"0.0": "Class 0", "1.0": "Class 1"})
        
        dir_results = DIR_RESULTS + f'results/{MODEL.str_name}_nS{NUM_SAMPLES_PER_CLASS}_Classification_Report.csv'
        report_df.to_csv(dir_results)
        
        print(report_train)
        print(report_df.transpose())
        
        cm = confusion_matrix(y_test, y_pred)
        dir_figure = DIR_RESULTS + f'/figures/{MODEL.str_name}_nS{NUM_SAMPLES_PER_CLASS}'
        save_confusion_matrix_sklearn(cm, dir_figure)

        # Save the model
        joblib.dump(grid_search, f'./models_pkl/{MODEL.str_name}_nS{NUM_SAMPLES_PER_CLASS}_Grid_Search_Model.pkl')
        del x_test, y_test, y_pred, y_pred_tr