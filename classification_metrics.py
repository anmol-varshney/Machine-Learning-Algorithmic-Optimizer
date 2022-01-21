BASE_DIR = 'https://github.com/Code-with-Palak/Machine-Learning-Algorithmic-Optimizer/'
import warnings
from numpy.lib.function_base import average
import matplotlib.pyplot as plt
from IPython.display import HTML
import streamlit as st
import re
warnings.filterwarnings("ignore")


from sklearn.metrics import *
import pandas as pd

def metrics(test_features, y_test, predictions, classifier, data_name, algorithm_selected):
    num_labels = len(set(list(y_test)))
    if num_labels==2:
        avg = 'binary'
    else: 
        avg = 'micro'
    accuracy=accuracy_score(y_test, predictions)
    jaccard = jaccard_score(y_test, predictions, average=avg)
    precision = jaccard_score(y_test, predictions, average=avg)
    recall = jaccard_score(y_test, predictions, average=avg)
    f1 = f1_score(y_test, predictions, average=avg)
    
 
    measures = pd.DataFrame(index = ['Accuracy',  'F1-Score', 'Precision', 'Recall', 'Jaccard Score'], columns=['Scores'])
    scores = [accuracy, f1, precision, recall, jaccard]

    measures['Scores'] = scores
    measures = measures.T
    
    #classification report
    report = classification_report(y_test, predictions, output_dict=True)
    df_classification_report = pd.DataFrame(report).transpose()
    
    report_df_html = df_classification_report.to_html()
    scores_df_html = measures.to_html()
    
    
    ####################
    con_matrix = confusion_matrix(y_true=y_test, y_pred=predictions)
    fig, ax = plt.subplots(figsize=(4.5,4.5))
    ax.matshow(con_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(con_matrix.shape[0]):
        for j in range(con_matrix.shape[1]):
            ax.text(x=j, y=i,s=con_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(BASE_DIR+'EDA and Performance reports/{} confusion_matrix of {}.png'.format(algorithm_selected, re.sub('.csv', '', str(data_name))))
             
    ###################
    
    
    htmlfile = open(BASE_DIR+'EDA and Performance reports\\{} performance_report of {} .html'.format(algorithm_selected, re.sub('.csv', '', str(data_name))),"w")
    html_template = """<html>
    <head>
    <title>Performance_report</title>
    </head>
    <body><div style="background: linear-gradient(to left, #008080 0%, #E0FFFF 100%); margin-left: 45px;">
    <h1 style="color: #E0FFFF; background: cadetblue;">Performance Report</h1>
    </div>
    </body>
    </html>
    """
    
    line_breaks = """
    <html>
    <br><br>
    </html>
    """
        
    htmlfile.write(html_template)
    htmlfile.write('<div style="margin-left: 45px;"> <hr> </div>')
    htmlfile.write(line_breaks)
    htmlfile.write('<div style="margin-left: 45px;"> <b>1. Classification Report</b>')
    htmlfile.write(line_breaks)
    htmlfile.write(report_df_html)
    htmlfile.write(line_breaks)
    htmlfile.write('<b>2. Model Scores</b>')
    htmlfile.write(line_breaks)
    htmlfile.write(scores_df_html)
    htmlfile.write(line_breaks)
    htmlfile.write(line_breaks)
    htmlfile.write("<img src =  '{}'> </div>".format(BASE_DIR+'EDA and Performance reports/{} confusion_matrix of {}.png'.format(algorithm_selected, re.sub('.csv', '', str(data_name)))))
    htmlfile.write(line_breaks)
    
    htmlfile.close()
    
    st.success('The performance report of algorithm ({}) on {}, with the selected preprocessing techniques, has been saved to this location; \n {}'.format(algorithm_selected, re.sub('.csv', '', str(data_name)), BASE_DIR+'EDA and Performance reports/{} performance_report of {} .html'.format(algorithm_selected, re.sub('.csv', '', str(data_name)))))
    
    return df_classification_report, measures, plot_confusion_matrix(estimator=classifier, X=test_features, y_true=y_test, labels=classifier.classes_, cmap='Blues')
