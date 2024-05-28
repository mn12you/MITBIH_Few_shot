import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix, average_precision_score,confusion_matrix
import wfdb


def split_data(seed=42):
    folds = range(1, 11)
    folds = np.random.RandomState(seed).permutation(folds)
    return folds[:8], folds[8:9], folds[9:]


def prepare_input(ecg_file: str):
    ecg_data=pd.read_csv((ecg_file+'.csv'),header=None).to_numpy()
    return ecg_data


def cal_scores(y_trues, y_scores):
    y_preds=apply_class(y_scores)
    precision = precision_score(y_trues, y_preds,average='macro')
    recall = recall_score(y_trues, y_preds,average='macro')
    f1=f1_score(y_trues, y_preds,average='macro')
    # auc = roc_auc_score(y_trues, y_scores,  average='macro',  multi_class='ovr')
    acc=accuracy_score(y_trues, y_preds)
    return precision, recall, f1, acc

def save_result(y_trues, y_scores,result_path,shot):
    precision, recall, f1, acc=cal_scores(y_trues, y_scores)
    y_preds=apply_class(y_scores)
    cm=confusion_matrix(y_trues.argmax(axis=1), y_preds.argmax(axis=1))
    # result_temp=np.concatenate(cm,axis=0)
    result_df=pd.DataFrame(cm)
    # result_all_df=pd.DataFrame({"Macro Precision":[precision],"Macro Recall":[recall],"Macro F1 score":[f1],"Macro AUC":[auc],"Thresholds":[thresholds]})
    result_all_df=pd.DataFrame({"Macro Precision":[precision],"Macro Recall":[recall],"Macro F1 score":[f1],"Accuracy":[acc]})
    result_df.to_csv(str(result_path)+"_cm_"+str(shot)+"_shot.csv")
    result_all_df.to_csv(str(result_path)+"_scores_"+str(shot)+"_shot.csv")


def cal_auc(y_trues,y_scores):
    return roc_auc_score(y_trues, y_scores,  average='macro',  multi_class='ovr')

# def find_optimal_threshold(y_trues, y_scores):
#     thresholds = np.linspace(0, 1, 100)
#     f1s=[]
#     for threshold in thresholds:
#         y_preds(y_scores,threshold) 
#         f1s.append(f1_score(y_trues, y_preds,average='macro'))
#     return thresholds[np.argmax(f1s)]

def confution_matrixs(y_trues,y_preds):
    cm=multilabel_confusion_matrix(y_trues, y_preds)
    return cm

def cal_acc(y_true, y_score):
    y_pred=apply_class(y_score)
    acc=accuracy_score(y_true, y_pred)
    return acc

def cal_f1(y_true, y_score):
    y_pred=apply_class(y_score)
    result=f1_score(y_true, y_pred,average='macro')
    return result

def apply_class( y_score):
    """
		apply class to prediction score in order to get binary format.
		BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
    """
    tmp = []	
    for p in y_score:
        tmp_p = (p ==np.max(p)).astype(int)
        tmp.append(tmp_p)
    tmp = np.array(tmp)
    return tmp

# def apply_thresholds( y_score,thresholds):
#     """
# 		apply class-wise thresholds to prediction score in order to get binary format.
# 		BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
#     """
#     tmp = []	
#     for p in y_score:
#         tmp_p = (p > thresholds).astype(int)
#         if np.sum(tmp_p) == 0:
#             tmp_p[np.argmax(p)] = 1
#         tmp.append(tmp_p)
#     tmp = np.array(tmp)
#     return tmp,thresholds

# def apply_optimal_thresholds(y_true, y_score):
#     """
# 		apply class-wise optimal thresholds to prediction score in order to get binary format.
# 		BUT: if no score is above threshold, pick maximum. This is needed due to metric issues.
#     """
#     thresholds=find_optimal_threshold(y_true, y_score)
#     tmp = []	
#     for p in y_score:
#         tmp_p = (p > thresholds).astype(int)
#         if np.sum(tmp_p) == 0:
#             tmp_p[np.argmax(p)] = 1
#         tmp.append(tmp_p)
#     tmp = np.array(tmp)
#     return tmp,thresholds

