# Imports
from __future__ import print_function
from matplotlib import pyplot
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from pandas import read_csv
from keras.models import Model
from keras.layers import Dense, Input
from keras.regularizers import l2
from keras.optimizers import Adam
from numpy import genfromtxt
from sklearn import preprocessing  # for normalization
import sklearn.metrics as metrics
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from numpy import set_printoptions
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from keras.layers import LeakyReLU
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
import copy
from sklearn.feature_selection import f_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from concrete_autoencoder import ConcreteAutoencoderFeatureSelector
from sklearn.feature_selection import mutual_info_classif
import time
from math import sqrt
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Loading test data
df_test = read_csv('test_imperson_without4n7_balanced_data.csv',header=0)
df_test_y = df_test.iloc[:, 152:153]   # keeping just the classifier (y)
df_test = df_test.iloc[:, :152]    # removing the classifer (y)

# Loading training data
df_train = read_csv('train_imperson_without4n7_balanced_data.csv',header=0)
df_train_y = df_train.iloc[:, 152:153]   # keeping just the classifier (y)
df_train = df_train.iloc[:, :152]    # removing the classifer (y)

# Turning to float32
x_train = copy.deepcopy(df_train).astype('float32')
x_test = copy.deepcopy(df_test).astype('float32')

# Normalization

min_max_scaler = preprocessing.MinMaxScaler()

x_train = min_max_scaler.fit_transform(x_train)
x_train = pd.DataFrame(x_train)  # turning back in df

x_test = min_max_scaler.fit_transform(x_test)
x_test = pd.DataFrame(x_test)

# NaNs to 0
x_train.fillna(0, inplace=True)
x_test.fillna(0, inplace=True)

def basic_autoencoder(b_size, en_activation, de_activation, optim):
    encoding_dim = 10

    # Input
    input_img = Input(shape=(152,))

    # Dense layers = fully connected layers (3 yields better results)
    encoder_1 = Dense(70, activation=en_activation)(input_img)
    encoder_2 = Dense(20, activation=en_activation)(encoder_1)
    encoder_3 = Dense(encoding_dim, activation=en_activation)(encoder_2)

    # Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=2)
    
    # Decoded and Model Fit (is this needed?)
    decoded = Dense(152, activation=de_activation)(encoder_3)
    basic_autoencoder = Model(input_img, decoded)
    basic_autoencoder.compile(optimizer=optim, loss = 'binary_crossentropy')

    basic_autoencoder.fit(x_train, x_train, epochs=100, batch_size=b_size, shuffle=True,
               validation_data=(x_test, x_test), callbacks=[early_stop])
    
    # Latent Vector (Encoder model)
    encoder = Model(input_img, encoder_3)

    # Features Engineered
    encoded_train_data = encoder.predict(x_train)
    encoded_test_data = encoder.predict(x_test)
    
    train_loss = basic_autoencoder.history.history['loss']
    test_loss = basic_autoencoder.history.history['val_loss']

    return(encoded_train_data, encoded_test_data, train_loss, test_loss)


def sparse_autoencoder(b_size, en_activation, de_activation, reg, optim):
    encoding_dim = 10

    # Input
    input_img = Input(shape=(152,))

    # Dense layers = fully connected layers
    encoder_1 = Dense(encoding_dim, activation=en_activation, 
                      activity_regularizer=l2(reg))(input_img)
    
    # Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=2)
    
    # Decoded and Model Fit
    decoded = Dense(152, activation=de_activation, 
                    activity_regularizer=l2(reg))(encoder_1)  
    sparse_autoencoder = Model(input_img, decoded)
    sparse_autoencoder.compile(optimizer=optim, loss = 'binary_crossentropy')
    sparse_autoencoder.fit(x_train, x_train, epochs=100, batch_size=b_size, shuffle=True,
               validation_data=(x_test, x_test), callbacks = [early_stop])

    # Latent Vector (Encoder model)
    encoder = Model(input_img, encoder_1)

    # Features Engineered
    encoded_train_data = encoder.predict(x_train)
    encoded_test_data = encoder.predict(x_test)
    
    train_loss = sparse_autoencoder.history.history['loss']
    test_loss = sparse_autoencoder.history.history['val_loss']

    return(encoded_train_data, encoded_test_data, train_loss, test_loss)


def stacked_autoencoder(b_size, en_activation, de_activation, reg, optim):
    encoding_dim = 10

    # Input
    input_img = Input(shape=(152,))

    # Dense layers = fully connected layers
    encoder_1 = Dense(150, activation=en_activation, 
                      activity_regularizer=l2(reg))(input_img)
    encoder_2 = Dense(50, activation=en_activation, 
                      activity_regularizer=l2(reg))(encoder_1)
    encoder_3 = Dense(20, activation=en_activation, 
                      activity_regularizer=l2(reg))(encoder_2)
    encoder_4 = Dense(encoding_dim, activation=en_activation, 
                      activity_regularizer=l2(reg))(encoder_3)
    
    # Early Stopping
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=2)
    
    # Decoded and Model Fit
    decoded = Dense(152, activation=de_activation, 
                    activity_regularizer=l2(reg))(encoder_4)  
    stacked_autoencoder = Model(input_img, decoded)
    stacked_autoencoder.compile(optimizer=optim, loss = 'binary_crossentropy')
    stacked_autoencoder.fit(x_train, x_train, epochs=100, batch_size=b_size, shuffle=True,
               validation_data=(x_test, x_test), callbacks = [early_stop])
    
    # Latent Vector (Encoder model)
    encoder = Model(input_img, encoder_4)
    
    # Features Engineered
    encoded_train_data = encoder.predict(x_train)
    encoded_test_data = encoder.predict(x_test)
        
    train_loss = stacked_autoencoder.history.history['loss']
    test_loss = stacked_autoencoder.history.history['val_loss']

    return(encoded_train_data, encoded_test_data, train_loss, test_loss)


def denoising_autoencoder(nf, b_size, en_activation, de_activation, optim):
    noise_factor = nf # 0.0005
    x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)
    
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
    
    encoding_dim = 10

    # Input
    input_img = Input(shape=(152,))

    # Dense layers = fully connected layers
    encoder_1 = Dense(encoding_dim, activation=en_activation)(input_img)
    
    # Early Stopping - to ensure no overfitting in model - https://stackoverflow.com/questions/47299624/how-to-understand-loss-acc-val-loss-val-acc-in-keras-model-fitting 
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=2)
    
    # Decoded and Model Fit
    decoded = Dense(152, activation=de_activation)(encoder_1)  
    denoising_autoencoder = Model(input_img, decoded)
    denoising_autoencoder.compile(optimizer=optim, loss = 'binary_crossentropy')
    denoising_autoencoder.fit(x_train_noisy, x_train_noisy, epochs=100, batch_size=b_size, shuffle=True,
               validation_data=(x_test_noisy, x_test_noisy), callbacks = [early_stop])

    # Latent Vector (Encoder model)
    encoder = Model(input_img, encoder_1)
    
    # Features Engineered
    encoded_train_data = encoder.predict(x_train)
    encoded_test_data = encoder.predict(x_test)
    
    train_loss = denoising_autoencoder.history.history['loss']
    test_loss = denoising_autoencoder.history.history['val_loss']

    return(encoded_train_data, encoded_test_data, train_loss, test_loss)


def stacked_denoising_autoencoder(nf, b_size, en_activation, de_activation, optim):
    noise_factor = nf
    x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)
    
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
    
    encoding_dim = 10

    # Input
    input_img = Input(shape=(152,))

    # Dense layers = fully connected layers
    encoder_1 = Dense(100, activation=en_activation)(input_img)
    encoder_2 = Dense(50, activation=en_activation)(encoder_1)
    encoder_3 = Dense(20, activation=en_activation)(encoder_2)
    encoder_4 = Dense(encoding_dim, activation=en_activation)(encoder_3)
    
    # Early Stopping - to ensure no overfitting in model - https://stackoverflow.com/questions/47299624/how-to-understand-loss-acc-val-loss-val-acc-in-keras-model-fitting 
    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=2)
    
    # Decoded and Model Fit
    decoded = Dense(152, activation=de_activation)(encoder_4)  
    denoising_autoencoder = Model(input_img, decoded)
    denoising_autoencoder.compile(optimizer=optim, loss = 'binary_crossentropy')
    # overfitting at batch_size=1000, changed to 700. Still overfitting. Changed to 300. trying 400
    denoising_autoencoder.fit(x_train_noisy, x_train_noisy, epochs=100, batch_size=b_size, shuffle=True,
               validation_data=(x_test_noisy, x_test_noisy), callbacks = [early_stop])

    # Latent Vector (Encoder model)
    encoder = Model(input_img, encoder_4)
    
    # Features Engineered
    encoded_train_data = encoder.predict(x_train)
    encoded_test_data = encoder.predict(x_test)
    
    train_loss = denoising_autoencoder.history.history['loss']
    test_loss = denoising_autoencoder.history.history['val_loss']

    return(encoded_train_data, encoded_test_data, train_loss, test_loss)


# Assigning best parameters per AE
basic_ae = basic_autoencoder(2000, 'elu', 'sigmoid', 'adam')
sparse_ae = sparse_autoencoder(2000, 'relu', 'sigmoid', 0.01, 'adam')
stacked_ae = stacked_autoencoder(2000, 'relu', 'sigmoid', 0.01,'adam')
denoising_ae = denoising_autoencoder(0.005, 2000, 'relu', 'sigmoid', 'adam')
stacked_denoising_ae = stacked_denoising_autoencoder(0.005, 2000, 'relu', 'sigmoid', 'adam')


# Combining features with original data

def combine_features(autoencoder):
    
    # Normalizing newly created features
    latent_rep_AE = min_max_scaler.fit_transform(autoencoder[0])
    encoded_test_data = min_max_scaler.fit_transform(autoencoder[1])
        
    # Turn data into pandas df
    latent_rep_AE_df = pd.DataFrame(data=latent_rep_AE)
    encoded_test_data_df = pd.DataFrame(data=encoded_test_data)
    latent_rep_AE_df = copy.deepcopy(latent_rep_AE_df)
    encoded_test_data_df = copy.deepcopy(encoded_test_data_df)

    # Combine additional features
    combined = pd.concat([x_train,latent_rep_AE_df],axis=1,ignore_index=True)
    combined_test_data = pd.concat([x_test,encoded_test_data_df],axis=1,ignore_index=True)
    
    return(combined, combined_test_data)


# Feature Selection - MI (Mutual Information)
def mutual_info(autoencoder, number_features):
        
    # Features
    combined = combine_features(autoencoder)[0]
    combined_test_data = combine_features(autoencoder)[1]

    mi_1 = SelectKBest(score_func=mutual_info_classif, k=number_features)
    mi_1.fit(combined, df_train_y)
    
    # Columns that have been selected
    cols = mi_1.get_support(indices=True)
    
    # Features that have been selected
    features_df_new = combined.iloc[:,cols]
    
    return(cols, features_df_new, combined_test_data)


# Feature Selection - RFE (Recursive Feature Elimination)

def wrapper_rfe(autoencoder, number_features):
    
    # Features
    combined = combine_features(autoencoder)[0]
    combined_test_data = combine_features(autoencoder)[1]
    
    # Feature extraction
    model = LogisticRegression(solver='liblinear')
    model = RFE(model, number_features)
    model = model.fit(combined, df_train_y)

    # Columns that have been selected
    cols = model.get_support(indices=True)
    
    # Features that have been selected
    features_df_new = combined.iloc[:,cols]
    
    return(cols, features_df_new, combined_test_data)


# Feature Selection - Feature Importance ExtraTrees
def selection_importance_ETC(autoencoder, number_of_features):
    
    # Features
    combined = combine_features(autoencoder)[0]
    combined_test_data = combine_features(autoencoder)[1]
    
    # feature extraction
    model = ExtraTreesClassifier()
    model.fit(combined, df_train_y)
    importances = model.feature_importances_

    # create a data frame for visualization.
    final_df = pd.DataFrame({"Features": combined.columns, "Importances":importances})
    final_df.set_index('Importances')

    # sort in ascending order to better visualization.
    final_df = final_df.sort_values('Importances', ascending=False)

    #
    final_df_cols = final_df.head(number_of_features)
    cols = final_df_cols.iloc[:,0]
    cols = cols.to_numpy()
    features_df_new = combined.iloc[:,cols]
    
    return(cols, features_df_new, combined_test_data)


# Logistic Regression Model (for comparing AE / feature selection)
def log_regress_model(feature_selection_model):
    
    start_time = time.time()
    
    # Data
    cols = feature_selection_model[0]
    features_df_new = feature_selection_model[1]
    combined_test_data = feature_selection_model[2]
    
    # Model
    lr_model = LogisticRegression()
    lr_model.fit(features_df_new, df_train_y)
    pred = lr_model.predict(combined_test_data[cols])
    
    cf = confusion_matrix(df_test_y, pred)
    
    ttm = (time.time() - start_time)
    
    return cf, cols, ttm, lr_model, features_df_new


# SVM - Linear - Model (for comparing AE / feature selection)
def svm_model(feature_selection_model):
    
    start_time = time.time()
    
    # Data
    cols = feature_selection_model[0]
    features_df_new = feature_selection_model[1]
    combined_test_data = feature_selection_model[2]
    
    #Create a svm Classifier - using only extracted features
    clf = svm.SVC(kernel='linear')

    #Train the model using the training sets
    clf.fit(features_df_new, df_train_y)

    #Predict the response for test dataset
    pred = clf.predict(combined_test_data[cols])
    
    cf = confusion_matrix(df_test_y, pred)
    
    ttm = (time.time() - start_time)
    
    return cf, cols, ttm, clf, features_df_new


# Comparing Accuracies per AE / feature selection pair
def performance_v_aefeaturesel(ml_model):
    
    cf = ml_model[0]
    cols = ml_model[1]
    ae_cols = ([x for x in cols if x > 151])
    ttm = ml_model[2]

    TN_test = cf[0][0]
    FP_test = cf[0][1]
    FN_test = cf[1][0]
    TP_test = cf[1][1]
    
    accuracy_test = round(((TP_test + TN_test) / (TP_test + TN_test + FP_test + FN_test)) * 100, 2)
    
    performance_test = {"Accuracy":accuracy_test, "Features Selected": cols, "AE Created Features Selected":ae_cols}

    return performance_test


# Function to plot performance_v_aefeaturesel results
def plotting_results(list_to_compare, titles):
    results = []
    for i in list_to_compare:
        results.append(i)
        
    results = pd.DataFrame(results, index=titles)
    results = results.sort_values(by=['Accuracy'], ascending=False)
    
    return display(results)

# Setting Display Options
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# Performance of Logistic Regression +  Mutual Information + various AEs
log_mi_basicae = performance_v_aefeaturesel(log_regress_model((mutual_info(basic_ae, 20))))
log_mi_sparseae = performance_v_aefeaturesel(log_regress_model((mutual_info(sparse_ae, 20))))
log_mi_stackedae = performance_v_aefeaturesel(log_regress_model((mutual_info(stacked_ae, 20))))
log_mi_denoisingae = performance_v_aefeaturesel(log_regress_model((mutual_info(denoising_ae, 20))))
log_mi_sta_denoisingae = performance_v_aefeaturesel(log_regress_model((mutual_info(stacked_denoising_ae, 20))))

# Plotting performance
list_lr_mi = [log_mi_basicae, log_mi_sparseae, log_mi_stackedae, log_mi_denoisingae,
           log_mi_sta_denoisingae]

titles_lr_mi = ['Basic AE - LR + MI', 'Sparse AE - LR + MI', 'Stacked AE - LR + MI', 
                'Denoising AE - LR + MI', 'SDAE - LR + MI']

plotting_results(list_lr_mi, titles_lr_mi) 


# Performance of Logistic Regression + ExtraTrees + various AEs
log_et_basicae = performance_v_aefeaturesel(log_regress_model((selection_importance_ETC(basic_ae, 20))))
log_et_sparseae = performance_v_aefeaturesel(log_regress_model((selection_importance_ETC(sparse_ae, 20))))
log_et_stackedae = performance_v_aefeaturesel(log_regress_model((selection_importance_ETC(stacked_ae, 20))))
log_et_denoisingae = performance_v_aefeaturesel(log_regress_model((selection_importance_ETC(denoising_ae, 20))))
log_et_sta_denoisingae = performance_v_aefeaturesel(log_regress_model((selection_importance_ETC(stacked_denoising_ae, 20))))

# Plotting performance
list_lr_et = [log_et_basicae, log_et_sparseae, log_et_stackedae, log_et_denoisingae,
           log_et_sta_denoisingae]

titles_lr_et = ['Basic AE - LR + ET', 'Sparse AE - LR + ET', 'Stacked AE - LR + ET', 
                'Denoising AE - LR + ET', 'SDAE - LR + ET']

plotting_results(list_lr_et,titles_lr_et)


# Performance of Logistic Regression + RFE + various AEs
log_rfe_basicae = performance_v_aefeaturesel(log_regress_model((wrapper_rfe(basic_ae, 20))))
log_rfe_sparseae = performance_v_aefeaturesel(log_regress_model((wrapper_rfe(sparse_ae, 20))))
log_rfe_stackedae = performance_v_aefeaturesel(log_regress_model((wrapper_rfe(stacked_ae, 20))))
log_rfe_denoisingae = performance_v_aefeaturesel(log_regress_model((wrapper_rfe(denoising_ae, 20))))
log_rfe_sta_denoisingae = performance_v_aefeaturesel(log_regress_model((wrapper_rfe(stacked_denoising_ae, 20))))

# Plotting performance
list_lr_rfe = [log_rfe_basicae, log_rfe_sparseae, log_rfe_stackedae, log_rfe_denoisingae,
           log_rfe_sta_denoisingae]

titles_lr_rfe = ['Basic AE - LR + RFE', 'Sparse AE - LR + RFE', 'Stacked AE - LR + RFE', 
                 'Denoising AE - LR + RFE', 'SDAE - LR + RFE']

plotting_results(list_lr_rfe,titles_lr_rfe)


# Performance of SVM + Mutual Information + various AEs
svm_mi_basicae = performance_v_aefeaturesel(svm_model((mutual_info(basic_ae, 20))))
svm_mi_sparseae = performance_v_aefeaturesel(svm_model((mutual_info(sparse_ae, 20))))
svm_mi_stackedae = performance_v_aefeaturesel(svm_model((mutual_info(stacked_ae, 20))))
svm_mi_denoisingae = performance_v_aefeaturesel(svm_model((mutual_info(denoising_ae, 20))))
svm_mi_sta_denoisingae = performance_v_aefeaturesel(svm_model((mutual_info(stacked_denoising_ae, 20))))

# Plotting performance
list_svm_mi = [svm_mi_basicae, svm_mi_sparseae, svm_mi_stackedae, svm_mi_denoisingae,
           svm_mi_sta_denoisingae]

titles_svm_mi = ['Basic AE - SVM + MI', 'Sparse AE - SVM + MI', 'Stacked AE - SVM + MI', 
                 'Denoising AE - SVM + MI', 'SDAE - SVM + MI']

plotting_results(list_svm_mi,titles_svm_mi)  


# Performance of SVM +  ExtraTress + various AEs
svm_et_basicae = performance_v_aefeaturesel(svm_model((selection_importance_ETC(basic_ae, 20))))
svm_et_sparseae = performance_v_aefeaturesel(svm_model((selection_importance_ETC(sparse_ae, 20))))
svm_et_stackedae = performance_v_aefeaturesel(svm_model((selection_importance_ETC(stacked_ae, 20))))
svm_et_denoisingae = performance_v_aefeaturesel(svm_model((selection_importance_ETC(denoising_ae, 20))))
svm_et_sta_denoisingae =performance_v_aefeaturesel(svm_model((selection_importance_ETC(stacked_denoising_ae, 20))))

# Plotting performance
list_svm_et = [svm_et_basicae, svm_et_sparseae, svm_et_stackedae, svm_et_denoisingae,
           svm_et_sta_denoisingae]

titles_svm_et = ['Basic AE - SVM + ET', 'Sparse AE - SVM + ET', 'Stacked AE - SVM + ET', 
                 'Denoising AE - SVM + ET', 'SDAE - SVM + ET']

plotting_results(list_svm_et,titles_svm_et)  


# Performance of SVM + RFE + various AEs
svm_rfe_basicae = performance_v_aefeaturesel(svm_model((wrapper_rfe(basic_ae, 20))))
svm_rfe_sparseae = performance_v_aefeaturesel(svm_model((wrapper_rfe(sparse_ae, 20))))
svm_rfe_stackedae = performance_v_aefeaturesel(svm_model((wrapper_rfe(stacked_ae, 20))))
svm_rfe_denoisingae = performance_v_aefeaturesel(svm_model((wrapper_rfe(denoising_ae, 20))))
svm_rfe_sta_denoisingae = performance_v_aefeaturesel(svm_model((wrapper_rfe(stacked_denoising_ae, 20))))

# Plotting performance
list_lr_rfe = [svm_rfe_basicae, svm_rfe_sparseae, svm_rfe_stackedae, svm_rfe_denoisingae,
           svm_rfe_sta_denoisingae]

titles_lr_rfe = ['Basic AE - SVM + RFE', 'Sparse AE - SVM + RFE', 'Stacked AE - SVM + RFE', 
                 'Denoising AE - SVM + RFE', 'SDAE - SVM + RFE']

plotting_results(list_lr_rfe,titles_lr_rfe)


# Best AE and feature selection combination assigned
data_best = selection_importance_ETC(denoising_ae, 20)


# Grid Search for Logistic Regression

# C (inverse of regularization strength), regularization penalty and solver considered
C = np.logspace(0, 4, num=10)
penalty = ['l1', 'l2']
solver = ['liblinear', 'saga', 'lbfgs']
hyperparameters = dict(C=C, penalty=penalty, solver=solver)

logisticregression_grid = GridSearchCV(log_regress_model(data_best)[3], hyperparameters)

# Fitting the model for grid search
best_model = logisticregression_grid.fit(log_regress_model(data_best)[4], df_train_y)

# Printing best parameters after hyperparameter tuning 
print("The best parameters after hyperparameter tuning our Logistic Regression model are:", '\n', best_model.best_params_, '\n') 
  
# Printing full model parameters after hyperparameter tuning 
print("The full model parameters after hyperparameter tuning our Logistic Regression model are:", '\n', best_model.best_estimator_) 


# Grid Search for SVM Model

# Cost and Gamma considered, rbf kernel
param_grid = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.0001],
              'kernel': ['rbf']}  

grid = GridSearchCV(svm_model(data_best)[3], param_grid=param_grid, refit = True, verbose = 3) 
  
# Fitting the model for grid search 
grid.fit(svm_model(data_best)[4], df_train_y)

# Printing best parameters after hyperparameter tuning 
print("The best parameters after hyperparameter tuning our SVM model are:", '\n', grid.best_params_, '\n') 
  
# Printing full model parameters after hyperparameter tuning 
print("The full model parameters after hyperparameter tuning our SVM model are:", '\n', grid.best_estimator_)


# MLP Model

def mlp_model(feature_selection_model, activation1, activation2, epo, bsize):
    
    start_time = time.time()
    
    # Data
    cols = feature_selection_model[0]
    features_df_new = feature_selection_model[1]
    combined_test_data = feature_selection_model[2]
    
    # Creating the model
    model = Sequential()
    model.add(Dense(30, input_dim=20, activation=activation1))
    model.add(Dense(15, activation=activation1))
    model.add(Dense(1, activation=activation2))
    
    # Compiling and fitting          
    model.compile(optimizer='sgd', loss='binary_crossentropy')
    model.fit(features_df_new, df_train_y, epochs = epo, batch_size=bsize)
    
    pred = model.predict_classes(combined_test_data[cols])
                  
    cf = confusion_matrix(df_test_y, pred)
        
    ttm = (time.time() - start_time)

    return cf, cols, ttm, model, features_df_new


# Checking impact of different batch_sizes on the MLP model
mlp150 = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=10, bsize=150))
mlp250 = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=10, bsize=250))
mlp500 = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=10, bsize=500))
mlp5000 = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=10, bsize=5000))

# Plotting
plotting_results([mlp150, mlp250, mlp500, mlp5000], 
                 ['MLP - Batch 150', 'MLP - Batch 250', 'MLP - Batch 500', 'MLP - Batch 5000'])


# Checking impact of different epochs on the MLP model
mlpe5 = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=5, bsize=150))
mlpe10 = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=10, bsize=150))
mlpe50 = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=50, bsize=150))
mlpe100 = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=100, bsize=150))
mlpe250 = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=250, bsize=150))

# Plotting
plotting_results([mlpe5, mlpe10, mlpe50, mlpe100, mlpe250], 
                 ['MLP - 5 epochs', 'MLP - 10 epochs', 'MLP - 50 epochs', 'MLP - 100 epochs', 'MLP - 250 epochs'])


# Comparing different activation functions on the MLP model
mlpelu = performance_v_aefeaturesel(mlp_model(data_best, 'elu','sigmoid',epo=10, bsize=150))
mlprelu = performance_v_aefeaturesel(mlp_model(data_best, 'relu','sigmoid',epo=10, bsize=150))
mlpsoftmax = performance_v_aefeaturesel(mlp_model(data_best, 'softmax','sigmoid',epo=10, bsize=150))
mlpselu = performance_v_aefeaturesel(mlp_model(data_best, 'selu','sigmoid',epo=10, bsize=150))

# Plotting
plotting_results([mlpelu, mlprelu, mlpsoftmax, mlpselu], 
                 ['MLP - Elu', 'MLP - Relu', 'MLP - Softmax', 'MLP - Selu'])


# Linear SVC Model

def model_linearsvc(feature_selection_model, c_reg):
    
    start_time = time.time()
    
    # Data
    cols = feature_selection_model[0]
    features_df_new = feature_selection_model[1]
    combined_test_data = feature_selection_model[2]
    
    #Create a svm Classifier - using only extracted features
    clf = svm.LinearSVC(C=c_reg)

    #Train the model using the training sets
    clf.fit(features_df_new, df_train_y)

    #Predict the response for test dataset
    pred = clf.predict(combined_test_data[cols])
    
    cf = confusion_matrix(df_test_y, pred)
    
    ttm = (time.time() - start_time)
    
    return cf, cols, ttm, clf, features_df_new, combined_test_data


# Checking impact of different C values - Regularization parameter - on the Linear SVC model.
linearsvc_05 = performance_v_aefeaturesel(model_linearsvc(data_best, c_reg=.05))
linearsvc_8 = performance_v_aefeaturesel(model_linearsvc(data_best, c_reg=.8))
linearsvc_12 = performance_v_aefeaturesel(model_linearsvc(data_best, c_reg=1.2))
linearsvc_15 = performance_v_aefeaturesel(model_linearsvc(data_best, c_reg=1.5))
linearsvc_20 = performance_v_aefeaturesel(model_linearsvc(data_best, c_reg=2.0))

# Plotting
plotting_results([linearsvc_05, linearsvc_8, linearsvc_12, linearsvc_15, linearsvc_20], 
                 ['Linear SVC - Regularization .05', 'Linear SVC - Regularization .8', 
                 'Linear SVC - Regularization 1.2', 'Linear SVC - Regularization 1.5', 
                 'Linear SVC - Regularization 2.0'])


# Perfomance function for calculating metrics from confusion matrix
# received from various ML models
def performance_(ml_model):
    
    cf = ml_model[0]
    cols = ml_model[1]
    ae_cols = ([x for x in cols if x > 151])
    ttm = ml_model[2]

    TN_test = cf[0][0]
    FP_test = cf[0][1]
    FN_test = cf[1][0]
    TP_test = cf[1][1]
    
    accuracy_test = round(((TP_test + TN_test) / (TP_test + TN_test + FP_test + FN_test)) * 100, 2)
    detectionrate_recall_test = round(((TP_test) / (TP_test + FN_test))* 100, 2)
    precision_test = round(((TP_test) / (TP_test + FP_test)) * 100, 2)
    false_alarm_test = round(((FP_test) / (TN_test + FP_test)) * 100, 2)
    fnr_unabletodetect_test = round(((FN_test) / (FN_test + TP_test)) * 100, 2)
    f1_meanofprecision_recall_test = round(((2 * (TP_test)) / ((2 * (TP_test)) + FP_test + FN_test)) * 100, 2)
    mcc = round(((((TP_test * TN_test) - (FP_test * FN_test)) / (sqrt((TP_test + FP_test)
                                                                     * (TP_test + FN_test)
                                                                     * (TN_test + FP_test)
                                                                     * (TN_test + FN_test))))) * 100, 2)
    
    performance_test = {"Accuracy":accuracy_test, "Detection Rate":detectionrate_recall_test,
                        "Precision":precision_test, "False Alarm Rate":false_alarm_test, 
                        "Unable to Detect":fnr_unabletodetect_test, "F1":f1_meanofprecision_recall_test, "MCC":mcc, "TTB":ttm, "Features Selected": cols, "AE Created Features Selected":ae_cols}

    return performance_test


# Performance of Logistic Regression model with best hyperparameters

def log_regress_model(feature_selection_model):
    
    start_time = time.time()
    
    # Data
    cols = feature_selection_model[0]
    features_df_new = feature_selection_model[1]
    combined_test_data = feature_selection_model[2]
    
    # Model
    lr_model = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')
    lr_model.fit(features_df_new, df_train_y)
    pred = lr_model.predict(combined_test_data[cols])
    
    cf = confusion_matrix(df_test_y, pred)
    
    ttm = (time.time() - start_time)
    
    return cf, cols, ttm, lr_model, features_df_new

logress_best = performance_(log_regress_model(data_best))


# Performance of RBF SVM model with best hyperparameters

def svm_model_rbf(feature_selection_model):
    
    start_time = time.time()
    
    # Data
    cols = feature_selection_model[0]
    features_df_new = feature_selection_model[1]
    combined_test_data = feature_selection_model[2]
    
    #Create a svm Classifier - using only extracted features
    clf = svm.SVC(C=0.1, gamma=1, kernel='rbf')

    #Train the model using the training sets
    clf.fit(features_df_new, df_train_y)

    #Predict the response for test dataset
    pred = clf.predict(combined_test_data[cols])
    
    cf = confusion_matrix(df_test_y, pred)
    
    ttm = (time.time() - start_time)
    
    return cf, cols, ttm, clf, features_df_new

svmrbf_best = performance_(svm_model_rbf(data_best))


# Performance of MLP model with best hyperparameters
mlp_best = performance_(mlp_model(data_best, 'elu','sigmoid',epo=15, bsize=150))


# Performance of Linear SVC model with best hyperparameters
linearsvc_best = performance_(model_linearsvc(data_best, c_reg=.05))


# Performance of SVM (Kernel=linear)
svmlinearkern = performance_(svm_model(data_best))


# Plotting results to compare all ML models

plotting_results([logress_best, svmrbf_best, mlp_best, linearsvc_best, svmlinearkern], 
                 ['Logistic Regression', 'SVM (RBF)', 'MLP', 'Linear SVC', 'SVM (Linear Kernel)'])

