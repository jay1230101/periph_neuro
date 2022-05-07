import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import altair as alt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.metrics import roc_auc_score,auc
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from numpy import where
from sklearn.preprocessing import LabelEncoder
from PIL import Image
data=pd.read_csv('peripheral_neuropathy_19_03-2022.csv',encoding = "ISO-8859-1")

st.sidebar.subheader('Presenter')
if st.sidebar.checkbox('Check to see Presenter'):

    col1,col2,col3=st.columns([1,6,1])
    with col1:
        st.write("")
    with col2:
        st.markdown ( "<h1 style='text-align: center; color: black;'>Johny El Achkar, MSBA.</h1>", unsafe_allow_html=True )
        st.markdown ( "<h3 style='text-align: center; color: black;'>Predictive Model: Peripheral Neuropathy</h3>", unsafe_allow_html=True )
    with col3:
        st.write("")
    col1,col2,col3=st.columns([2.3,3.7,2])
    with col1:
        st.write("")
    with col2:
        image = Image.open ( 'johny.jpg' )
        new_width = 300
        new_height = 300
        image = image.resize ( (new_width, new_height), Image.ANTIALIAS )
        st.image(image)
    with col3:
        st.write("")

data= data.drop(['Sx Onset','RLS','Anesthesia','PY','ID','MP-A','MP-CV','Me-CV','P-A','P-CV','S-CV','U-CV','Vib','PinP','U-A','LT','Tgait','Alcohol','Tg'],axis=1)
data=data.rename(columns={'HR-S':'HR_S','Renal disease':'Renal_disease','Family stress':'Family_stress','Driving hours':'Driving_hours','Working hours':'Working_hours','BP-L Dias':'BP_L_Dias','BP-S Sys':'BP_S_Sys','BP- S Dias':'BP_S_Dias','HR - L':'HR_L','BP-L Sys':'BP_L_Sys','Diurnal\xa0':'Diurnal'})
# remove rows having the following special characters
data = data[(data.Diet != 'MISSING') & (data.Diet != 'EXCLUDED') &(data.Diet !='EXCLUDED CHEMO') &(data.Diet !='#DIV/0!')
& (data.Diet != '/\xa0') & (data.Diet != '#VALUE!') &(data.Diet != 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Age != 'MISSING') & (data.Age != 'EXCLUDED') &(data.Age !='EXCLUDED CHEMO') &(data.Age !='#DIV/0!')
& (data.Age != '/\xa0') & (data.Age != '#VALUE!') &(data.Age != 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Sex != 'MISSING') & (data.Sex != 'EXCLUDED') &(data.Sex !='EXCLUDED CHEMO') &(data.Sex !='#DIV/0!')
& (data.Sex != '/\xa0') & (data.Sex != '#VALUE!') &(data.Sex != 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Weight != 'MISSING') & (data.Weight != 'EXCLUDED') &(data.Weight !='EXCLUDED CHEMO') &(data.Weight !='#DIV/0!')
& (data.Weight != '/\xa0') & (data.Weight != '#VALUE!') &(data.Weight != 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Height!= 'MISSING') & (data.Height!= 'EXCLUDED') &(data.Height!='EXCLUDED CHEMO') &(data.Height!='#DIV/0!')
& (data.Height!= '/\xa0') & (data.Height!= '#VALUE!') &(data.Height!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.BMI!= 'MISSING') & (data.BMI!= 'EXCLUDED') &(data.BMI!='EXCLUDED CHEMO') &(data.BMI!='#DIV/0!')
& (data.BMI!= '/\xa0') & (data.BMI!= '#VALUE!') &(data.BMI!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Smoker!= 'MISSING') & (data.Smoker!= 'EXCLUDED') &(data.Smoker!='EXCLUDED CHEMO') &(data.Smoker!='#DIV/0!')
& (data.Smoker!= '/\xa0') & (data.Smoker!= '#VALUE!') &(data.Smoker!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data['Activity Level']!= 'MISSING') & (data['Activity Level']!= 'EXCLUDED') &(data['Activity Level']!='EXCLUDED CHEMO') &(data['Activity Level']!='#DIV/0!')
& (data['Activity Level']!= '/\xa0') & (data['Activity Level']!= '#VALUE!') &(data['Activity Level']!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Caffeine!= 'MISSING') & (data.Caffeine!= 'EXCLUDED') &(data.Caffeine!='EXCLUDED CHEMO') &(data.Caffeine!='#DIV/0!')
& (data.Caffeine!= '/\xa0') & (data.Caffeine!= '#VALUE!') &(data.Caffeine!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Working_hours!= 'MISSING') & (data.Working_hours!= 'EXCLUDED') &(data.Working_hours!='EXCLUDED CHEMO') &(data.Working_hours!='#DIV/0!')
& (data.Working_hours!= '/\xa0') & (data.Working_hours!= '#VALUE!') &(data.Working_hours!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Driving_hours!= 'MISSING') & (data.Driving_hours!= 'EXCLUDED') &(data.Driving_hours!='EXCLUDED CHEMO') &(data.Driving_hours!='#DIV/0!')
& (data.Driving_hours!= '/\xa0') & (data.Driving_hours!= '#VALUE!') &(data.Driving_hours!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Family_stress!= 'MISSING') & (data.Family_stress!= 'EXCLUDED') &(data.Family_stress!='EXCLUDED CHEMO') &(data.Family_stress!='#DIV/0!')
& (data.Family_stress!= '/\xa0') & (data.Family_stress!= '#VALUE!') &(data.Family_stress!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Hypertension!= 'MISSING') & (data.Hypertension!= 'EXCLUDED') &(data.Hypertension!='EXCLUDED CHEMO') &(data.Hypertension!='#DIV/0!')
& (data.Hypertension!= '/\xa0') & (data.Hypertension!= '#VALUE!') &(data.Hypertension!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Renal_disease!= 'MISSING') & (data.Renal_disease!= 'EXCLUDED') &(data.Renal_disease!='EXCLUDED CHEMO') &(data.Renal_disease!='#DIV/0!')
& (data.Renal_disease!= '/\xa0') & (data.Renal_disease!= '#VALUE!') &(data.Renal_disease!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Hyperlipidemia!= 'MISSING') & (data.Hyperlipidemia!= 'EXCLUDED') &(data.Hyperlipidemia!='EXCLUDED CHEMO') &(data.Hyperlipidemia!='#DIV/0!')
& (data.Hyperlipidemia!= '/\xa0') & (data.Hyperlipidemia!= '#VALUE!') &(data.Hyperlipidemia!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Hyperthyroidism!= 'MISSING') & (data.Hyperthyroidism!= 'EXCLUDED') &(data.Hyperthyroidism!='EXCLUDED CHEMO') &(data.Hyperthyroidism!='#DIV/0!')
& (data.Hyperthyroidism!= '/\xa0') & (data.Hyperthyroidism!= '#VALUE!') &(data.Hyperthyroidism!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Hypothyroidism!= 'MISSING') & (data.Hypothyroidism!= 'EXCLUDED') &(data.Hypothyroidism!='EXCLUDED CHEMO') &(data.Hypothyroidism!='#DIV/0!')
& (data.Hypothyroidism!= '/\xa0') & (data.Hypothyroidism!= '#VALUE!') &(data.Hypothyroidism!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Retinopathy!= 'MISSING') & (data.Retinopathy!= 'EXCLUDED') &(data.Retinopathy!='EXCLUDED CHEMO') &(data.Retinopathy!='#DIV/0!')
& (data.Retinopathy!= '/\xa0') & (data.Retinopathy!= '#VALUE!') &(data.Retinopathy!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.DM!= 'MISSING') & (data.DM!= 'EXCLUDED') &(data.DM!='EXCLUDED CHEMO') &(data.DM!='#DIV/0!')
& (data.DM!= '/\xa0') & (data.DM!= '#VALUE!') &(data.DM!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.AgeO!= 'MISSING') & (data.AgeO!= 'EXCLUDED') &(data.AgeO!='EXCLUDED CHEMO') &(data.AgeO!='#DIV/0!')
& (data.AgeO!= '/\xa0') & (data.AgeO!= '#VALUE!') &(data.AgeO!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Modification!= 'MISSING') & (data.Modification!= 'EXCLUDED') &(data.Modification!='EXCLUDED CHEMO') &(data.Modification!='#DIV/0!')
& (data.Modification!= '/\xa0') & (data.Modification!= '#VALUE!') &(data.Modification!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Pain!= 'MISSING') & (data.Pain!= 'EXCLUDED') &(data.Pain!='EXCLUDED CHEMO') &(data.Pain!='#DIV/0!')
& (data.Pain!= '/\xa0') & (data.Pain!= '#VALUE!') &(data.Pain!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Diurnal!= 'MISSING') & (data.Diurnal!= 'EXCLUDED') &(data.Diurnal!='EXCLUDED CHEMO') &(data.Diurnal!='#DIV/0!')
& (data.Diurnal!= '/\xa0') & (data.Diurnal!= '#VALUE!') &(data.Diurnal!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Coldness!= 'MISSING') & (data.Coldness!= 'EXCLUDED') &(data.Coldness!='EXCLUDED CHEMO') &(data.Coldness!='#DIV/0!')
& (data.Coldness!= '/\xa0') & (data.Coldness!= '#VALUE!') &(data.Coldness!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Tingling!= 'MISSING') & (data.Tingling!= 'EXCLUDED') &(data.Tingling!='EXCLUDED CHEMO') &(data.Tingling!='#DIV/0!')
& (data.Tingling!= '/\xa0') & (data.Tingling!= '#VALUE!') &(data.Tingling!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Alloydnia!= 'MISSING') & (data.Alloydnia!= 'EXCLUDED') &(data.Alloydnia!='EXCLUDED CHEMO') &(data.Alloydnia!='#DIV/0!')
& (data.Alloydnia!= '/\xa0') & (data.Alloydnia!= '#VALUE!') &(data.Alloydnia!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.HR_L!= 'MISSING') & (data.HR_L!= 'EXCLUDED') &(data.HR_L!='EXCLUDED CHEMO') &(data.HR_L!='#DIV/0!')
& (data.HR_L!= '/\xa0') & (data.HR_L!= '#VALUE!') &(data.HR_L!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.BP_L_Sys!= 'MISSING') & (data.BP_L_Sys!= 'EXCLUDED') &(data.BP_L_Sys!='EXCLUDED CHEMO') &(data.BP_L_Sys!='#DIV/0!')
& (data.BP_L_Sys!= '/\xa0') & (data.BP_L_Sys!= '#VALUE!') &(data.BP_L_Sys!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.BP_L_Dias!= 'MISSING') & (data.BP_L_Dias!= 'EXCLUDED') &(data.BP_L_Dias!='EXCLUDED CHEMO') &(data.BP_L_Dias!='#DIV/0!')
& (data.BP_L_Dias!= '/\xa0') & (data.BP_L_Dias!= '#VALUE!') &(data.BP_L_Dias!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.HR_S!= 'MISSING') & (data.HR_S!= 'EXCLUDED') &(data.HR_S!='EXCLUDED CHEMO') &(data.HR_S!='#DIV/0!')
& (data.HR_S!= '/\xa0') & (data.HR_S!= '#VALUE!') &(data.HR_S!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.BP_S_Sys!= 'MISSING') & (data.BP_S_Sys!= 'EXCLUDED') &(data.BP_S_Sys!='EXCLUDED CHEMO') &(data.BP_S_Sys!='#DIV/0!')
& (data.BP_S_Sys!= '/\xa0') & (data.BP_S_Sys!= '#VALUE!') &(data.BP_S_Sys!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.BP_S_Dias!= 'MISSING') & (data.BP_S_Dias!= 'EXCLUDED') &(data.BP_S_Dias!='EXCLUDED CHEMO') &(data.BP_S_Dias!='#DIV/0!')
& (data.BP_S_Dias!= '/\xa0') & (data.BP_S_Dias!= '#VALUE!') &(data.BP_S_Dias!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Ohtn!= 'MISSING') & (data.Ohtn!= 'EXCLUDED') &(data.Ohtn!='EXCLUDED CHEMO') &(data.Ohtn!='#DIV/0!')
& (data.Ohtn!= '/\xa0') & (data.Ohtn!= '#VALUE!') &(data.Ohtn!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.DTRs!= 'MISSING') & (data.DTRs!= 'EXCLUDED') &(data.DTRs!='EXCLUDED CHEMO') &(data.DTRs!='#DIV/0!')
& (data.DTRs!= '/\xa0') & (data.DTRs!= '#VALUE!') &(data.DTRs!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.Rom!= 'MISSING') & (data.Rom!= 'EXCLUDED') &(data.Rom!='EXCLUDED CHEMO') &(data.Rom!='#DIV/0!')
& (data.Rom!= '/\xa0') & (data.Rom!= '#VALUE!') &(data.Rom!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.FBS!= 'MISSING') & (data.FBS!= 'EXCLUDED') &(data.FBS!='EXCLUDED CHEMO') &(data.FBS!='#DIV/0!')
& (data.FBS!= '/\xa0') & (data.FBS!= '#VALUE!') &(data.FBS!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.TSH!= 'MISSING') & (data.TSH!= 'EXCLUDED') &(data.TSH!='EXCLUDED CHEMO') &(data.TSH!='#DIV/0!')
& (data.TSH!= '/\xa0') & (data.TSH!= '#VALUE!') &(data.TSH!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.VitB12!= 'MISSING') & (data.VitB12!= 'EXCLUDED') &(data.VitB12!='EXCLUDED CHEMO') &(data.VitB12!='#DIV/0!')
& (data.VitB12!= '/\xa0') & (data.VitB12!= '#VALUE!') &(data.VitB12!= 'EXCLUDED RENAL TRANSPLANT')]
data = data[(data.HDL!= 'MISSING') & (data.HDL!= 'EXCLUDED') &(data.HDL!='EXCLUDED CHEMO') &(data.HDL!='#DIV/0!')
& (data.HDL!= '/\xa0') & (data.HDL!= '#VALUE!') &(data.HDL!= 'EXCLUDED RENAL TRANSPLANT')]
data['S_A '].isnull().sum()
data = data[data['S_A '].notnull()]

bins =[-1,0.1,20,np.inf]
labels=('Severe_PN','Moderate_PN','No_PN')
data['S_A_categories']=pd.cut(data['S_A '],bins=bins,labels=labels)
data = data.replace(['-','/'],np.nan)
data['TSH']=data['TSH'].map(lambda x: str(x).lstrip('<'))
data['TSH']=data['TSH'].astype('float32')
# use appropriate data type
data['Age']=data['Age'].astype('float32')
data['Weight']=data['Weight'].astype('float32')
data['Height']=data['Height'].astype('float32')
data['BMI']=data['BMI'].astype('float32')
data['Working_hours']=data['Working_hours'].astype('float32')
data['Driving_hours']=data['Driving_hours'].astype('float32')
data['AgeO']=data['AgeO'].astype('float32')
data['HR_L']=data['HR_L'].astype('float32')
data['BP_L_Sys']=data['BP_L_Sys'].astype('float32')
data['BP_L_Dias']=data['BP_L_Dias'].astype('float32')
data['HR_S']=data['HR_S'].astype('float32')
data['BP_S_Sys']=data['BP_S_Sys'].astype('float32')
data['BP_S_Dias']=data['BP_S_Dias'].astype('float32')
data=data[['Age', 'Sex', 'HR_L', 'BP_L_Sys', 'BP_L_Dias', 'HR_S', 'BP_S_Sys', 'BP_S_Dias',
       'Ohtn', 'DTRs', 'Rom', 'FBS', 'TSH', 'VitB12', 'HDL', 'LDL','Weight', 'Height', 'BMI', 'Diet', 'Smoker',
       'Activity Level', 'Caffeine', 'Working_hours', 'Driving_hours',
       'Family_stress', 'Hypertension', 'Renal_disease', 'Hyperlipidemia',
       'Hyperthyroidism', 'Hypothyroidism', 'Retinopathy', 'DM', 'AgeO',
       'Modification', 'Pain', 'Diurnal', 'Coldness', 'Tingling', 'Alloydnia','S_A ',
       'S_A_categories']]

st.sidebar.subheader('Data Cleaning')
if st.sidebar.checkbox('Data Cleaning Process'):
    st.markdown ( "<h3 style='text-align: center; color: black;'>Data Frame Sample</h3>",
                  unsafe_allow_html=True )
    st.write()
    st.write(data.head(3))
    st.markdown( "<h3 style='text-align: center; color: black;'>Number of Initial Features = 60</h3>", unsafe_allow_html=True )
    st.markdown ( "<h3 style='text-align: center; color: black;'>Number of Features reduced = 41</h3>", unsafe_allow_html=True )

st.sidebar.subheader('Data Splitting')
if st.sidebar.checkbox('Data Splitting'):
    col1, col2, col3 = st.columns ( [1, 6, 1] )
    with col1:
        st.write ( "" )
    with col2:
        st.markdown ( "<h1 style='text-align: center; color: black;'>3 Folds Splitting.</h1>",
                      unsafe_allow_html=True )
        st.markdown ( "<h3 style='text-align: center; color: black;'>60% Training & Validation and 40% Testing</h3>",
                      unsafe_allow_html=True )
    with col3:
        st.write ( "" )
    col1, col2, col3 = st.columns ( [2.3, 3.7, 2] )
    with col1:
        st.write ( "" )
    with col2:
        image1 = Image.open ( 'hold.jpg' )
        new_width = 650
        new_height = 650
        image1 = image1.resize ( (new_width, new_height), Image.ANTIALIAS )
        st.image ( image1 )
    with col3:
        st.write ( "" )

X=data.drop(['S_A_categories','S_A '],axis=1)
y_cat=data['S_A_categories']
y_num=data['S_A ']

y_cat=data['S_A_categories'].map({
    'Severe_PN':0,
    'Moderate_PN':1,
    'No_PN':2
})
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y_cat,test_size=0.4,stratify=y_cat,random_state=0)
X_train_val,X_val,y_train_val,y_val=train_test_split(X_train,y_train,test_size=0.4,stratify=y_train,random_state=0)

X_train_val['DTRs']=X_train_val['DTRs'].replace('3','2')
X_train_val['DTRs']=X_train_val['DTRs'].astype(str)
X_train_val['Diet']=X_train_val['Diet'].replace(['5','3','9','2'],'6')
X_train_val['Diet']=X_train_val['Diet'].astype(str)
# using new variables names just for exploration
from sklearn.model_selection import train_test_split
X_train1,X_test1,y_train1,y_test1 = train_test_split(X,y_num,test_size=0.4,random_state=0)
X_train_val1,X_val1,y_train_val1,y_val1=train_test_split(X_train1,y_train1,test_size=0.4,random_state=0)
X_train_val_explor_numeric= X_train_val1.select_dtypes(include=[np.number])
X_train_val_explor_numeric =X_train_val_explor_numeric.reset_index()

y_train_val_explor = y_train_val1.reset_index()
train_val_explor_final = X_train_val_explor_numeric.merge(y_train_val_explor,on='index',how='outer')
train_val_explor_final = train_val_explor_final.fillna(train_val_explor_final.mean())


X_train_val_explor_cat= X_train_val1.select_dtypes(exclude=[np.number])
X_train_val_explor_cat=X_train_val_explor_cat.reset_index()
X_train_val_explor_cat= X_train_val_explor_cat.apply(lambda x: x.fillna(x.value_counts().index[0]))
X_train_val_explor_cat=X_train_val_explor_cat.reset_index()


data_final_num_cat = train_val_explor_final.merge(X_train_val_explor_cat,on='index',how='outer')
data_final_num_cat=data_final_num_cat.drop('index',axis=1)
bins1 =[-1,0.1,20,np.inf]
labels1=('Severe_PN','Moderate_PN','No_PN')
data_final_num_cat['S_A_categories']=pd.cut(data_final_num_cat['S_A '],bins=bins,labels=labels)
scale = StandardScaler()
data_final_num_cat1 = data_final_num_cat.drop('S_A ',axis=1)
x_explo = data_final_num_cat1.drop('S_A_categories',axis=1)
x_explo1 = scale.fit_transform(x_explo)

# Pipeline without RFECV

BP_L_Dias_ix,HR_S_ix,BP_S_Sys_ix,TSH_ix = [
                                                 list(X_train_val.columns).index(col) for col in ('BP_L_Dias','HR_S','BP_S_Sys','TSH')
]
def add_extra_features(X,add_BP_S_Sys=True):
  BP_L_Dias_TSH = X[:,BP_L_Dias_ix]/X[:,TSH_ix]
  HR_S_TSH=X[:,HR_S_ix]/X[:,TSH_ix]
  if add_BP_S_Sys:
    BP_S_Sys=X[:,BP_S_Sys_ix]/X[:,TSH_ix]
    return np.c_[X,BP_L_Dias_TSH,HR_S_TSH,BP_S_Sys]
  else:
    return np.c_[X,BP_L_Dias_TSH,HR_S_TSH]

attr_adder = FunctionTransformer(add_extra_features,validate=False,kw_args={'add_BP_S_Sys':False})
train_extra_attribs=attr_adder.fit_transform(X_train_val.values)
train_extra_attribs=pd.DataFrame(
    train_extra_attribs,
    columns=list(X_train_val.columns)+['BP_L_Dias_TSH','HR_S_TSH'],
    index=X_train_val.index

)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
num_pipeline = Pipeline([
                         ('imputer',SimpleImputer(strategy='median')),
                         ('attribs_adder',FunctionTransformer(add_extra_features)),
                         ('std_scaler',StandardScaler()),
])
train_num = X_train_val.select_dtypes(include=[np.number])
train_cat=X_train_val.select_dtypes(exclude=[np.number])
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
num_attribs = list(train_num)
cat_attribs = list(train_cat)

full_pipeline = ColumnTransformer([
                                   ('num',num_pipeline,num_attribs ),
                                   ('cat',OneHotEncoder(handle_unknown = "ignore"),cat_attribs),
])
train_prepared = full_pipeline.fit_transform(X_train_val)
valid_prepared = full_pipeline.transform(X_val)
# Basline Models - datafarme without RFECV
classifiers = [LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),SVC(),KNeighborsClassifier()]
estimators = ['Logistic Regression','Random Forest Classifier','Decision Tree Classifier','SVC','KNN']
Accuracy=[]
Estimator=[]
Model =[]
for c,e in zip(classifiers,estimators):
  cvs=c.fit(train_prepared,y_train_val)
  y_predict1 = cvs.predict(valid_prepared)
  accuracy1 = accuracy_score(y_val,y_predict1)
  accuracy1=round(accuracy1,2)
  Accuracy.append(accuracy1)
  Estimator.append(e)
  Model.append(c)

dtc_cvs=pd.DataFrame({'Model':Model,'Estimator':Estimator,'Accuracy':Accuracy})
dtc_cvs=dtc_cvs.sort_values(by='Accuracy',ascending=False)
dtc_cvs=dtc_cvs.drop('Model',axis=1)
X_train_prepared = full_pipeline.transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)
y_test_bin=label_binarize(y_test,classes=[0,1,2])
n_classes1=y_test_bin.shape[1]

# dataframe with RFECV
RFECV_Data = data[['Age','VitB12','TSH','FBS','BP_S_Dias','BP_S_Sys','HR_S','BP_L_Dias','BP_L_Sys','HR_L','LDL','Diet','DTRs','Coldness','AgeO','Driving_hours', 'Working_hours','BMI','Caffeine','Weight','Height','HDL','S_A_categories']]
X_RFE =RFECV_Data.drop('S_A_categories',axis=1)
y_RFE= RFECV_Data['S_A_categories']
y_RFE_num=RFECV_Data['S_A_categories'].map({
    'Severe_PN':0,
    'Moderate_PN':1,
    'No_PN':2
})
X_RFE['DTRs']=X_RFE['DTRs'].replace('3','2')
X_RFE['DTRs']=X_RFE['DTRs'].astype(str)
X_train_r,X_test_r,y_train_r,y_test_R = train_test_split(X_RFE,y_RFE_num,test_size=0.4,stratify=y_RFE_num)
X_train_rfe,X_val_rfe,y_train_rfe,y_val_rfe=train_test_split(X_train_r,y_train_r,test_size=0.4,stratify=y_train_r)
BP_L_Dias_ix,HR_S_ix,BP_S_Sys_ix,TSH_ix = [
                                                 list(X_train_rfe.columns).index(col) for col in ('BP_L_Dias','HR_S','BP_S_Sys','TSH')
]
def add_extra_features1(X,add_BP_S_Sys=True):
  BP_L_Dias_TSH1 = X[:,BP_L_Dias_ix]/X[:,TSH_ix]
  HR_S_TSH1=X[:,HR_S_ix]/X[:,TSH_ix]
  if add_BP_S_Sys:
    BP_S_Sys1=X[:,BP_S_Sys_ix]/X[:,TSH_ix]
    return np.c_[X,BP_L_Dias_TSH1,HR_S_TSH1,BP_S_Sys1]
  else:
    return np.c_[X,BP_L_Dias_TSH1,HR_S_TSH1]

attr_adder1 = FunctionTransformer(add_extra_features1,validate=False,kw_args={'add_BP_S_Sys':False})
train_extra_attribs1=attr_adder1.fit_transform(X_train_rfe.values)
train_extra_attribs1=pd.DataFrame(
    train_extra_attribs1,
    columns=list(X_train_rfe.columns)+['BP_L_Dias_TSH1','HR_S_TSH1'],
    index=X_train_rfe.index

)

num_pipeline1 = Pipeline([
                         ('imputer',SimpleImputer(strategy='median')),
                         ('attribs_adder',FunctionTransformer(add_extra_features1)),
                         ('std_scaler',StandardScaler()),
])

train_num1 = X_train_rfe.select_dtypes(include=[np.number])
train_cat1=X_train_rfe.select_dtypes(exclude=[np.number])

num_attribs1 = list(train_num1)
cat_attribs1 = list(train_cat1)

full_pipeline1 = ColumnTransformer([
                                   ('num',num_pipeline1,num_attribs1 ),
                                   ('cat',OneHotEncoder(handle_unknown = "ignore"),cat_attribs1),
])
train_prepared1 = full_pipeline1.fit_transform(X_train_rfe)
valid_prepared1 = full_pipeline1.transform(X_val_rfe)
# classification models with RFECV
# Basline Model comparison - datafarme without RFECV
classifiers_rfe = [LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),SVC(),KNeighborsClassifier()]
estimators_rfe = ['Logistic Regression','Random Forest Classifier','Decision Tree Classifier','SVC','KNN']
Accuracy_rfe=[]
Estimator_rfe=[]
Model_rfe =[]
for c,e in zip(classifiers_rfe,estimators_rfe):
  cvs_rfe=c.fit(train_prepared1,y_train_rfe)
  y_predict_rfe = cvs_rfe.predict(valid_prepared1)
  accuracy_r = accuracy_score(y_val_rfe,y_predict_rfe)
  accuracy_r=round(accuracy_r,2)
  Accuracy_rfe.append(accuracy_r)
  Estimator_rfe.append(e)
  Model_rfe.append(c)


dtc_cvs1=pd.DataFrame({'Model':Model_rfe,'Estimator':Estimator_rfe,'Accuracy':Accuracy_rfe})
dtc_cvs1=dtc_cvs1.sort_values(by='Accuracy',ascending=False)
dtc_cvs1=dtc_cvs1.drop('Model',axis=1)
dtc_cvs1= dtc_cvs1.reset_index()
dtc_cvs= dtc_cvs.reset_index()
# Merging the two dataset for comparison
baseline_models=dtc_cvs.merge(dtc_cvs1,on='index',how='outer')
baseline_models = baseline_models.drop(['index','Estimator_y'],axis=1)
baseline_models = baseline_models.rename({'Estimator_x':'Classifier','Accuracy_x':'Accuracy_Without_RFECV','Accuracy_y':'Accuracy_with_RFECV'},axis=1)

# SMOTE
df_smote = data
y_smote = LabelEncoder().fit_transform(y_cat)
counter = Counter(y_smote)
# replace NaN of numerical features with Mean
data_numeric = data.select_dtypes(include=[np.number])
data_numeric = data_numeric.fillna(data_numeric.mean())
data_numeric=data_numeric.reset_index()
# replace NaN of categorical features with Mode or most frequent value of each column
data_categorical = data.select_dtypes(exclude=[np.number])
data_categorical = data_categorical.apply(lambda x: x.fillna(x.value_counts().index[0]))
data_categorical=data_categorical.reset_index()
data_final = data_numeric.merge(data_categorical,on='index',how='outer')
data_final=data_final.drop(['index','S_A '],axis=1)
y_cat1=data_final['S_A_categories'].map({
    'Severe_PN':0,
    'Moderate_PN':1,
    'No_PN':2
})

# Oversample minority classes with SMOTE
df_smote1 = data_final
data_smote = df_smote1.values
X_smote1,y_smote1 = data_smote[:,:-1],data_smote[:,-1]
y_smote1 = LabelEncoder().fit_transform(y_cat1)
#transform teh dataset
oversample=SMOTE()
X_smote1,y_smote1=oversample.fit_resample(X_smote1,y_smote1)
#summarize distribution
counter1 = Counter(y_smote1)


# let us split our data after SMOTE
X_train_smote,X_test_smote,y_train_smote,y_test_smote = train_test_split(X_smote1,y_smote1,test_size=0.3,stratify=y_smote1,shuffle=True,random_state=0)
X_train1_smote,X_valid1_smote,y_train1_smote,y_valid1_smote=train_test_split(X_train_smote,y_train_smote,test_size=0.3,stratify=y_train_smote,shuffle=True,random_state=0)





import seaborn as sns
st.sidebar.subheader('Data Exploration')
if st.sidebar.checkbox('Data Exploration'):
    selectbox1=st.sidebar.selectbox('Data Exploration',['Correlations 1','Correlations 2','HeatMap','TSNE'])
    if selectbox1=='Correlations 1':
        col1,col2 =st.columns(2)
        with col1:
            st.subheader('SA and Age')
            c = alt.Chart ( data_final_num_cat).mark_circle ().encode (
                            x='S_A ', y='Age', tooltip=['S_A ', 'Age'] )
            st.altair_chart(c,use_container_width=True)
        with col2:
            st.subheader('SA and Age of Onset')
            c = alt.Chart ( data_final_num_cat).mark_circle ().encode (
                            x='S_A ', y='AgeO', tooltip=['S_A ', 'AgeO'] )
            st.altair_chart(c,use_container_width=True)
        col1,col2=st.columns(2)
        with col1:
            image2=Image.open ( 'age.jpg' )
            st.image(image2)
        with col2:
            image3=Image.open ( 'ageo.jpg' )
            st.image(image3)
    if selectbox1 =='Correlations 2':
        col1,col2=st.columns(2)
        with col1:
            st.subheader ( 'SA and Fasting Blood Sugar' )
            data3 = data_final_num_cat
            data3['Sex'] = data3['Sex'].astype ( int )
            bins = [0, 1.1, 2.1]
            labels = ('Male', 'Female')
            data3['Gender'] = pd.cut ( data3['Sex'], bins=bins, labels=labels )
            data3.head ( 2 )
            c = alt.Chart (data3).mark_circle ().encode (
                            x='S_A ', y='FBS',size='Gender',color='Gender', tooltip=['S_A ', 'FBS'] )
            st.altair_chart(c,use_container_width=True)

        with col2:
            st.subheader ( 'SA and Vitamin B12' )
            data3 = data_final_num_cat
            data3['Sex'] = data3['Sex'].astype ( int )
            bins = [0, 1.1, 2.1]
            labels = ('Male', 'Female')
            data3['Gender'] = pd.cut ( data3['Sex'], bins=bins, labels=labels )
            data3.head ( 2 )
            c = alt.Chart (data3).mark_circle ().encode (
                            x='S_A ', y='VitB12',size='Gender',color='Gender', tooltip=['S_A ', 'VitB12'] )
            st.altair_chart(c,use_container_width=True)
        with col1:
            image2=Image.open ( 'fbs.jpg' )
            st.image(image2)
        with col2:
            image3=Image.open ( 'vitb.jpg' )
            st.image(image3)

    if selectbox1 =='HeatMap':
        st.markdown ( "<h1 style='text-align: center; color: black;'>HeatMap</h1>",
                      unsafe_allow_html=True )
        corr_table = train_val_explor_final.corr()
        mask = np.triu ( np.ones_like ( corr_table, dtype=bool ) )
        fig, ax = plt.subplots ()
        sns.heatmap ( corr_table, ax=ax, fmt='.2g',mask=mask,annot=False)
        st.write ( fig )

    if selectbox1 =='TSNE':
        st.markdown ( "<h1 style='text-align: center; color: black;'>TSNE</h1>",
                      unsafe_allow_html=True )
        bins1 = [-1, 0.1, 20, np.inf]
        labels1 = ('Severe_PN', 'Moderate_PN', 'No_PN')
        data_final_num_cat['S_A_categories'] = pd.cut ( data_final_num_cat['S_A '], bins=bins, labels=labels )
        scale = StandardScaler ()
        data_final_num_cat1 = data_final_num_cat.drop ( 'S_A ', axis=1 )
        x_explo = data_final_num_cat1.drop ( 'S_A_categories', axis=1 )
        x_explo1 = scale.fit_transform ( x_explo )
        y_explo_cat = data_final_num_cat1['S_A_categories'].map ( {
            'Severe_PN': 0,
            'Moderate_PN': 1,
            'No_PN': 2
        } )
        tsne = TSNE ( n_components=2 )
        reduced_features = tsne.fit_transform ( x_explo1 )
        plt.figure ( figsize=(5, 5) )
        reduced_features0 = reduced_features[y_explo_cat == 0]
        reduced_features1 = reduced_features[y_explo_cat == 1]
        reduced_features2 = reduced_features[y_explo_cat == 2]
        x1 = reduced_features0[:, 0]
        x2 = reduced_features0[:, 1]
        tsne1 = pd.DataFrame ( x1, x2 )
        tsne1 = tsne1.reset_index ()
        tsne1 = tsne1.rename ( {'index': 'X1', 0: 'X2'}, axis=1 )
        tsne1['class'] = 'Severe PN'
        x1 = reduced_features1[:, 0]
        x2 = reduced_features1[:, 1]
        tsne2 = pd.DataFrame ( x1, x2 )
        tsne2 = tsne2.reset_index ()
        tsne2 = tsne2.rename ( {'index': 'X1', 0: 'X2'}, axis=1 )
        tsne2['class'] = 'Moderate PN'
        x1 = reduced_features2[:, 0]
        x2 = reduced_features2[:, 1]
        tsne3 = pd.DataFrame ( x1, x2 )
        tsne3 = tsne3.reset_index ()
        tsne3 = tsne3.rename ( {'index': 'X1', 0: 'X2'}, axis=1 )
        tsne3['class'] = 'No PN'
        chart1 = alt.Chart ( tsne1).mark_point ().encode (
            x='X1',
            y='X2',
            color='class'
        ).properties (
            height=300,
            width=800
        )

        chart2 = alt.Chart ( tsne2).mark_point ().encode (
            x='X1',
            y='X2',
            color='class'
        ).properties (
            height=300,
            width=800
        )

        chart3 = alt.Chart ( tsne3).mark_point ().encode (
            x='X1',
            y='X2',
            color='class'
        ).properties (
            height=300,
            width=800
        )

        st.altair_chart(chart1 + chart2 + chart3)
st.sidebar.subheader('RFECV')
if st.sidebar.checkbox('RFECV'):
    bins1 = [-1, 0.1, 20, np.inf]
    labels1 = ('Severe_PN', 'Moderate_PN', 'No_PN')
    data_final_num_cat['S_A_categories'] = pd.cut ( data_final_num_cat['S_A '], bins=bins, labels=labels )
    scale = StandardScaler ()
    data_final_num_cat1 = data_final_num_cat.drop ( 'S_A ', axis=1 )
    x_explo = data_final_num_cat1.drop ( 'S_A_categories', axis=1 )
    x_explo1 = scale.fit_transform ( x_explo )
    y_explo_cat = data_final_num_cat1['S_A_categories'].map ( {
        'Severe_PN': 0,
        'Moderate_PN': 1,
        'No_PN': 2
    } )
    estimator = RandomForestClassifier ( random_state=0 )
    selector = RFECV ( estimator, step=1, cv=StratifiedKFold ( 7 ) )
    selector = selector.fit ( x_explo, y_explo_cat )
    features_ranking = selector.ranking_
    features_names = x_explo.columns
    important_features = pd.DataFrame ( {'Features Names': features_names, 'Features Ranking': features_ranking} )
    important_features = important_features.sort_values ( by='Features Ranking', ascending=True )
    st.markdown( "<h3 style='text-align: center; color: black;'>RFECV</h3>", unsafe_allow_html=True )
    st.write(important_features)

st.sidebar.subheader('Traditional Models')
if st.sidebar.checkbox('Classification Models'):
    st.markdown( "<h3 style='text-align: center; color: black;'>Baseline Accuracy: Without Vs With RFECV </h3>", unsafe_allow_html=True )
    st.write ( baseline_models )

st.sidebar.subheader('Parameters Optimization')
if st.sidebar.checkbox('Parameter Optimization'):
    radio1 = st.sidebar.radio('Select Model',['SVC','Random Forest Classifier'])
    if radio1=='SVC':
        st.subheader('SVC Accuracy - Validation Dataset')
        def get_user_input():
            Cost=st.sidebar.slider('Select Cost',0.5,3.0,0.5)
            Kernel=st.sidebar.selectbox('Select Kernel',['linear','poly','rbf','sigmoid'])
            Gamma=st.sidebar.radio('Select Gamma',['scale','auto'])
            user_input ={
                'Cost':Cost,
                'Kernel':Kernel,
                'Gamma':Gamma
            }
            user_df = pd.DataFrame ( user_input, index=[0] )
            return user_df
        user_input_p = get_user_input ()
        svc = SVC ( C=user_input_p.iloc[0]['Cost'], kernel=user_input_p.iloc[0]['Kernel'],
                    gamma=user_input_p.iloc[0]['Gamma'] ).fit ( train_prepared, y_train_val )
        button1 = st.button('Press for Accuracy')
        if button1:
            prediction=svc.predict(valid_prepared)
            accuracy=accuracy_score(y_val,prediction)
            accuracy=round(accuracy,2)
            st.write(f"SVC Accuracy on Validation dataset is {accuracy *100}%")

        # plot overfitting
        button2 = st.button('Press to See Overfitting Plot')
        if button2:
            st.subheader('SVC: Training Vs Validation Plot')
            cost = [i for i in range ( 1, 10, 1 )]
            train_scores, val_scores = list (), list ()
            for i in cost:
                model = SVC ( C=i )
                # fit model on the training dataset
                model.fit ( train_prepared, y_train_val )
                # evaluate on the train dataset
                train_yhat = model.predict ( train_prepared )
                train_acc = accuracy_score ( y_train_val, train_yhat )
                train_scores.append ( train_acc )
                # evaluate on the valid dataset
                val_yhat = model.predict ( valid_prepared )
                val_acc = accuracy_score ( y_val, val_yhat )
                val_scores.append ( val_acc )
            fig1=plt.figure()
            plt.plot ( cost, train_scores, '-o', label='Train' )
            plt.plot ( cost, val_scores, '-o', label='Val' )
            plt.xlabel ( 'Cost' )
            plt.ylabel ( 'Accuracy' )
            plt.legend ()
            st.pyplot(fig1)

    if radio1=='Random Forest Classifier':
        st.subheader ( 'Random Forest Classifier Accuracy - Validation Dataset' )
        def get_user_input():
            N_Estimators = st.sidebar.slider ( 'Select Number of Estimators', 10, 50, 10 )
            Criterion= st.sidebar.radio ( 'Select Criterion', ['gini', 'entropy'] )
            Min_Samples_Split = st.sidebar.selectbox ( 'Select Number of Splits', [1.0,2,3,4,5,6] )
            user_input = {
                'N_Estimators': N_Estimators,
                'Criterion': Criterion,
                'Min_Samples_Split': Min_Samples_Split
            }
            user_df = pd.DataFrame ( user_input, index=[0] )
            return user_df


        user_input_p = get_user_input ()
        rfc1 = RandomForestClassifier( n_estimators=user_input_p.iloc[0]['N_Estimators'], criterion=user_input_p.iloc[0]['Criterion'],
                    min_samples_split=user_input_p.iloc[0]['Min_Samples_Split'] ).fit ( train_prepared, y_train_val )
        col1,col2,col3=st.columns([2.5,3.5,2.5])
        with col1:
            st.write("")
        with col2:
            button3 = st.button ( 'Press for Accuracy' )
            if button3:
                prediction1 = rfc1.predict ( valid_prepared )
                accuracy1 = accuracy_score ( y_val, prediction1 )
                accuracy1 = np.round(accuracy1,2)
                st.write ( f"Random Forest Accuracy on Validation dataset is {accuracy1*100}%" )
        with col3:
            st.write("")
        # plot overfitting
        col1,col2=st.columns(2)
        with col1:
            button4=st.button('Press to See Overfitting Plot')
            if button4:
                st.subheader ( 'Random Forest Classifier: Training Vs Validation Plot' )
                n_estimators = [i for i in range ( 1, 60, 10 )]
                train_scores, val_scores = list (), list ()
                for i in n_estimators:
                    model = RandomForestClassifier( n_estimators=i )
                    # fit model on the training dataset
                    model.fit ( train_prepared, y_train_val )
                    # evaluate on the train dataset
                    train_yhat = model.predict ( train_prepared )
                    train_acc = accuracy_score ( y_train_val, train_yhat )
                    train_scores.append ( train_acc )
                    # evaluate on the valid dataset
                    val_yhat = model.predict ( valid_prepared )
                    val_acc = accuracy_score ( y_val, val_yhat )
                    val_scores.append ( val_acc )
                fig2 = plt.figure ()
                plt.plot ( n_estimators, train_scores, '-o', label='Train' )
                plt.plot ( n_estimators, val_scores, '-o', label='Val' )
                plt.xlabel ( 'n_estimators' )
                plt.ylabel ( 'Accuracy' )
                plt.legend ()
                st.pyplot ( fig2 )
        with col2:
            button5=st.button('Press to see Highest Accuracy')
            if button5:
                st.subheader('DataFrame: sorted by Highest Accuracy')
                n_estimators = [10, 20, 30, 40, 50]
                criterion = ['gini', 'entropy']
                min_samples_split = [2, 3, 4, 5]
                Estimators = []
                Criter = []
                Min_Samples = []
                Accurate_valid = []
                Accurate_train = []
                for n in n_estimators:
                    for c in criterion:
                        for m in min_samples_split:
                            rfc2 = RandomForestClassifier ( n_estimators=n, criterion=c, min_samples_split=m ).fit (
                                train_prepared, y_train_val )
                            y_predict_train = rfc2.predict ( train_prepared )
                            y_predict_valid = rfc2.predict ( valid_prepared )
                            accur_valid = accuracy_score ( y_predict_valid, y_val )
                            accur_train = accuracy_score ( y_predict_train, y_train_val )
                            Accurate_valid.append ( accur_valid )
                            Accurate_train.append ( accur_train )
                            Estimators.append ( n )
                            Criter.append ( c )
                            Min_Samples.append ( m )

                random_forest_dataframe=pd.DataFrame({'Numb_of_Estimators':Estimators,'Criterion':Criter,'Min_Samples':Min_Samples,'Accuracy_valid':Accurate_valid,'Accuracy_train':Accurate_train})
                random_forest_dataframe=random_forest_dataframe.sort_values(by='Accuracy_valid',ascending=False)
                st.write(random_forest_dataframe)

st.sidebar.subheader('Deep Neural Network')
DNN = st.sidebar.checkbox('Deep Neural Network')
if DNN:
    st.markdown ( "<h3 style='text-align: center; color: black;'>Deep Neural Network with SKLearn</h3>",
                  unsafe_allow_html=True )

    hidden_layer_sizes = [2, 4, 6]
    activation = ['logistic', 'relu', 'tanh']
    solver = ['sgd', 'adam']
    accuracy = []
    parameters = []
    for h in hidden_layer_sizes:
        for a in activation:
            for s in solver:
                mlp = MLPClassifier ( hidden_layer_sizes=(h), activation=a, solver=s )
                mlp.fit ( train_prepared, y_train_val )
                y_predict = mlp.predict ( valid_prepared )
                acc_score = round ( accuracy_score ( y_val, y_predict ), 2 )
                parameters.append ( "Hidden_Layers:{}, Activation:{}, Solver:{}".format ( h, a, s ) )
                accuracy.append ( acc_score )
    mlp_dataframe = pd.DataFrame({'Parameters':parameters,'Accuracy':accuracy})
    mlp_dataframe =mlp_dataframe.sort_values(by='Accuracy',ascending=False)
    st.write(mlp_dataframe)

tensor=st.sidebar.checkbox('Tensorflow')
if tensor:
    st.markdown ( "<h3 style='text-align: center; color: black;'>Deep Neural Network with Tensorflow & Keras</h3>",
                  unsafe_allow_html=True )
    col1,col2=st.columns(2)
    with col1:
        image4=Image.open('tensor_model.jpg' )
        st.image(image4)
    with col2:
        image5=Image.open('tensor_epoch.jpg' )
        st.image(image5)

st.sidebar.subheader('ROC & AUC')
conf=st.sidebar.checkbox('Confusion Matrix')
roc_auc = st.sidebar.checkbox('ROC & AUC Comparison')

if conf:
    col1,col2,col3=st.columns([1,6,1])
    with col1:
        st.write("")
    with col2:
        st.markdown ( "<h3 style='text-align: center; color: black;'>Confusion Matrix</h3>",
                      unsafe_allow_html=True )
        rfc3= RandomForestClassifier(n_estimators=25,max_features=7,min_samples_split=2).fit(train_prepared,y_train_val)
        plot_confusion_matrix(rfc3, valid_prepared, y_val)
        st.set_option ( 'deprecation.showPyplotGlobalUse', False )
        st.pyplot()
    with col3:
        st.write("")
if roc_auc:
    col1,col2=st.columns(2)
    with col1:
        st.markdown ( "<h3 style='text-align: center; color: black;'>ROC-AUC: Validation Dataset</h3>",
                      unsafe_allow_html=True )
        y_val_bin = label_binarize ( y_val, classes=[0, 1, 2] )
        n_classes = y_val_bin.shape[1]
        classifier = OneVsRestClassifier (
            RandomForestClassifier ( n_estimators=25, max_features=7, min_samples_split=2 )
        )
        y_score = classifier.fit ( train_prepared, y_train_val ).predict_proba ( valid_prepared )
        # compute ROC curve and ROC area for each class
        fpr = dict ()
        tpr = dict ()
        roc_auc = dict ()
        for i in range ( n_classes ):
            fpr[i], tpr[i], _ = roc_curve ( y_val_bin[:, i], y_score[:, i] )
            roc_auc[i] = auc ( fpr[i], tpr[i] )

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve ( y_val_bin.ravel (), y_score.ravel () )
        roc_auc["micro"] = auc ( fpr["micro"], tpr["micro"] )
        # plot ROC curves for the multiclass problem
        # First aggregate all false positive rates
        all_fpr = np.unique ( np.concatenate ( [fpr[i] for i in range ( n_classes )] ) )

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like ( all_fpr )
        for i in range ( n_classes ):
            mean_tpr += np.interp ( all_fpr, fpr[i], tpr[i] )

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc ( fpr["macro"], tpr["macro"] )

        # Plot all ROC curves
        fig4=plt.figure ()
        plt.plot (
            fpr["micro"],
            tpr["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format ( roc_auc["micro"] ),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot (
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format ( roc_auc["macro"] ),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle ( ["aqua", "darkorange", "cornflowerblue"] )
        for i, color in zip ( range ( n_classes ), colors ):
            plt.plot (
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})".format ( i, roc_auc[i] ),
            )

        plt.plot ( [0, 1], [0, 1], "k--", lw=2 )
        plt.xlim ( [0.0, 1.0] )
        plt.ylim ( [0.0, 1.05] )
        plt.xlabel ( "False Positive Rate" )
        plt.ylabel ( "True Positive Rate" )
        # plt.title ( "Some extension of Receiver operating characteristic to multiclass" )
        plt.legend ( loc="lower right" )
        st.pyplot(fig4)
        # A macro-average will compute the metric independently for each class and then take the average hence treating all classes equally,
        # whereas a micro-average will aggregate the contributions of all classes to compute the average metric.

    with col2:
        st.markdown ( "<h3 style='text-align: center; color: black;'>ROC-AUC: Test Dataset</h3>",
                      unsafe_allow_html=True )
        classifier1 = OneVsRestClassifier (
            RandomForestClassifier ( n_estimators=25, max_features=7, min_samples_split=2 )
        )
        y_score1 = classifier1.fit ( X_train_prepared, y_train ).predict_proba ( X_test_prepared )
        # compute ROC curve and ROC area for each class
        fpr1 = dict ()
        tpr1 = dict ()
        roc_auc1 = dict ()
        for i in range ( n_classes1 ):
            fpr1[i], tpr1[i], _ = roc_curve ( y_test_bin[:, i], y_score1[:, i] )
            roc_auc1[i] = auc ( fpr1[i], tpr1[i] )

        # Compute micro-average ROC curve and ROC area
        fpr1["micro"], tpr1["micro"], _ = roc_curve ( y_test_bin.ravel (), y_score1.ravel () )
        roc_auc1["micro"] = auc ( fpr1["micro"], tpr1["micro"] )
        # plot ROC curves for the multiclass problem
        # First aggregate all false positive rates
        all_fpr1 = np.unique ( np.concatenate ( [fpr1[i] for i in range ( n_classes1 )] ) )

        # Then interpolate all ROC curves at this points
        mean_tpr1 = np.zeros_like ( all_fpr1 )
        for i in range ( n_classes1 ):
            mean_tpr1 += np.interp ( all_fpr1, fpr1[i], tpr1[i] )

        # Finally average it and compute AUC
        mean_tpr1 /= n_classes1

        fpr1["macro"] = all_fpr1
        tpr1["macro"] = mean_tpr1
        roc_auc1["macro"] = auc ( fpr1["macro"], tpr1["macro"] )

        # Plot all ROC curves
        fig5=plt.figure ()
        plt.plot (
            fpr1["micro"],
            tpr1["micro"],
            label="micro-average ROC curve (area = {0:0.2f})".format ( roc_auc1["micro"] ),
            color="deeppink",
            linestyle=":",
            linewidth=4,
        )

        plt.plot (
            fpr1["macro"],
            tpr1["macro"],
            label="macro-average ROC curve (area = {0:0.2f})".format ( roc_auc1["macro"] ),
            color="navy",
            linestyle=":",
            linewidth=4,
        )

        colors = cycle ( ["aqua", "darkorange", "cornflowerblue"] )
        for i, color in zip ( range ( n_classes ), colors ):
            plt.plot (
                fpr1[i],
                tpr1[i],
                color=color,
                lw=2,
                label="ROC curve of class {0} (area = {1:0.2f})".format ( i, roc_auc1[i] ),
            )

        plt.plot ( [0, 1], [0, 1], "k--", lw=2 )
        plt.xlim ( [0.0, 1.0] )
        plt.ylim ( [0.0, 1.05] )
        plt.xlabel ( "False Positive Rate" )
        plt.ylabel ( "True Positive Rate" )
        plt.legend ( loc="lower right" )
        st.pyplot(fig5)
    st.markdown ( "<h3 style='text-align: center; color: black;'>AUC achieved by (Shin et al.2021) is 82.5%</h3>",
                      unsafe_allow_html=True )

st.sidebar.subheader('Imbalanced Data')
comp=st.sidebar.checkbox('Balanced Vs Imbalanced')
acc = st.sidebar.checkbox('Accuracy Comparison')
opt = st.sidebar.checkbox('Model Optimization')
test = st.sidebar.checkbox('Final Accuracy')
if comp:
    col1,col2=st.columns(2)
    with col1:
        st.subheader ( "Imbalanced Plot"
                      )
        for k, v in counter.items ():
            per = v / len ( y_smote ) * 100
            st.write( 'Class =%d, n=%d (%.3f%%)' % (k, v, per) )

        # plot the distribution
        fig6=plt.figure()
        plt.bar ( counter.keys (), counter.values () )
        plt.title ( 'Class Breakdown before SMOTE' )
        st.pyplot(fig6)

    with col2:
        st.subheader( "Balanced Plot"
                      )
        for k, v in counter1.items ():
            per = v / len ( y_smote1 ) * 100
            st.write ( 'Class =%d, n=%d (%.3f%%)' % (k, v, per) )

        # plot the distribution
        fig7=plt.figure()
        plt.bar ( counter1.keys (), counter1.values () )
        plt.title ( 'Class Break Down after SMOTE' )
        st.pyplot(fig7)

if acc:
    st.markdown ( "<h3 style='text-align: center; color: black;'>Accuracy Comparison</h3>",
                      unsafe_allow_html=True )
    classifiers_smote = [LogisticRegression (), RandomForestClassifier (), DecisionTreeClassifier (), SVC (),
                             KNeighborsClassifier ()]
    Accuracy_smote = []
    estimator_smote = []
    for c in classifiers_smote:
        m = c.fit ( X_train1_smote, y_train1_smote )
        pred_smote = m.predict ( X_valid1_smote )
        acc_smote = accuracy_score ( y_valid1_smote, pred_smote )
        acc_smote = round ( acc_smote, 2 )
        Accuracy_smote.append ( acc_smote )
        estimator_smote.append ( c )

    data_frame_smote = pd.DataFrame ( {'Model': estimator_smote, 'Accuracy': Accuracy_smote} )
    data_frame_smote = data_frame_smote.sort_values ( by='Accuracy', ascending=False )
    data_frame_smote = data_frame_smote.reset_index ()
    data_b = dtc_cvs1.merge ( dtc_cvs, on='index', how='outer' )
    data_b = data_b.drop ( 'Estimator_y', axis=1 )
    data_b = data_b.rename (
            {'Estimator_x': 'Model', 'Accuracy_x': 'With RFECV', 'Accuracy_y': 'Imbalanced Dataset'},
            axis=1 )
    # join all dataframes for comparison
    data_f = data_b.merge ( data_frame_smote, on='index', how='outer' )
    data_f = data_f.drop ( ['Model_y', 'index'], axis=1 )
    data_f = data_f.rename ( {'Model_x': 'Model', 'Accuracy': 'Balanced Dataset'}, axis=1 )
    data_f=data_f.drop('With RFECV',axis=1)
    data_f = data_f.sort_values ( by='Balanced Dataset', ascending=False )
    st.write(data_f)
    col1,col2=st.columns(2)
    with col1:
        st.markdown ( "<h3 style='text-align: center; color: black;'>Imbalanced Dataset</h3>",
                      unsafe_allow_html=True )
        data_f1=data_f[['Model','Imbalanced Dataset']]
        fig8=px.bar(data_f1,x='Model',y='Imbalanced Dataset',color='Model',width=400,height=400)
        px.bar()
        st.write(fig8)
    with col2:
        st.markdown ( "<h3 style='text-align: center; color: black;'>Balanced Dataset</h3>",
                      unsafe_allow_html=True )
        data_f2=data_f[['Model','Balanced Dataset']]
        fig9=px.bar(data_f2,x='Model',y='Balanced Dataset',color='Model',width=400,height=400)
        st.write(fig9)

if opt:
    st.markdown ( "<h3 style='text-align: center; color: black;'>Random Forest Classifier Optimization</h3>",
                      unsafe_allow_html=True )
    n_estimators2 = [10, 20, 30, 40, 50]
    criterion2 = ['gini', 'entropy']
    min_samples_split2 = [2, 3, 4, 5]
    Estimators2 = []
    Criter2 = []
    Min_Samples2 = []
    Accurate_valid2 = []
    for n in n_estimators2:
        for c in criterion2:
            for m in min_samples_split2:
                rfc2 = RandomForestClassifier ( n_estimators=n, criterion=c, min_samples_split=m ).fit ( X_train1_smote,
                                                                                                        y_train1_smote )
                y_predict_valid = rfc2.predict ( X_valid1_smote )
                accur_valid = accuracy_score ( y_predict_valid, y_valid1_smote )
                Accurate_valid2.append ( accur_valid )
                Estimators2.append ( n )
                Criter2.append ( c )
                Min_Samples2.append ( m )

    rfc_smote_opt = pd.DataFrame ( {'Numb_Estimators': Estimators2, 'Min_Samples_split': Min_Samples2, 'Criteria': Criter2,
                                    'Accuracy_valid': Accurate_valid2} )
    rfc_smote_opt = rfc_smote_opt.sort_values ( by='Accuracy_valid', ascending=False )
    rfc_smote_opt = rfc_smote_opt.reset_index ()
    st.write(rfc_smote_opt)
if test:
    button6=st.button('Press for Test Accuracy')
    if button6:
        st.markdown ( "<h3 style='text-align: center; color: black;'>Test Accuracy on Balanced Dataset</h3>",
                      unsafe_allow_html=True )
        rfc_test = RandomForestClassifier ( n_estimators=40, criterion='gini', min_samples_split=2 )
        rfc_test.fit ( X_train_smote, y_train_smote )
        predict_rfc = rfc_test.predict ( X_test_smote )
        accuracy_test = accuracy_score ( y_test_smote, predict_rfc )
        accuracy_test = round ( accuracy_test, 2 )
        # my_bar = st.progress ( 0 )
        # for percent_complete in range ( 30 ):
        #     time.sleep ( 0.1 )
        #     my_bar.progress ( percent_complete + 1 )
        # st.subheader(f"The Test Accuracy is {accuracy_test*100}%")

        with st.spinner ( 'Calculating Accuracy...' ):
            time.sleep ( 5 )
        st.subheader(f"Our Test Accuracy is {accuracy_test*100}% compared to 76% achieved by (Kazemi et al, 2016)")

st.sidebar.subheader('Predictive Model')
pred=st.sidebar.checkbox('Patient Predictions')
if pred:
    col1,col2,col3=st.columns([1,6,1])
    with col1:
        st.write("")
    with col2:
        st.markdown ( "<h3 style='text-align: center; color: black;'>Peripheral Neuropathy</h3>",
                      unsafe_allow_html=True )
        image6=Image.open('C:/Users/j.elachkar/Desktop/peripheral_pic.jpg')
        st.image(image6)


        def get_user_input():
            Age = st.slider("Age",10,105,25)
            FBS = st.slider("Glucose",32,400,150)
            VitB12 = st.slider("VitB12",10,150,25)
            BMI=st.slider("BMI",0.00,250.00,20.00)
            HDL=st.slider("HDL",20.0,220.0,80.0)
            LDL=st.slider('LDL',10.0,209.0,50.0)

            features ={
                'Age':Age,
                'FBS':FBS,
                'VitB12':VitB12,
                'BMI':BMI,
                'HDL':HDL,
                'LDL':LDL
            }
            user_input=pd.DataFrame(features,index=[0])
            return user_input
        user = get_user_input()
        rfc_patient=RandomForestClassifier()
        data_patient=data[['Age','FBS', 'VitB12','BMI','HDL','LDL','S_A ','S_A_categories']]
        X_pt = data_patient.drop ( ['S_A_categories', 'S_A '], axis=1 )
        y_cat_pt = data_patient['S_A_categories']
        y_num_pt = data_patient['S_A ']
        y_cat_pt = data_patient['S_A_categories'].map ( {
            'Severe_PN': 0,
            'Moderate_PN': 1,
            'No_PN': 2
        } )
        X_train_pt, X_test_pt, y_train_pt, y_test_pt = train_test_split ( X_pt, y_cat_pt, test_size=0.4, stratify=y_cat_pt, random_state=0 )
        X_train_pt_na = X_train_pt.fillna ( X_train_pt.mean () )
        rfc_patient.fit(X_train_pt_na,y_train_pt)
        X_test_pt_na = X_test_pt.fillna ( X_test_pt.mean () )
        predict = rfc_patient.predict(user)
        button8=st.button('Press to see result')
        if button8:
            my_bar = st.progress ( 0 )
            for percent_complete in range ( 30 ):
                time.sleep ( 0.1 )
                my_bar.progress ( percent_complete + 1 )
            if predict==0:
                st.write('You are most likely diagnosed with Severe Peripheral Neuropathy')
            elif predict ==1:
                st.write("You are most likely diagnosed with Moderate Periphearl Neuropathy")
            else:
                st.write("You don't seem to have Peripheral Neuropathy")


    with col3:
        st.write("")
























































