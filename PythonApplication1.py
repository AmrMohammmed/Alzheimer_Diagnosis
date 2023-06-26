import  streamlit as st
import  pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn as sk
import pandas as pd
import os
import warnings
from PIL import Image
from streamlit_option_menu import option_menu
import webbrowser
import sys
from streamlit.components.v1 import html

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
warnings.simplefilter(action='ignore', category=FutureWarning)

image = Image.open('a2.png')

st.set_page_config(
    page_title="Alzheimer's Detection",
    page_icon=image,
    initial_sidebar_state='collapsed',
)

st.image(image, caption='',width=700)

st.markdown(
    """
    <style>
         .main {
   
         }
    </style>
    """,
    unsafe_allow_html=True
)


# Define your javascript
my_js1 = """
window.open('https://adni.loni.usc.edu/', '_blank');
"""
my_js2 = """
window.open('https://medlineplus.gov/genetics/condition/alzheimer-disease/', '_blank');
"""
my_js3 = """
window.open('https://github.com/amrMohammmed/AD-Project.git', '_blank');
"""

# Wrapt the javascript as html code
my_html1 = f"<script>{my_js1}</script>"
my_html2 = f"<script>{my_js2}</script>"
my_html3 = f"<script>{my_js3}</script>"

# Execute your app


with st.sidebar:
    st.title("AD Project")
    selected = option_menu("Main", ["Home","ADNI",'Alzheimer\'s Disease','Github Link'], 
     menu_icon="cast", default_index=0)
    
    if(selected =='ADNI'):
        html(my_html1)

    elif(selected =='Alzheimer\'s Disease'):
        html(my_html2)

    elif(selected =='Github Link'):
       html(my_html3)

    st.info('Before closing this make sure \'Home\' is selected')


header=st.container()
datset=st.container()
features=st.container()
model_training=st.container()


def mode_act(dat,tnum):
    dat.isnull().sum().sum()
    dat = dat.dropna()
    # """### Remove ID's and other features"""
    X = dat
    Y = dat['DX.bl']
    del dat

    # Remove unnecessary columns (features), remove first 9 columns and 'Dx codes for submission'
    remove_columns = list(X.columns)[0:9]

    remove_columns.append('Dx Codes for Submission')
    # print('Removing columns:', remove_columns)
    X = X.drop(remove_columns, axis=1)
    features = list(X.columns)

    Y = Y.replace('LMCI', 'CN')
    Y = Y.replace('CN', 'NOT AD')

    # """## Exploratory Data Analysis (EDA)"""

    numerical_vars = ['AGE', 'MMSE', 'PTEDUCAT']
    cat_vars = list(set(features) - set(numerical_vars))

    # for each categorical var, convert to 1-hot encoding
    for var in cat_vars:
        #  print('Converting', var, 'to 1-hot encoding')

        # get 1-hot and replace original column with the >= 2 categories as columns
        one_hot_df = pd.get_dummies(X[var])
        X = pd.concat([X, one_hot_df], axis=1)
        X = X.drop(var, axis=1)

    accuracy_tests = []
    for i in range(tnum):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2)

        num_test = X_test.shape[0]

        log_clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000000, multi_class='multinomial')

        log_clf.fit(X_train, y_train)

        log_clf_preds = log_clf.predict(X_test)
        log_clf_accuracy = (log_clf_preds == y_test)
        test_accurecy = format(np.sum(log_clf_accuracy) / num_test, '.2%')
        accuracy_tests.append(test_accurecy)

    display_col.subheader("Minimum Test Accurecy")
    display_col.write(min(accuracy_tests))
    display_col.subheader("Maximum Test Accurecy")
    display_col.write(max(accuracy_tests))

@st.cache
def get_data(filename):
    patients_data=pd.read_csv(filename)
    return patients_data

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(index=False)


def show(file):
 if st.button("Display Result"):
  if file is not None:
    st.dataframe(file)



with header :
    st.title('Welcome to our awesome project !')
    st.text('In this project we are looking forward to providing a simple tool\n'
            'for predicting the presence of Alzheimer’s disease.')

with datset:
    st.header('ADNI data set')
    st.text('We found this data set at adni.loni.usc.edu,\n'
            'ADNI refers to Alzheimer’s Disease Neuroimaging Initiative.')

    train_data=get_data('train_data.csv')
    st.write(train_data)
    display_col,selection_col = st.columns(2)

    expander = st.expander("Show Data Distribution")
   # if st.button("Show Data Distribution"):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(train_data['AGE'], bins=20)
    #plt.title("Age Distribution")

    expander.subheader("Age Distribution")
    expander.pyplot(fig)    
    
    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(train_data['MMSE'], bins=20)

    expander.subheader("MMSE Distribution")
    expander.pyplot(fig)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.hist(train_data['PTEDUCAT'], bins=20)
    expander.subheader("Education Distribution")
    expander.pyplot(fig)

    expander.subheader("Diagnosis Distribution")
    dx_bl=pd.DataFrame(train_data['DX.bl'].value_counts())
    expander.bar_chart(dx_bl)



with features:
    st.header('The features we created')
    st.text('Let\'s take a look into the features we generated.')
    st.markdown('* Detect AD Presence:This is the main feature of or project that predicts the diagnosis of your patients')
    st.markdown('* Test Model Accuracy:This feature was made to enable the user to run tests on the model to realise the expected range of model accuracy')

with model_training:
    st.header('Time to train the model !')
    st.text('In this section you can select the hyperparameters !')
    selection_col, display_col = st.columns(2)
    dat = train_data

    if(selected =='Home'):
        selection_col.subheader('Model Accuracy')
        tnum = selection_col.slider('Choose the number of test to run on the model', min_value=1, max_value=10, value=5, step=1)
   
        if  selection_col.button("Run Tests"):
         mode_act(train_data,tnum)

        st.subheader('Time to put the model in actual use!')
      
        try:
            uploaded_file = st.file_uploader("Choose a file")
            if uploaded_file is not None:
                dataframe = pd.read_csv(uploaded_file)
                #st.write(dataframe)
                testp= dataframe
                #"""### Remove NA"""
                dat.isnull().sum().sum()
                dat = dat.dropna()

                testp.isnull().sum().sum()
                testp = testp.dropna()

                tesClone = testp
                # Peek at data

                #"""### Remove ID's and other features"""
                X = dat
                Y = dat['DX.bl']
                del dat

                # Remove unnecessary columns (features), remove first 9 columns and 'Dx codes for submission'
                remove_columns = list(X.columns)[0:9]

                remove_columns.append('Dx Codes for Submission')
                # print('Removing columns:', remove_columns)
                X = X.drop(remove_columns, axis=1)
                features = list(X.columns)

                allTest_colums = list(testp.columns)
                remove_From_testp = list(set(allTest_colums) - set(features))
                testp = testp.drop(remove_From_testp, axis=1)

                Y = Y.replace('LMCI', 'CN')
                Y = Y.replace('CN', 'NOT AD')

                #"""## Exploratory Data Analysis (EDA)"""

                numerical_vars = ['AGE', 'MMSE', 'PTEDUCAT']
                cat_vars = list(set(features) - set(numerical_vars))

                # for each categorical var, convert to 1-hot encoding
                for var in cat_vars:
                    #  print('Converting', var, 'to 1-hot encoding')

                    # get 1-hot and replace original column with the >= 2 categories as columns
                    one_hot_df = pd.get_dummies(X[var])
                    X = pd.concat([X, one_hot_df], axis=1)
                    X = X.drop(var, axis=1)

                    one_hot_df2 = pd.get_dummies(testp[var])
                    testp = pd.concat([testp, one_hot_df2], axis=1)
                    testp = testp.drop(var, axis=1)

                more_to_drop = list(set(X.columns) - set(testp.columns))
                # print(more_to_drop)
                X = X.drop(more_to_drop, axis=1)


                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=None)

                # num_test = X_test.shape[0]

                """## Predicted Diagnosis"""

                log_clf = LogisticRegression(solver='lbfgs', penalty='l2', max_iter=1000000, multi_class='multinomial')
                # print('Validation Accuracy = ', format(cross_val_score(log_clf, X_train, y_train, cv=5).mean(), '.2%'))

                log_clf.fit(X_train, y_train)

                a = log_clf_preds = log_clf.predict(testp)

                a = pd.DataFrame(a)
                a.columns = ["PDX"]

                tesClone = pd.concat([tesClone, a], axis=1)
       
                # tesClone.to_csv('sample.csv',index=False)
           
               
                show(tesClone)

                csv = convert_df(tesClone)
                #csv.drop(columns=csv.columns[0],axis=1, inplace=True)

                st.download_button(
                        label="Download result as CSV",
                        data=csv,
                        file_name='large_df.csv',
                        mime='text/csv',
                        
                    )
        
                st.success('Model worked successfully') 
             
        except:
            st.error('''Oops ! error occured\n
            Make sure .csv file is selected ,and your data is valid for this model\n
            \"Consider ADNI dataset above is a sample of valid data\"''')
            

      

    else:
         st.error('''Make sure \'Home\' is selected !\n
         Check the menue of the side bar''')
