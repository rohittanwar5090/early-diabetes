import numpy as np
import pandas as pd
import streamlit as st


actual_patient_data = pd.read_csv('diabetes_data_upload.csv')

converted_data=pd.get_dummies(actual_patient_data, prefix=['Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
       'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
       'Itching', 'Irritability', 'delayed healing', 'partial paresis',
       'muscle stiffness', 'Alopecia', 'Obesity', 'class'], drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(converted_data.drop('class_Positive', axis=1),converted_data['class_Positive'], test_size=0.3, random_state=0)
   
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.ensemble import RandomForestClassifier
RF_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_classifier.fit(X_train, y_train)

def predict_note_authentication(age,gender,polyuria,polydipsia,weight,weakness,polyphagia,genital_thrush,visual_blurring,itching,irritability, delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity):

    prediction=RF_classifier.predict(sc.transform(np.array([[int(age),int(gender),int(polyuria),int(polydipsia),int(weight),int(weakness),int(polyphagia),int(genital_thrush),int(visual_blurring),int(itching),int(irritability), int(delayed_healing),int(partial_paresis),int(muscle_stiffness),int(alopecia),int(obesity)]])))
    print(prediction)
    return prediction

def main():
   

    html_temp = """
    <div style="background-color:#eee;padding:10px;border:1px solid red">
    <h2 style="color:black;text-align:center;">Early Stage Diabetes Predictor </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    html_subheader = """
    <div>
    <br>
    <h5 style="color:black;text-align:center;">Choose one of the alternatives below based on your knowledge and click 'Predict' to find out how is diabetes status</h2>
    </div>
    """
    st.markdown(html_subheader,unsafe_allow_html=True)

    # age
    age = st.number_input("What is your Age?")
    if age>0:
         st.warning('valid')
    else :
         st.error('invalid age')

    # gender
    gender = st.selectbox("What is your Gender?",('Male', 'Female'))
    st.write('You selected:', gender)
    # print(gender)

    if gender == 'Male':
        gender = 1
    else:
        gender = 0
    
    polyuria = st.selectbox("Do you have Polyuria?",("Yes","No"))
    if polyuria == 'Yes':
        polyuria = 1
    else:
        polyuria = 0
    
  
       
    polydipsia = st.selectbox("Do you have Polydipsia?",("Yes","No"))
    if polydipsia == 'Yes':
        polydipsia = 1
    else:
        polydipsia = 0
  

    weight = st.selectbox("Recently do you observe sudden weight loss?",("Yes","No"))
    if weight == 'Yes':
        weight = 1
    else:
        weight = 0

    weakness = st.selectbox("Do you feel any Weekness?",("Yes","No"))
    if weakness == 'Yes':
        weakness = 1
    else:
        weakness = 0

    polyphagia = st.selectbox("Do you have Polyphagia?",("Yes","No"))
    if polyphagia == 'Yes':
        polyphagia = 1
    else:
        polyphagia = 0
   
        
    genital_thrush = st.selectbox("Do you have Genital thrush?",("Yes","No"))
    if genital_thrush == 'Yes':
        genital_thrush = 1
    else:
        genital_thrush = 0


    visual_blurring = st.selectbox("Do you have Visual blurring?",("Yes","No"))
    if visual_blurring == 'Yes':
        visual_blurring = 1
    else:
        visual_blurring = 0
  

    itching = st.selectbox("Do you have Itching?",("Yes","No"))
    if itching == 'Yes':
        itching = 1
    else:
        itching = 0
  

    irritability = st.selectbox("Do you have Irritability?",("Yes","No"))
    if irritability == 'Yes':
        irritability = 1
    else:
        irritability = 0
   

    delayed_healing = st.selectbox("Do you have Delayed healing?",("Yes","No"))
    if delayed_healing == 'Yes':
        delayed_healing = 1
    else:
        delayed_healing = 0

    partial_paresis = st.selectbox("Do you have Partial paresis?",("Yes","No"))
    if partial_paresis == 'Yes':
        partial_paresis = 1
    else:
        partial_paresis = 0
   

    muscle_stiffness = st.selectbox("Do you have Muscle stiffness?",("Yes","No"))
    if muscle_stiffness == 'Yes':
        muscle_stiffness = 1
    else:
        muscle_stiffness = 0

    alopecia = st.selectbox("Do you have Alopecia?",("Yes","No"))
    if alopecia == 'Yes':
        alopecia = 1
    else:
        alopecia = 0
   

    obesity = st.selectbox("Do you have Obesity?",("Yes","No"))
    if obesity == 'Yes':
        obesity = 1
    else:
        obesity = 0

    result=""
    if st.button("Predict"):
        result=predict_note_authentication(age,gender,polyuria,polydipsia,weight,weakness,polyphagia,genital_thrush,visual_blurring,itching,irritability, delayed_healing,partial_paresis,muscle_stiffness,alopecia,obesity)
        if result ==1:
            st.warning('You might have Diabeties. Please consult with a Doctor.')
            html_positive = """
                <div style='display:flex;justify-content:'>
                    <img style='width:200px;margin:0 auto' src='https://i0.wp.com/dedipic.com/wp-content/uploads/2019/10/YouAreSick.gif?resize=498%2C498&ssl=1'/>
                </div>
                """
         
            st.markdown(html_positive,unsafe_allow_html=True)
        else:
            st.success("Hurray! You don't have Diabeties.")
            html_negative = """
                <div style='display:flex;justify-content:'>
                    <img style='width:200px;margin:0 auto' src='https://webstockreview.net/images/clipart-smile-smile-gif-2.gif'/>
                </div>
                """
            st.markdown(html_negative,unsafe_allow_html=True)

    # github = '[GitHub](https://github.com/soumyabrataroy)'
    # st.markdown(github, unsafe_allow_html=True)

if __name__=='__main__':
    main()


html_temp1 = """
    <div style="background-color:white;padding:10px">
    
    </div>
    """
st.markdown(html_temp1,unsafe_allow_html=True)

# #sidebars
# st.sidebar.header("Diabeties Predictor V2")
# st.sidebar.text("Developed by Soumyabrata Roy")
# st.sidebar.text("This is just a predictor based on ML ")
# st.sidebar.text("model. Before taking any decisions, ")
# st.sidebar.text("please consult with your Doctor.")
