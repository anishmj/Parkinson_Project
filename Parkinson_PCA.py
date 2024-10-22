import numpy as np
import pickle
import streamlit as st

# Load the scaler and model
scaler = pickle.load(open('scaler.sav', 'rb'))  # Adjust the path if necessary
loaded_model = pickle.load(open('parkinsons_hybrid_model.sav', 'rb'))  # Adjust the path if necessary

def Parkinson(input_data):
    # changing input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # reshape the numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # standardize the data
    std_data = scaler.transform(input_data_reshaped)
    
    # make prediction
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return "The Person does not have Parkinson's Disease"
    else:
        return "The Person has Parkinson's Disease"
    
def main():
    st.title("Parkinson Prediction Web App")
    col1, col2, col3, col4, col5 = st.columns(5)

    # Create input fields for the required features
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)', '0')
    
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)', '0')
    
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)', '0')
    
    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)', '0')
    
    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)', '0')
    
    with col1:
        RAP = st.text_input('MDVP:RAP', '0')
    
    with col2:
        PPQ = st.text_input('MDVP:PPQ', '0')
    
    with col3:
        DDP = st.text_input('Jitter:DDP', '0')
    
    with col4:
        Shimmer = st.text_input('MDVP:Shimmer', '0')
    
    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)', '0')
    
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3', '0')
    
    with col2:
        APQ5 = st.text_input('Shimmer:APQ5', '0')
    
    with col3:
        APQ = st.text_input('MDVP:APQ', '0')
    
    with col4:
        DDA = st.text_input('Shimmer:DDA', '0')
    
    with col5:
        NHR = st.text_input('NHR', '0')
    
    with col1:
        HNR = st.text_input('HNR', '0')
    
    with col2:
        RPDE = st.text_input('RPDE', '0')
    
    with col3:
        DFA = st.text_input('DFA', '0')
    
    with col4:
        spread1 = st.text_input('spread1', '0')
    
    with col5:
        spread2 = st.text_input('spread2', '0')
    
    with col1:
        D2 = st.text_input('D2', '0')
    
    with col2:
        PPE = st.text_input('PPE', '0')
    
    # Code for Prediction
    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        # Convert inputs to float and pass to the model
        parkinsons_diagnosis = Parkinson([fo, fhi, flo, Jitter_percent, Jitter_Abs,
                                          RAP, PPQ, DDP, Shimmer, Shimmer_dB, 
                                          APQ3, APQ5, APQ, DDA, NHR, HNR, 
                                          RPDE, DFA, spread1, spread2, D2, PPE])
    
    # Display the prediction result
    st.success(parkinsons_diagnosis)

if __name__ == '__main__':
    main()
