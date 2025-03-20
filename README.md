This task was our first experience with machine learning.This application aims to develop and evaluate an EEG-based 
machine learning model to classify eye states, providing a foundation for real-world applications. -When choosing our eeg data, we searched 
https://www.kaggle.com/ for eeg eye state data. We then found 
various studies done that include eye state (opened vs closed). After 
looking at the accuracy of each notebook inside the datasets. We 
settled on this one: 
https://www.kaggle.com/datasets/ppb00x/eeg-neuroheadset . 
Since it had the most appropriate accuracy in each notebook. 
2. Signal Preprocessing: -Inside the dataset we found 14 channels and 1 column for eye 
state.  
The columns labeled AF3, F7, F3, 
FC5, T7, P, O1, O2, P8, T8, FC6, 
F4, F8, and AF4 correspond to  
different electrodes placed on the  
scalp according to the 10-20 EEG 
system: 
● AF3, AF4: Anterior frontal electrodes 
● F7, F3, F4, F8: Frontal lobe electrodes 
● FC5, FC6: Frontal-central electrodes 
● T7, T8: Temporal electrodes 
● P: Parietal electrode 
● O1, O2: Occipital electrodes 
● P8: Another parietal-related electrode -For our application, we found the most appropriate channels we 
chose were: 
● Frontal Electrodes: AF3 and AF4 are closer to the prefrontal 
cortex, which is active during eye-related tasks. 
● Occipital Electrodes (O1, O2): Responsible for visual 
processing. 
● Temporal Electrodes (T7, T8): Can capture eye-related artifacts 
due to their proximity to the eyes. 
3. Feature Extraction: 
1. Mean (df_cleaned.mean()) 
Measures the average EEG signal amplitude over time. 
When eyes are closed, alpha waves (8-13 Hz) increase, leading to a 
higher mean in EEG signals. 
When eyes are open, more Beta (13-30 Hz) and Gamma (30-50 Hz) 
waves dominate, leading to a lower mean in some channels. 
2. Standard Deviation (df_cleaned.std()) 
Measures how much the EEG signal varies from its mean. 
Higher standard deviation indicates a more active brain state (e.g., 
eyes open). Lower standard deviation occurs in a relaxed state (e.g., 
eyes closed, more alpha waves). 
3. Skewness (df_cleaned.skew()) 
Measures the asymmetry of the EEG signal distribution. Some EEG 
channels shift their symmetry when transitioning between eyes-open 
and eyes-closed states. Certain waveforms become more dominant, 
altering the signal’s skewness. 
4. Kurtosis (df_cleaned.kurtosis()) 
Measures how sharp or flat the EEG signal distribution is. A higher 
kurtosis suggests EEG signals contain more spikes and peaks, often 
seen in high-frequency brain activity (eyes open). A lower kurtosis 
indicates a more uniform signal (eyes closed, dominated by slow 
alpha waves). 
5. Root Mean Square (RMS) (np.sqrt((df_cleaned ** 2).mean())) 
Measures the overall energy of the EEG signal. Higher RMS 
suggests more overall brain activity (eyes open). Lower RMS 
suggests a more relaxed state (eyes closed).
-The machine learning model we chose was Support Vector 
Machine SVM (SVC). After doing our research, we found that it is 
effective for binary classification like eye-open vs. eye-closed and 
works well with high-dimensional data like EEG. We gave Random 
Forest a try but found lower and unstable accuracy, that may be due 
to random forest being more accurate on larger datasets.
-The model divides the given data into 80%-20% way. 80% for 
training the model, then it uses the last 20% to test the model and 
accuracy and work the actuator. -Our model had accuracy of 60 percent at first. We then found out it 
could be related to features extracted, filtering and built in functions 
we can use. We then tried unit testing and found out that the low 
accuracy was due to the model misreading a lot of the eyes opened 
states as eyes closed. Which led us to using SMOTE technique. A 
data balancing technique used to handle imbalanced datasets, 
especially in classification problems where one class has 
significantly fewer samples than the other.The accuracy then 
reached 65% and we settled on that number. -The output is classified into two classes. 1-eyes opened 2-eyes 
closed and it shows precision, recall, f1-score, support then accuracy 
(weighted moto and avg).
Real time response to our predicted data using an actuator 
We decided on implementing the actuator application by playing a 
sound on the computer upon receiving the data: 
As shown above in this part of the code we have used the pygame 
library to call the mixer function that is supposed to load the sound 
files we want to use.

