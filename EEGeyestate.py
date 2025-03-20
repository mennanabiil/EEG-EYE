import joblib
import numpy as np
import pandas as pd
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, detrend, iirnotch, welch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE 
import pygame
import tkinter as tk
import threading

# **Load EEG Dataset**
file_path = r"C:\Users\asmaa\Uni\Junior 2\Monitors\eeg-headset.csv"
df = pd.read_csv(file_path)

# **Define EEG Channels and Target Column**
feature_columns = df.columns[:-1]  # # All EEG channels except target
target_column = "eye_state"

# **Extract EEG Features and Labels**
X = df[feature_columns]
y = df[target_column]

# **Step 1: Bandpass Filtering (1-50 Hz)**
def bandpass_filter(data, lowcut=1, highcut=50, fs=128, order=4):
    nyq = 0.5 * fs  # Nyquist Frequency
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

df_filtered = X.copy() #to insure the data stays unchanged
for col in feature_columns:
    df_filtered[col] = bandpass_filter(X[col]) #applies the eeg signal to all columns in the eeg dataset

# **Step 2: Baseline Correction (Detrending)**
df_corrected = df_filtered.copy()
for col in feature_columns:
    df_corrected[col] = detrend(df_filtered[col])  # Removes slow drift and slow fluctuations due to skin potentials and movements, etc

# **Step 3: Notch Filtering (50 Hz Noise Removal)**
def notch_filter(data, freq=50, fs=128, quality_factor=30):
    b, a = iirnotch(freq / (fs / 2), quality_factor)
    return filtfilt(b, a, data)  #filtfilt() applies the filter twice (once forward, once backward)

df_cleaned = df_corrected.copy()
for col in feature_columns:
    df_cleaned[col] = notch_filter(df_corrected[col])

# **Step 4: Plot Before & After Processing (Only AF3)**
plt.figure(figsize=(12, 5))
plt.plot(X["AF3"][:500], label="Raw Signal", alpha=0.5)
plt.plot(df_cleaned["AF3"][:500], label="Processed Signal", linewidth=2)
plt.legend()
plt.title("EEG Signal Processing (AF3)")
plt.xlabel("Samples")
plt.show()

# **Step 5: Feature Extraction**
mean_features = df_cleaned.mean()
std_dev = df_cleaned.std()
skewness = df_cleaned.skew()
kurtosis = df_cleaned.kurtosis()
rms = np.sqrt((df_cleaned ** 2).mean())

print("\nMean EEG Values After Processing:\n", mean_features)
print("\nStandard Deviation of EEG Values:\n", std_dev)
print("\nSkewness of EEG Values:\n", skewness)
print("\nKurtosis of EEG Values:\n", kurtosis)
print("\nRoot Mean Square (RMS) of EEG Values:\n", rms)

# **Step 6: Compute Power Spectral Density (PSD)**
def power_dens(data, fs=128, nperseg=128):       #how many samples per segment are used when computing 
    freqs, psd = welch(data, fs=fs, nperseg=nperseg)
    return freqs, psd

psd_data = {}
for col in feature_columns:
    freqs, psd = power_dens(df_cleaned[col])
    psd_data[col] = (freqs, psd)

# **Step 7: Compute Band Power**
bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 50)
}

def band_power(psd, freqs, band):
    low, high = bands[band]
    idx = np.logical_and(freqs >= low, freqs <= high)
    return np.trapz(psd[idx], freqs[idx])  # Integrate PSD over the band

band_power_df = pd.DataFrame(index=feature_columns, columns=bands.keys())

for col in feature_columns:
    freqs, psd = psd_data[col]
    for band in bands:
        band_power_df.at[col, band] = band_power(psd, freqs, band)

# **Compute Band Power Ratios**
band_power_df["Alpha/Beta"] = band_power_df["Alpha"] / band_power_df["Beta"]
band_power_df["Theta/Alpha"] = band_power_df["Theta"] / band_power_df["Alpha"]

print("\nBand Power Values:\n", band_power_df)
print("\nBand Power Ratios:\n", band_power_df[["Alpha/Beta", "Theta/Alpha"]])

# **Step 8: Standardize Features**
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)   #standardizes the dataset by removing the mean and scaling to unit variance.

smote = SMOTE(random_state=42)    #more eye-open samples than eye-closed samples, the model might learn to favor the majority class SMOTE improves model performance by ensuring the classifier learns features from both classes more equally.
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train_res, X_test_res, y_train_res, y_test_res = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled   #Splits the balanced dataset into training (80%) and testing (20%) sets.
)

# **Step 10: Train an SVC Model**
svc_model = SVC(kernel="rbf", C=1.0, gamma="scale")
svc_model.fit(X_train_res, y_train_res)


y_pred = svc_model.predict(X_test_res)
accuracy = accuracy_score(y_test_res, y_pred)
print(f"\nModel Accuracy on Test Set: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test_res, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_res, y_pred))

model_path = r"C:\Users\asmaa\Uni\Junior 2\Monitors\svc_model.pkl"


joblib.dump(svc_model, model_path)
print(f"Model saved at: {model_path}")

# Load trained model
svc_model = joblib.load(r"C:\Users\asmaa\Uni\Junior 2\Monitors\svc_model.pkl")
  # Ensure model file exists


# Initialize pygame mixer
pygame.init()
pygame.mixer.init()

#  Sound playback using threading (non-blocking)
def play_sound_thread(eye_state):
    def sound_task():
        if eye_state == 1:
            pygame.mixer.music.load("C:/Users/asmaa/Downloads/03.Lama_El_Shams_Trwah.wav")
        else:
            pygame.mixer.music.load("C:/Users/asmaa/Downloads/m30310.wav")
        pygame.mixer.music.play()
    threading.Thread(target=sound_task, daemon=True).start()

# GUI 
num_features = X_train_res.shape[1]
print(f" Model expects {num_features} features")

root = tk.Tk()
root.title("EEG Eye State Prediction")
root.geometry("400x200")
root.configure(bg="white")

status_label = tk.Label(root, text="Waiting for prediction...", font=("Arial", 20), fg="blue", bg="white")
status_label.pack(pady=40)

# Update GUI & play sound after prediction
def update_gui(prediction):
    if prediction == 1:
        status_label.config(text="Eyes Closed", fg="red")
    else:
        status_label.config(text="Eyes Open", fg="green")
    play_sound_thread(prediction)  #  Call sound function from here

# Prediction loop (runs in a separate thread)
def prediction_loop():
    while True:
        random_eeg_sample = np.random.rand(1, num_features)
        prediction = svc_model.predict(random_eeg_sample)[0]
        root.after(0, update_gui, prediction)
        time.sleep(5)

t = threading.Thread(target=prediction_loop, daemon=True)
t.start()

root.mainloop()

# **Real-time EEG Simulation**
def predict_eye_state(eeg_data):
    """
    Simulate real-time EEG prediction.
    Takes EEG data (a single sample) and predicts eye state.
    """
    eeg_data = np.array(eeg_data).reshape(1, -1)  # Reshape for model
    predicted_eye_state = svc_model.predict(eeg_data)[0]  # Predict
    return predicted_eye_state

# # **Simulated Real-Time Streaming**
# for _ in range(10):  # Simulate 10 EEG readings
#     # Generate a random EEG sample (Replace this with real EEG input)
#     random_eeg_sample = np.random.rand(1, num_features)

#     # Predict eye state
#     eye_state_prediction = predict_eye_state(random_eeg_sample)

#     # Play corresponding sound
#     play_sound(eye_state_prediction)

#     # Print result
#     print(f"Predicted Eye State: {eye_state_prediction}")

#     time.sleep(10)  # Simulate delay between readings (adjust as needed)