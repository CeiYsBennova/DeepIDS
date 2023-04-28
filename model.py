import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import load_model

feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

#load dataset
train_data = pd.read_csv('/content/drive/MyDrive/ML/NSL-KDD/KDDTrain+.txt',names=feature)
test_data = pd.read_csv('/content/drive/MyDrive/ML/NSL-KDD/KDDTest+.txt',names=feature)

#drop difficulty column
train_data.drop('difficulty',axis=1,inplace=True)

# change labels to Dos, Probe, R2L, U2R
train_data['label'] = train_data['label'].replace(['back','land','neptune','pod','smurf','teardrop','apache2','udpstorm','processtable','worm','mailbomb'],'DoS')
train_data['label'] = train_data['label'].replace(['ipsweep','nmap','portsweep','satan','mscan','saint'],'Probe')
train_data['label'] = train_data['label'].replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L')
train_data['label'] = train_data['label'].replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R')

train_data['label'].value_counts()

# use standard scaler to scale the data except colums 1,2,3
scaler = StandardScaler()
train_data.iloc[:,4:41] = scaler.fit_transform(train_data.iloc[:,4:41])

# label encoding: DoS=0, Probe=1, R2L=2, U2R=3, Normal=4
train_data['label'] = train_data['label'].replace(['DoS','Probe','R2L','U2R', 'normal'],[0,1,2,3,4])

# one hot encoding
train_data = pd.get_dummies(train_data,columns=['protocol_type','service','flag'])

# split data into train and test
X = train_data.drop('label',axis=1)
y = train_data['label']

# one hot encoding for y
y = pd.get_dummies(y).values

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# reshape data to 3D array
X_train = np.array(X_train).reshape(X_train.shape[0],X_train.shape[1],1)
X_test = np.array(X_test).reshape(X_test.shape[0],X_test.shape[1],1)

# build model
model = Sequential()
model.add(Conv1D(32,2,activation='relu',input_shape=(X_train.shape[1],1),padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(64,2,activation='relu',padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(128,2,activation='relu',padding='same'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(5,activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# train model
model.fit(X_train,y_train,epochs=10,batch_size=32,validation_data=(X_test,y_test))

# save model
model.save('/content/drive/MyDrive/ML/NSL-KDD/deepids.h5')

# load model
model = load_model('/content/drive/MyDrive/ML/NSL-KDD/deepids.h5')



