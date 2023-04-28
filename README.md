# DeepIDS

## NSL-KDD
Là bộ dataset chuẩn cho training IDS, gồm 43 thuộc tính, trong đó:
- 41 thuộc tính liên quan đến lưu lượng
- 2 thuộc tính gồm label: không tấn công hoặc tấn công kiểu gì và difficulty: mức độ nghiêm trọng của lưu lượng đầu vào

Có 4 lớp tấn công chính là Tấn công từ chối dịch vụ (Denial of Services – DoS), Do thám (Probe), User to Root (U2R) và Remote to Local (R2L), mỗi lớp tấn công lại có các lớp con như dưới đây
|Lớp|DoS|Probe|R2L|U2R|
|---|---|-----|---|---|
|Lớp con|back<br/>land<br/>neptune<br/>pod<br/>smurf<br/>teardrop<br/>apache2<br/>udpstorm<br/>processtable<br/>worm<br/>mailbomb|ipsweep<br/>nmap<br/>portsweep<br/>satan<br/>mscan<br/>saint|ftp_write<br/>guess_passwd<br/>httptunnel<br/>imap<br/>multihop<br/>named<br/>phf<br/>sendmail<br/>snmpgetattack<br/>snmpguess<br/>spy<br/>warezclient<br/>warezmaster<br/>xlock<br/>xsnoop|buffer_overflow<br/>loadmodule<br/>perl<br/>ps<br/>rootkit<br/>sqlattack<br/>xterm|
|Tổng|11|6|15|7|

## Preprocessing data
Do dữ liệu thuần data, các cột chưa có tên nên ta cần phải "lắp" tên vào nó:
```
feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]

#load dataset
train_data = pd.read_csv('/content/drive/MyDrive/ML/NSL-KDD/KDDTrain+.txt',names=feature)
```

Sau đó, ta sẽ cần loại bỏ cột difficulty, vì cột này không có giá trị để training:
```
train_data.drop('difficulty',axis=1,inplace=True)
```

Đổi các class con về 4 class chính:
```
train_data['label'] = train_data['label'].replace(['back','land','neptune','pod','smurf','teardrop','apache2','udpstorm','processtable','worm','mailbomb'],'DoS')
train_data['label'] = train_data['label'].replace(['ipsweep','nmap','portsweep','satan','mscan','saint'],'Probe')
train_data['label'] = train_data['label'].replace(['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack','snmpguess','spy','warezclient','warezmaster','xlock','xsnoop'],'R2L')
train_data['label'] = train_data['label'].replace(['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'],'U2R')
```

Do các dữ liệu dạng số trong dataset không giống nhau nên chúng ta cần chuẩn hóa bằng module `StandardScaler` của `sklearn` ngoại trừ 3 cột là `protocol_type`, `service` , `flag`
```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data.iloc[:,4:41] = scaler.fit_transform(train_data.iloc[:,4:41])
```

Sau đó thực hiện quá trình encoding label thành số, sau đó chuyển thành vector, tương tự với 3 cột `protocol_type`, `service` , `flag`, ta sử dụng one hot encoding
```
# label encoding: DoS=0, Probe=1, R2L=2, U2R=3, Normal=4
train_data['label'] = train_data['label'].replace(['DoS','Probe','R2L','U2R', 'normal'],[0,1,2,3,4])

# one hot encoding
train_data = pd.get_dummies(train_data,columns=['protocol_type','service','flag'])

X = train_data.drop('label',axis=1)
y = train_data['label']

# one hot encoding for y
y = pd.get_dummies(y).values
```

Cuối cùng, dùng `train_test_split` của `sklearn` để chia thành train data và test data, sau đó reshape (thêm 1 chiều) để làm đầu vào cho lớp `Conv1D`
```
# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# reshape data to 3D array
X_train = np.array(X_train).reshape(X_train.shape[0],X_train.shape[1],1)
X_test = np.array(X_test).reshape(X_test.shape[0],X_test.shape[1],1)
```

## Train model
Xây dựng một mạng CNN đơn giản gồm 3 lớp `Conv1D` + `MaxPooling1D` với `activation` là `relu`, sau đó dùng `flatten` để duỗi ma trận thành vector để làm đầu vào cho `fully connected network`. Dùng `dense` để cô đọng kết quả dần dần và `softmax` để phân loại:
```
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
```

Compile model với `optimizer=adam`, hàm loss sử dụng `categorical_crossentropy` và phương pháp đánh giá là `accuracy` 
```
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
```

Cuối cùng là train model với 50 epochs:
```
model.fit(X_train,y_train,epochs=50,batch_size=3000,validation_data=(X_test,y_test))
```
