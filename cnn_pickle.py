import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image
from PIL import Image
import cv2

num_classes = 3
im_rows = 184
im_cols = 216
im_color = 3
in_shape = (im_rows, im_cols, im_color)

# pickleファイルの読み込み
data_file =  "pickleファイルの場所のパス/pose_emission.pickle"
data = pickle.load(open(data_file,"rb"))

#データをimageとlabelに分けつつ保存
X = []
y = []
for d in data:
    (label,img)=d 

    y.append(label)
    X.append(img)

X = np.array(X)
y = np.array(y)

# train dataとtest dataに分類
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, train_size=0.8,shuffle=True)
# # データを正規化 --- (*2)

print(X_train.size)
print(y_train.size)

resize_img = cv2.resize(img, (im_rows,im_cols))
im = Image.fromarray((X_train[20] * 255).astype(np.uint8))
src = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
plt.imshow(src)
plt.show()

X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255


#One-Hot形式前のy_testを保存
y_test_org = y_test
# ラベルデータをOne-Hot形式に変換
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)



# モデルを定義 --- (*3)
model = Sequential()
model.add(Conv2D(32, (4, 4), padding='same',
                input_shape=in_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (4, 4), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (4, 4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()


# 小規模モデル
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=(in_shape)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(4, activation='softmax'))
# model.summary()


model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])

#コールバックを宣言
# callbacks_filepath='/tmp/callbacks'
# callbacks= keras.callbacks.ModelCheckpoint(filepath= callbacks_filepath, 
#                                                verbose=1, 
#                                                save_weights_only=False, 
#                                                mode='auto',
#                                                monitor='val_accuracy', 
#                                                patience=10,
#                                                save_best_only=True)

history = model.fit(X_train, y_train,
          batch_size=32, 
          epochs=50,
          verbose=1,
          #validation_data=(X_test, y_test),callbacks=callbacks)
          validation_data=(X_test, y_test))



label_list = ["fall","sitting","standing"]

# モデルを評価
score = model.evaluate(X_test, y_test, verbose=1)

print('正解率=', score[1], 'loss=', score[0])

#各ラベルの評価
pred = model.predict(X_test) # modelは学習させたもの
correct_count = [0] * num_classes
count = [0] * num_classes


for i in range(pred.shape[0]):
    prediction = np.argmax(pred[i]) # モデルの予測ラベル取得
    answer = y_test_org[i]
    count[answer] += 1
    if prediction == answer:
        correct_count[answer] += 1 # 正解数カウント

accuracy = [correct/N for correct, N in zip(correct_count, count)] # 精度算出

for label, acc in enumerate(accuracy):
    print(f'accuracy for label {label_list[label]} :        {acc}')


# 学習の様子をグラフへ描画 --- (*7)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.show()
plt.legend(['train', 'test'], loc='upper left')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 予測
y_test = np.argmax(y_test, axis=1)  # 正解ラベル.one_hotから変換
pred = np.argmax(model.predict(X_test), axis=1)   # モデルの予測

cm = confusion_matrix(y_test, pred)
sns.heatmap(cm,  annot=True, fmt='d')  # annotでセルに値を表示, fmt='d'で整数表示
plt.show()

#学習済みモデルの保存
model.save('モデルの保存先のパスを指定/pose_emission.h5')
print("model save Done!")