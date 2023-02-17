from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


folder = ["fall","sitting","standing"]
image_size = 200 #　画像サイズの指定（正方形）
num_class = 3 # 分類する数
 
X = []
Y = []
for index, name in enumerate(folder):
    dir = "データセットのパスを入力" + name
    files = glob.glob(dir + "/*.jpg")
    for i, file in enumerate(files):
        image = Image.open(file)
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)
        X.append(data)
        Y.append(index)
 
X = np.array(X)
Y = np.array(Y)


X = X.astype('float32')
X = X / 255.0


# 正解ラベルの形式を変換
Y = np_utils.to_categorical(Y, num_class) #初期値3が入っていた。

# 学習用データとテストデータ
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
            
y_test_org = y_test
                                        
# CNNを構築
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_class))
model.add(Activation('softmax'))
 
# コンパイル
model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])


#訓練
history = model.fit(X_train, y_train,
          batch_size=16, 
          epochs=40,
          verbose=1,
          #validation_data=(X_test, y_test),callbacks=callbacks)
          validation_data=(X_test, y_test))


#評価 & 評価結果出力
score = model.evaluate(X_test, y_test)
print(score)

print('正解率=', score[1], 'loss=', score[0])

#各ラベルの評価
pred = model.predict(X_test) # modelは学習させたもの
correct_count = [0] * num_class
count = [0] * num_class 

# 学習の様子をグラフへ描画 --- (*7)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 予測
y_test = np.argmax(y_test, axis=1)  # 正解ラベル.one_hotから変換
print(X_test)
pred = np.argmax(model.predict(X_test), axis=1)   # モデルの予測
cm = confusion_matrix(y_test, pred)
sns.heatmap(cm,  annot=True, fmt='d')  # annotでセルに値を表示, fmt='d'で整数表示
plt.show()

#学習済みモデルの保存
model.save('モデルの保存のパスを入力/pose_emission.h5')
print("model save Done!")