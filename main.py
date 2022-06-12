import scipy.io
import numpy as np 
import tensorflow.keras
import sklearn.model_selection
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from keras import backend as K
def import_matlab(filename):
    data=scipy.io.loadmat(filename)
    fea=data['fea']
    labels=data['gnd']
    images=[]
    for image in fea:
        images.append(np.reshape(image,(32,32,1)))
    images=np.array(images)
    return images,labels

class FilterCNN():
    def __init__(self,input_shape,output_shape):
        self.CNN = Sequential([
            Conv2D(32,(3,3),input_shape=input_shape,activation='relu'),
            BatchNormalization(),
            MaxPooling2D(),
            Conv2D(64,(3,3),input_shape=input_shape,activation='relu'),
            BatchNormalization(),
            Flatten(),
            Dense(200,activation='relu'),
            Dense(output_shape, activation='softmax')
        ])
        adam = Adam(lr=0.001, decay=1e-6)#taxa de aprendizagem
        self.CNN.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=adam)
    def fit(self,X,Y):
        self.CNN.fit(X,Y, batch_size=50, epochs=165,verbose=False)#numero de epocas e numero de amostras por iteracao(batch_size)
    def transform(self,images):
        layers=self.CNN.layers[:-1]
        c_filter=Sequential(layers)
        return c_filter.predict(images)



X,Y=import_matlab('Yale_32x32.mat')

precisao_SVM_sem_filtro=[]
precisao_SVM_PCA=[]
precisao_SVM_CNN=[]

for i in range(100):
    K.clear_session()
    #divisao do conjunto de treinamento
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.2,stratify=Y)
    #preparando o label pro NN
    y_train_onehot=to_categorical(y_train)
    #preparando os dados pro PCA e SVM sem filtro
    x_train_flat=np.array([image.flatten() for image in x_train])
    x_test_flat=np.array([image.flatten() for image in x_test])
    #preprando os labels pro svm
    y_train_svm=np.reshape(y_train,(len(y_train)))
    y_test_svm=np.reshape(y_test,(len(y_test)))



    #PCA Filter
    pca=PCA(0.95)#Porcentagem Significativa
    pca.fit(x_train_flat)
    #dados filtrados PCA
    x_train_pca=pca.transform(x_train_flat)
    x_test_pca=pca.transform(x_test_flat)
    #CNN Filter
    filtercnn=FilterCNN((x_train.shape[1],x_train.shape[2],1),16)
    filtercnn.fit(x_train,y_train_onehot)
    #dados filtrados CNN
    x_train_filtcnn=filtercnn.transform(x_train)
    x_test_filtcnn=filtercnn.transform(x_test)

    #SVM sem Filtro
    print("\ntreinando sem Filtro %i"%(i),end="\r")
    modelSVM=SVC(kernel='rbf',decision_function_shape='ovo',gamma=1)
    modelSVM.fit(x_train_flat,y_train_svm)
    precisao_SVM_sem_filtro.append(sum(modelSVM.predict(x_test_flat)==y_test_svm)/len(y_test))
    #SVM Filtrado PCA
    print("\ntreinando PCA %i"%(i),end="\r")
    modelSVM=SVC(kernel='rbf',decision_function_shape='ovo',gamma=1)
    modelSVM.fit(x_train_pca,y_train_svm)
    precisao_SVM_PCA.append(sum(modelSVM.predict(x_test_pca)==y_test_svm)/len(y_test))
    #SVM Filtrado CNN
    print("\ntreinando SVM-CNN %i\r"%(i),end="\r")
    modelSVM=SVC(kernel='rbf',decision_function_shape='ovo',gamma=1)
    modelSVM.fit(x_train_filtcnn,y_train_svm)
    precisao_SVM_CNN.append(sum(modelSVM.predict(x_test_filtcnn)==y_test_svm)/len(y_test))

print("Precisao Filtrado PCA: %f"%(sum(precisao_SVM_PCA)/len(precisao_SVM_PCA)))
print("Precisao Filtrado CNN: %f"%(sum(precisao_SVM_CNN)/len(precisao_SVM_CNN)))
print("Precisao sem filtro: %f"%(sum(precisao_SVM_sem_filtro)/len(precisao_SVM_sem_filtro)))


#NN

# modelNN = Sequential([
#     Dense(300, activation='relu',input_shape=(1024,1)),
#     BatchNormalization(),
#     Dense(16, activation='softmax')
# ])
#adam = Adam(lr=0.001, decay=1e-6)

# modelNN.compile(loss='categorical_crossentropy',metrics=['accuracy'], optimizer=adam)
# x_train_nn=np.reshape(x_train_flat,(x_train_flat.shape[0],x_train_flat.shape[1],1))
# modelNN.fit(x_train_nn, y_train_onehot, batch_size=50, epochs=300)

# sum(modelNN.predict_classes(np.reshape(x_test_flat,(x_test_flat.shape[0],x_test_flat.shape[1],1))==np.reshape(y_test,(33)))/len(y_test))
