import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, AvgPool2D, Dropout, Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2
from keras.models import load_model
from utils import batch_generator

np.random.seed(42)

def dataloader():
    df = pd.read_csv('data/driving_log.csv')
    X = df[['center', 'left', 'right']].values
    y = df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_valid, y_train, y_valid

def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Conv2D(24, 5, 5, activation='relu', subsample=(2, 2), border_mode='valid', W_regularizer=l2(1e-3)))
    model.add(Conv2D(36, 5, 5, activation='relu', subsample=(2, 2), border_mode='valid', W_regularizer=l2(1e-3)))
    model.add(Conv2D(48, 5, 5, activation='relu', subsample=(2, 2), border_mode='valid', W_regularizer=l2(1e-3)))
    model.add(Conv2D(64, 3, 3, activation='relu', subsample=(1, 1), border_mode='valid', W_regularizer=l2(1e-3)))
    model.add(Conv2D(64, 3, 3, activation='relu', subsample=(1, 1), border_mode='valid', W_regularizer=l2(1e-3)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model
              
def train_model(model, X_train, X_valid, y_train, y_valid):
    checkpoint = ModelCheckpoint('model-new-{epoch:03d}.h5',
                                 monitor='val_loss',
                                 verbose=2,
                                 save_best_only=True,
                                 mode='auto')
              
    reducelr = ReduceLROnPlateau(monitor='val_loss', 
                                 factor=0.2, 
                                 patience=2, 
                                 verbose=2, 
                                 mode='auto',
                                 min_lr=1e-6)
              
    earlystop = EarlyStopping(monitor='val_loss', 
                              min_delta=0.001, 
                              patience=4, 
                              verbose=2, 
                              mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-2))

    model.fit_generator(batch_generator('data', X_train, y_train, 64, True),
                        steps_per_epoch=2000,
                        epochs=10,
                        verbose=1,
                        validation_data=batch_generator('data', X_valid, y_valid, 64, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint, reducelr, earlystop]
                       )             

def main():
    data = dataloader()
    model = get_model()
#     model = load_model('model-005.h5')
    train_model(model, *data)

if __name__ == '__main__':
    main()