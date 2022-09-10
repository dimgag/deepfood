import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import SGD


def train_model(model, model_name, train, val, nb_train_samples, nb_validation_samples, epochs, batch_size):
    model.trainable = True
    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath=model_name+'.hdf5', verbose=1, save_best_only=True)
    csv_logger = CSVLogger(model_name +'.log') #, append = True)

    history = model.fit_generator(train,
                                  steps_per_epoch = nb_train_samples // batch_size,
                                  validation_data=val,
                                  validation_steps=nb_validation_samples // batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  callbacks=[csv_logger, checkpointer])
    model.save(model_name)    
    return model, history
