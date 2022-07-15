# Visualize the Network Architecture
# from tensorflow.keras.utils import plot_model
# import pydot
# plot_model(reconstructed_model, to_file='InceptionV3.png', show_shapes=True)



# Visualize Accuracy/Validation Accuracy & Loss/Validation Loss from .log files
model_log = pd.read_csv('/Users/dim__gag/python/food-101/DSRI_MODELS/InceptionV3.log', sep=',')


def plot_Acc_and_Loss(model_log,title):
    # plot model accuracy
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title(title)
    plt.plot(model_log['accuracy'])
    plt.plot(model_log['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    # plot model loss
    plt.subplot(1,2,2)
    plt.title(title)
    plt.plot(model_log['loss'])
    plt.plot(model_log['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()




plot_Acc_and_Loss(model_log, "Name of the Plot")