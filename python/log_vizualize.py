import pandas as pd
import matplotlib.pyplot as plt

def log_plot_Acc_and_Loss(log,title):
    # plot model accuracy
    data = pd.read_csv(log, sep=',')
    plt.figure(figsize=(16,8))
    plt.subplot(1,2,1)
    plt.title(title)
    plt.plot(log['accuracy'])
    plt.plot(log['val_accuracy'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_accuracy', 'validation_accuracy'], loc='best')
    # plot model loss
    plt.subplot(1,2,2)
    plt.title(title)
    plt.plot(log['loss'])
    plt.plot(log['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss', 'validation_loss'], loc='best')
    plt.show()

# Example of usage:
log_plot_Acc_and_Loss('EfficientNetV2S.log', 'EfficientNetV2S')