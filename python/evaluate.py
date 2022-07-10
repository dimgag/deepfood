# Print training, test accuracy and loss of the model
def model_eval(model, train, val):
    # evaluate the model
    train_loss, train_acc = model.evaluate(train, verbose=0)
    val_loss, val_acc = model.evaluate(val, verbose=0)
    print('Train loss:', train_loss)
    print('Train accuracy:', train_acc)
    print('Validation loss:', val_loss)
    print('Validation accuracy:', val_acc)