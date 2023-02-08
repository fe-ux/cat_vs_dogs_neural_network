import matplotlib.pyplot as plt

def main(acc,loss,val_acc,val_loss):
    n=range(1,len(acc)+1)
    plt.plot(n,acc,"bo",label="Train acc")
    plt.plot(n,val_acc,"b",label="Val acc")
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.clf()

    plt.plot(n,loss,"bo",label="Train loss")
    plt.plot(n,val_loss,"b",label="Val loss")
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()