
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Analysis/results_droprate.csv') #batch sizes!
print(df.head())


def reorder(order, list):
    return [list[i] for i in order]


from tensorflow import keras
model = keras.models.load_model('Models/batchsize_nobn_64')
model.summary()

#Loss plot
fig, ax = plt.subplots(figsize=(8,6))
for name, group in df.groupby('Model'):
    group.plot(x='Epoch', y='val_loss', ax=ax, label = name)
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.savefig('figures/val_loss_cnn_dropout.eps')
plt.show()


#Acc plot
fig, ax = plt.subplots(figsize=(8,6))
for name, group in df.groupby('Model'):
    group.plot(x='Epoch', y='val_accuracy', ax=ax, label=name)
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.savefig('figures/val_acc_cnn_dropout.eps')
plt.show()

