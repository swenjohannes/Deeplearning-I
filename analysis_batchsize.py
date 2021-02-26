
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Analysis/results_batchsize.csv') #batch sizes!
print(df.head())


def reorder(order, list):
    return [list[i] for i in order]

#Loss plot
fig, ax = plt.subplots(figsize=(8,6))
for name, group in df.groupby('Model'):
    group.plot(x='Epoch', y='val_loss', ax=ax, label = name)
#get current labels
current_handles, current_labels = plt.gca().get_legend_handles_labels()
# sort or reorder the labels and handles
new_handles = reorder([2,4,0,1,3], current_handles)
new_labels = reorder([2,4,0,1,3], current_labels)
plt.legend(new_handles, new_labels)    # call plt.legend() with the new values
plt.xlabel('Epochs')
plt.ylabel('Validation loss')
plt.savefig('figures/val_loss_cnn_batchsize.eps')
plt.show()


#Acc plot
fig, ax = plt.subplots(figsize=(8,6))
for name, group in df.groupby('Model'):
    group.plot(x='Epoch', y='val_accuracy', ax=ax, label=name)
#get current labels
current_handles, current_labels = plt.gca().get_legend_handles_labels()
# sort or reorder the labels and handles
new_handles = reorder([2,4,0,1,3], current_handles)
new_labels = reorder([2,4,0,1,3], current_labels)
plt.legend(new_handles, new_labels)    # call plt.legend() with the new values
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.savefig('figures/val_acc_cnn_batchsize.eps')
plt.show()

