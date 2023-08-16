import pickle
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

f = open('objs.pkl', 'rb')
data = pickle.load(f)
f.close()

loss_tracker = np.array(data[0])
avg_loss_tracker = np.array(data[1])
valid_loss_tracker = np.array(data[2])
acc_tracker = np.array(data[3])
patience_tracker = data[4]
fpr = data[5]
tpr = data[6]
auc = data[7]

min_training_loss = loss_tracker.min()
min_avg_loss = avg_loss_tracker.min()
min_valid_loss = valid_loss_tracker.min()
max_accuracy = acc_tracker.max()

min_training_loss_idx = loss_tracker.argmin()
min_avg_loss_idx = avg_loss_tracker.argmin()
min_valid_loss_idx = valid_loss_tracker.argmin()
max_accuracy_idx = acc_tracker.argmax()

plt.figure(1)
plt.plot(loss_tracker)
plt.plot(min_training_loss_idx, min_training_loss, marker="o", color="blue")
plt.title("Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend([Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='blue', markersize=15)],
			[f'{min_training_loss:.4f}'])

plt.figure(2)
plt.plot(avg_loss_tracker, color="blue")
plt.plot(min_avg_loss_idx, min_avg_loss, marker="o", color="blue")
plt.title("Average Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend([Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='blue', markersize=15)],
			[f'{min_avg_loss:.4f}'])

fig, ax = plt.subplots()
ax.plot(valid_loss_tracker, color="red")
ax.plot(min_valid_loss_idx, min_valid_loss, marker="o", color="red")
plt.title("Validation Loss and Accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax2 = ax.twinx()
ax2.plot(acc_tracker, color='green')
ax2.plot(max_accuracy_idx, max_accuracy, marker="o", color="green")
ax2.set_ylabel("Accuracy")
plt.legend([Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='red', markersize=15),
			Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='green', markersize=15)],
			[f'Validation: {min_valid_loss:.4f}', f'Accuracy: {max_accuracy:.4f}'])

fig2, ax3 = plt.subplots()
ax3.plot(valid_loss_tracker, color='red')
ax3.plot(min_valid_loss_idx, min_valid_loss, marker="o", color="red")
plt.title("Validation Loss and Patience")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Loss")
ax4 = ax3.twinx()
ax4.plot(patience_tracker, color='cyan')
ax4.set_yticks([0, 1, 2, 3])
ax4.set_ylabel("Patience")
plt.legend([Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='red', markersize=15)],
			[f'Validation: {min_valid_loss:.4f}'])

fig3, ax4 = plt.subplots()
ax4.plot(avg_loss_tracker, color='blue')
ax4.plot(min_avg_loss_idx, min_avg_loss, marker="o", color="blue")
plt.title("Training and Validation Loss")
ax4.set_xlabel("Epoch")
ax4.set_ylabel("Loss")
ax4.plot(valid_loss_tracker, color='red')
ax4.plot(min_valid_loss_idx, min_valid_loss, marker="o", color="red")
plt.legend([Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='blue', markersize=15),
			Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='red', markersize=15)],
			[f'Training: {min_avg_loss:.4f}', f'Validation: {min_valid_loss:.4f}'])

fig4, ax5 = plt.subplots()
ax5.plot(avg_loss_tracker[0:10], color="blue")
ax5.plot(valid_loss_tracker[0:10], color="red")
ax5.plot(min_valid_loss_idx, min_valid_loss, marker="o", color="red")
plt.title("Loss and Accuracy")
ax5.set_xlabel("Epoch")
ax5.set_ylabel("Loss")
ax6 = ax5.twinx()
ax6.plot(acc_tracker[0:10], color='green')
ax6.plot(max_accuracy_idx, max_accuracy, marker="o", color="green")
ax6.set_ylabel("Accuracy")
plt.legend([Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='blue', markersize=15),
			Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='red', markersize=15),
			Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='green', markersize=15)],
			[f'Training: {min_avg_loss:.4f}', f'Validation: {min_valid_loss:.4f}',f'Accuracy: {max_accuracy:.4f}'])

plt.figure(7)
plt.plot(fpr, tpr, label='ROC Curve')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend([Line2D([0], [0], marker='s', color='w', label='Scatter', markerfacecolor='#AECCE4', markersize=15)], [f'AUC: {auc:.4f}'])
plt.fill_between(fpr, tpr, color="#AECCE4")

fig5, ax7 = plt.subplots()
ax8 = ax7.twiny()
ax8.plot(loss_tracker[0:200], 'b.')
ax7.plot(avg_loss_tracker[0:10], color="cyan")
ax7.set_xlabel("Epoch")
plt.title("Training Loss")
ax7.set_ylabel("Loss")


plt.show()
