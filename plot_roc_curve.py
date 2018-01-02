from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_test, y_score):
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	y_test = y_test.reshape(y_test.shape[0], 1)
	y_score = y_score.reshape(y_score.shape[0], 1)
	classes = 1
	for i in range(classes):
	    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
	    roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

	plt.figure()
	# change key 2 ---> "micro"
	plt.plot(fpr["micro"], tpr["micro"], color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc["micro"])
	plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()