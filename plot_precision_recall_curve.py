from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

def plot_precision_recall_curve(y_test, y_score):
	from sklearn.metrics import average_precision_score
	average_precision = average_precision_score(y_test, y_score)
	precision, recall, _ = precision_recall_curve(y_test, y_score)

	plt.step(recall, precision, color='b', alpha=0.2,
	         where='post')
	plt.fill_between(recall, precision, step='post', alpha=0.2,
	                 color='b')

	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
	
	return precision, recall