#import sklearn
from plot_learning_curve import *


# do some prepare work
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)


def train(X_train, Y_train, learner, title):

	print('start training...')
	learner.fit(X_train, Y_train)
	#plot_learning_curve(learner, title, X_train, Y_train, cv=5, n_jobs=4)
	#plt.show()

	print('saving model...')
	from sklearn.externals import joblib
	with open('trained_' + title +'.fzy', 'wb') as fo: 
		joblib.dump(learner, fo)
	print('done')


def test(X_test, Y_test, title):
	from sklearn.externals import joblib
	learner = joblib.load('trained_' + title +'.fzy')
	Y_pred = learner.predict(X_test)

	from sklearn.metrics import accuracy_score
	accuracy = accuracy_score(Y_test, Y_pred)

	from plot_precision_recall_curve import plot_precision_recall_curve
	precision, recall = plot_precision_recall_curve(Y_test, Y_pred)

	from sklearn.metrics import f1_score
	f1 = f1_score(Y_test, Y_pred)

	from sklearn.metrics import confusion_matrix
	cnf_matrix = confusion_matrix(Y_test, Y_pred)

	from plot_confusion_matrix import plot_confusion_matrix
	plot_confusion_matrix(cnf_matrix, classes=['pular', 'non-pular'], normalize=True, title='Normalized confusion matrix')
	
	from plot_roc_curve import plot_roc_curve
	plot_roc_curve(Y_test, Y_pred)

	print('accuracy: ' + str(accuracy))
	print('f1: ' + str(f1))
	print('precision: ' + str(precision))
	print('recall: ' + str(recall))
	#print('threshold: ' + str(threshold))
