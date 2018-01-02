import train
import numpy as np
from PFDFile import *
import os

DIRS = ['./p309p_pfd', './p309n_pfd']

def make_data(dir, pular, most_file_num=1000):
	X = 0
	cnt = 0
	for file in os.listdir(dir):
		if cnt >= most_file_num:
			break
		print(cnt, file)
		cnt += 1

		cand = PFD(dir + './' + file)

		# np.array (32, ?) ? 
		subband = cand.get_subbands()
		subband_sum = np.sum(subband, axis=1)

		# np.array (64, ?) ? 
		#subints = cand.get_subints()
		#subints_sum = np.sum(subints, axis=1)

		# list of length ?
		# each element stands for the peak of the profile
		profile = cand.getprofile()
		profile_sum = np.sum(profile)

		# a number 
		# the calculated reduced-chi^2 of the current summed profile.
		chi2 = cand.calc_redchi2()

		# row and column axis number 2x100
		#chi2dm = cand.plot_chi2_vs_DM()

		x = np.array([profile_sum, chi2])
		x = np.hstack((x, subband_sum))

		if X is 0:
			X = x
		else:
			X = np.vstack((X, x))
	if pular is True:
		Y = np.repeat(1, X.shape[0]).reshape(X.shape[0], 1)
	else:
		Y = np.repeat(0, X.shape[0]).reshape(X.shape[0], 1)
	return X, Y


def save_data(X, Y):
	from sklearn.externals import joblib
	with open('saved_data.fzy', 'wb') as fo: 
		joblib.dump((X, Y), fo)
	print('data saved')

def get_data():
	from sklearn.externals import joblib
	X, Y = joblib.load('saved_data.fzy')
	return X, Y

def main():
	X1, Y1 = make_data('./p309p_pfd', pular=True)
	X2, Y2 = make_data('./p309n_pfd', pular=False)
	X = np.vstack((X1, X2))
	Y = np.vstack((Y1, Y2))
	Y = Y.reshape(Y.shape[0])

	save_data(X, Y)

	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=0)
	
	from sklearn.svm import SVC
	train.train(X_train, y_train, SVC(), 'SVC')
	train.test(X_test, y_test, 'SVC')

	from sklearn.ensemble import RandomForestClassifier
	train.train(X_train, y_train, RandomForestClassifier(n_estimators=10), 'RandomForest')
	tain.test(X_test, y_test, 'RandomForest')

	from sklearn.ensemble import GradientBoostingClassifier
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
	train.train(X_train, y_train, clf, 'GradientBoosting')
	tain.test(X_test, y_test, 'GradientBoosting')

	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.tree import DecisionTreeClassifier
	bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), algorithm="SAMME", n_estimators=200)
	train.train(X_train, y_train, bdt, 'AdaBoost')
	tain.test(X_test, y_test, 'AdaBoost')

	from sklearn.neural_network import MLPClassifier
	MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(8, 8, 8), random_state=1)


main()