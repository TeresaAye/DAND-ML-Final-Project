# DAND-ML-Final-Project
Final Project for Udacity DataAnalytics Nanodegree Machine Learning Module

Mar 5, 2018 resubmission:

The only item that I could not resubmit is the last comment. This is Udacity code, not my code. Tester.py works fine when I run it.
If tester.py does not work for the reviewer, then perhaps reviewer environment is not set up properly.

I checked the forum and google and could not find answer on how to fix udacity's code.

Or please give me an idea of what you expect, since this is not my code.

Rubric is:

When tester.py is used to evaluate performance, precision and recall are both at least 0.3.
Required

Reviewer wrote:

The tester.py file threw the following error:

Traceback (most recent call last):
  File "tester.py", line 103, in <module>
    main()
  File "tester.py", line 100, in main
    test_classifier(clf, dataset, feature_list)
  File "tester.py", line 50, in test_classifier
    clf.fit(features_train, labels_train)
  File "lib\site-packages\sklearn\pipeline.py", line 257, in fit
    Xt, fit_params = self._fit(X, y, **fit_params)
  File "lib\site-packages\sklearn\pipeline.py", line 190, in _fit
    memory = self.memory
AttributeError: 'Pipeline' object has no attribute 'memory'

The script was run using the latest version of scikit-learn (0.19.0).

