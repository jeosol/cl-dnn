(uiop/package:define-package :src/utilities/utilities
    (:use :cl)
  (:export #:layer-sizes))

(in-package :src/utilities/utilities)

(defun num-rows (X)
  "Returns the number of rows in matrix X."
  (nth 0 (array-dimensions X)))

(defun num-cols (X)
  "Returns the number of rows in matrix X."
  (nth 1 (array-dimensions X)))

(defun layer-sizes (X, Y, &optional (n-h 4))
  "Arguments:
   X -- input dataset of shape (input size, number of examples)
   Y -- labels of shape (output size, number of examples

   Returns:
   n-x -- the size of the input layer
   n-h -- the size of the hidden layer
   n-y -- the size of the output layer"
  (let ((n-x (num-rows X))
        (n-y (num-rows Y)))
    (assert (= (num-cols X) (= num-cols Y) nil "Different number of samples in X and Y.")))
  (values n-x n-h n-y))
