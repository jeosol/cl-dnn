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

(defun layer-sizes (X, Y)
  "Arguments:
   X -- input dataset of shape (input size, number of examples)
   Y -- labels of shape (output size, number of examples

   Returns:
   n-x -- the size of the input layer
   n-h -- the size of the hidden layer
   n-y -- the size of the output layer"
  (values (num-rows X) (num-cols X) (num-rows)))
