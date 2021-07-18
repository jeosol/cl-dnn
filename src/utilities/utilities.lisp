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

(defun rand ()
  (random 1.0))

(defun rand-double (&optional (min 0) (max 1.0))
  (declare (type number min max))
  (+ min (* (- max min) (rand))))

(defun rand-int (&optional (min 0) (max 1))
  (declare (type number min max))
  (floor (+ min (* (- max min) (rand)))))

(defun make-random-array (num-rows num-cols &optional (scale 1.0))
  (let* ((arr (make-array (list num-rows num-cols) :element-type 'single-float)))
    (dotimes (i num-rows)
      (dotimes (j num-cols)
        (setf (aref arr i j) (* (random 1.0) scale))))
    arr))

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

(defun initialize-parameters-one-hidden-layer (n-x n-h n-y)
  "
  Arguments:
  n-x -- size of the input layer
  n-h -- size of the hidden layer
  n-y -- size of the output layer
 
  Returns:
  parameters -- hashmap containing the weight and bias parameters
                W1 -- weight matrix of shape (n-h, n-x)
                b1 -- bias vector of shape (n-h, 1)
                W2 -- weight matrix of shape (n-y, n-h)
                b2 -- bias vector of shape (n-y, 1)"
  (let* ((w1 (make-random-array n-h n-x 0.01))
         (b1 (make-random-array n-h 1))
         (w2 (make-random-array n-y n-h 0.01))
         (b2 (make-random-array n-y 1))
         (parameters (make-hash-table :size 4 :test #'equal)))
    (setf (gethash "w1" parameters) w1
          (gethash "b1" parameters) b1
          (gethash "w2" parameters) w2
          (gethash "b2" parameters) b2)
    parameters))
