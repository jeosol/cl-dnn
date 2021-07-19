(uiop/package:define-package :src/utilities/utilities
    (:use :cl)
  (:nicknames :cl-dnn-utilities)
  (:export #:layer-sizes))

(in-package :src/utilities/utilities)

(defun tanh-activation (arr)
  "Implements the hyperbolic tangent (tanh) function"
  (let* ((rows (num-rows arr))
         (cols (num-cols arr))
         (new-arr (make-array (list rows cols) :element-type 'single-float)))
    (dotimes (i rows)
      (dotimes (j cols)
        (setf (aref new-arr i j) (/ (- (exp (aref arr i j)) (exp (- (aref arr i j))))
                                      (+ (exp (aref arr i j)) (exp (- (aref arr i j))))))))
    new-arr))

(defun relu-activation (arr)
  "Implements the rectified linear unit (relu) function."
  (let* ((rows (num-rows arr))
         (cols (num-cols arr))
         (new-arr (make-array (list rows cols) :element-type 'single-float)))
    (dotimes (i rows)
      (dotimes (j cols)
        (setf (aref new-arr i j) (if (< (aref arr i j) 0.0) 0.0 (aref arr i j)))))
    new-arr))

(defun sigmoid-activation (arr)
  "Implements the logistic/sigmoid function"
  (let* ((rows (num-rows arr))
         (cols (num-cols arr))
         (new-arr (make-array (list rows cols) :element-type 'single-float)))
    (dotimes (i rows)
      (dotimes (j cols)
        (setf (aref new-arr i j) (/ 1 (+ 1 (exp (- (aref arr i j))))))))
    new-arr))

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

(defun make-matrix (num-rows num-cols)
  "Simple function to create a matrix of size NUM-ROWS by NUM-COLS"
  (make-array (list num-rows num-cols) :element-type 'single-float :initial-element 0.0))

(defun make-random-array (num-rows num-cols &optional (scale 1.0))
  (let* ((arr (make-array (list num-rows num-cols) :element-type 'single-float)))
    (dotimes (i num-rows)
      (dotimes (j num-cols)
        (setf (aref arr i j) (* (random 1.0) scale))))
    arr))

(defun matrix-matrix-multiply (a b)
  "C = A*B"
  (let* ((c (make-array (list (num-rows a) (num-cols b)) :element-type 'single-float))
         (sum 0.0))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols b))
        (setf sum 0.0)
        (dotimes (k (num-cols a))
          (incf sum (* (aref a i k) (aref b k j))))
        (setf (aref c i j) sum)))
    c))

(defun matrix-matrix-elementwise-multiply (a b)
  "Matrix element multiply c[i,j] = a[i,j] * b[i,j]"
  (let* ((c (make-matrix (num-rows a) (num-rows b))))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref c i j) (* (aref a i j) (aref b i j)))))
    c))

(defun matrix-matrix-add (a b)
  "Returns c = a + b"
  (let* ( (c (make-array (list (num-rows a) (num-cols b)) :element-type 'single-float)))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref c i j) (+ (aref a i j) (aref b i j)))))
    c))

(defun matrix-matrix-subtract (a b)
  "Returns c = a - b"
  (let* ( (c (make-array (list (num-rows a) (num-cols b)) :element-type 'single-float)))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref c i j) (- (aref a i j) (aref b i j)))))
    c))

(defun scalar-matrix-subtract (scalar matrix)
  "Computes a new matrix with elements c[i,j] = 1 - matrix[i,j]"
  (let* ((c (make-array (list (num-rows matrix) (num-cols matrix)) :element-type 'single-float)))
    (dotimes (i (num-rows matrix))
      (dotimes (j (num-rows matrix))
        (setf (aref c i j) (- scalar (aref matrix i j)))))
    c))

(defun matrix-scalar-multiply (matrix-a scalar)
  "Returns c = a*scalar - multiplies the elements of matrix a by SCALAR"
  (let* ((c (make-matrix (num-rows matrix-a) (num-cols matrix-a))))
    (dotimes (i (num-rows matrix-a))
      (dotimes (j (num-rows matrix-a))
        (setf (aref c i j) (* scalar (aref matrix-a i j)))))
    c))

(defun matrix-power (matrix-a power)
  "Computes the elements of matrix MATRIX-A raised to power POWER"
  (let* ((c (make-array (list (num-rows matrix-a) (num-cols matrix-a)) :element-type 'single-float)))
    (dotimes (i (num-rows matrix-a))
      (dotimes (j (num-rows matrix-a))
        (setf (aref c i j) (expt (aref matrix-a i j) power))))
    c))

(defun transpose-matrix (a)
  "Returns the transpose of matrix A"
  (let* ((c (make-array (list (num-cols a) (num-rows a)) :element-type 'single-float)))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref c j i) (aref a i j))))
    c))

(defun matrix-row-sum (a)
  "Sums the elements on each row of matrix a"
  (let* ((c (make-array (list (num-rows a) 1) :element-type 'single-float :initial-element 0.0)))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (incf (aref c i 0) (aref a i j))))
    c))

(defun matrix-col-sum (a)
  "Sums the elements on each column of matrix a"
  (let* ((c (make-array (list (num-cols a) 1) :element-type 'single-float :initial-element 0.0)))
    (dotimes (j (num-cols a))
      (dotimes (i (num-rows a))
        (incf (aref c j 0) (aref a i j))))
    c))

(defun vector-vector-dot-product (a b)
  "Returns the sum of the products of components in vectors a and b"
  (loop :for ai :across a :for bi :across b :summing (* ai bi) :into sum
        :finally (return sum)))

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
