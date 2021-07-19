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

(defun layer-sizes (x y &optional (n-h 4))
  "Arguments:
   X -- input dataset of shape (input size, number of examples)
   Y -- labels of shape (output size, number of examples

   Returns:
   n-x -- the size of the input layer
   n-h -- the size of the hidden layer
   n-y -- the size of the output layer"
  (let ((n-x (num-rows x))
        (n-y (num-rows y)))
    (assert (= (num-cols x) (= num-cols y) nil "Different number of samples in X and Y.")))
  (values n-x n-h n-y))

;;; TODO:
;;; Change the uniform random function for the weight matrices to
;;; normalized random function 
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

;;; FORWARD PROPAGATION
(defun forward-propagation-one-hidden-layer (x parameters
                                             &optional (activation-functions '(#'tanh-activation
                                                                               #'sigmoid-activation)))
  "Performs forward propagation
  Argument:
  X -- input data of size (n-x, m)
  parameters - hashtable containing weight and bias matrices.

  Returns:
  A2 -- The sigmoid output of the second activation
  cache -- a hashmap containing z1, a1, z2, and a2"
  (let* ((w1 (gethash "w1" parameters))
         (b1 (gethash "b1" parameters))
         (w2 (gethash "w2" parameters))
         (b1 (gethash "b1" parameters))
         (z1 (matrix-matrix-sum (matrix-matrix-multiply w1 x) b1))
         (a1 (funcall (nth 0 activation-functions) z1))
         (z2 (matrix-matrix-sum (matrix-matrix-multiply w2 a1) b2))
         (a2 (funcall (nth 1 activation-functions) z2))
         (cache (make-hash-table :test 'equal)))
    (setf (gethash "z1" cache) z1
          (gethash "a1" cache) a1
          (gethash "z2" cache) z2
          (gethash "a2" cache) a2)
    a2 cache))

(defun compute-cost-one-hidden-layer (a2 y)
  "Computes the cross-entropy cost
   Arguments:
   A2 -- The sigmoid output of the scond activation of shape (1, number of examples)
   Y  -- 'True' labels vector of shape (1, number of examples)

   Returns:
   Cost -- cross-entropy"
  (let* ((m (num-cols y))
         (sum 0.0))
    (dotimes (i m)
      (incf sum (+ (* (aref y 0 i) (log (aref a2 0 i)))
                   (* (- 1.0 (aref y 0 i)) (log (- 1.0 (aref a2 0 i)))))))
    (* (/ 1.0 m) sum)))

(defun backward-propagation-one-hidden-layer (parameters cache x y)
  "Implements the backward propagation algorithm

   Arguments:
   parameters - hashmap containing the weights and biases
   cache - hashtable containing z1, a1, z2, a2
   x - input data of shape(nx, number of examples)
   y - true labels vector of shape (1, number of exapmles)

   Returns:
   grads - hashtable containing gradients with respect to different parameters
  "
  (let* ((m (num-cols x))
         (w1 (gethash "w1" parameters))
         (w2 (gethash "w2" parameters))
         (a1 (gethash "a1" cache))
         (a2 (gethash "a2" cache))
         ;; backward propagation to calculate dw1, dw2, db1, db2, dz1, dz2
         (dz2 (matrix-matrix-subtract a2 y))
         (dw2 (matrix-scalar-multiply (matrix-matrix-multiply dz2 (transpose-matrix a1 mmatrix-t)) (/ 1.0 m)))
         (db2 (matrix-scalar-multiply (matrix-row-sum dz2) (/ 1.0 m)))
         (dz1 (matrix-matrix-elementwise-multiply (matrix-matrix-multiply (transpose-matrix w2) dz2)
                                                  (scalar-matrix-subtract 1.0 (matrix-power a1 2.0))))
         (dw1 (matrix-scalar-multiply (matrix-matrix-multiple dz1 (transpose-matrix x))
                                      (/ 1.0 m)))
         (db2 (matrix-scalar-multiply (matrix-row-sum dz1) (/ 1.0 m)))
         (grads (make-hash-table :test 'equal)))
    (setf (gethash "dw1" grads) dw1
          (gethash "db1" grads) db1
          (gethash "dw2" grads) dw2
          (gethash "db2" grads) db2)

    grads))

;;; Update parameters using gradient descent procedure
(defun update-parameters (parameters grads &optional (learning-rate 1.2))
  "Update parameters using the gradient descent update rule given above

   Arguments:
   parameters - hashtable containing the parameters
   grads      - hashtable containing the gradients

   Returns:
   parameters - hashtable containing updated parameters"
  (let* (;; retrive the parameters
         (w1 (gethash "w1" parameters))
         (b1 (gethash "b1" parameters))
         (w2 (gethash "w2" parameters))
         (b2 (gethash "b2" parameters))
         ;; retrive the gradient of each parameters
         (dw1 (gethash "dw1" grads))
         (db1 (gethash "db1" grads))
         (dw2 (gethash "dw2" grads))
         (db2 (gethash "db2" grads)))
    ;; update the parameters
    (setf w1 (- w1 (* learning-rate dw1))
          b1 (- b1 (* learning-rate db1))
          w2 (- w2 (* learning-rate dw2))
          b2 (- b2 (* learning-rate db2)))
    ;; save the updated parameters in the hash and return
    (setf (gethash "w1" parameters) w1
          (gethash "b1" parameters) b1
          (gethash "w2" parameters) w2
          (gethash "b2" parameters) b2)
    parameters))
