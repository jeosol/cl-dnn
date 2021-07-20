(uiop/package:define-package :src/utilities/utilities
    (:use :cl)
  (:nicknames :cl-dnn-utilities)
  (:export #:layer-sizes))

(in-package :src/utilities/utilities)

(defun make-matrix (num-rows num-cols)
  "Create a matrix of size NUM-ROWS by NUM-COLS."
  (make-array (list num-rows num-cols) :element-type 'single-float :initial-element 0.0))

(defun tanh-activation (matrix)
  "Compute hyperbolic tangent function tanh(x) where x is an element of matrix MATRIX."
  (let* ((rows (num-rows matrix))
         (cols (num-cols matrix))
         (new-matrix (make-matrix rows cols)))
    (dotimes (i rows)
      (dotimes (j cols)
        (setf (aref new-matrix i j) (/ (- (exp (aref matrix i j)) (exp (- (aref matrix i j))))
                                       (+ (exp (aref matrix i j)) (exp (- (aref matrix i j))))))))
    new-matrix))

(defun relu-activation (matrix)
  "Compute the RELU(x) where x is an element of matrix MATRIX."
  (let* ((rows (num-rows matrix))
         (cols (num-cols matrix))
         (new-matrix (make-matrix rows cols)))
    (dotimes (i rows)
      (dotimes (j cols)
        (setf (aref new-matrix i j) (if (< (aref matrix i j) 0.0) 0.0 (aref matrix i j)))))
    new-matrix))

(defun sigmoid-activation (matrix)
  "Implements the logistic/sigmoid function"
  (let* ((rows (num-rows matrix))
         (cols (num-cols matrix))
         (new-matrix (make-matrix rows cols)))
    (dotimes (i rows)
      (dotimes (j cols)
        (setf (aref new-matrix i j) (/ 1 (+ 1 (exp (- (aref matrix i j))))))))
    new-matrix))

(defun num-rows (matrix)
  "Return the number of rows in matrix MATRIX"
  (nth 0 (array-dimensions X)))

(defun num-cols (matrix)
  "Return the number of rows in matrix MATRIX."
  (nth 1 (array-dimensions X)))

(defun rand ()
  "Return a random float between 0 and 1."
  (random 1.0))

(defun rand-double (&optional (min 0) (max 1.0))
  "Return a random float between MIN and MAX."
  (declare (type number min max))
  (+ min (* (- max min) (rand))))

(defun rand-int (&optional (min 0) (max 1))
  "Return a random integer between MIN and MAX."
  (declare (type number min max))
  (floor (+ min (* (- max min) (rand)))))

(defun make-random-array (num-rows num-cols &optional (scale 1.0))
  "Create a random matrix with elements initialized to random numbers in [0,1]."
  (let* ((arr (make-matrix num-rows numcols)))
    (dotimes (i num-rows)
      (dotimes (j num-cols)
        (setf (aref arr i j) (* (random 1.0) scale))))
    arr))

(defun matrix-matrix-multiply (a b)
  "Compute the product of matrices: C = A*B."
  (let* ((c (make-matrix (num-rows a) (num-cols b)))
         (sum 0.0))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols b))
        (setf sum 0.0)
        (dotimes (k (num-cols a))
          (incf sum (* (aref a i k) (aref b k j))))
        (setf (aref c i j) sum)))
    c))

(defun matrix-matrix-elementwise-multiply (a b)
  "Perform elementwise multiplication of two matrices: c[i,j] = a[i,j] * b[i,j]."
  (let* ((c (make-matrix (num-rows a) (num-rows b))))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref c i j) (* (aref a i j) (aref b i j)))))
    c))

(defun matrix-matrix-add (a b)
  "Perform elementwise addition of two matrices: c[i,j] = a[i,j] + b[i,j]."
  (let* ( (c (make-marix (num-rows a) (num-cols a))))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref c i j) (+ (aref a i j) (aref b i j)))))
    c))

(defun matrix-matrix-subtract (a b)
  "Perform elementwise subtraction of two matrices: c[i,j] = a[i,j] - b[i,j]."
  (let* ( (c (make-matrix (num-rows a) (num-cols a))))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref c i j) (- (aref a i j) (aref b i j)))))
    c))

(defun scalar-matrix-subtract (scalar matrix)
  "Compute c[i,j] = scalar - matrix[i,j] for each element of matrix."
  (let* ((c (make-matrix (num-rows matrix) (num-cols matrix))))
    (dotimes (i (num-rows matrix))
      (dotimes (j (num-rows matrix))
        (setf (aref c i j) (- scalar (aref matrix i j)))))
    c))

(defun matrix-scalar-multiply (matrix scalar)
  "Scale the elementns of matrix by scalar: Returns c[i,j] = matrix[i,j] * scalar."
  (let* ((c (make-matrix (num-rows matrix) (num-cols matrix))))
    (dotimes (i (num-rows matrix))
      (dotimes (j (num-rows matrix))
        (setf (aref c i j) (* scalar (aref matrix i j)))))
    c))

(defun matrix-power (matrix power)
  "Compute the elements of matrix MATRIX raised to power POWER."
  (let* ((c (make-matrix (num-rows matrix) (num-cols matrix))))
    (dotimes (i (num-rows matrix))
      (dotimes (j (num-rows matrix))
        (setf (aref c i j) (expt (aref matrix i j) power))))
    c))

(defun transpose-matrix (matrix)
  "Return the transpose of matrix MATRIX."
  (let* ((new-matrix (make-matrix (num-cols matrix) (num-rows matrix))))
    (dotimes (i (num-rows matrix))
      (dotimes (j (num-cols matrix))
        (setf (aref new-matrix j i) (aref matrix i j))))
    new-matrix))

(defun matrix-row-sum (matrix)
  "Sum the elements on each row of matrix MATRIX"
  (let* ((new-matrix (make-matrix (num-rows matrix) 1)))
    (dotimes (i (num-rows matrix))
      (dotimes (j (num-cols matrix))
        (incf (aref new-matrix i 0) (aref matrix i j))))
    new-matrix))

(defun matrix-col-sum (matrix)
  "Sum the elements on each column of matrix MATRIX"
  (let* ((new-matrix (make-matrix (num-cols matrix) 1)))
    (dotimes (j (num-cols matrix))
      (dotimes (i (num-rows matrix))
        (incf (aref new-matrix j 0) (aref matrix i j))))
    new-matrix))

(defun vector-vector-dot-product (a b)
  "Return the dotproduct: sum of the products of components in vectors a and b"
  (loop :for ai :across a :for bi :across b :summing (* ai bi) :into sum
        :finally (return sum)))

(defun layer-sizes (x y &optional (n-h 4))
  "Return the neural network layer sizes for input, hidden, and output layers,

   Arguments:
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
  "Perform initialization of the layer weights and bias.

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
  "Perform forward propagation.

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
  "Compute the cross-entropy cost.

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
  "Perform backward propagation.

   Arguments:
   parameters - hashmap containing the weights and biases
   cache - hashtable containing z1, a1, z2, a2
   x - input data of shape(nx, number of examples)
   y - true labels vector of shape (1, number of exapmles)

   Returns:
   grads - hashtable containing gradients with respect to different parameters
  "
  (let* ((m (num-cols x))
         ;;(w1 (gethash "w1" parameters))
         (w2 (gethash "w2" parameters))
         (a1 (gethash "a1" cache))
         (a2 (gethash "a2" cache))
         ;; backward propagation to calculate dw1, dw2, db1, db2, dz1, dz2
         (dz2 (matrix-matrix-subtract a2 y))
         (dw2 (matrix-scalar-multiply (matrix-matrix-multiply dz2 (transpose-matrix a1)) (/ 1.0 m)))
         (db2 (matrix-scalar-multiply (matrix-row-sum dz2) (/ 1.0 m)))
         (dz1 (matrix-matrix-elementwise-multiply (matrix-matrix-multiply (transpose-matrix w2) dz2)
                                                  (scalar-matrix-subtract 1.0 (matrix-power a1 2.0))))
         (dw1 (matrix-scalar-multiply (matrix-matrix-multiple dz1 (transpose-matrix x))
                                      (/ 1.0 m)))
         (db1 (matrix-scalar-multiply (matrix-row-sum dz1) (/ 1.0 m)))
         (grads (make-hash-table :test 'equal)))
    ;; update the gradients hashtable
    (setf (gethash "dw1" grads) dw1
          (gethash "db1" grads) db1
          (gethash "dw2" grads) dw2
          (gethash "db2" grads) db2)
    ;; return gradients hashtable
    grads))

;;; Update parameters using gradient descent procedure
(defun update-parameters-one-hidden-layer (parameters grads &optional (learning-rate 1.2))
  "Update parameters using the gradient descent: x = x - (learning_rate * dL/dx).

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

;;; Neural network model with one hidden layer
(defun nn-model-one-hidden-layer (x y n-h &optional (num-iterations 10000) (print-cost nil))
  "Run the one-hidden layer neural newtwork model.

  Arguments:
  X -- dataset of shape (n_x, number of examples) (n_x dimension of each sample)
  Y -- labels of shape (1, number of examples
  n-h -- number of units in hiden layer
  num-iterations -- number of iterations in gradient descent loop
  print-cost -- if T, print the cost every 1000 iterations"
  (let* (dummy n-x n-y parameters a2 cache cost grads) 
    (setf (values n-x dummy n-y) (layer-sizes x y n-h))
    ;; Initialize parameters
    (setf parameters (initialize-parameters-one-hidden-layer n-x n-h n-y))
    ;; Loop for gradient descent
    (loop :for i from 0 :below num-iterations :do
             (setf (values a2 cache) (forward-propagation-one-hidden-layer x parameters))
             (setf cost (compute-cost-one-hidden-layer a2 y))
             (setf grads (backward-propagation-one-hidden-layer parameters cache x y))
             (setf parameters (update-parameters parameters grads))
             (when (and print-cost (zerop (mod i 1000)))
               (format t "~&Cost after iteration ~d: ~18,12f" i cost)))
    ;; Return final parameters
    parameters))
