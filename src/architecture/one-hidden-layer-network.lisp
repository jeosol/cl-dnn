(uiop/package:define-package :src/architecture/one-hidden-layer-network
    (:use :cl)
  (:nicknames :two-layer-model)
  (:use-reexport :src/array/array)
  (:use-reexport :src/activations/activations)
  (:export #:nn-model-one-hidden-layer
           #:get-batch-start-end-indices))

(in-package :src/architecture/one-hidden-layer-network)

(defun layer-sizes (x y &optional (n-h 4))
  "Return the neural network layer sizes for input, hidden, and output layers,

   Arguments:
   X -- input dataset of shape (input size, number of examples)
   Y -- labels of shape (output size, number of examples

   Returns:
   n-x -- the size of the input layer
   n-h -- the size of the hidden layer
   n-y -- the size of the output layer"
  (let ((n-x (num-cols x)) ;; x is of shape (m n-x)
        (n-y (length (aref y 0))))
    ;(assert (= (num-rows x) n-y) nil "Different number of samples in X and Y.")
    (values n-x n-h n-y)))

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
  (let* ((w1 (make-random-matrix n-h n-x 0.01))
         (b1 (make-random-matrix n-h 1))
         (w2 (make-random-matrix n-y n-h 0.01))
         (b2 (make-random-matrix n-y 1))
         (parameters (make-hash-table :size 4 :test #'equal)))
    (setf (gethash "w1" parameters) w1
          (gethash "b1" parameters) b1
          (gethash "w2" parameters) w2
          (gethash "b2" parameters) b2)
    parameters))

;;; FORWARD PROPAGATION
(defun forward-propagation-one-hidden-layer (x parameters
                                             &optional (activation-functions '(relu-activation
                                                                               softmax-activation)))
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
         (b2 (gethash "b2" parameters)) 
         (z1 (matrix-matrix-add (matrix-matrix-multiply w1 x) b1))
         (a1 (funcall (nth 0 activation-functions) z1))
         (z2 (matrix-matrix-add (matrix-matrix-multiply w2 a1) b2))
         (a2 (funcall (nth 1 activation-functions) z2))
         (cache (make-hash-table :test 'equal)))
    (setf (gethash "z1" cache) z1
          (gethash "a1" cache) a1
          (gethash "z2" cache) z2
          (gethash "a2" cache) a2)
    (values a2 cache)))

(defun compute-cost-one-hidden-layer (a2 y)
  "Compute the cross-entropy cost.

   Arguments:
   A2 -- The sigmoid output of the scond activation of shape (1, number of examples)
   Y  -- 'True' labels vector of shape (1, number of examples)

   Returns:
   Cost -- cross-entropy"
  (format t "~&Computing cost: dimensions a2=~d, dimensions y=~d" (dimensions a2) (dimensions y))
  (let* ((m (num-cols y))
         (num-outputs (num-rows y))
         (sum 0.0))
    ;; (dotimes (i m)
    ;;   (dotimes (k num-outputs)
    ;;     ;(print (list i k m num-outputs (num-cols a2) (aref (aref y k) i) (aref (aref a2 k) i)))
    ;;     (incf sum (+ (* (aref (aref y k) i) (log (aref (aref a2 k) i)))
    ;;                  (* (- 1.0 (aref (aref y k) i)) (log (- 1.0 (aref (aref a2 k) i))))))))
    (dotimes (i m)
      (dotimes (k num-outputs)
        (incf sum (+ (* (aref (aref y k) i) (log (aref (aref a2 k) i)))))))
    (* (- (/ 1.0 m) sum))))

(defun backward-propagation-one-hidden-layer (parameters cache x y)
  "Perform backward propagation.

   Arguments:
   parameters - hashtable containing the weights and biases
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
         (a2 (gethash "a2" cache)) ;(dummy (print (list (print 'here) (dimensions a2) (dimensions y))))
         ;; backward propagation to calculate dw1, dw2, db1, db2, dz1, dz2
         (dz2 (matrix-matrix-subtract a2 y)) ;(dummy (print 'here2))
         ;(dummy (print (list "dim sz2" (dimensions dz2) "dim a1" (dimensions a1))))
         (dw2 (matrix-scalar-multiply (matrix-matrix-multiply dz2 (transpose-matrix a1)) (/ 1.0 m)))

         (db2 (matrix-scalar-multiply (matrix-row-sum dz2) (/ 1.0 m)))
         ;(dummy (print 'here))
         ;; (dummy (print (list "dim w2" (dimensions (transpose-matrix w2)) "dim dz2" (dimensions dz2)
         ;;                     "dim a1" (dimensions a1))))
         (dz1 (matrix-matrix-elementwise-multiply (matrix-matrix-multiply (transpose-matrix w2) dz2)
                                                  (scalar-matrix-subtract 1.0 (matrix-power a1 2.0))))
         ;; (dummy (print (list "dim dz1" (dimensions dz1))))
         ;; (dummy (print (list "dim x" (dimensions (transpose-matrix x)) "dim dz1" (dimensions dz1))))
         (dw1 (matrix-scalar-multiply (matrix-matrix-multiply dz1 (transpose-matrix x))
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
    (setf w1 (matrix-matrix-subtract w1 (matrix-scalar-multiply dw1 learning-rate))
          b1 (matrix-matrix-subtract b1 (matrix-scalar-multiply db1 learning-rate))
          w2 (matrix-matrix-subtract w2 (matrix-scalar-multiply dw2 learning-rate))
          b2 (matrix-matrix-subtract b2 (matrix-scalar-multiply db2 learning-rate)))
    ;; save the updated parameters in the hash and return
    (setf (gethash "w1" parameters) w1
          (gethash "b1" parameters) b1
          (gethash "w2" parameters) w2
          (gethash "b2" parameters) b2)
    parameters))

;;; Determine batch parameters
(defun get-batch-start-end-indices (num-samples batch-size)
  "Return a list with elementthe start and end indices"
  (let* ((num-batches (floor (/ num-samples batch-size)))
         (start-end-indices))
    (loop :for i :from 0 :below num-batches
          :for start :from 0 :by batch-size
          :for end   :from batch-size by batch-size
          :do
             (push (list start end) start-end-indices))
    (unless (= (* num-batches batch-size) num-samples)
      (push (list (* num-batches batch-size) num-samples)
            start-end-indices)
      (incf num-batches))
    (values (reverse start-end-indices) num-batches)))

;; Compute accuracy
(defun compute-accuracy (predicted-labels actual-labels)
  "Compute the accuracy of the predicted labels

   Arguments:
   predicted-labels -- labels as predicted by the network
   true-labels -- 'True' labels vector of shape (1, number of examples)

   Returns:
   accuracy -- accuracy expressed as fraction"
  ;; (format t "~%Computing accuracy: dimensions predicted-labels=~d, dimensions actual-labels~d"
  ;;         (dimensions predicted-labels) (dimensions actual-labels))
  (assert (= (num-cols predicted-labels) (num-cols actual-labels))
          nil "The length of predicted and actual label data are not equal")
  (let* ((m (num-cols predicted-labels))
         (num-outputs (num-rows predicted-labels))
         (correct 0)
         (pred-label) (true-label))
    (dotimes (i m)
      (setf pred-label (argmax (map 'vector #'identity
                                    (loop :for k :from 0 :below num-outputs
                                          :collect (aref (aref predicted-labels k) i))))
            true-label (argmax (map 'vector #'identity
                                    (loop :for k :from 0 :below num-outputs
                                          :collect (aref (aref actual-labels k) i)))))
      ;(print (list "accuracy-info = " i pred-label true-label))
      (when (= pred-label true-label)
        (incf correct)))
    (float  (/ correct m))))


;;; Neural network model with one hidden layer
(defun nn-model-one-hidden-layer (train-x train-y n-h &key
                                                        (batch-size 64) (num-epochs 100)
                                                        valid-x valid-y
                                                        (print-output-p nil))
  "Run the one-hidden layer neural newtwork model.
  Arguments:
  X              -- dataset of shape (n_x, number of examples) (n_x dimension of each sample)
  Y              -- labels of shape (1, number of examples
  n-h            -- number of units in hiden layer
  num-iterations -- number of iterations in gradient descent loop
  print-cost     -- if T, print the cost every 1000 iterations"
  (let* (dummy n-x n-y parameters a2 cache cost grads cost-history
         batch-indices (mini-batch-counter 0) (epoch 1)
         (num-batches 0)
         (num-data (num-rows train-x))
         (iter-counter 0)
         shuffle-indices
         (learning-rate)
         (learning-rate-init 0.199) (decay-rate 0.999) (decay-steps 10)
         (validation-accuracy nil)
         (valid-x (transpose-matrix valid-x))
         (valid-y (transpose-matrix valid-y)))
    ;; Get dimensions for the input and output layers
    (setf (values n-x dummy n-y) (layer-sizes train-x train-y n-h))
    ;; Get shuffled indices
    (setf shuffle-indices (map 'vector #'identity (loop :for i :from 0 :below num-data :collect i)))
    ;; Determine the batch start and end indices
    (setf (values batch-indices num-batches) (get-batch-start-end-indices num-data batch-size))
    (format t "~&Number of batches = ~d" (length batch-indices))
    ;; Initialize parameters
    (setf parameters (initialize-parameters-one-hidden-layer n-x n-h n-y))
    ;; Loop over the training data
    (loop :for k :from 1 :upto num-epochs :do
      ;; Generate one vector represented shuffle indices of the training data
      (setf shuffle-indices (alexandria:shuffle shuffle-indices))
      ;; Do batch gradient descent
      (loop :for (start-index end-index) :in batch-indices
            :for batch-counter :from 1
            :for batch-x = (transpose-matrix (slice-data train-x start-index end-index shuffle-indices))
            :for batch-y = (transpose-matrix (slice-data train-y start-index end-index shuffle-indices))
            :do
               ;(print (length batch-x))
               (format t "~&Batch information: start=~6d, end=~6d: dimensions: batch-x=~s, batch-y=~s"
                       start-index end-index (num-cols batch-x) (num-cols batch-y))
               ;; increment iteration counter
               (incf iter-counter)
               ;; Forward propagation step
               (setf (values a2 cache) (forward-propagation-one-hidden-layer batch-x parameters))
               ;; Compute the cost
               (setf cost (compute-cost-one-hidden-layer a2 batch-y))
               (push (list epoch (incf mini-batch-counter) cost) cost-history)
               ;; Backward propagation step
               (setf grads (backward-propagation-one-hidden-layer parameters cache batch-x batch-y))
               ;; reduce the learning parameter
               ;;(setf learning-rate (* learning-rate-init (/ 1.0 (+ 1.0 (* decay-rate iter-counter)))))
               (setf learning-rate (* learning-rate-init (expt decay-rate (/ iter-counter decay-steps))))
               ;; Update parameters
               (setf parameters (update-parameters-one-hidden-layer parameters grads learning-rate))
               ;; compute accuracy of the validation data using the current network parameters
               (when (and valid-x valid-y)
                 (let* ((predictions))
                   (setf predictions (forward-propagation-one-hidden-layer valid-x parameters))
                   (setf validation-accuracy (compute-accuracy predictions valid-y))))
               (when print-output-p
                 (format t "~&Epoch ~5d of ~5d: Iteration = ~5d, Batch ~5d of ~5d, LR=~7,4f, Cost=~18,12f~a~&"
                         k num-epochs iter-counter batch-counter num-batches learning-rate cost
                         (if validation-accuracy
                             (format nil ", Accuracy = ~6,4f" validation-accuracy) "")))))
    ;; Return final parameters
    ;(values parameters (reverse cost-history))
    ))

;; TODO
;; Create example folders
;; Create option to decay the learning rate
;; Add option to including different gradient-based methods (e.g., momentum, RMS, Adam)
;;
;; Simple test with XOR data
(defparameter *xor-xdata* #(#(0.0 0.0)
                            #(0.0 1.0)
                            #(1.0 0.0)
                            #(1.0 1.0)))

(defvar *xor-ydata* #(#(0) #(1) #(1) #(0)))

(defun test-xor (&optional (num-epochs 10))
  (nn-model-one-hidden-layer *xor-xdata* *xor-ydata* 10 :num-epochs num-epochs  :print-cost t))

