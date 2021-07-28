(uiop/package:define-package :src/activations/activations
    (:use :cl)
  (:use-reexport :src/array/array)
  (:export #:tanh-activation
           #:relu-activation
           #:sigmoid-activation
           #:softmax-activation))

(in-package :src/activations/activations)

(defun tanh-activation (matrix)
  "Compute hyperbolic tangent function tanh(x) where x is an element of matrix MATRIX."
  (let* ((rows (num-rows matrix))
         (cols (num-cols matrix))
         (new-matrix (make-matrix rows cols)))
    (dotimes (j cols)
      (dotimes (i rows)
        (setf (aref (aref new-matrix i) j) (/ (- (exp (aref (aref matrix i) j))
                                                 (exp (- (aref (aref matrix i) j))))
                                              (+ (exp (aref (aref matrix i) j))
                                                 (exp (- (aref (aref matrix i) j))))))))
    new-matrix))

(defun relu-activation (matrix)
  "Compute the RELU(x) where x is an element of matrix MATRIX."
  (let* ((rows (num-rows matrix))
         (cols (num-cols matrix))
         (new-matrix (make-matrix rows cols)))
    (dotimes (j cols)
      (dotimes (i rows)
        (setf (aref (aref new-matrix i) j) (if (< (aref (aref matrix i) j) 0.0) 0.0 (aref (aref matrix i) j)))))
    new-matrix))

(defun sigmoid-activation (matrix)
  "Implements the logistic/sigmoid function"
  (let* ((rows (num-rows matrix))
         (cols (num-cols matrix))
         (new-matrix (make-matrix rows cols)))
    (dotimes (i rows)
      (dotimes (j cols)
        (setf (aref (aref new-matrix i) j) (/ 1 (+ 1 (exp (- (aref (aref matrix i) j))))))))
    new-matrix))

(defun softmax-activation (matrix)
  "Implements the softmax activation function"
  (let* ((rows (num-rows matrix))
         (cols (num-cols matrix))
         (new-matrix (make-matrix rows cols))
         (sum 0.0))
    (dotimes (j cols)
      (setf sum 0.0)
      (dotimes (i rows)
        (incf sum (exp (aref (aref matrix i) j))))
      (dotimes (i rows)
        (setf (aref (aref new-matrix i) j) (/ (exp (aref (aref matrix i) j)) sum))))
    new-matrix))
