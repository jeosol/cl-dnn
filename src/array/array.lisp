(uiop/package:define-package :src/array/array
    (:use :cl)
  (:export #:make-vector
           #:make-matrix
           #:make-random-matrix
           #:num-rows
           #:num-cols
           #:rand-double
           #:rand-int
           #:matrix-matrix-multiply
           #:matrix-matrix-elementwise-multiply
           #:matrix-matrix-add
           #:matrix-matrix-subtract
           #:matrix-scalar-multiply
           #:matrix-power
           #:transpose-matrix
           #:matrix-row-sum
           #:matrix-col-sum
           #:vector-vector-dot-product
           #:shuffle-sequence
           #:generate-random-sequence))

(in-package :src/array/array)

;; 23:19pm July 23, 2021
;; Comment:
;; Change matrix format from using (make-array '(num-rows num-cols) ...)
;; to use vector of vector format,
;; The new matrix representation is more flexible as it allows us to reshuffle
;; the rows easily and to extract the training data for a given instance easily

;; However, we will now need to use a slighly more verbose array access syntax: (aref (aref matrix i) j)
;; to access  the element at location (i,j) instead of (aref matrix i j)
;;

(defun make-vector (num-elements)
  "Create a vector with NUM-ELEMENTS."
  (make-array num-elements :element-type 'single-float :initial-element 0.0))

(defun make-matrix (num-rows num-cols)
  "Create a matrix of size NUM-ROWS by NUM-COLS."
  (let* ((matrix (make-array num-rows)))
    (dotimes (i num-rows)
      (setf (aref matrix i) (make-vector num-cols)))
    matrix))

(defun num-rows (matrix)
  "Return the number of rows in matrix MATRIX"
  (length matrix))

(defun num-cols (matrix)
  "Return the number of rows in matrix MATRIX."
  (length (aref matrix 0)))

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

(defun make-random-matrix (num-rows num-cols &optional (scale 1.0))
  "Create a random matrix with elements initialized to random numbers in [0,1]."
  (let* ((matrix (make-matrix num-rows num-cols)))
    (dotimes (i num-rows)
      (dotimes (j num-cols)
        (setf (aref (aref matrix i) j) (* (random 1.0) scale))))
    matrix))

(defun matrix-matrix-multiply (a b)
  "Compute the product of matrices: C = A*B."
  (let* ((c (make-matrix (num-rows a) (num-cols b)))
         (sum 0.0))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols b))
        (setf sum 0.0)
        (dotimes (k (num-cols a))
          (incf sum (* (aref (aref a i) k) (aref (aref b k) j))))
        (setf (aref (aref c i) j) sum)))
    c))

(defun matrix-matrix-elementwise-multiply (a b)
  "Perform elementwise multiplication of two matrices: c[i,j] = a[i,j] * b[i,j]."
  (let* ((c (make-matrix (num-rows a) (num-rows b))))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref (aref c i) j) (* (aref (aref a i) j) (aref (aref b i) j)))))
    c))

(defun matrix-matrix-add (a b)
  "Perform elementwise addition of two matrices: c[i,j] = a[i,j] + b[i,j]."
  (let* ( (c (make-matrix (num-rows a) (num-cols a))))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref (aref c i) j) (+ (aref (aref a i) j) (aref (aref b i) j)))))
    c))

(defun matrix-matrix-subtract (a b)
  "Perform elementwise subtraction of two matrices: c[i,j] = a[i,j] - b[i,j]."
  (let* ( (c (make-matrix (num-rows a) (num-cols a))))
    (dotimes (i (num-rows a))
      (dotimes (j (num-cols a))
        (setf (aref (aref c i) j) (- (aref (aref a i) j) (aref (aref b i) j)))))
    c))

(defun scalar-matrix-subtract (scalar matrix)
  "Compute c[i,j] = scalar - matrix[i,j] for each element of matrix."
  (let* ((c (make-matrix (num-rows matrix) (num-cols matrix))))
    (dotimes (i (num-rows matrix))
      (dotimes (j (num-rows matrix))
        (setf (aref (aref c i) j) (- scalar (aref (aref matrix i) j)))))
    c))

(defun matrix-scalar-multiply (matrix scalar)
  "Scale the elementns of matrix by scalar: Returns c[i,j] = matrix[i,j] * scalar."
  (let* ((c (make-matrix (num-rows matrix) (num-cols matrix))))
    (dotimes (i (num-rows matrix))
      (dotimes (j (num-rows matrix))
        (setf (aref (aref c i) j) (* scalar (aref (aref matrix i) j)))))
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
        (setf (aref (aref new-matrix j) i) (aref (aref matrix i) j))))
    new-matrix))

(defun matrix-row-sum (matrix)
  "Sum the elements on each row of matrix MATRIX"
  (let* ((new-matrix (make-matrix (num-rows matrix) 1)))
    (dotimes (i (num-rows matrix))
      (dotimes (j (num-cols matrix))
        (incf (aref (aref new-matrix i) 0) (aref (aref matrix i) j))))
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

(defun shuffle-sequence (sequence)
  "Return a shufle sequence"
  (alexandria:shuffle sequence))

(defun generate-random-sequence (length)
  "Generate a random sequence"
  (alexandria:shuffle (map 'vector #'identity (loop :for i :from 0 :below length :collect i))))
