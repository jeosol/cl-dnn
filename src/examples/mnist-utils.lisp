(uiop/package:define-package :src/examples/mnist-utils
    (:use :cl)
  (:nicknames :mnist)
  (:use-reexport :src/array/array)
  (:export #:read-mnist-xdata
           #:run-mnist-classification))

(in-package :src/examples/mnist-utils)

(defun read-mnist-xdata (filename)
  (cl-csv:read-csv filename
                   :skip-first-p t
                   :separator #\,
                   :map-fn #'(lambda (row)
                               (map 'vector #'parse-integer row))))

(defun read-mnist-data (filename &optional (skip-firstrow-p t))
  (let* ((data (cl-simple-table:read-csv filename))
         (start (if skip-firstrow-p 1 0))
         (end (length data)))
    (map 'vector #'identity
         (loop :for i :from start :below end
               :collect
               (map 'vector #'parse-integer (aref data i))))))

(print "Reading training and testing data ...")
(defparameter *train-xdata-filename*
  (uiop:merge-pathnames* "/data/datasets/machine-learning/mnist/mnist/mnist_train_xdata.csv"
                         (user-homedir-pathname)))
(defparameter *train-ydata-filename*
  (uiop:merge-pathnames* "/data/datasets/machine-learning/mnist/mnist/mnist_train_ydata.csv"
                         (user-homedir-pathname)))
(defparameter *test-xdata-filename*
  (uiop:merge-pathnames* "/data/datasets/machine-learning/mnist/mnist/mnist_test_xdata.csv"
                         (user-homedir-pathname)))
(defparameter *test-ydata-filename*
  (uiop:merge-pathnames* "/data/datasets/machine-learning/mnist/mnist/mnist_test_ydata.csv"
                         (user-homedir-pathname)))

;; Read the training data and split it later to training and validation sets (16.667%)
(defvar *all-train-x* (read-mnist-data *train-xdata-filename*))
(defvar *all-train-y* (read-mnist-data *train-ydata-filename*))

;; Read the test data
(defvar *test-x* (read-mnist-data *train-xdata-filename*))
(defvar *test-y* (read-mnist-data *train-ydata-filename*))

(defun run-mnist-classification ()
  (let* ((train-x) (train-y)
         (valid-x) (valid-y))
    (setf (values train-x train-y valid-x valid-y) (train-test-split 0.16667))))


