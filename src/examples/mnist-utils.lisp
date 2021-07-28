(uiop/package:define-package :src/examples/mnist-utils
    (:use :cl)
  (:nicknames :mnist)
  (:use-reexport :src/array/array)
  (:use-reexport :src/architecture/one-hidden-layer-network)
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
  (if (probe-file filename)
      (let* ((data (cl-simple-table:read-csv filename))
             (start (if skip-firstrow-p 1 0))
             (end (length data)))
        (map 'vector #'identity
             (loop :for i :from start :below end
                   :do
                      (when (zerop (mod i 1000))
                        (format t "~% ~d of ~d" i (- end start)))
                   :collect
                   (map 'vector #'parse-integer (aref data i)))))
      (warn "Filename ~s does not exist" filename)))


(defparameter *train-xdata-filename*
  (uiop:merge-pathnames* "data/datasets/machine-learning/mnist/mnist_train_xdata.csv"
                         (user-homedir-pathname)))
(defparameter *train-ydata-filename*
  (uiop:merge-pathnames* "data/datasets/machine-learning/mnist/mnist_train_ydata.csv"
                         (user-homedir-pathname)))
(defparameter *test-xdata-filename*
  (uiop:merge-pathnames* "data/datasets/machine-learning/mnist/mnist_test_xdata.csv"
                         (user-homedir-pathname)))
(defparameter *test-ydata-filename*
  (uiop:merge-pathnames* "data/datasets/machine-learning/mnist/mnist_test_ydata.csv"
                         (user-homedir-pathname)))

(print "Reading training and testing data ...")
;; Read the training data and split it later to training and validation sets (16.667%)
(print "Reading training x data")
(defvar *all-train-x* (read-mnist-data *train-xdata-filename*))
(print "Reading training y data")
(defvar *all-train-y* (read-mnist-data *train-ydata-filename*))

;; Read the test data
(print "Reading test x data")
(defvar *test-x* (read-mnist-data *test-xdata-filename*))
(print "Reading test y data")
(defvar *test-y* (read-mnist-data *test-ydata-filename*))

(defun scale-image-data (data-row &optional (scaler (/ 1.0 255.0)))
  (map 'vector #'(lambda (x) (* scaler x)) data-row))

(defun run-mnist-classification ()
  (let* ((train-x) (train-y)
         (valid-x) (valid-y))
    (setf (values train-x train-y valid-x valid-y) (train-test-split *all-train-x* *all-train-y* 0.16667))
    (format t "~%Length of train-x: ~d" (length train-x))
    (format t "~%Length of valid-x: ~d" (length valid-x))
    ;; scale training and validation data
    (setf train-x (map 'vector #'scale-image-data train-x)
          valid-x (map 'vector #'scale-image-data valid-x))
    ;; one-hot-encode the y-data
    (setf train-y (map 'vector #'(lambda (x) (one-hot-encode x 10)) train-y)
          valid-y (map 'vector #'(lambda (x) (one-hot-encode x 10)) valid-y))
    (nn-model-one-hidden-layer train-x train-y 300 :valid-x valid-x :valid-y valid-y :print-output-p t)))


