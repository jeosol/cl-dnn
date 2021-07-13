(defpackage cl-dnn/tests/main
  (:use :cl
        :cl-dnn
        :rove))
(in-package :cl-dnn/tests/main)

;; NOTE: To run this test file, execute `(asdf:test-system :cl-dnn)' in your Lisp.

(deftest test-target-1
  (testing "should (= 1 1) to be true"
    (ok (= 1 1))))
