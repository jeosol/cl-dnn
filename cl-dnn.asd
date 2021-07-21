(defsystem "cl-dnn"
  :version "0.1.0"
  :author "Jerome E. Onwunalu, PhD <jeronwunalu@gmail.com>"
  :license "Thedibia LLC"
  :depends-on ()
  :components ((:module "src"
                :components
                ((:file "main")
                 (:module "array"
                  :components
                  ((:file "array")))
                 (:module "activations"
                  :components
                  ((:file "activations")))
                 (:module "architecture"
                  :components
                  ((:file :one-hidden-layer-network))))))
  :description "CL-DNN - Implementation of Deep Neural Networks using Common Lisp"
  :in-order-to ((test-op (test-op "cl-dnn/tests"))))

(defsystem "cl-dnn/tests"
  :author "Jerome E. Onwunalu, PhD <jeronwunalu@gmail.com>"
  :license "Thedibia LLC"
  :depends-on ("cl-dnn"
               "rove")
  :components ((:module "tests"
                :components
                ((:file "main"))))
  :description "Test system for cl-dnn"
  :perform (test-op (op c) (symbol-call :rove :run c)))
