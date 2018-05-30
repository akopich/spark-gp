package org.apache.spark.ml.regression

package object kernel {
  implicit def toScalar(C: Double) = new Scalar(C)

  implicit def toSummable(kernel: Kernel) = new SummableKernel(kernel)
}
