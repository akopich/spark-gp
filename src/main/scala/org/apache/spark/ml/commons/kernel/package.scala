package org.apache.spark.ml.commons

package object kernel {
  implicit def toScalar(C: Double) = new Scalar(C)

  implicit class SummableKernel private[kernel](private val kernel: Kernel) {
    def +(other: Kernel) = new SumOfKernels(kernel, other)
  }
}
