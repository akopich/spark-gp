package org.apache.spark.ml.regression.kernel
import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import breeze.numerics._
import org.apache.spark.ml.linalg

class ScalarTimesKernel(private val kernel: Kernel,
                        private var C: Double,
                        private val Clower: Double = 1e-6,
                        private val Cupper: Double = inf) extends Kernel {

  override def getHyperparameters: BDV[Double] = prependToVector(C, kernel.getHyperparameters)

  override def setHyperparameters(value: BDV[Double]): ScalarTimesKernel.this.type = {
    C = value(0)
    kernel.setHyperparameters(value(1 until value.length))
    this
  }

  override def numberOfHyperparameters: Int = 1 + kernel.numberOfHyperparameters

  private def prependToVector(c : Double, v : BDV[Double]) = BDV[Double](c +: v.data)

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = {
    val (lower, upper) = kernel.hyperparameterBoundaries
    (prependToVector(Clower, lower), prependToVector(Cupper, upper))
  }

  override def getTrainingVectors: Option[Array[linalg.Vector]] = kernel.getTrainingVectors

  override def setTrainingVectors(vectors: Array[linalg.Vector]): ScalarTimesKernel.this.type = {
    kernel.setTrainingVectors(vectors)
    this
  }

  override def trainingKernel(): DenseMatrix[Double] = kernel.trainingKernel() * C

  override def trainingKernelDiag(): Array[Double] = kernel.trainingKernelDiag().map(_ * C)

  override def trainingKernelAndDerivative(): (DenseMatrix[Double], Array[DenseMatrix[Double]]) = {
    val (kernelMatrix, derivative) = kernel.trainingKernelAndDerivative()

    (kernelMatrix * C, kernelMatrix +: derivative.map(_ * C))
  }

  override def crossKernel(test: Array[linalg.Vector]): DenseMatrix[Double] = C * kernel.crossKernel(test)
}

class Scalar private[kernel](private val C : Double) extends AnyVal {
  def *(kernel : Kernel) = new ScalarTimesKernel(kernel, C)
}

object ScalarTimesKernel {
  implicit def toScalar(C: Double) = new Scalar(C)
}