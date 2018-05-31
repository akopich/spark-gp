package org.apache.spark.ml.regression.kernel
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.spark.ml.linalg

trait ScalarTimesKernel extends Kernel {
  protected val kernel: Kernel

  protected def C: Double

  override def getTrainingVectors: Array[linalg.Vector] = kernel.getTrainingVectors

  override def setTrainingVectors(vectors: Array[linalg.Vector]): this.type = {
    kernel.setTrainingVectors(vectors)
    this
  }

  override def trainingKernel(): BDM[Double] = kernel.trainingKernel() *= C

  override def trainingKernelDiag(): Array[Double] = kernel.trainingKernelDiag().map(_ * C)

  override def crossKernel(test: Array[linalg.Vector]): BDM[Double] = kernel.crossKernel(test) *= C

  override def whiteNoiseVar: Double = C * kernel.whiteNoiseVar

  override def toString = if (C != 0) s"$C * $kernel" else ""
}

class ConstantTimesKernel(protected val kernel: Kernel, protected val C: Double) extends ScalarTimesKernel {
  override def getHyperparameters: BDV[Double] = kernel.getHyperparameters

  override def setHyperparameters(value: BDV[Double]): this.type = {
    kernel.setHyperparameters(value)
    this
  }

  override def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]]) = {
    val (kernelMatrix, derivative) = kernel.trainingKernelAndDerivative()

    (kernelMatrix * C, derivative.map(_ *= C))
  }

  override def numberOfHyperparameters: Int = kernel.numberOfHyperparameters

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = kernel.hyperparameterBoundaries
}

class TrainableScalarTimesKernel(protected val kernel: Kernel,
                                 protected var C: Double,
                                 private val Clower: Double = 0,
                                 private val Cupper: Double = inf) extends ScalarTimesKernel {

  override def getHyperparameters: BDV[Double] = prependToVector(C, kernel.getHyperparameters)

  override def setHyperparameters(value: BDV[Double]): TrainableScalarTimesKernel.this.type = {
    C = value(0)
    kernel.setHyperparameters(value(1 until value.length))
    this
  }

  override def numberOfHyperparameters: Int = 1 + kernel.numberOfHyperparameters

  private def prependToVector(c : Double, v : BDV[Double]) = BDV[Double](c +: v.toArray)

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = {
    val (lower, upper) = kernel.hyperparameterBoundaries
    (prependToVector(Clower, lower), prependToVector(Cupper, upper))
  }

  override def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]]) = {
    val (kernelMatrix, derivative) = kernel.trainingKernelAndDerivative()

    (kernelMatrix * C, kernelMatrix +: derivative.map(_ *= C))
  }
}

class Scalar private[kernel](private val C: Double,
                             private val lower: Double = 0,
                             private val upper: Double = inf,
                             private val isTrainable: Boolean = true) {

  require(lower < upper && isTrainable || !isTrainable,
    "The scalar should either have its lower limit below its upper limit or not be trainable")

  def *(kernel : Kernel) = if (isTrainable) new TrainableScalarTimesKernel(kernel, C, lower, upper)
                            else new ConstantTimesKernel(kernel, C)

  /**
    * Intended use is
    * `1 between 0 and 30`
    * which means "create a scalar with initial value of 1 which should be optimized in the range from 0 to 30"
    *
    * @param lower
    * @return
    */
  def between(lower: Double) = new AnyRef {
    private val low: Double = lower

    def and(upper: Double) = new Scalar(C, low, upper, isTrainable)
  }

  /**
    * Intended use is
    * `1 below 10`
    *
    * which means "create a scalar with initial value of 1 which should be optimized but never exceed 10"
    *
    * @param newUpper
    * @return
    */
  def below(newUpper: Double) = new Scalar(C, lower, newUpper, isTrainable)

  /**
    *
    * @return the same scalar but constant
    */
  def const = new Scalar(C, C, C, false)
}
