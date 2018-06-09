package org.apache.spark.ml.commons.kernel
import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import org.apache.spark.ml.linalg

/**
  * The class implements a kernel which is a sum of kernels.
  *
  * k'(x_1, x_2) = k_1(x_1, x_2) + k_2(x_1, x_2)
  *
  * The kernels are assumed to have no shared hyperparameters.
  *
  * @param kernel1
  * @param kernel2
  */
class SumOfKernels(private val kernel1: Kernel,
                   private val kernel2: Kernel) extends Kernel {
  private def concat(v: BDV[Double], u: BDV[Double]) = BDV[Double](v.toArray ++ u.toArray)

  override def getHyperparameters: BDV[Double] =
    concat(kernel1.getHyperparameters, kernel2.getHyperparameters)

  override def setHyperparameters(value: BDV[Double]): SumOfKernels.this.type = {
    kernel1.setHyperparameters(value(0 until kernel1.numberOfHyperparameters))
    kernel2.setHyperparameters(value(kernel1.numberOfHyperparameters until value.length))
    this
  }

  override def numberOfHyperparameters: Int = kernel1.numberOfHyperparameters + kernel2.numberOfHyperparameters

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = {
    val (lower1, upper1) = kernel1.hyperparameterBoundaries
    val (lower2, upper2) = kernel2.hyperparameterBoundaries

    (concat(lower1, lower2), concat(upper1, upper2))
  }

  override def getTrainingVectors(): Array[linalg.Vector] = kernel1.getTrainingVectors

  override def setTrainingVectors(vectors: Array[linalg.Vector]): SumOfKernels.this.type = {
    kernel1.setTrainingVectors(vectors)
    kernel2.setTrainingVectors(vectors)
    this
  }

  override def trainingKernel(): DenseMatrix[Double] = kernel1.trainingKernel() += kernel2.trainingKernel()

  override def trainingKernelDiag(): Array[Double] =
    kernel1.trainingKernelDiag() zip kernel2.trainingKernelDiag() map{case(a,b) => a+b}

  override def trainingKernelAndDerivative(): (DenseMatrix[Double], Array[DenseMatrix[Double]]) = {
    val (k1, derivatives1) = kernel1.trainingKernelAndDerivative()
    val (k2, derivatives2) = kernel2.trainingKernelAndDerivative()

    (k1 += k2, derivatives1 ++ derivatives2)
  }

  override def crossKernel(test: Array[linalg.Vector]): DenseMatrix[Double] =
    kernel1.crossKernel(test) + kernel2.crossKernel(test)

  override def selfKernel(test: linalg.Vector): Double = kernel1.selfKernel(test) + kernel2.selfKernel(test)

  override def whiteNoiseVar: Double = kernel1.whiteNoiseVar + kernel2.whiteNoiseVar

  override def toString: String = List(kernel1, kernel2).map(_.toString).filter(_.length > 0).mkString(" + ")
}
