package org.apache.spark.ml.regression.kernel
import breeze.linalg.{DenseMatrix, DenseVector => BDV}
import org.apache.spark.ml.linalg

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

  override def trainingKernel(): DenseMatrix[Double] = kernel1.trainingKernel() + kernel2.trainingKernel()

  override def trainingKernelDiag(): Array[Double] =
    kernel1.trainingKernelDiag() zip kernel2.trainingKernelDiag() map{case(a,b) => a+b}

  override def trainingKernelAndDerivative(): (DenseMatrix[Double], Array[DenseMatrix[Double]]) = {
    val (k1, derivatives1) = kernel1.trainingKernelAndDerivative()
    val (k2, derivatives2) = kernel2.trainingKernelAndDerivative()

    (k1 + k2, derivatives1 ++ derivatives2)
  }

  override def crossKernel(test: Array[linalg.Vector]): DenseMatrix[Double] =
    kernel1.crossKernel(test) + kernel2.crossKernel(test)

  override def iidNoise: Double = kernel1.iidNoise + kernel2.iidNoise
}

class SummableKernel private[kernel](private val kernel: Kernel) {
  def +(other: Kernel) = new SumOfKernels(kernel, other)
}

object SummableKernel {
  implicit def toSummable(kernel: Kernel) = new SummableKernel(kernel)
}
