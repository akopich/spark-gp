package org.apache.spark.ml.commons.kernel

import breeze.linalg.{norm, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import breeze.numerics.{exp, inf}
import org.apache.spark.ml.linalg.Vector

/**
  * Automatic Relevance Determination Kernel.
  *
  * Is a straightforward generalization of RBF kernel.
  *
  * `k(x_i, k_j) = exp(||(x_i - x_j) \otimes beta||^2)`,
  *
  * where beta is a vector and \otimes stands for element-wise product
  *
  * @param beta  the vector of the same dimensionality as inputs
  * @param lower element-wise lower bound
  * @param upper element-upper lower bound
  */
class ARDRBFKernel(private var beta: BDV[Double],
                   private val lower: BDV[Double],
                   private val upper: BDV[Double]) extends TrainDatasetBearingKernel
  with NoiselessKernel with SameOnDiagonalKernel {

  def this(beta: BDV[Double]) = this(beta, beta * 0d, beta * inf)

  def this(p : Int, beta: Double = 1, lower: Double = 0, upper : Double = inf) =
    this(BDV.zeros[Double](p) + beta,
      BDV.zeros[Double](p) + lower,
      BDV.zeros[Double](p) + upper)

  override def setHyperparameters(value: BDV[Double]): ARDRBFKernel.this.type = {
    beta = value
    this
  }

  override def getHyperparameters: BDV[Double] = beta

  override def numberOfHyperparameters: Int = beta.length

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = (lower, upper)

  private def kernelElement(a: BV[Double], b: BV[Double]) : Double = {
    val weightedDistance = norm((a - b) *:* beta)
    exp(- weightedDistance * weightedDistance)
  }

  override def trainingKernel(): BDM[Double] = {
    val train = getTrainingVectors

    val result = BDM.zeros[Double](train.length, train.length)
    for (i <- train.indices; j <- 0 to i) {
      val k = kernelElement(train(i).asBreeze, train(j).asBreeze)
      result(i, j) = k
      result(j, i) = k
    }

    result
  }

  override def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]]) = {
    val train = getTrainingVectors
    val K = trainingKernel()
    val minus2Kernel = -2d * K
    val result = Array.fill[BDM[Double]](beta.length)(BDM.zeros[Double](train.length, train.length))

    for (i <- train.indices; j <- 0 to i) {
      val diff = train(i).asBreeze - train(j).asBreeze
      diff :*= diff
      diff :*= beta
      val betaXi_Xj = diff
      for (k <- 0 until beta.length) {
        result(k)(i, j) = betaXi_Xj(k)
        result(k)(j, i) = betaXi_Xj(k)
      }
    }

    (K, result.map(derivative => derivative *:* minus2Kernel))
  }

  override def crossKernel(test: Array[Vector]): BDM[Double] = {
    val train = getTrainingVectors
    val result = BDM.zeros[Double](test.length, train.length)

    for (testIndx <- test.indices; trainIndex <- train.indices)
      result(testIndx, trainIndex) = kernelElement(train(trainIndex).asBreeze, test(testIndx).asBreeze)

    result
  }

  override def selfKernel(test: Vector): Double = 1d

  override def toString = "ARDRBFKernel(beta=" + BDV2String(beta) + ")"

  private def BDV2String(v : BDV[Double]) = v.valuesIterator.map(e => f"$e%1.1e").mkString("[", ", " , "]")
}
