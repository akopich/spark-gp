package org.apache.spark.ml.commons.kernel

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.{exp, inf}
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * Implements traditional RBF kernel `k(x_i, k_j) = exp(||x_i - x_j||^2 / sigma^2)`
  *
  * @param sigma
  * @param lower
  * @param upper
  */
class RBFKernel(private var sigma: Double,
                private val lower: Double = 1e-6,
                private val upper: Double = inf) extends TrainDatasetBearingKernel
  with NoiselessKernel with SameOnDiagonalKernel {
  def this() = this(1)

  override def setHyperparameters(value: BDV[Double]): RBFKernel.this.type = {
    sigma = value(0)
    this
  }

  override def getHyperparameters: BDV[Double] = BDV[Double](sigma)

  override def numberOfHyperparameters: Int = 1

  private def getSigma() = sigma

  private var squaredDistances: Option[BDM[Double]] = None

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = {
    (BDV[Double](lower), BDV[Double](upper))
  }

  override def setTrainingVectors(vectors: Array[Vector]): this.type = {
    super.setTrainingVectors(vectors)
    val sqd = BDM.zeros[Double](vectors.length, vectors.length)
    for (i <- vectors.indices; j <- 0 to i) {
      val dist = Vectors.sqdist(vectors(i), vectors(j))
      sqd(i, j) = dist
      sqd(j, i) = dist
    }

    squaredDistances = Some(sqd)
    this
  }

  override def trainingKernel(): BDM[Double] = {
    val result = squaredDistances.getOrElse(throw new TrainingVectorsNotInitializedException) / (-2d * sqr(getSigma()))
    exp.inPlace(result)
    result
  }

  override def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]]) = {
    val sqd = squaredDistances.getOrElse(throw new TrainingVectorsNotInitializedException)

    val kernel = trainingKernel()
    val derivative = sqd *:* kernel
    derivative /= cube(getSigma())

    (kernel, Array(derivative))
  }

  override def crossKernel(test: Array[Vector]): BDM[Double] = {
    val train = getTrainingVectors
    val result = BDM.zeros[Double](test.length, train.length)

    for (i <- test.indices; j <- train.indices)
      result(i, j) = Vectors.sqdist(test(i), train(j)) / (-2d * sqr(getSigma()))

    exp.inPlace(result)

    result
  }

  override def selfKernel(test: Vector): Double = 1d

  private def sqr(x: Double) = x * x

  private def cube(x: Double) = x * x * x

  override def toString = f"RBFKernel(sigma=$sigma%1.1e)"
}
