package org.apache.spark.ml.regression.kernel

import breeze.linalg.{DenseMatrix => BDM}
import breeze.numerics._

import org.apache.spark.ml.linalg.{Vector, Vectors}


trait Kernel {
  var hyperparameters: Vector

  def setTrainingVectors(vectors: Array[Vector]): this.type

  def trainingKernel(): BDM[Double]

  def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]])
}

class TrainingVectorsNotInitializedException
  extends Exception("setTrainingVectors method should have been called first")

class RBFKernel(sigma: Double) extends Kernel {
  var hyperparameters : Vector = Vectors.dense(Array(sigma))

  private def getSigma() = hyperparameters(0)

  private var squaredDistances: Option[BDM[Double]] = None

  def this() = this(1)

  override def setTrainingVectors(vectors: Array[Vector]): this.type = {
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
    exp(squaredDistances.getOrElse(throw new TrainingVectorsNotInitializedException)
      / (-2d * getSigma()*getSigma()))
  }

  override def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]]) = {
    val sqd = squaredDistances.getOrElse(throw new TrainingVectorsNotInitializedException)

    val kernel = trainingKernel()
    val derivative = (sqd *:* kernel) / (getSigma() * getSigma() * getSigma())

    (kernel, Array(derivative))
  }
}