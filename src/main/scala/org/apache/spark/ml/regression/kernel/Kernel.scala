package org.apache.spark.ml.regression.kernel

import breeze.linalg.{DenseMatrix => DBM}
import breeze.numerics._

import org.apache.spark.ml.linalg.{Vector, Vectors}


trait Kernel {
  var hyperparameters: Vector

  def setTrainingVectors(vectors: Array[Vector]): Unit

  def trainingKernel(): DBM[Double]

  def derivative(): Array[DBM[Double]]
}

class TrainingVectorsNotInitializedException
  extends Exception("setTrainingVectors method should have been called first")

class RBFKernel(sigma: Double) extends Kernel {
  var hyperparameters : Vector = Vectors.dense(Array(sigma))

  private def getSigma() = hyperparameters(0)

  private var squaredDistances: Option[DBM[Double]] = None

  def this() = this(1)

  override def setTrainingVectors(vectors: Array[Vector]): Unit = {
    val sqd = DBM.zeros[Double](vectors.length, vectors.length)
    for (i <- vectors.indices; j <- 0 to i) {
      val dist = Vectors.sqdist(vectors(i), vectors(j))
      sqd(i, j) = dist
      sqd(j, i) = dist
    }

    squaredDistances = Some(sqd)
  }

  override def trainingKernel(): DBM[Double] = {
    exp(squaredDistances.getOrElse(throw new TrainingVectorsNotInitializedException)
      / (-2d * getSigma()*getSigma()))
  }

  override def derivative(): Array[DBM[Double]] = {
    val sqd = squaredDistances.getOrElse(throw new TrainingVectorsNotInitializedException)

    val derivative = (sqd *:* trainingKernel()) / (getSigma() * getSigma() * getSigma())

    Array(derivative)
  }
}