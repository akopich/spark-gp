package org.apache.spark.ml.regression.kernel

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.spark.ml.linalg.{Vector, Vectors}


trait Kernel {
  var hyperparameters: BDV[Double]

  def setHyperparameters(value: BDV[Double]): this.type = {
    hyperparameters = value
    this
  }

  def setTrainingVectors(vectors: Array[Vector]): this.type

  def trainingKernel(): BDM[Double]

  def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]])

  def crossKernel(test: Array[Vector]): BDM[Double]
}

class TrainingVectorsNotInitializedException
  extends Exception("setTrainingVectors method should have been called first")

class RBFKernel(sigma: Double) extends Kernel {
  var hyperparameters : BDV[Double] = BDV[Double](sigma)

  private def getSigma() = hyperparameters(0)

  private var squaredDistances: Option[BDM[Double]] = None

  private var trainOption: Option[Array[Vector]] = None

  def this() = this(1)

  override def setTrainingVectors(vectors: Array[Vector]): this.type = {
    trainOption = Some(vectors)
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

  override def crossKernel(test: Array[Vector]): BDM[Double] = {
    val train = trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    val values = train.flatMap(trainVector =>
      test.map(testVector =>
        Vectors.sqdist(trainVector, testVector)/ (-2d * getSigma()*getSigma()))
    )

    exp(BDM.create(test.length, train.length, values))
  }
}