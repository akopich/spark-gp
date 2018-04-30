package org.apache.spark.ml.regression.kernel

import breeze.linalg.{Transpose, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics._
import org.apache.spark.ml.linalg.{Vector, Vectors}


trait Kernel extends Serializable {
  var hyperparameters: BDV[Double]

  def setHyperparameters(value: BDV[Double]): this.type = {
    hyperparameters = value
    this
  }

  var trainOption: Option[Array[Vector]]

  def hyperparameterBoundaries : (BDV[Double], BDV[Double])

  def setTrainingVectors(vectors: Array[Vector]): this.type

  def trainingKernel(): BDM[Double]

  def trainingKernelDiag(): Array[Double]

  def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]])

  def crossKernel(test: Array[Vector]): BDM[Double]

  def crossKernel(test: Vector): Transpose[BDV[Double]] = {
    val k = crossKernel(Array(test))
    k(0, ::)
  }
}

class TrainingVectorsNotInitializedException
  extends Exception("setTrainingVectors method should have been called first")

class RBFKernel(sigma: Double,
                private var lower: Double = 1e-6,
                private var upper: Double = inf) extends Kernel {
  override var hyperparameters : BDV[Double] = BDV[Double](sigma)

  private def getSigma() = hyperparameters(0)

  private var squaredDistances: Option[BDM[Double]] = None

  var trainOption: Option[Array[Vector]] = None

  def this() = this(1)

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = {
    (BDV[Double](lower), BDV[Double](upper))
  }

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
      / (-2d * sqr(getSigma()) ))
  }

  override def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]]) = {
    val sqd = squaredDistances.getOrElse(throw new TrainingVectorsNotInitializedException)

    val kernel = trainingKernel()
    val derivative = (sqd *:* kernel) / cube(getSigma())

    (kernel, Array(derivative))
  }

  override def crossKernel(test: Array[Vector]): BDM[Double] = {
    val train = trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    val values = train.flatMap(trainVector =>
      test.map(testVector =>
        Vectors.sqdist(trainVector, testVector)/ (-2d * sqr(getSigma())))
    )

    exp(BDM.create(test.length, train.length, values))
  }


  override def trainingKernelDiag(): Array[Double] = {
    val train = trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    train.map(_ => 1d)
  }

  private def sqr(x: Double) = x * x

  private def cube(x: Double) = x * x * x
}