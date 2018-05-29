package org.apache.spark.ml.regression.kernel

import breeze.linalg.{Transpose, norm, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import breeze.numerics._
import org.apache.spark.ml.linalg.{Vector, Vectors}

/**
  * Trait defining the covariance function `k` of a Gaussian Process
  *
  * The kernel should be differentiable with respect to the hyperparemeters
  *
  */
trait Kernel extends Serializable {
  /**
    * Returns a vector of hyperparameters of the kernel
    */
  def getHyperparameters: BDV[Double]

  /**
    * Setter
    *
    * @param value
    * @return this
    */
  def setHyperparameters(value: BDV[Double]): this.type

  /**
    *
    * @return boundaries lower and upper: lower <= hyperparameter <= upper (inequalities are element-wise).
    */
  def hyperparameterBoundaries : (BDV[Double], BDV[Double])


  /**
    * Returns the portion of the training sample the kernel is computed with respect to.
    */
  def getTrainingVectors: Option[Array[Vector]]

  /**
    *
    * @param vectors
    * @return this
    */
  def setTrainingVectors(vectors: Array[Vector]): this.type

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @return a matrix K such that `K_{ij} = k(trainingVectros(i), trainingVectros(j))`
    */
  def trainingKernel(): BDM[Double]

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @return the diagonal of the matrix which would be returned by `trainingKernel`.
    */
  def trainingKernelDiag(): Array[Double]

  /**
    *`setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @return a pair of a kernel K (returned by `trainingKernel`)
    *         and an array of partial derivatives \partial K / \partial hyperparameters(i)
    */
  def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]])

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @param test vectors (typically, those we want a prediction for)
    * @return a matrix K of size `test.size * trainVectors.size` such that K_{ij} = k(test(i), trainVectors(j))
    */
  def crossKernel(test: Array[Vector]): BDM[Double]

  /**
    * `setTrainingVectors` should be called beforehand. Otherwise TrainingVectorsNotInitializedException is thrown.
    * @param test a vector (typically, the one we want a prediction for)
    * @return a row-vector K such that K_{i} = k(test, trainVectors(i))
    */
  def crossKernel(test: Vector): Transpose[BDV[Double]] = {
    val k = crossKernel(Array(test))
    k(0, ::)
  }
}

class TrainingVectorsNotInitializedException
  extends Exception("setTrainingVectors method should have been called first")

/**
  * Implements traditional RBF kernel `k(x_i, k_j) = exp(||x_i - x_j||^2 / sigma^2)`
  *
  * @param sigma
  * @param lower
  * @param upper
  */
class RBFKernel(sigma: Double,
                private val lower: Double = 1e-6,
                private val upper: Double = inf) extends Kernel {
  private var hyperparameters : BDV[Double] = BDV[Double](sigma)

  override def setHyperparameters(value: BDV[Double]): RBFKernel.this.type = {
    hyperparameters = value
    this
  }

  override def getHyperparameters: BDV[Double] = hyperparameters

  private def getSigma() = hyperparameters(0)

  private var squaredDistances: Option[BDM[Double]] = None

  var trainOption: Option[Array[Vector]] = None

  def this() = this(1)

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = {
    (BDV[Double](lower), BDV[Double](upper))
  }

  override def getTrainingVectors: Option[Array[Vector]] = trainOption

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
    val train = trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    val result = BDM.zeros[Double](test.length, train.length)

    for (i <- test.indices; j <- train.indices)
      result(i, j) = Vectors.sqdist(test(i), train(j)) / (-2d * sqr(getSigma()))

    exp.inPlace(result)

    result
  }


  override def trainingKernelDiag(): Array[Double] = {
    trainOption.getOrElse(throw new TrainingVectorsNotInitializedException).map(_ => 1d)
  }

  private def sqr(x: Double) = x * x

  private def cube(x: Double) = x * x * x
}


class ARDRBFKernel(private var hyperparameters: BDV[Double],
                   private val lower: BDV[Double],
                   private val upper: BDV[Double]) extends Kernel {

  def this(hyperparameters: BDV[Double]) = this(hyperparameters, hyperparameters * 0d, hyperparameters * inf)

  override def setHyperparameters(value: BDV[Double]): ARDRBFKernel.this.type = {
    hyperparameters = value
    this
  }

  override def getHyperparameters: BDV[Double] = hyperparameters

  def this(p : Int, beta: Double = 1, lower: Double = 0, upper : Double = inf) =
    this(BDV.zeros[Double](p) + beta,
      BDV.zeros[Double](p) + lower,
      BDV.zeros[Double](p) + upper)

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = (lower, upper)

  private var trainOption: Option[Array[Vector]] = _

  override def getTrainingVectors: Option[Array[Vector]] = trainOption

  override def setTrainingVectors(vectors: Array[Vector]): this.type = {
    trainOption = Some(vectors)
    this
  }

  private def kernelElement(a: BV[Double], b: BV[Double]) : Double = {
    val weightedDistance = norm((a - b) *:* hyperparameters)
    exp(- weightedDistance * weightedDistance)
  }

  override def trainingKernelDiag(): Array[Double] = {
    val train = trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    train.map(_ => 1d)
  }

  override def trainingKernel(): BDM[Double] = {
    val train = trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)

    val result = BDM.zeros[Double](train.length, train.length)
    for (i <- train.indices; j <- 0 to i) {
      val k = kernelElement(train(i).asBreeze, train(j).asBreeze)
      result(i, j) = k
      result(j, i) = k
    }

    result
  }

  override def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]]) = {
    val train = trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    val K = trainingKernel()
    val minus2Kernel = -2d * K
    val result = Array.fill[BDM[Double]](hyperparameters.length)(BDM.zeros[Double](train.length, train.length))

    for (i <- train.indices; j <- 0 to i) {
      val diff = train(i).asBreeze - train(j).asBreeze
      diff :*= diff
      diff :*= hyperparameters
      val betaXi_Xj = diff
      for (k <- 0 until hyperparameters.length) {
        result(k)(i, j) = betaXi_Xj(k)
        result(k)(j, i) = betaXi_Xj(k)
      }
    }

    (K, result.map(derivative => derivative *:* minus2Kernel))
  }

  override def crossKernel(test: Array[Vector]): BDM[Double] = {
    val train = trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)
    val result = BDM.zeros[Double](test.length, train.length)

    for (testIndx <- test.indices; trainIndex <- train.indices)
      result(testIndx, trainIndex) = kernelElement(train(trainIndex).asBreeze, test(testIndx).asBreeze)

    result
  }
}





















