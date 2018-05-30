package org.apache.spark.ml.regression.kernel

import breeze.linalg.{Transpose, norm, DenseMatrix => BDM, DenseVector => BDV, Vector => BV}
import breeze.numerics._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.regression.kernel.ScalarTimesKernel._

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

  def numberOfHyperparameters: Int

  /**
    *
    * @return boundaries lower and upper: lower <= hyperparameter <= upper (inequalities are element-wise).
    */
  def hyperparameterBoundaries : (BDV[Double], BDV[Double])


  /**
    * Returns the portion of the training sample the kernel is computed with respect to.
    *
    * The method `setTrainingVectors` should be called beforehand.
    * `TrainingVectorsNotInitializedException` is thrown otherwise.
    */
  def getTrainingVectors: Array[Vector]

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

  /**
    *
    * @return variance of the white noise presumed by the kernel
    */
  def whiteNoiseVar: Double
}

/**
  * Many kernels do not presume any white noise. The trait provides an implementation of `whiteNoiseVar` for them.
  */
trait NoiselessKernel extends Kernel {
  override def whiteNoiseVar: Double = 0
}

class TrainingVectorsNotInitializedException
  extends Exception("setTrainingVectors method should have been called first")

/**
  * Most of the basic (non-composite ones like `SumOfKernels`) kernels have own a portion of a dataset.
  * The trait does exactly this.
  */
trait TrainDatasetBearingKernel extends Kernel {
  private var trainOption: Option[Array[Vector]] = None

  override def getTrainingVectors: Array[Vector] =
    trainOption.getOrElse(throw new TrainingVectorsNotInitializedException)

  override def setTrainingVectors(vectors: Array[Vector]): this.type = {
    trainOption = Some(vectors)
    this
  }
}

/**
  * Implements traditional RBF kernel `k(x_i, k_j) = exp(||x_i - x_j||^2 / sigma^2)`
  *
  * @param sigma
  * @param lower
  * @param upper
  */
class RBFKernel(private var sigma: Double,
                private val lower: Double = 1e-6,
                private val upper: Double = inf) extends TrainDatasetBearingKernel with NoiselessKernel {
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

  override def trainingKernelDiag(): Array[Double] = getTrainingVectors.map(_ => 1d)

  private def sqr(x: Double) = x * x

  private def cube(x: Double) = x * x * x
}

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
                   private val upper: BDV[Double]) extends TrainDatasetBearingKernel with NoiselessKernel {

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

  override def trainingKernelDiag(): Array[Double] = {
    getTrainingVectors.map(_ => 1d)
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
}

class EyeKernel extends TrainDatasetBearingKernel {
  override def getHyperparameters: BDV[Double] = BDV[Double]()

  override def setHyperparameters(value: BDV[Double]): EyeKernel.this.type = this

  override def numberOfHyperparameters: Int = 0

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = (BDV[Double](), BDV[Double]())

  override def trainingKernel(): BDM[Double] = BDM.eye[Double](getTrainingVectors.length)

  override def trainingKernelDiag(): Array[Double] = {
    getTrainingVectors.map(_ => 1d)
  }

  override def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]]) = {
    (trainingKernel(), Array[BDM[Double]]())
  }

  override def crossKernel(test: Array[Vector]): BDM[Double] = BDM.zeros[Double](test.length, getTrainingVectors.length)

  override def whiteNoiseVar: Double = 1
}

object WhiteNoiseKernel {
  def apply(initial: Double, lower: Double, upper: Double): Kernel =
    (initial between lower and upper) * new EyeKernel
}