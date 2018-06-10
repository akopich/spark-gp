package org.apache.spark.ml.commons.kernel

import breeze.linalg.{Transpose, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.ml.linalg.Vector

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
  def trainingKernelDiag(): BDV[Double]

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
    * returns k(x, x)
    * @param test
    * @return returns a single value k(test, test)
    */
  def selfKernel(test: Vector): Double

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

/**
  * Some kernels have equal values on the diagonal. The trait implements `trainingKernelDiag` using a single call of
  * `selfKernel`
  */
trait SameOnDiagonalKernel extends Kernel {
  override def trainingKernelDiag(): BDV[Double] =
    BDV.zeros[Double](getTrainingVectors.length) + selfKernel(getTrainingVectors.head)
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
  * Identity matrix kernel.
  */
class EyeKernel extends TrainDatasetBearingKernel with SameOnDiagonalKernel {
  override def getHyperparameters: BDV[Double] = BDV[Double]()

  override def setHyperparameters(value: BDV[Double]): EyeKernel.this.type = this

  override def numberOfHyperparameters: Int = 0

  override def hyperparameterBoundaries: (BDV[Double], BDV[Double]) = (BDV[Double](), BDV[Double]())

  override def trainingKernel(): BDM[Double] = BDM.eye[Double](getTrainingVectors.length)

  override def trainingKernelAndDerivative(): (BDM[Double], Array[BDM[Double]]) = {
    (trainingKernel(), Array[BDM[Double]]())
  }

  override def crossKernel(test: Array[Vector]): BDM[Double] = BDM.zeros[Double](test.length, getTrainingVectors.length)

  override def whiteNoiseVar: Double = 1d

  override def selfKernel(test: Vector): Double = 1d

  override def toString = "I"
}

object WhiteNoiseKernel {
  def apply(initial: Double, lower: Double, upper: Double): Kernel =
    (initial between lower and upper) * new EyeKernel
}


