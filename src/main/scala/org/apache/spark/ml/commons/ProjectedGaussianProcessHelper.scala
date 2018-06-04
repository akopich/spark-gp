package org.apache.spark.ml.commons

import breeze.linalg.{any, eigSym, DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.ml.commons.kernel.Kernel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD

trait ProjectedGaussianProcessHelper {
  class NotPositiveDefiniteException extends Exception("Some matrix which is supposed to be " +
    "positive definite is not. This probably happened due to `sigma2` parameter being too small." +
    " Try to gradually increase it.")

  /**
    * Does some matrix multiplications in a distributed manner.
    *
    * @param expertLabelsAndKernels
    * @param activeSet
    * @return (K_mn * K_nm, K_mn * y)
    */
  def getMatrixKmnKnmAndVectorKmny(expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
                                   activeSet: Array[Vector]): (BDM[Double], BDV[Double]) = {
    val activeSetSize = activeSet.length
    val activeSetBC = expertLabelsAndKernels.sparkContext.broadcast(activeSet)

    expertLabelsAndKernels.treeAggregate(BDM.zeros[Double](activeSetSize, activeSetSize),
      BDV.zeros[Double](activeSetSize))({ case (u, (y, k)) =>
      val kernelMatrix = k.crossKernel(activeSetBC.value)
      u._1 += kernelMatrix * kernelMatrix.t
      u._2 += kernelMatrix * y
      u
    }, { case (u, v) =>
      u._1 += v._1
      u._2 += v._2
      u
    })
  }

  /**
    * Computes inv(sigma^2^ K_mm + K_mn * K_nm) * K_mn * y
    *
    * @param kernel Should be fully initialized: both `setHyperparameters` and `setTrainingVectors`
    *               should be called beforehand
    * @param matrixKmnKnm
    * @param vectorKmny
    * @param activeSet
    * @param optimalHyperparameter
    * @return
    */
  def getMagicVector(kernel: Kernel,
                     matrixKmnKnm: BDM[Double],
                     vectorKmny: BDV[Double],
                     activeSet: Array[Vector],
                     optimalHyperparameter: BDV[Double]) = {
    val positiveDefiniteMatrix = kernel.trainingKernel() //K_mm
    positiveDefiniteMatrix *= kernel.whiteNoiseVar // sigma^2 K_mm
    positiveDefiniteMatrix += matrixKmnKnm // sigma^2 K_mm + K_mn * K_nm

    assertSymPositiveDefinite(positiveDefiniteMatrix)
    positiveDefiniteMatrix \ vectorKmny
  }

  protected def assertSymPositiveDefinite(matrix: BDM[Double]): Unit = {
    if (any(eigSym(matrix).eigenvalues <:< 0d))
      throw new NotPositiveDefiniteException
  }
}
