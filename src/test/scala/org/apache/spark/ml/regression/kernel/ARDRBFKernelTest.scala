package org.apache.spark.ml.regression.kernel

import breeze.linalg.{DenseMatrix, all}
import breeze.numerics.abs
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.FunSuite

class ARDRBFKernelTest extends FunSuite {
  private val dataset = Array(Array(1d, 2d), Array(2d, 3d), Array(5d, 7d)).map(Vectors.dense)

  private def computationalDerivative(beta: Double, h: Double): DenseMatrix[Double] = {
    val left = new ARDRBFKernel(2, beta - h)
    val right = new ARDRBFKernel(2, beta + h)

    left.setTrainingVectors(dataset)
    right.setTrainingVectors(dataset)

    (right.trainingKernel() - left.trainingKernel()) / (2 * h)
  }

  test("being called after `setTrainingVector`," +
    " `derivative` should return the correct kernel matrix derivative") {
    val ard = new ARDRBFKernel(2, 0.2)
    ard.setTrainingVectors(dataset)

    val analytical = ard.trainingKernelAndDerivative()._2.reduce(_ + _)
    val computational = computationalDerivative(0.2, 1e-3)

    assert(all(abs(analytical - computational) <:< 1e-3))
  }

}
