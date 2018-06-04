package org.apache.spark.ml.commons.kernel

import breeze.linalg.{all, DenseMatrix => BDM, DenseVector => BDV}
import breeze.numerics.abs
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.FunSuite

class ARDRBFKernelTest extends FunSuite {
  private val dataset = Array(Array(1d, 2d), Array(2d, 3d), Array(5d, 7d)).map(Vectors.dense)

  private def computationalDerivative(beta: BDV[Double], h: Double): BDM[Double] = {
    val left = new ARDRBFKernel(beta - h)
    val right = new ARDRBFKernel(beta + h)

    left.setTrainingVectors(dataset)
    right.setTrainingVectors(dataset)

    (right.trainingKernel() - left.trainingKernel()) / (2 * h)
  }

  test("being called after `setTrainingVector`," +
    " `derivative` should return the correct kernel matrix derivative") {
    val beta = BDV[Double](0.2, 0.3)
    val ard = new ARDRBFKernel(beta)
    ard.setTrainingVectors(dataset)

    val analytical = ard.trainingKernelAndDerivative()._2.reduce(_ + _)
    val computational = computationalDerivative(beta, 1e-3)

    assert(all(abs(analytical - computational) <:< 1e-3))
  }

}
