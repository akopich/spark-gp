package org.apache.spark.ml.regression.kernel

import breeze.linalg.{DenseMatrix, all}
import breeze.numerics.abs
import org.apache.spark.ml.linalg.Vectors
import org.scalatest.FunSuite

class RBFKernelTest extends FunSuite {
  test("Calling `trainingKernel` before `setTrainingVectors` " +
    "yields `TrainingVectorsNotInitializedException") {
    val rbf = new RBFKernel()

    assertThrows[TrainingVectorsNotInitializedException] {
      rbf.trainingKernel()
    }
  }

  test("Calling `derivative` before `setTrainingVectors` " +
    "yields `TrainingVectorsNotInitializedException") {
    val rbf = new RBFKernel()

    assertThrows[TrainingVectorsNotInitializedException] {
      rbf.derivative()
    }
  }

  private val dataset = Array(Array(1d, 2d), Array(2d, 3d), Array(5d, 7d)).map(Vectors.dense)

  test("being called after `setTrainingVector`," +
    " `trainingKernel` should return the correct kernel matrix") {
    val rbf = new RBFKernel(math.sqrt(0.2))
    rbf.setTrainingVectors(dataset)

    val correctKernelMatrix = DenseMatrix((1.000000e+00, 6.737947e-03, 3.053624e-45),
                                          (6.737947e-03, 1.000000e+00, 7.187782e-28),
                                          (3.053624e-45, 7.187782e-28, 1.000000e+00))

    assert(all(abs(rbf.trainingKernel() - correctKernelMatrix) <:< 1e-4))
  }

  private def computationalDerivative(sigma: Double, h: Double) = {
    val rbfLeft = new RBFKernel(sigma - h)
    val rbfRight = new RBFKernel(sigma + h)

    rbfLeft.setTrainingVectors(dataset)
    rbfRight.setTrainingVectors(dataset)

    (rbfRight.trainingKernel() - rbfLeft.trainingKernel()) / (2 * h)
  }

  test("derivative") {
    val rbf = new RBFKernel(0.2)
    rbf.setTrainingVectors(dataset)

    val analytical = rbf.derivative()(0)
    val computational = computationalDerivative(0.2, 1e-3)

    assert(all(abs(analytical - computational) <:< 1e-3))
  }
}
