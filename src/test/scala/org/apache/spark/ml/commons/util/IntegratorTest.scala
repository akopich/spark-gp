package org.apache.spark.ml.commons.util

import breeze.numerics.{abs, sigmoid, sqrt}
import breeze.stats.distributions.{Gaussian, RandBasis}
import org.apache.commons.math3.random.MersenneTwister
import org.scalatest.FunSuite


class IntegratorTest extends FunSuite {

  test("testExpectedOfFunctionOfNormal") {
    val f = (x: Double) => sigmoid(x)
    val integrator = new Integrator(100)
    val mean = 0.5
    val variance = 3
    val sd = sqrt(variance)

    val testResult = integrator.expectedOfFunctionOfNormal(mean, variance, f)

    val gg = new Gaussian(mean, sd)(new RandBasis(new MersenneTwister()))
    val mcIters = 100000
    val values = gg.sample(mcIters).map(f)
    val mcResult = values.sum / mcIters
    val mcSD = sqrt(values.map(_ - mcResult).map(x => x * x).sum / mcIters) / sqrt(mcIters)
    assert(abs(mcResult - testResult) < 3 * mcSD)
  }

}
