package org.apache.spark.ml.commons.util

import breeze.numerics.sqrt
import org.apache.commons.math3.analysis.UnivariateFunction
import org.apache.commons.math3.analysis.integration.gauss.GaussIntegratorFactory

class Integrator(private val n: Int) {
  private val integrator = new GaussIntegratorFactory().hermite(n)

  def expectedOfFunctionOfNormal(mean: Double, variance: Double, f: (Double) => Double) = {
    val sd = sqrt(variance)
    integrator.integrate(new UnivariateFunction {
      override def value(x: Double): Double = f(sqrt(2) * sd * x + mean)
    }) / sqrt(math.Pi)
  }
}
