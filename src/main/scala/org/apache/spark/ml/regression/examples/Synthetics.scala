package org.apache.spark.ml.regression.examples

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.ml.commons.RandomActiveSetProvider
import org.apache.spark.ml.commons.kernel.{RBFKernel, WhiteNoiseKernel, _}
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GaussianProcessRegression

object Synthetics extends App with GPExample {
  import spark.sqlContext.implicits._

  override def name = "Synthetics"

  val noiseVar = 0.01
  val g = breeze.stats.distributions.Gaussian(0, math.sqrt(noiseVar))

  val X = linspace(0, 1, length = 2000).toDenseMatrix
  val Y = sin(X).toArray.map(y => y + g.sample())

  val instances = spark.sparkContext.parallelize(X.toArray.zip(Y).map { case(v, y) =>
    LabeledPoint(y, Vectors.dense(Array(v)))}).toDF

  val gp = new GaussianProcessRegression()
    .setKernel(() => 1*new RBFKernel(0.1, 1e-6, 10) + WhiteNoiseKernel(0.5, 0, 1))
    .setDatasetSizeForExpert(100)
    .setActiveSetProvider(RandomActiveSetProvider)
    .setActiveSetSize(100)
    .setSeed(13)
    .setSigma2(1e-3)

  cv(gp, instances, 0.11)
}
