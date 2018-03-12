package org.apache.spark.ml.regression.examples

import breeze.linalg._
import breeze.numerics._
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GaussianProcessRegression
import org.apache.spark.ml.regression.kernel.RBFKernel

object Synthetics extends App with GPExample {
  import spark.sqlContext.implicits._

  override def name = "Synthetics"

  val X = linspace(0, 1, length = 2000).toDenseMatrix
  val Y = sin(X).toArray

  val instances = spark.sparkContext.parallelize(X.toArray.zip(Y).map { case(v, y) =>
    LabeledPoint(y, Vectors.dense(Array(v)))}).toDF

  val gp = new GaussianProcessRegression()
    .setKernel(() => new RBFKernel(0.1))
    .setDatasetSizeForExpert(100)
    .setActiveSetSize(100)
    .setSeed(13)
    .setSigma2(1e-3)

  cv(gp, instances, 7e-5)
}
