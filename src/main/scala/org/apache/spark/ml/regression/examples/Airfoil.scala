package org.apache.spark.ml.regression.examples

import org.apache.spark.ml.commons.kernel.{ARDRBFKernel, _}
import org.apache.spark.ml.commons.util.Scaling
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GaussianProcessRegression

object Airfoil extends App with GPExample with Scaling {
  import spark.sqlContext.implicits._

  override def name = "Airfoil"

  val airfoil = readSCV("data/airfoil.csv")

  val scaled = scale(airfoil).toDF

  val gp = new GaussianProcessRegression()
    .setActiveSetSize(1000)
    .setSigma2(1e-4)
    .setKernel(() => 1 * new ARDRBFKernel(5) + 1.const * new EyeKernel)

  cv(gp, scaled, 2.1)

  def readSCV(path : String) = {
    spark.read.format("csv").load(path).rdd.map(row => {
      val features = Vectors.dense(Array("_c0", "_c1", "_c2", "_c3", "_c4")
        .map(col => row.getAs[String](col).toDouble))
      LabeledPoint(row.getAs[String]("_c5").toDouble, features)
    })
  }
}
