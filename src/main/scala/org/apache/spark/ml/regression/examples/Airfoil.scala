package org.apache.spark.ml.regression.examples

import breeze.linalg.DenseVector
import breeze.numerics.sqrt
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.regression.GaussianProcessRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


object Airfoil extends App {
  val spark = SparkSession
    .builder()
    .appName("Airfoil")
    .master("local[4]")
    .getOrCreate()

  val sqlContext= new org.apache.spark.sql.SQLContext(spark.sparkContext)
  import sqlContext.implicits._

  val airfoil = readSCV("data/airfoil.csv")

  val scaled = scale(airfoil).toDF

  val gp = new GaussianProcessRegression().setActiveSetSize(1000).setSigma2(1e-4)

  val cv = new CrossValidator()
    .setEstimator(gp)
    .setEvaluator(new RegressionEvaluator())
    .setEstimatorParamMaps(new ParamGridBuilder().build())
    .setNumFolds(10)

  println("RMSE: " + cv.fit(scaled).avgMetrics.head)

  def readSCV(path : String) = {
    spark.read.format("csv").load(path).rdd.map(row => {
      val features = Vectors.dense(Array("_c0", "_c1", "_c2", "_c3", "_c4")
        .map(col => row.getAs[String](col).toDouble))
      LabeledPoint(row.getAs[String]("_c5").toDouble, features)
    })
  }

  def scale(data: RDD[LabeledPoint]) = {
    val x = data.map(x => DenseVector(x.features.toArray)).cache()
    val y = data.map(_.label)
    val mean = x.reduce(_+_) / x.count().toDouble
    val variance = x.map(xx => (xx-mean) *:* (xx-mean) ).reduce(_+_) / x.count.toDouble
    val features = x.map(xx => (xx-mean) /:/ sqrt(variance)).map(_.toArray).map(Vectors.dense)
    features zip y map {
      case(f, y) => LabeledPoint(y, f)
    }
  }
}
