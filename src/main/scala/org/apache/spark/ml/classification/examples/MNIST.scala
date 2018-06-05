package org.apache.spark.ml.classification.examples

import breeze.linalg.DenseVector
import breeze.numerics.sqrt
import org.apache.spark.ml.classification.{GaussianProcessClassification, OneVsRest}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object MNIST extends App {
  val name = "MNIST"
  val spark = SparkSession.builder().appName(name).master(s"local[${args(0)}]").getOrCreate()
  val path = args(1)
  val parallelism = args(0).toInt * 4
  val forExpert = args(2).toInt
  val activeSet = args(3).toInt

  import spark.sqlContext.implicits._
  val dataset = scale(spark.read.format("csv").load(path).rdd.map(row => {
    val features = Vectors.dense((1 until row.length).map("_c" + _).map(row.getAs[String]).map(_.toDouble).toArray)
    val label = row.getAs[String]("_c0").toDouble
    LabeledPoint(label, features)
  }).cache()).toDF.repartition(parallelism).cache()

  val gp = new GaussianProcessClassification().setDatasetSizeForExpert(forExpert).setActiveSetSize(activeSet)
  val ovr = new OneVsRest().setClassifier(gp)

  val cv = new TrainValidationSplit()
    .setEstimator(ovr)
    .setEvaluator(new MulticlassClassificationEvaluator().setMetricName("accuracy"))
    .setEstimatorParamMaps(new ParamGridBuilder().build())
    .setTrainRatio(0.8)

  println("Accuracy: " + cv.fit(dataset).validationMetrics.toList)

  def scale(data: RDD[LabeledPoint]) = {
    val x = data.map(x => DenseVector(x.features.toArray)).cache()
    val y = data.map(_.label)
    val mean = x.reduce(_+_) / x.count().toDouble
    val variance = x.map(xx => (xx-mean) *:* (xx-mean) ).reduce(_+_) / x.count.toDouble
    val varianceNoZeroes = for (v <- variance) yield if (v > 0d) v else 1d
    val features = x.map(xx => (xx-mean) /:/ sqrt(varianceNoZeroes)).map(_.toArray).map(Vectors.dense)
    x.unpersist()
    features zip y map {
      case(f, y) => LabeledPoint(y, f)
    }
  }
}
