package org.apache.spark.ml.classification.examples

import org.apache.spark.ml.classification.GaussianProcessClassification
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.SparkSession

object Iris extends App  {
  val name = "Iris"
  val spark = SparkSession.builder().appName(name).master("local[4]").getOrCreate()

  import spark.sqlContext.implicits._

  val dataset = spark.read.format("csv").load("data/iris.csv").rdd.map(row => {
    val features = Vectors.dense(Array("_c0", "_c1", "_c2", "_c3")
      .map(col => row.getAs[String](col).toDouble))

    val label = if (row.getAs[String]("_c4") == "Iris-setosa") 1d else -1d
    LabeledPoint(label, features)
  }).toDF

  val gp = new GaussianProcessClassification().setDatasetSizeForExpert(20)

  gp.fit(dataset)
}
