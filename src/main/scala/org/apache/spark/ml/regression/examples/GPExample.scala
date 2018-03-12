package org.apache.spark.ml.regression.examples

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.GaussianProcessRegression
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, SparkSession}

trait GPExample {
  def name : String

  val spark = SparkSession.builder().appName(name).master("local[4]").getOrCreate()

  /*
   * It takes `gp`, runs 10-fold cross-validation on `instances` and returns the rmse
   * if it's below `expectedRMSE`. Exception is generated otherwise.
   */
  def cv(gp: GaussianProcessRegression, instances: DataFrame, expectedRMSE: Double) = {
    val cv = new CrossValidator()
      .setEstimator(gp)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(new ParamGridBuilder().build())
      .setNumFolds(10)

    val rmse = cv.fit(instances).avgMetrics.head
    assert(rmse < expectedRMSE)
    println("RMSE: " + rmse)
  }
}
