package org.apache.spark.ml.commons.util

import breeze.linalg.DenseVector
import breeze.numerics.sqrt
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.rdd.RDD

private[ml] trait Scaling {
  def scale(data: RDD[LabeledPoint]) = {
    val x = data.map(x => DenseVector(x.features.toArray)).cache()
    val y = data.map(_.label)
    val n = x.count().toDouble
    val mean = x.reduce(_ + _) / n
    val centered = x.map(_ - mean).cache()
    val variance = centered.map(xx => xx *:* xx).reduce(_ + _) / n
    x.unpersist()
    val varianceNoZeroes = variance.map(v => if (v > 0d) v else 1d)
    val scaled = centered.map(_ /:/ sqrt(varianceNoZeroes)).map(_.toArray).map(Vectors.dense).zip(y).map {
      case(f, y) => LabeledPoint(y, f)
    }.cache()
    if (scaled.count() > 0) // ensure scaled is materialized
      centered.unpersist()
    scaled
  }
}
