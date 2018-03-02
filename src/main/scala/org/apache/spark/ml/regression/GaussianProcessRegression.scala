package org.apache.spark.ml.regression

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{ParamMap, Params}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.regression.kernel.{Kernel, RBFKernel}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

private[regression] trait GaussianProcessRegressionParams extends PredictorParams
  with HasMaxIter with HasTol with HasStandardization with HasAggregationDepth

class GaussianProcessRegression(override val uid: String,
                                private var kernel: Kernel)
  extends Regressor[Vector, GaussianProcessRegression, GaussianProcessRegressionModel]
    with GaussianProcessRegressionParams {

  def this() = this(Identifiable.randomUID("gaussProcessReg"), new RBFKernel())

  def setKernel(kernel : Kernel) : Unit = {
    this.kernel = kernel
  }

  override protected def train(dataset: Dataset[_]): GaussianProcessRegressionModel = ???

  override def copy(extra: ParamMap): GaussianProcessRegression = ???
}

class GaussianProcessRegressionModel private(override val uid: String)
  extends RegressionModel[Vector, GaussianProcessRegressionModel] {

  override protected def predict(features: Vector): Double = ???

  override def copy(extra: ParamMap): GaussianProcessRegressionModel = ???
}