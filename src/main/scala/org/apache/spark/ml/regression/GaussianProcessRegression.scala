package org.apache.spark.ml.regression

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.{IntParam, Param, ParamMap}
import org.apache.spark.ml.param.shared._
import org.apache.spark.ml.regression.kernel.{Kernel, RBFKernel}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

private[regression] trait GaussianProcessRegressionParams extends PredictorParams
  with HasMaxIter with HasTol with HasStandardization with HasAggregationDepth {

  final val kernel = new Param[Kernel](this,
    "kernel", "the kernel of the prior Gaussian Process")

  final val datasetSizeForExpert = new IntParam(this,
    "datasetSizeForExpert", "the number of data points fed to each expert")

  def setDatasetSizeForExpert(value: Int): this.type = {
    set(datasetSizeForExpert, value)
    this
  }
  setDefault(datasetSizeForExpert -> 100)

  def setKernel(value: Kernel): this.type = {
    set(kernel, value)
    this
  }
  setDefault(kernel -> new RBFKernel())
}

class GaussianProcessRegression(override val uid: String)
  extends Regressor[Vector, GaussianProcessRegression, GaussianProcessRegressionModel]
    with GaussianProcessRegressionParams {

  def this() = this(Identifiable.randomUID("gaussProcessReg"))

  override protected def train(dataset: Dataset[_]): GaussianProcessRegressionModel = ???

  override def copy(extra: ParamMap): GaussianProcessRegression = ???
}

class GaussianProcessRegressionModel private(override val uid: String)
  extends RegressionModel[Vector, GaussianProcessRegressionModel] {

  override protected def predict(features: Vector): Double = ???

  override def copy(extra: ParamMap): GaussianProcessRegressionModel = ???
}