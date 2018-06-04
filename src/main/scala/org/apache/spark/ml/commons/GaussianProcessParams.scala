package org.apache.spark.ml.commons

import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.commons.kernel.{Kernel, RBFKernel}
import org.apache.spark.ml.param.shared.{HasAggregationDepth, HasMaxIter, HasSeed, HasTol}
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param}

private[ml] trait GaussianProcessParams extends PredictorParams
  with HasMaxIter with HasTol with HasAggregationDepth with HasSeed {

  final val activeSetProvider = new Param[ActiveSetProvider](this, "activeSetProvider",
    "the class which provides the active set used by Projected Process Approximation")

  final val kernel = new Param[() => Kernel](this,
    "kernel", "function of no arguments which returns " +
      "the kernel of the prior Gaussian Process")

  final val datasetSizeForExpert = new IntParam(this,
    "datasetSizeForExpert", "The number of data points fed to each expert. " +
      "Time and space complexity of training quadratically grows with it.")

  final val sigma2 = new DoubleParam(this,
    "sigma2", "The variance of noise in the inputs. The value is added to the diagonal of the " +
      "kernel Matrix. Also prevents numerical issues associated with inversion " +
      "of a computationally singular matrix ")

  final val activeSetSize = new IntParam(this,
    "activeSetSize", "Number of latent functions to project the process onto. " +
      "The size of the produced model and prediction complexity " +
      "linearly depend on this value.")

  def setActiveSetProvider(value : ActiveSetProvider): this.type = set(activeSetProvider, value)
  setDefault(activeSetProvider -> RandomActiveSetProvider)

  def setDatasetSizeForExpert(value: Int): this.type = set(datasetSizeForExpert, value)
  setDefault(datasetSizeForExpert -> 100)

  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  def setSigma2(value: Double): this.type = set(sigma2, value)
  setDefault(sigma2 -> 1e-3)

  def setKernel(value: () => Kernel): this.type = set(kernel, value)
  setDefault(kernel -> (() => new RBFKernel()))

  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  def setActiveSetSize(value: Int): this.type = set(activeSetSize, value)
  setDefault(activeSetSize -> 100)

  def setSeed(value: Long): this.type = set(seed, value)
}
