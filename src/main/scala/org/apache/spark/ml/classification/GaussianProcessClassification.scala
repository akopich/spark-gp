package org.apache.spark.ml.classification

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, _}
import breeze.numerics
import breeze.numerics.{abs, exp, sigmoid, sqrt}
import breeze.optimize.LBFGSB
import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression._
import org.apache.spark.ml.regression.kernel.Kernel
import org.apache.spark.ml.regression.util.{DiffFunctionMemoized, GaussianProcessCommons, GaussianProjectedProcessRawPredictor, ProjectedGaussianProcessHelper}
import org.apache.spark.ml.util.{Identifiable, Instrumentation}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset

class GaussianProcessClassification(override val uid: String)
  extends ProbabilisticClassifier[Vector, GaussianProcessClassification, GaussianProcessClassificationModel]
    with GaussianProcessParams with ProjectedGaussianProcessHelper with GaussianProcessCommons with Logging {
  def this() = this(Identifiable.randomUID("gaussProcessClass"))

  override protected def train(dataset: Dataset[_]): GaussianProcessClassificationModel = {
    val instr = Instrumentation.create(this, dataset)
    val points: RDD[LabeledPoint] = getPoints(dataset)

    // RDD of (y, f, kernel)
    val expertLabelsHiddensAndKernels: RDD[(BDV[Double], BDV[Double], Kernel)] = getExpertLabelsAndKernels(points)
      .map {case(y, kernel) => (y, BDV.zeros[Double](y.size), kernel)}
      .cache()

    instr.log("Optimising the kernel hyperparameters")
    val optimalHyperparameters = optimizeHyperparameters(expertLabelsHiddensAndKernels)
    val optimalKernel = getKernel().setHyperparameters(optimalHyperparameters)
    instr.log("Optimal kernel: " + optimalKernel)

    expertLabelsHiddensAndKernels.foreach(_._3.setHyperparameters(optimalHyperparameters))

    val rawPredictor = projectedProcess(expertLabelsHiddensAndKernels.map {case(_, f, kernel) => (f, kernel) },
      points, optimalHyperparameters, optimalKernel)

    val model = new GaussianProcessClassificationModel(uid, rawPredictor)
    instr.logSuccess(model)
    model
  }

  private def optimizeHyperparameters(expertLabelsHiddensAndKernels: RDD[(BDV[Double], BDV[Double], Kernel)]) = {
    val function = new DiffFunctionMemoized[BDV[Double]] with Serializable {
      override protected def calculateNoMemory(x: BDV[Double]): (Double, BDV[Double]) = {
        expertLabelsHiddensAndKernels.treeAggregate((0d, BDV.zeros[Double](x.length)))({ case (u, (y, f, k)) =>
          k.setHyperparameters(x)
          val (likelihood, gradient) = likelihoodAndGradient(y, f, k)
          (u._1 + likelihood, u._2 += gradient)
        }, { case (u, v) =>
          (u._1 + v._1, u._2 += v._2)
        })
      }
    }

    val x0 = getKernel().getHyperparameters
    val (lower, upper) = getKernel().hyperparameterBoundaries
    val solver = new LBFGSB(lower, upper, maxIter = $(maxIter), tolerance = $(tol))

    solver.minimize(function, x0)
  }

  private def likelihoodAndGradient(y: BDV[Double], f: BDV[Double], kernel: Kernel) = {
    val (kernelMatrix, derivatives) = kernel.trainingKernelAndDerivative()

    var oldObj = Double.NegativeInfinity
    var newObj = 0d

    val L : BDM[Double] = BDM.zeros[Double](y.length, y.length)
    val sqrtW : BDM[Double] = BDM.zeros[Double](y.length, y.length)
    val pi : BDV[Double] = BDV.zeros[Double](y.length)
    val a : BDV[Double] = BDV.zeros[Double](y.length)
    val gradLogP : BDV[Double] = BDV.zeros[Double](y.length)
    val I = BDM.eye[Double](y.length)

    while (abs(oldObj - newObj) > $(tol)) {
      pi := sigmoid(f)
      val W = diag(pi * (1d - pi)) // optimize me
      sqrtW := sqrt(W)
      val B = I + sqrtW * kernelMatrix * sqrtW
      L := cholesky(B)
      gradLogP := y - pi
      val b = W * f + gradLogP
      a := b - sqrtW * L.t \ (L \ (sqrtW * (kernelMatrix * b)))
      f := kernelMatrix * a
      oldObj = newObj
      newObj = -a.t * f / 2d + sum(numerics.log(sigmoid((y * 2d - 1d) *:* f)))
    }

    val logZ = newObj - sum(numerics.log(diag(L)))

    val R = sqrtW * L.t \ (L \ sqrtW)
    val C = L \ (sqrtW * kernelMatrix)
    val d3logP = -(2d * pi - 1d) *:* pi *:* pi *:* exp(-f)
    val s2 = - 0.5 * diag(diag(kernelMatrix) - diag(C.t * C)) * d3logP // optimize me

    val gradLogZ = BDV[Double](derivatives.map(C => {
      val s1 = 0.5 * (a.t * C * a) - 0.5 * trace(R * C)
      val b = C * gradLogP
      val s3 = b - kernelMatrix * R * b
      s1 + s2.t * s3
    }))

    (-logZ, -gradLogZ)
  }

  override def copy(extra: ParamMap): GaussianProcessClassification = defaultCopy(extra)
}

class GaussianProcessClassificationModel private[classification](override val uid: String,
              private val gaussianProjectedProcessRawPredictor: GaussianProjectedProcessRawPredictor)
  extends ProbabilisticClassificationModel[Vector, GaussianProcessClassificationModel] {

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = rawPrediction match {
    case dv : DenseVector =>
      dv.values(0) = sigmoid(-dv.values(0))
      dv.values(1) = sigmoid(dv.values(1))
      rawPrediction
    case sv: SparseVector =>
      throw new RuntimeException("Unexpected error in GaussianProcessClassificationModel:" +
        " raw2probabilitiesInPlace encountered SparseVector")
  }

  override def numClasses: Int = 2

  override protected def predictRaw(features: Vector): Vector = {
    val f = gaussianProjectedProcessRawPredictor.predict(features)
    Vectors.dense(-f, f)
  }

  override def copy(extra: ParamMap): GaussianProcessClassificationModel = {
    val newModel = copyValues(new GaussianProcessClassificationModel(uid, gaussianProjectedProcessRawPredictor), extra)
    newModel.setParent(parent)
  }
}