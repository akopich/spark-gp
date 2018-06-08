package org.apache.spark.ml.commons

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import breeze.optimize.LBFGSB
import org.apache.spark.ml.commons.kernel.{EyeKernel, Kernel, _}
import org.apache.spark.ml.commons.util.DiffFunctionMemoized
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Dataset, Row}

private[ml] trait GaussianProcessCommons[F, E <: Predictor[F, E, M], M <: PredictionModel[F, M]]
  extends ProjectedGaussianProcessHelper {  this: Predictor[F, E, M] with GaussianProcessParams =>

  protected val getKernel : () => Kernel = () => $(kernel)() + $(sigma2).const * new EyeKernel

  protected def getPoints(dataset: Dataset[_]) = {
    dataset.select(col($(labelCol)), col($(featuresCol))).rdd.map {
      case Row(label: Double, features: Vector) => LabeledPoint(label, features)
    }
  }

  protected def groupForExperts(points: RDD[LabeledPoint]) = {
    val numberOfExperts = Math.round(points.count().toDouble / $(datasetSizeForExpert))
    points.zipWithIndex.map { case(instance, index) =>
      (index % numberOfExperts, instance)
    }.groupByKey().map(_._2)
  }

  protected def getExpertLabelsAndKernels(points: RDD[LabeledPoint]): RDD[(BDV[Double], Kernel)] = {
    groupForExperts(points).map { chunk =>
      val (labels, trainingVectors) = chunk.map(lp => (lp.label, lp.features)).toArray.unzip
      (BDV(labels: _*), getKernel().setTrainingVectors(trainingVectors))
    }
  }

  protected def projectedProcess(expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
                                 points: RDD[LabeledPoint],
                                 optimalHyperparameters: BDV[Double]) = {
    val activeSet = $(activeSetProvider)($(activeSetSize), expertLabelsAndKernels, points,
      getKernel, optimalHyperparameters, $(seed))

    points.unpersist()

    val (matrixKmnKnm, vectorKmny) = getMatrixKmnKnmAndVectorKmny(expertLabelsAndKernels, activeSet)

    expertLabelsAndKernels.unpersist()

    val optimalKernel = getKernel().setHyperparameters(optimalHyperparameters).setTrainingVectors(activeSet)

    // inv(sigma^2 K_mm + K_mn * K_nm) * K_mn * y
    val (magicVector, magicMatrix) = getMagicVector(optimalKernel,
      matrixKmnKnm, vectorKmny, activeSet, optimalHyperparameters)

    new GaussianProjectedProcessRawPredictor(magicVector, magicMatrix, optimalKernel)
  }

  /**
    *
    * @tparam T for GPR it's (y, kernel), for GPC it's (y, f, kernel)
    * @return the optimal hyperparameters estimates
    */
  protected def optimizeHypers[T](instr: Instrumentation[E],
                                  expertValuesAndKernels: RDD[T],
                                  likelihoodAndGradient: (T, BDV[Double]) => (Double, BDV[Double])) = {
    instr.log("Optimising the kernel hyperparameters")

    val f = new DiffFunctionMemoized[BDV[Double]] with Serializable {
      override protected def calculateNoMemory(x: BDV[Double]): (Double, BDV[Double]) = {
        expertValuesAndKernels.treeAggregate((0d, BDV.zeros[Double](x.length)))({ case (u, yAndK) =>
          val (likelihood, gradient) = likelihoodAndGradient(yAndK, x)
          (u._1 + likelihood, u._2 += gradient)
        }, { case (u, v) =>
          (u._1 + v._1, u._2 += v._2)
        })
      }
    }

    val x0 = getKernel().getHyperparameters
    val (lower, upper) = getKernel().hyperparameterBoundaries
    val solver = new LBFGSB(lower, upper, maxIter = $(maxIter), tolerance = $(tol))

    val optimalHyperparameters = solver.minimize(f, x0)

    val optimalKernel = getKernel().setHyperparameters(optimalHyperparameters)
    instr.log("Optimal kernel: " + optimalKernel)

    optimalHyperparameters
  }

  /**
    *
    * @param instr
    * @param points
    * @param expertLabelsAndKernels the kernels contained in the RDD should have the optimal hyperparameters set
    * @param optimalHyperparameters
    * @return the model
    */
  protected def produceModel(instr: Instrumentation[E],
                                                points: RDD[LabeledPoint],
                                                expertLabelsAndKernels: RDD[(BDV[Double], Kernel)],
                                                optimalHyperparameters: BDV[Double]) = {
    val rawPredictor = projectedProcess(expertLabelsAndKernels, points, optimalHyperparameters)
    val model = createModel(uid, rawPredictor)
    instr.logSuccess(model)
    model
  }

  /**
    * just calls the constructor
    */
  protected def createModel(uid: String, rawPredictor: GaussianProjectedProcessRawPredictor) : M
}

class GaussianProjectedProcessRawPredictor private[commons] (val magicVector: BDV[Double],
                                                             val magicMatrix: BDM[Double],
                                                             val kernel: Kernel) extends Serializable {
  def predict(features: Vector): (Double, Double) = {
    val cross = kernel.crossKernel(features)
    val selfKernel = kernel.trainingKernelDiag().head // TODO this is not fair
    (cross * magicVector, selfKernel + cross * magicMatrix * cross.t)
  }
}

