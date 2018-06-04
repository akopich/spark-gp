package org.apache.spark.ml.commons.util

import breeze.linalg.{LU, MatrixSingularException, DenseMatrix => BDM, DenseVector => BDV}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.netlib.util.intW
import spire.implicits.cforRange

object logDetAndInv {
  /**
    *
    * This code is adopted from breeze with minimal changes.
    */
  private def LU2inv(m: BDM[Double], ipiv:Array[Int]): BDM[Double] = {
    val N         = m.rows
    val lwork     = scala.math.max(1, N)
    val work      = Array.ofDim[Double](lwork)
    val info      = new intW(0)
    lapack.dgetri(
      N, m.data, scala.math.max(1, N) /* LDA */,
      ipiv,
      work /* workspace */, lwork /* workspace size */,
      info
    )
    assert(info.`val` >= 0, "Malformed argument %d (LAPACK)".format(-info.`val`))

    if (info.`val` > 0)
      throw new MatrixSingularException
    m
  }

  /**
    *
    * This code is adopted from breeze with minimal changes.
    */
  private def LU2logdet(m: BDM[Double], ipiv:Array[Int]): (Double, Double) = {
    val numExchangedRows = ipiv.map(_ - 1).zipWithIndex.count { piv => piv._1 != piv._2 }

    var sign = if (numExchangedRows % 2 == 1) -1.0 else 1.0

    var acc = 0.0
    cforRange(0 until m.rows){ i =>
      val mii = m(i, i)
      if(mii == 0.0) return (0.0, Double.NegativeInfinity)
      acc += math.log(math.abs(mii))
      sign *= math.signum(mii)
    }

    (sign, acc)
  }

  /**
    * Computes logdet and inverse of the given matrix relying on LU-decomposition which takes place only once.
    *
    * @param X
    * @return (sign, logdet, inverse)
    */
  def apply(X: BDM[Double]) = {
    val (m: BDM[Double], ipiv: Array[Int]) = LU(X)
    val (sign: Double, logdet: Double) = LU2logdet(m, ipiv)
    val inverse = LU2inv(m, ipiv)
    (sign, logdet, inverse)
  }
}
