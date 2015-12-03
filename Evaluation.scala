package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import java.io.File

import scala.collection.mutable.HashMap
/**
 * Created by Georgios on 03/11/2015.
 */
/**
 * This class exposes various common classification evaluation metrics
 * @param ys: gold labels
 * @param preds: predicted labels
 * @param exclude: set of labels to exclude from overages (e.g. "None")
 * @tparam Y
 */
case class Evaluation[Y](ys:Seq[Y],preds:Seq[Y],exclude:Set[Y]=Set.empty[Y]){
  lazy val labels = ((ys++preds).toSet).toSeq
  /**
   * The confusion matrix is the basic structure from which the various evaluation metrics will be derived
   */
  lazy val confusion:Map[(Y,Y),Double]={
    // (true_label,pred_label)->count
    val matrix = new HashMap[(Y,Y),Double] withDefaultValue 0.0
    // count
    for ((y,pred) <- ys.zip(preds)){
      matrix((y,pred)) += 1.0
    }
    matrix.toMap withDefaultValue 0.0
  }

  /**
   * Helper functions to take counts over elements of the confusion matrix
   */
  private lazy val correctPredictionCounts:Map[Y,Double] = labels.map(key=>key -> confusion((key,key))).toMap
  private lazy val predictedLabelCounts:Map[Y,Double] = confusion.toSeq.groupBy(_._1._2).mapValues(s=>s.map(_._2).sum) withDefaultValue 0.0
  private lazy val goldLabelCounts:Map[Y,Double] = confusion.toSeq.groupBy(_._1._1).mapValues(s=>s.map(_._2).sum) withDefaultValue 0.0

  /**
   * Other helper functions
   */
  private def safeDiv(a:Double,b:Double):Double = if (b!=0.0) a/b else 0.0
  private def divideMaps(nom:Map[Y,Double],denom:Map[Y,Double])=nom.map{case (k,v) => k -> safeDiv(v,denom(k))}
  private def harmonic(p:Double,r:Double)= safeDiv(p*r,p+r)*2.0
  private def sumDivide(nom:Map[Y,Double],denom:Map[Y,Double])={
    val a = nom.filter{case (k,_) => !exclude.contains(k)}.values.sum
    val b = denom.filter{case (k,_) => !exclude.contains(k)}.values.sum
    safeDiv(a,b)
  }

  /**
   * Implementation of various evaluation metrics
   */
  lazy val precision:Map[Y,Double] = divideMaps(correctPredictionCounts,predictedLabelCounts)
  lazy val recall:Map[Y,Double] = divideMaps(correctPredictionCounts,goldLabelCounts)
  lazy val f1:Map[Y,Double] = precision map {case (k,p) => val r=recall(k); k-> harmonic(p,r)}
  lazy val averagePrecision:Double = sumDivide(correctPredictionCounts,predictedLabelCounts)
  lazy val averageRecall:Double = sumDivide(correctPredictionCounts,goldLabelCounts)
  lazy val averageF1:Double = harmonic(averagePrecision,averageRecall)

  /**
   * Creates a verbose report on various evaluation metrics
   * @return
   */
  override def toString={
    val excluded = if (exclude.isEmpty) "" else s"(excluding: ${exclude})"
    val sb = new StringBuilder()
    sb.append("Precision\n")
    sb.append("---------\n")
    sb.append(s"Average${excluded}: ${averagePrecision}\n")
    sb.append(s"Per class: ${precision}\n")
    sb.append("Recall\n")
    sb.append("---------\n")
    sb.append(s"Average${excluded}: ${averageRecall}\n")
    sb.append(s"Per class: ${recall}\n")
    sb.append("F1 score\n")
    sb.append("---------\n")
    sb.append(s"Average${excluded}: ${averageF1}}\n")
    sb.append(s"Per class: ${f1}\n")
    sb.toString
  }

}
object Evaluation {
  /**
   * Convenience method that writes the predictions to a file, one line each time
   * @param preds
   * @param dir
   * @tparam Y
   */
  def toFile[Y](preds:Seq[Y],dir: String): Unit = {
    val fileDir = new File(dir)
    fileDir.getParentFile.mkdirs()
    val pw = new java.io.PrintWriter(fileDir)
    try for (pred <- preds) pw.write(pred.toString+"\n") finally pw.close()
  }
}