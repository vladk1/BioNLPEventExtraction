package uk.ac.ucl.cs.mr.statnlpbook.assignment2


import uk.ac.ucl.cs.mr.statnlpbook.assignment2._

import scala.collection.immutable.ListMap
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
 * Created by Georgios on 30/10/2015.
 */

object Problem5{

  def main (args: Array[String]) {
    println("Joint Extraction")

    val train_dir = "./data/assignment2/bionlp/train"
    val test_dir = "./data/assignment2/bionlp/test"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir,0.8,500)
    // load test
    val testDocs = BioNLP.getTestDocuments(test_dir)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> (e.gold,e.arguments.map(_.gold)))

    // ================= Joint Classification =================
    // get candidates and make tuples with gold
    // read the specifications of the method for different subsampling thresholds
    // no subsampling for dev/test!
    def getJointCandidates(docs: Seq[Document]) = docs.flatMap(_.jointCandidates(0.02,0.4))
    def getTestJointCandidates(docs: Seq[Document]) = docs.flatMap(_.jointCandidates())
    val jointTrain = preprocess(getJointCandidates(trainDocs))
    val jointDev = preprocess(getTestJointCandidates(devDocs))
    val jointTest = preprocess(getTestJointCandidates(testDocs))

    // show statistics for counts of true labels, useful for deciding on subsampling
    println("True label counts (trigger - train):")
    println(jointTrain.unzip._2.unzip._1.groupBy(x=>x).mapValues(_.length))
    println("True label counts (trigger - dev):")
    println(jointDev.unzip._2.unzip._1.groupBy(x=>x).mapValues(_.length))
    println("True label counts (argument - train):")
    println(jointTrain.unzip._2.unzip._2.flatten.groupBy(x=>x).mapValues(_.length))
    println("True label counts (argument - dev):")
    println(jointDev.unzip._2.unzip._2.flatten.groupBy(x=>x).mapValues(_.length))


    // get label sets
    val triggerLabels = jointTrain.map(_._2._1).toSet
    val argumentLabels = jointTrain.flatMap(_._2._2).toSet

    // define model
    //TODO: change the features function to explore different types of features
    //TODO: experiment with the unconstrained and constrained (you need to implement the inner search) models
//    val jointModel = JointUnconstrainedClassifier(triggerLabels,argumentLabels,Features.myTriggerFeatures,Features.myArgumentFeatures)
    val jointModel = SimpleJointConstrainedClassifier(triggerLabels,argumentLabels,Features.myTriggerFeatures,Features.myArgumentFeatures)

    // use training algorithm to get weights of model
    val jointWeights = PrecompiledTrainers.trainPerceptron(jointTrain,jointModel.feat,jointModel.predict,10)

    // get predictions on dev
    val jointDevPred = jointDev.unzip._1.map { case e => jointModel.predict(e,jointWeights) }
    val jointDevGold = jointDev.unzip._2

    // Triggers (dev)
    val triggerDevPred = jointDevPred.unzip._1
    val triggerDevGold = jointDevGold.unzip._1
    val triggerDevEval = Evaluation(triggerDevGold,triggerDevPred,Set("None"))
    println("Evaluation for trigger classification:")
    println(triggerDevEval.toString)

    // Arguments (dev)
    val argumentDevPred = jointDevPred.unzip._2.flatten
    val argumentDevGold = jointDevGold.unzip._2.flatten
    val argumentDevEval = Evaluation(argumentDevGold,argumentDevPred,Set("None"))
    println("Evaluation for argument classification:")
    println(argumentDevEval.toString)

    // get predictions on test
    val jointTestPred = jointTest.unzip._1.map { case e => jointModel.predict(e,jointWeights) }
    // Triggers (test)
    val triggerTestPred = jointTestPred.unzip._1
    // write to file
    Evaluation.toFile(triggerTestPred,"./data/assignment2/out/joint_trigger_test.txt")
    // Arguments (test)
    val argumentTestPred = jointTestPred.unzip._2.flatten
    // write to file
    Evaluation.toFile(argumentTestPred,"./data/assignment2/out/joint_argument_test.txt")
  }
}


case class SimpleJointConstrainedClassifier(triggerLabels:Set[Label],
                                      argumentLabels:Set[Label],
                                      triggerFeature:(Candidate,Label)=>FeatureVector,
                                      argumentFeature:(Candidate,Label)=>FeatureVector
                                       ) extends JointModel {
  def predict(x: Candidate, weights: Weights) = {
    def argmax(labels: Set[Label], x: Candidate, weights: Weights, feat: (Candidate, Label) => FeatureVector) = {
      val scores = labels.toSeq.map(y => y -> dot(feat(x, y), weights)).toMap withDefaultValue 0.0
      scores.maxBy(_._2)._1
    }

    var bestTrigger = argmax(triggerLabels, x, weights, triggerFeature)

    var bestArguments: Seq[Label] = List()

    // Constraint 1 :  A trigger can only have arguments if its own label is not NONE
    if (bestTrigger.toString == "None") {
      bestArguments = for (arg<-x.arguments) yield "None"
    } else {
      // Constraint 3: Only regulation events can have CAUSE arguments
      if(bestTrigger.toString.contains("egulation")) {
        bestArguments = for (arg<-x.arguments) yield argmax(argumentLabels,arg,weights,argumentFeature)
      } else {
        bestArguments = for (arg<-x.arguments) yield argmax(argumentLabels.filter(_.toString != "Cause"),arg,weights,argumentFeature)
      }
      // Constraint 2: A trigger with a label other than NONE must have at least one THEME
      // check count of label "Theme"
      if (bestArguments.count(_.toString == "Theme") == 0) {
        // make map of highest feat score for "Theme"
        var overallScore = new ArrayBuffer[Double]()
        for (i <- 0 until x.arguments.size) {
          val realArgScores = dot(argumentFeature(x.arguments(i), bestArguments(i)), weights)
          val replacedThemeScore = dot(argumentFeature(x.arguments(i), argumentLabels.filter(_.toString == "Theme").head), weights)
          val replacedCauseScore = dot(argumentFeature(x.arguments(i), argumentLabels.filter(_.toString == "Cause").head), weights)
          val replacedNoneScore = dot(argumentFeature(x.arguments(i), argumentLabels.filter(_.toString == "None").head), weights)
          //          println("realArgScores="+realArgScores+" ThemeScore="+replacedThemeScore
          //            +" CauseScore="+replacedCauseScore+" NoneScore="+replacedNoneScore)
          overallScore += replacedThemeScore - realArgScores
        }
        val maxScoreIndex = overallScore.zipWithIndex.max._2
        bestArguments = bestArguments.updated(maxScoreIndex, "Theme")

        //        val themeScores = x.arguments.map(x => x -> dot(argumentFeature(x, argumentLabels.filterNot(_.toString() == "Theme").head), weights)).toMap withDefaultValue 0.0
        //        var count = 0
        //
        //          for (arg <- x.arguments) {
        //            if (arg == themeScores.maxBy(_._2)._1 && count == 0) {
        //              // println("updated at " + count)
        //              bestArguments = bestArguments.updated(count, "Theme")
        //            }
        //            count = count + 1
        //          }
        //
        //        var replacedTotalScore = 0.0
        //        var totalScoreNone = 0.0
        //
        //        for (i <- 0 until x.arguments.size) {
        //          replacedTotalScore += dot(argumentFeature(x.arguments(i), bestArguments(i)), weights)
        //          totalScoreNone += dot(argumentFeature(x.arguments(i), "None"), weights)
        //        }
        //
        //        val totalScoreReplace = dot(triggerFeature(x, bestTrigger), weights) + replacedTotalScore
        //
        //        totalScoreNone += dot(triggerFeature(x, "None"), weights)

        //        if(totalScoreNone > totalScoreReplace){
        //          bestTrigger = "None"
        //          bestArguments.map(arg => "None")
        //        }
      }
    }
    (bestTrigger,bestArguments)
  }
}

/**
 * A joint event classifier (both triggers and arguments).
 * It predicts the structured event.
 * It's predict method should only produce the best solution that respects the constraints on the event structure.
 * @param triggerLabels
 * @param argumentLabels
 * @param triggerFeature
 * @param argumentFeature
 */
case class JointConstrainedClassifier(triggerLabels:Set[Label],
                                      argumentLabels:Set[Label],
                                      triggerFeature:(Candidate,Label)=>FeatureVector,
                                      argumentFeature:(Candidate,Label)=>FeatureVector
                                       ) extends JointModel {
  def predict(x: Candidate, weights: Weights) = {
    //TODO
    def argmax(labels: Set[Label], x: Candidate, weights: Weights, feat:(Candidate,Label)=>FeatureVector) = {
      val scores = labels.toSeq.map(y => y -> dot(feat(x, y), weights)).toMap withDefaultValue 0.0
      scores.maxBy(_._2)._1
    }

    def getBestTriggerLabels(labels: Set[Label], x: Candidate, weights: Weights, feat:(Candidate,Label)=>FeatureVector) = {
      val scores = labels.toSeq.map(y => y -> dot(feat(x, y), weights)).toMap withDefaultValue 0.0
      val sortedSc = ListMap(scores.toSeq.sortBy(_._2):_*)
      sortedSc.drop(sortedSc.size - 3)
    }

    // there are 3 categories: I -> None, II -> Regulations, III -> Rest of labels for triggers
    def triggerCategory(label: Label): Int = label match {
      case "None" => 1
      case "Regulation" | "Positive_regulation" | "Negative_regulation" => 2
      case _ => 3
    }

    def validLabels(category: Int):Seq[String] = category match {
      case 1 => List("None")
      case 2 => List("Theme, Cause, None")
      case 3 => List("Theme, None")
    }

//    def initMatrix (nRows: Int, nCols: Int) = Array.tabulate(nRows,nCols)( (x,y) => 0f )

    def buildScoreMatrix(possibleLabels: Seq[Label], x: IndexedSeq[Candidate], weights: Weights, feat:(Candidate,Label)=>FeatureVector) = {

      val possibLab = possibleLabels.toIndexedSeq

      var scoresMap = new mutable.HashMap[(Int,Int), Double]()

      for(i<- 0 until possibleLabels.size) {
        for(j<- 0 until x.size) {
          scoresMap.put((i,j),dot(feat(x(j), possibLab(i)), weights))
        }
      }
      println(scoresMap)
    }

    var bestTriggers = getBestTriggerLabels(triggerLabels, x, weights, triggerFeature)

    val trigCategMap = bestTriggers.toSeq.map(trig => trig -> triggerCategory(trig._1)).toMap
    val uniqueCategories = trigCategMap.toSeq.map(trig => trig._2).toSet + 1
    val validArgLabels = uniqueCategories.map(cat => cat -> validLabels(cat)).toMap

    validArgLabels.foreach(label => {
      buildScoreMatrix(label._2, x.arguments.toIndexedSeq, weights, argumentFeature); println("for category: " + label._1)
    })




    // at the end return the best trigger label and the best argument labels
    (bestTriggers.head._1, List())
  }
}



/**
 * A joint event classifier (both triggers and arguments).
 * It predicts the structured event.
 * It treats triggers and arguments independently, i.e. it ignores any solution constraints.
 * @param triggerLabels
 * @param argumentLabels
 * @param triggerFeature
 * @param argumentFeature
 */
case class JointUnconstrainedClassifier(triggerLabels:Set[Label],
                                        argumentLabels:Set[Label],
                                        triggerFeature:(Candidate,Label)=>FeatureVector,
                                        argumentFeature:(Candidate,Label)=>FeatureVector
                                         ) extends JointModel{
  /**
   * Constraint 1: if e=None, all a=None
   * Constraint 2: if e!=None, at least one a=Theme
   * Constraint 3: only e=Regulation can have a=Cause
   * @param x
   * @param weights
   * @return
   */
  def predict(x: Candidate, weights: Weights) = {
    def argmax(labels: Set[Label], x: Candidate, weights: Weights, feat:(Candidate,Label)=>FeatureVector) = {
      val scores = labels.toSeq.map(y => y -> dot(feat(x, y), weights)).toMap withDefaultValue 0.0
      scores.maxBy(_._2)._1
    }
    val bestTrigger = argmax(triggerLabels,x,weights,triggerFeature)
    val bestArguments = for (arg<-x.arguments) yield argmax(argumentLabels,arg,weights,argumentFeature)
    (bestTrigger,bestArguments)
  }
}

trait JointModel extends Model[Candidate,StructuredLabels]{
  def triggerFeature:(Candidate,Label)=>FeatureVector
  def argumentFeature:(Candidate,Label)=>FeatureVector
  def feat(x: Candidate, y: StructuredLabels): FeatureVector ={
    val f = new mutable.HashMap[FeatureKey, Double] withDefaultValue 0.0
    addInPlace(triggerFeature(x,y._1),f,1)
    for ((a,label)<- x.arguments zip y._2){
      addInPlace(argumentFeature(a,label),f,1)
    }
    f
  }
}


