package uk.ac.ucl.cs.mr.statnlpbook.assignment2

import scala.collection.immutable.ListMap
import scala.collection.mutable

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
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir,0.8) // 500
    // load test
    val testDocs = BioNLP.getTestDocuments(test_dir)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> (e.gold,e.arguments.map(_.gold)))

    // ================= Joint Classification =================
    // get candidates and make tuples with gold
    // read the specifications of the method for different subsampling thresholds
    // no subsampling for dev/test!
    val TUNED_TRIGGER_THRESHOLD = 0.2
    val DEFAULT_TRIGGER_THRESHOLD = 0.02
    def getJointCandidates(docs: Seq[Document]) = docs.flatMap(_.jointCandidates(TUNED_TRIGGER_THRESHOLD,0.4)) //0.02,0.4
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
    val jointModel = JointConstrainedClassifier(triggerLabels,argumentLabels,Features.myTriggerFeatures,Features.myArgumentFeatures)
//    val jointModel = SimpleJointConstrainedClassifier(triggerLabels,argumentLabels,Features.myTriggerFeatures,Features.myArgumentFeatures)

    // use training algorithm to get weights of model
    val jointWeights = Problem1.trainPerceptron(jointTrain,jointModel.feat,jointModel.predict,10)

    println("get prediction on dev")
    // get predictions on dev
    val jointDevPred = jointDev.unzip._1.map { case e => jointModel.predict(e,jointWeights) }
    val jointDevGold = jointDev.unzip._2

    // Triggers (dev)
    val triggerDevPred = jointDevPred.unzip._1
    val triggerDevGold: Seq[String] = jointDevGold.unzip._1
    val triggerDevEval = Evaluation(triggerDevGold, triggerDevPred, Set("None"))
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


case class SimpleJointConstrainedClassifier(triggerLabels: Set[Label],
                                            argumentLabels: Set[Label],
                                            triggerFeature: (Candidate, Label) => FeatureVector,
                                            argumentFeature: (Candidate, Label) => FeatureVector
                                             ) extends JointModel {
  def predict(x: Candidate, weights: Weights) = {
    def argmax(labels: Set[Label], x: Candidate, weights: Weights, feat: (Candidate, Label) => FeatureVector) = {
      val scores = labels.toSeq.map(y => y -> dot(feat(x, y), weights)).toMap withDefaultValue 0.0
      scores.maxBy(_._2)._1
    }

    val bestTrigger = argmax(triggerLabels, x, weights, triggerFeature)

    var bestArguments: Seq[Label] = List()

    // Constraint 1 :  A trigger can only have arguments if its own label is not NONE
    if (bestTrigger.toString == "None") {
      bestArguments = for (arg <- x.arguments) yield "None"
    } else {
      // Constraint 3: Only regulation events can have CAUSE arguments
      if (bestTrigger.toString.contains("egulation")) {
        bestArguments = for (arg <- x.arguments) yield argmax(argumentLabels, arg, weights, argumentFeature)
      } else {
        bestArguments = for (arg <- x.arguments) yield argmax(argumentLabels.filter(_.toString != "Cause"), arg, weights, argumentFeature)
      }
      // Constraint 2: A trigger with a label other than NONE must have at least one THEME
      // check count of label "Theme"
      if (bestArguments.count(_.toString == "Theme") == 0) {
        // make map of scores of arguments being "Theme"
        val themeScores = x.arguments.map(x => x -> dot(argumentFeature(x, "Theme"), weights)).toMap withDefaultValue 0.0
        var count = 0

        for (arg <- x.arguments) {
          if (arg == themeScores.maxBy(_._2)._1) {
            bestArguments = bestArguments.updated(count, "Theme")
          }
          count = count + 1
        }
      }
    }
    (bestTrigger, bestArguments)
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

case class JointConstrainedClassifier(triggerLabels: Set[Label],
                                      argumentLabels: Set[Label],
                                      triggerFeature: (Candidate, Label) => FeatureVector,
                                      argumentFeature: (Candidate, Label) => FeatureVector
                                       ) extends JointModel {
  def predict(x: Candidate, weights: Weights) = {
    // gets best trigger labels for each category
    def getBestTriggerLabels(labels: Set[Label], x: Candidate, weights: Weights, feat: (Candidate, Label) => FeatureVector) = {
      val scores = labels.toSeq.map(y => y -> dot(feat(x, y), weights)).toMap withDefaultValue 0.0
      val sortedSc = ListMap(scores.toSeq.sortBy(_._2): _*)

//      the best with features 1,2
      val bestFromTriggerCategList = mutable.MutableList[(Label, Double)]()
      for (i <- 1 to 3) {
        val bestListForCateg = sortedSc.filter(label => triggerCategory(label._1) == i)
        if (bestListForCateg.size > 0) {
          bestFromTriggerCategList += bestListForCateg.last
        }
      }
      bestFromTriggerCategList
    }

    // there are 3 categories: I -> None, II -> Regulations, III -> Rest of labels for triggers
    def triggerCategory(label: Label): Int = label match {
      case "None" => 1
      case "Regulation" | "Positive_regulation" | "Negative_regulation" => 2
      case _ => 3
    }

    def validLabels(category: Int): Seq[String] = category match {
      case 1 => List("None")
      case 2 => List("None", "Theme", "Cause") // has to have at least one theme and the only one which can have cause
      case 3 => List("None", "Theme") // has to have at least one theme
    }

    val bestTriggers = getBestTriggerLabels(triggerLabels, x, weights, triggerFeature)
    val trigCategMap = bestTriggers.toSeq.map(trig => trig -> triggerCategory(trig._1)).toMap // to use

    val uniqueCategories = trigCategMap.toSeq.map(trig => trig._2).toSet // we can add default category here e.g. None + 1
    val validArgLabels = uniqueCategories.map(cat => cat -> validLabels(cat)).toMap

    val catValidScorePath = validArgLabels.map(argLabelType => {
      if (argLabelType._1 != 1) {
        val validScorePaths = mutable.MutableList[(List[(String, Int)], Double)]()
        scorePathsInPlace(x.arguments.toIndexedSeq, weights,argumentFeature, (List(), 0.0), validScorePaths, hasTheme = false, 0, argLabelType._2.toIndexedSeq)
        val bestValidScorePath = validScorePaths.maxBy(_._2)
        ((argLabelType._1, bestValidScorePath._1), bestValidScorePath._2)
      } else {
        var score = 0.0
        var path = List[(String, Int)]()
        for (i <- 0 until x.arguments.size) {
          score += dot(argumentFeature(x.arguments(i), "None"), weights)
          path = ("None", i) :: path
        }
        ((argLabelType._1, path), score)
      }
    })
    val bestOverall = trigCategMap.map(trigger => {
       val triggerScore = trigger._1._2
       val triggerType = trigger._2
       val bestArgsForTrigger = catValidScorePath.filter(_._1._1==triggerType).maxBy(_._2)
       val totalScore = triggerScore + bestArgsForTrigger._2
       ((trigger._1._1, bestArgsForTrigger._1._2), totalScore)
    })
    val best = bestOverall.maxBy(_._2)
    (best._1._1, best._1._2.sortBy(_._2).map(pathitem => pathitem._1))
  }


  def scorePathsInPlace(args: IndexedSeq[Candidate], weights: Weights, feat:(Candidate,Label)=>FeatureVector, pathToScoreMap:(List[(String, Int)],Double), finalValidScorePath:mutable.MutableList[(List[(String, Int)],Double)],
                       hasTheme: Boolean, curArgInd:Int, possibleLabels: IndexedSeq[Label]):Unit = {
    if (curArgInd<args.size) {
      val labelCandMap = possibleLabels.map(curLabel => {
        val newScore = pathToScoreMap._2 + dot(feat(args(curArgInd), curLabel), weights)
        val newPath = (curLabel, curArgInd) :: pathToScoreMap._1
        ((newPath, newScore), curLabel.contains("Theme") || hasTheme)
      })
      // Main change is that we are not interested in arguments that are not max
      // because arguments are independent, and there is no way those which are not max, won't ever be max
      // => hence not that wxhaustive

      // always continue with max valid candidate
      val maxValidCand = labelCandMap.filter(_._2).maxBy(_._1._2)
      val maxCand = labelCandMap.maxBy(_._1._2)

      scorePathsInPlace(args, weights, feat, maxValidCand._1, finalValidScorePath, maxValidCand._2, curArgInd + 1, possibleLabels)

      // if not valid path has bigger score, it might be valid in future, so continue this path
      if (maxCand._1._2 > maxValidCand._1._2) {
        scorePathsInPlace(args, weights, feat, maxCand._1, finalValidScorePath, maxCand._2, curArgInd + 1, possibleLabels)
      }
    } else {
      //base case
      if (hasTheme) finalValidScorePath += pathToScoreMap
    }
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


