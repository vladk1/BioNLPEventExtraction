package uk.ac.ucl.cs.mr.statnlpbook.assignment2

/**
 * Created by Georgios on 06/11/2015.
 */
object Problem3Triggers {

  def main(args: Array[String]) {
    println("Trigger Extraction")
    val train_dir = "./data/assignment2/bionlp/train"
    val test_dir = "./data/assignment2/bionlp/test"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir, 0.8, 500)
    // load test
    val testDocs = BioNLP.getTestDocuments(test_dir)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> e.gold)

    // ================= Trigger Classification =================

    // get candidates and make tuples with gold
    // read the specifications of the method for different subsampling thresholds
    // no subsampling for dev/test!
    def getTriggerCandidates(docs: Seq[Document]) = docs.flatMap(_.triggerCandidates(0.02))
    def getTestTriggerCandidates(docs: Seq[Document]) = docs.flatMap(_.triggerCandidates())
    val triggerTrain = preprocess(getTriggerCandidates(trainDocs))
    val triggerDev = preprocess(getTestTriggerCandidates(devDocs))
    val triggerTest = preprocess(getTestTriggerCandidates(testDocs))

    // show statistics for counts of true labels, useful for deciding on subsampling
    println("True label counts (trigger - train):")
    println(triggerTrain.unzip._2.groupBy(x => x).mapValues(_.length))
    println("True label counts (trigger - dev):")
    println(triggerDev.unzip._2.groupBy(x => x).mapValues(_.length))

    // get label set
    val triggerLabels = triggerTrain.map(_._2).toSet

    // define model
    //TODO: change the features function to explore different types of features
    val triggerModel = SimpleClassifier(triggerLabels, Features.defaultTriggerFeatures)

    // use training algorithm to get weights of model
    //TODO: change the trainer to explore different training algorithms
    //val triggerWeights = PrecompiledTrainers.trainNB(triggerTrain,triggerModel.feat)
    val triggerWeights = PrecompiledTrainers.trainPerceptron(triggerTrain, triggerModel.feat, triggerModel.predict, 2)

    // get predictions on dev
    val (triggerDevPred, triggerDevGold) = triggerDev.map { case (trigger, gold) => (triggerModel.predict(trigger, triggerWeights), gold) }.unzip
    // evaluate on dev
    val triggerDevEval = Evaluation(triggerDevGold, triggerDevPred, Set("None"))
    // print evaluation results
    println("Evaluation for trigger classification:")
    println(triggerDevEval.toString)

    ErrorAnalysis(triggerDev.unzip._1,triggerDevGold,triggerDevPred).showErrors(5)

    // get predictions on test
    val triggerTestPred = triggerTest.map { case (trigger, dummy) => triggerModel.predict(trigger, triggerWeights) }
    // write to file
    Evaluation.toFile(triggerTestPred, "./data/assignment2/out/simple_trigger_test.txt")
  }
}

object Problem3Arguments {
  def main (args: Array[String] ) {
    println("Arguments Extraction")
    val train_dir = "./data/assignment2/bionlp/train"
    val test_dir = "./data/assignment2/bionlp/test"

    // load train and dev data
    // read the specification of the method to load more/less data for debugging speedup
    val (trainDocs, devDocs) = BioNLP.getTrainDevDocuments(train_dir,0.8,500)
    // load test
    val testDocs = BioNLP.getTestDocuments(test_dir)
    // make tuples (Candidate,Gold)
    def preprocess(candidates: Seq[Candidate]) = candidates.map(e => e -> e.gold)

    // ================= Argument Classification =================

    // get candidates and make tuples with gold
    // no subsampling for dev/test!
    def getArgumentCandidates(docs:Seq[Document]) = docs.flatMap(_.argumentCandidates(0.008))
    def getTestArgumentCandidates(docs:Seq[Document]) = docs.flatMap(_.argumentCandidates())
    val argumentTrain =  preprocess(getArgumentCandidates(trainDocs))
    val argumentDev = preprocess(getTestArgumentCandidates(devDocs))
    val argumentTest = preprocess(getTestArgumentCandidates(testDocs))

    // show statistics for counts of true labels, useful for deciding on subsampling
    println("True label counts (argument - train):")
    println(argumentTrain.unzip._2.groupBy(x=>x).mapValues(_.length))
    println("True label counts (argument - dev):")
    println(argumentDev.unzip._2.groupBy(x=>x).mapValues(_.length))

    // get label set
    val argumentLabels = argumentTrain.map(_._2).toSet

    // define model
    val argumentModel = SimpleClassifier(argumentLabels, Features.defaultArgumentFeatures)

    //val argumentWeights = PrecompiledTrainers.trainNB(argumentTrain,argumentModel.feat)
    val argumentWeights = PrecompiledTrainers.trainPerceptron(argumentTrain,argumentModel.feat,argumentModel.predict,2)

    // get predictions on dev
    val (argumentDevPred, argumentDevGold) = argumentDev.map { case (arg, gold) => (argumentModel.predict(arg,argumentWeights), gold) }.unzip
    // evaluate on dev
    val argumentDevEval = Evaluation(argumentDevGold, argumentDevPred, Set("None"))
    println("Evaluation for argument classification:")
    println(argumentDevEval.toString)


    ErrorAnalysis(argumentDev.unzip._1,argumentDevGold,argumentDevPred).showErrors(5)

    // get predictions on test
    val argumentTestPred = argumentTest.map { case (arg, dummy) => argumentModel.predict(arg,argumentWeights) }
    // write to file
    Evaluation.toFile(argumentTestPred,"./data/assignment2/out/simple_argument_test.txt")
  }

}