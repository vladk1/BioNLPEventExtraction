package uk.ac.ucl.cs.mr.statnlpbook.assignment2

/**
 * Created by Georgios on 10/11/2015.
 */

trait Model[X,Y]{
  def predict(x:X,weights:Weights):Y
  def feat(x:X,y:Y):FeatureVector
}

/**
 * A baseline classifier that uniformly returns a random label.
 * It is not possible to be worse that this model!
 * @param labels
 * @param features
 */
case class RandomSimpleClassifier(labels:Set[Label],features:(Candidate,Label)=>FeatureVector) extends Model[Candidate,Label]{
  def predict(x: Candidate, weights: Weights) = {
    val scored = labels.map(y => y -> math.random)
    scored.maxBy(_._2)._1
  }
  def feat(x: Candidate, y: Label): FeatureVector =  features(x,y)
}

/**
 * A simple classifier (not joint) that predicts the highest scoring label.
 * The classification weights must be learned, e.g. from a Naive Bayes or Log Linear trainer.
 * @param labels: the possible labels (for triggers or arguments)
 * @param features a feature function f(candidate,label)
 */
case class SimpleClassifier(labels:Set[Label],features:(Candidate,Label)=>FeatureVector) extends Model[Candidate,Label]{
  //the predict function returns the highest scoring label.
  def predict(x: Candidate, weights: Weights) = {
    val scored = labels.map(y => y -> dot(feat(x, y), weights))
    scored.maxBy(_._2)._1
  }
  def feat(x: Candidate, y: Label): FeatureVector =  features(x,y)
}
